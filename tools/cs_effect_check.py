#!/usr/bin/env python3
"""cs_effect_check.py - the closed loop's code adapter for C#: each requirement's EFFECT held to the code, and to the TESTS.

WHY (2026-10-29). The loop's adapter until
now was the Blueprint view's (its logic_map.py EffectCheck), and it checks SHAPE: an assertion holds
when a statement at its place reads like it (`sets X`, `calls F(args)`, `branches on X`), and its plant (L8) mutates that
statement's TEXT and requires the matcher to stop matching. That proves the check can fail. It does not prove the code is
right - a statement that pays the wrong amount, or pays under the wrong condition, has the right shape. This adapter does both:

  SHAPE   the same verdict over C#: the place is `Type.Method`; its body is cut into statements; holds / plant as EffectCheck.
  KILL    the behavioural verdict (usecase_kg.py loop L14 KILLED): the statement the assertion matched is MUTATED in the real
          source - a branch's condition negated, an assignment's value replaced by `default`, a call statement removed - the
          mutant compiled in an overlay (never the live project), and the model's tests run against it. The requirement's code
          is falsified only when a test FAILS. A mutant that does not compile is not a verdict (the next mutant is tried); a
          place no runner can execute (view code needing UnityEngine.UI/TMP) is said so, never counted as killed.

THE MODEL'S loop block names what to run:
  meta.loop: {closed: true, entry: [...],
              code: {roots: [<dir or .cs> ...]},                       where `Type.Method` places are looked up
              tests: [{runner: headless_nunit, test: <Editor test .cs>, covers: [<dir or .cs> ...]}]}
  An assertion: {sets|calls|branches|reads: <name>, to, args, by, at: <Type.Method>}

    python -X utf8 cs_effect_check.py <model folder> --out <verdicts.json> [--no-kill] [--only REQ-ID]
    python -X utf8 cs_effect_check.py --selftest

Output: effect-verdicts/1 (the format usecase_kg.py loop reads) - assertions {holds, plant, fp, why, matched, kill}, names, says.
"""
import argparse
import collections
import glob
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
# the project the model's paths are relative to: this repo, or a developer's own (CLOSED_LOOP_ROOT, set by the kit's installer)
ROOT = os.path.abspath(os.environ.get("CLOSED_LOOP_ROOT") or os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import usecase_model as ucm  # noqa: E402  the one loader's Assertion: an effect is parsed once, the same everywhere

HEADLESS = os.path.join(ROOT, ".opus-tools", "headless_nunit", "headless_nunit.py")
# THIS repo's settings (site.json beside the tools; never shipped): the Unity project folder. Without one there is no Unity runner
# and paths are the model's own, relative to ROOT - the kit's case
SITE = json.load(io.open(os.path.join(HERE, "site.json"), encoding="utf-8")) if os.path.exists(os.path.join(HERE, "site.json")) else {}
UNITY_DIR = SITE.get("unity_project") or os.environ.get("CLOSED_LOOP_UNITY_PROJECT") or ""
UNITY_PROJECT = os.path.join(ROOT, UNITY_DIR) if UNITY_DIR else ROOT
# A SANDBOX (10-29, the on/off comparison): an overlay laid out as the Unity project (DIR/Assets/...) holding one agent's copy of
# the model's code and tests. Set, every `<project>/...` path the model names is read from it when it holds the file, and the mutants
# are compiled over it - so two agents can each change the code and run the whole loop while the live project is never touched.
OVERLAY = None


def src_path(rel):
    """a model path (`<project>/Assets/...`, or absolute) as the sandbox holds it, else as the live project does - a lookup order,
    not a fallback: the sandbox only replaces what it holds, exactly as headless_nunit --overlay does"""
    if os.path.isabs(rel):
        return os.path.normpath(rel)
    rel = rel.replace("\\", "/")
    head = UNITY_DIR + "/"
    if OVERLAY and UNITY_DIR and rel.startswith(head) and os.path.exists(os.path.join(OVERLAY, rel[len(head):])):
        return os.path.normpath(os.path.join(OVERLAY, rel[len(head):]))
    return os.path.normpath(os.path.join(ROOT, rel))


def shown(p):
    """a path as the reports show it: relative to the repo when it is on the repo's drive, else whole - os.path.relpath raises
    across Windows drives (the A/B ON arm of R1, 10-29: a sandbox on C: against the tool on D: crashed three reports)"""
    try:
        return os.path.relpath(p, ROOT).replace("\\", "/")
    except ValueError:
        return os.path.normpath(p).replace("\\", "/")


def proj_rel(f):
    """a file's path inside the Unity project (Assets/...), from the sandbox or the live project"""
    if OVERLAY and os.path.normpath(f).startswith(os.path.normpath(OVERLAY) + os.sep):
        return os.path.relpath(f, OVERLAY)
    return os.path.relpath(f, UNITY_PROJECT)
WORD = re.compile(r"[A-Za-z_]\w*")
SHAPED = re.compile(r"`([A-Za-z_]\w*)`|\b([A-Za-z]\w*?(?:[a-z0-9][A-Z]|_[A-Za-z0-9])\w*)\b")
BRANCH_KW = ("if", "else if", "while", "for", "foreach", "switch", "case", "do")
ASSIGN = re.compile(r"^(?P<lhs>[^=!<>]*?[\w\]\)])\s*(?P<op>[-+*/%|&^]?=|\?\?=)(?!=|>)\s*(?P<rhs>.+)$", re.S)


IL_READER = os.path.join(HERE, "il_bodies")


def il_reader():
    """[dotnet, il_bodies.dll]: the IL reader (il_bodies/, C#), built once per version of its source by the dotnet the C# adapter
    already needs, in a temp folder of its own (its build output never lands beside the tools); None when there is no dotnet.
    KT-9, 10-29: it was a PowerShell 7 script, and without pwsh the kit's own C# example could not hold - on Windows too, which
    ships PowerShell 5.1. A build that fails raises: an equivalence proof that silently stopped working would count survivors as
    holes or, worse, the other way round."""
    dotnet = shutil.which("dotnet")
    if not dotnet:
        return None
    src = "".join(io.open(os.path.join(IL_READER, f), encoding="utf-8").read() for f in ("il_bodies.csproj", "Program.cs"))
    home = os.path.join(tempfile.gettempdir(), "closed_loop_il_bodies", hashlib.sha1(src.encode("utf-8")).hexdigest()[:12])
    dll = os.path.join(home, "bin", "il_bodies.dll")
    if not os.path.exists(dll):
        os.makedirs(os.path.join(home, "src"), exist_ok=True)
        for f in ("il_bodies.csproj", "Program.cs"):
            shutil.copyfile(os.path.join(IL_READER, f), os.path.join(home, "src", f))
        r = subprocess.run([dotnet, "build", os.path.join(home, "src", "il_bodies.csproj"), "-o", os.path.join(home, "bin"), "-nologo", "-v", "q"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0 or not os.path.exists(dll):
            raise RuntimeError("the IL reader (il_bodies/) did not build: %s" % (r.stdout + r.stderr)[-800:])
    return [dotnet, dll]


def sha(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


# ── reading C# well enough to find methods and statements ─────────────────────────────────────────

def blank_out(src):
    """the source with every comment and string/char literal's CONTENT replaced by spaces, same length - so braces, semicolons and
    names inside them are never read as code. Verbatim (@"..") and interpolated ($"..") strings included."""
    out, i, n = list(src), 0, len(src)
    while i < n:
        c, nx = src[i], src[i + 1] if i + 1 < n else ""
        if c == "/" and nx == "/":
            j = src.find("\n", i)
            j = n if j < 0 else j
            out[i:j] = " " * (j - i)
            i = j
        elif c == "/" and nx == "*":
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out[i:j] = [ch if ch == "\n" else " " for ch in src[i:j]]
            i = j
        elif c == '"' or (c in "@$" and nx == '"') or (c in "@$" and nx in "@$" and i + 2 < n and src[i + 2] == '"'):
            k = i
            while src[k] in "@$":
                k += 1
            verbatim = "@" in src[i:k]
            j = k + 1
            while j < n:
                if verbatim and src[j] == '"' and j + 1 < n and src[j + 1] == '"':
                    j += 2
                    continue
                if not verbatim and src[j] == "\\":
                    j += 2
                    continue
                if src[j] == '"':
                    break
                j += 1
            out[k + 1:j] = " " * (j - k - 1)
            i = j + 1
        elif c == "'":
            j = i + 1
            while j < n and src[j] != "'":
                j += 2 if src[j] == "\\" else 1
            out[i + 1:j] = " " * (j - i - 1)
            i = j + 1
        else:
            i += 1
    return "".join(out)


def match_brace(code, open_at):
    depth = 0
    for j in range(open_at, len(code)):
        if code[j] == "{":
            depth += 1
        elif code[j] == "}":
            depth -= 1
            if depth == 0:
                return j
    return -1


class Statement(object):
    __slots__ = ("file", "start", "end", "text", "kind")

    def __init__(self, file, start, end, text, kind):
        self.file, self.start, self.end, self.text, self.kind = file, start, end, text, kind


class CsCode(object):
    """the C# files under the model's code roots, with `Type.Method` -> its statements"""

    def __init__(self, roots):
        self.files = {}
        for r in roots:
            p = src_path(r)
            paths = [p] if p.endswith(".cs") else [os.path.join(d, f) for d, _, fs in os.walk(p) for f in fs if f.endswith(".cs")]
            for f in paths:
                self.files[os.path.normpath(f)] = io.open(f, encoding="utf-8-sig").read()
        if not self.files:
            raise RuntimeError("the model's code roots hold no .cs file: %s" % roots)
        self.blank = {f: blank_out(s) for f, s in self.files.items()}
        self.words = set()
        for b in self.blank.values():
            self.words.update(WORD.findall(b))

    def _locate(self, place):
        """(file, name_at, generics, paren_open, paren_close, body_at, body kind) of `Type.Method`'s one declaration"""
        if "." not in place:
            raise KeyError("a C# place is Type.Method: %r" % place)
        tname, mname = place.rsplit(".", 1)
        mname, _, which = mname.partition("~")
        pattern = r"\b%s\s*(<[^>]*>)?\s*\(" % re.escape(mname)
        if mname.startswith("operator") and mname != "operator":
            pattern = r"\boperator\s*%s()\s*\(" % re.escape(mname[len("operator"):])
        found = []
        for f, b in self.blank.items():
            for tm in re.finditer(r"\b(?:class|struct|record|interface)\s+%s\b[^{;]*\{" % re.escape(tname.split(".")[-1]), b):
                t_open = tm.end() - 1
                t_close = match_brace(b, t_open)
                for mm in re.finditer(pattern, b[t_open:t_close]):
                    at = t_open + mm.end() - 1
                    # the parameter list, then a body: '{' or '=>'
                    depth, j = 0, at
                    while j < t_close:
                        depth += b[j] == "("
                        depth -= b[j] == ")"
                        if depth == 0:
                            break
                        j += 1
                    rest = b[j + 1:t_close]
                    head = re.match(r"\s*(?:where\b[^{=]*)?(\{|=>)", rest)
                    if not head:
                        continue
                    pre = b[max(0, t_open + mm.start() - 120):t_open + mm.start()]
                    if re.search(r"(?:\bnew|\.|=|\breturn)\s*$", pre):
                        continue                                    # a call or a construction, not the declaration
                    body_at = j + 1 + head.end() - 1
                    found.append((f, t_open + mm.start(), mm.group(1) or "", at, j, body_at, head.group(1)))
        if not found:
            for f, b in self.blank.items():
                for tm in re.finditer(r"\b(?:class|struct|record|interface)\s+%s\b[^{;]*\{" % re.escape(tname.split(".")[-1]), b):
                    for name, at, kind, body_at in self._properties(b, tm.end() - 1, match_brace(b, tm.end() - 1)):
                        if name == mname:
                            found.append((f, at, "", at, at, body_at, kind))
        if not found:
            raise KeyError("no method %s in the code roots" % place)
        if which:
            if not 1 <= int(which) <= len(found):
                raise KeyError("no overload %s of %s: it has %d" % (which, place, len(found)))
            return found[int(which) - 1]
        if len(found) > 1:
            raise KeyError("%d methods named %s in the code roots (overloads): name one with ~N, in source order" % (len(found), place))
        return found[0]

    @staticmethod
    def _members(b, t_open, t_close):
        """(start, text before it in this member) of each position at the type's own level: a nested body or a parameter list is skipped"""
        depth, paren, start = 0, 0, t_open + 1
        for i in range(t_open + 1, t_close):
            c = b[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    start = i + 1
            elif c == "(":
                paren += 1
            elif c == ")":
                paren -= 1
            elif c == ";" and depth == 0 and paren == 0:
                start = i + 1
            if depth == 0 and paren == 0:
                yield i, start

    def _properties(self, b, t_open, t_close):
        """(name, name_at, kind, body_at) of each property WITH a body declared directly in this type: `T Name => expr;` or `T Name { get { ... } }`.
        An auto-property holds no statement. body_at is what statements() reads: the '>' of the arrow, or the '{' of the accessors."""
        out = []
        for i, start in self._members(b, t_open, t_close):
            m = re.compile(r"\b(\w+)\s*(=>|\{\s*(?:(?:public|private|protected|internal)\s+)*(?:get|set|init)\b)").match(b, i)
            if not m or (i > 0 and (b[i - 1].isalnum() or b[i - 1] == "_")):
                continue
            head = b[start:i]
            if "=" in head or "(" in head or not re.search(r"[\w>\]\?]\s+$", head):
                continue                                     # a lambda in a field's value, or no type before the name
            if m.group(2) == "=>":
                out.append((m.group(1), i, "=>", m.end() - 1))
            else:
                brace = b.index("{", i)
                if re.search(r"\b(?:get|set|init)\s*(?:\{|=>)", b[brace:match_brace(b, brace)]):
                    out.append((m.group(1), i, "{", brace))
        return out

    def not_walked(self, files):
        """what holds code the walk does not reach: a field's value and an indexer, by name. Said in the report, so 'every statement' stays true"""
        out = []
        for f in sorted(files):
            b = self.blank[f]
            for tm in re.finditer(r"\b(?:class|struct|record)\s+(\w+)\b[^{;]*\{", b):
                close = match_brace(b, tm.end() - 1)
                seen_starts = set()
                for i, start in self._members(b, tm.end() - 1, close):
                    if b[i] == "=" and b[i + 1:i + 2] not in ("=", ">") and start not in seen_starts:
                        seen_starts.add(start)
                        names = re.findall(r"\w+", b[start:i])
                        if names and "operator" not in names and "(" in b[i:b.find(";", i)]:
                            out.append("%s.%s (a field's value)" % (tm.group(1), names[-1]))
                    if b.startswith("this", i) and b[i + 4:].lstrip().startswith("["):
                        out.append("%s.this[] (an indexer)" % tm.group(1))
        return out

    def overloads(self, place):
        """how many declarations `Type.Method` has: 0 none, 1 one, more an overload set (each its own place, Type.Method~N)"""
        n = 0
        while True:
            try:
                self._locate("%s~%d" % (place, n + 1))
            except KeyError:
                return n
            n += 1

    def method(self, place):
        """the statements of `Type.Method` (a constructor is `Type.Type`); KeyError when no type or method has the name, or two do"""
        f, _, _, _, _, at, kind = self._locate(place)
        return self.statements(f, at, kind)

    def decl(self, place):
        """the MethodDecl of `Type.Method`: where it sits, enough to copy it and call the copy"""
        return MethodDecl(self, place, *self._locate(place))

    def places(self, files):
        """every `Type.Method` declared in these files, in source order - the units the branch space and the mutants walk"""
        out = []
        for f in sorted(files):
            b = self.blank[f]
            for tm in re.finditer(r"\b(?:class|struct|record)\s+(\w+)\b[^{;]*\{", b):
                close = match_brace(b, tm.end() - 1)
                for name, at, kind, body_at in self._properties(b, tm.end() - 1, close):
                    q = "%s.%s" % (tm.group(1), name)
                    if q not in out:
                        out.append(q)
                for mm in re.finditer(r"\b(\w+)\s*(?:<[^>]*>)?\s*\([^;{]*\)\s*(?:where\b[^{=;]*)?(\{|=>)", b[tm.end():close]):
                    name = mm.group(1)
                    if name in ("if", "while", "for", "foreach", "switch", "using", "lock", "catch", "return", "new", "nameof", "typeof"):
                        continue
                    place = "%s.%s" % (tm.group(1), name)
                    count = self.overloads(place)
                    each = [] if count == 0 else [place] if count == 1 else ["%s~%d" % (place, i + 1) for i in range(count)]
                    for q in each:
                        if q not in out:
                            out.append(q)
                for mm in re.finditer(r"\boperator\s*([^\s(\w]+)\s*\(", b[tm.end():close]):
                    q = "%s.operator%s" % (tm.group(1), mm.group(1))
                    if q not in out:
                        out.append(q)
        return out

    def branches(self, files):
        """[(key, place, statement)] - every DECISION of every method in these files: a branch, and since REVIEW_1 #4 (10-29) any
        statement that compares, joins conditions, picks by a ternary, returns, or is a case or an else-arm (an expression body
        like CanPay's `price >= 0 && Money >= price` decides as much as an `if` does). The key `Type.Method#bK` (K the statement's
        index in its method) is the branch space's cell, the same in closed_loop.py and here."""
        out, seen = [], set()
        for place in self.places(files):
            for k, st in enumerate(self.method(place)):
                # a nested type's method is found under its outer type's name too: one statement, one cell
                # EVERY statement (REVIEW_2 #1, 10-29): `p.Rank++;` decides nothing yet changes what the game does - a
                # decision-only space let it in unclaimed and unfingerprinted
                if st.text.strip() and (st.file, st.start) not in seen:
                    seen.add((st.file, st.start))
                    out.append(("%s#b%d" % (place, k), place, st))
        return out

    def statements(self, f, at, kind):
        b, src = self.blank[f], self.files[f]
        if kind == "=>":
            end = b.find(";", at)
            s0 = at + 2 + (len(b[at + 2:end]) - len(b[at + 2:end].lstrip()))
            return [Statement(f, s0, end, src[s0:end].rstrip(), "expr")]   # an expression body: the member's whole value
        close = match_brace(b, at)
        out, i, start = [], at + 1, at + 1
        while i < close:
            c = b[i]
            if c in ";{}":
                seg = b[start:i].strip()
                if seg and not re.fullmatch(r"(?:(?:public|private|protected|internal)\s+)?(?:get|set|init)", seg):
                    s0 = start + (len(b[start:i]) - len(b[start:i].lstrip()))
                    s1 = s0 + len(seg)
                    out.append(Statement(f, s0, s1, src[s0:s1], self.kind(seg)))
                start = i + 1
            i += 1
        return out

    @staticmethod
    def kind(seg):
        s = re.sub(r"\s+", " ", seg)
        for kw in BRANCH_KW:
            if re.match(r"%s\b" % kw, s):
                return "branch"
        if re.match(r"return\b", s):
            return "return"
        if ASSIGN.match(s):
            return "set"
        if "(" in s:
            return "call"
        return "other"


# ── the verdicts ─────────────────────────────────────────────────────────────────────────────────

def lhs_names(text):
    m = ASSIGN.match(re.sub(r"\s+", " ", text.strip()))
    if not m:
        return None, None, None
    lhs = m.group("lhs").strip()
    lhs = re.sub(r"^(?:var|[\w<>\[\],.?]+)\s+(?=\w+$)", "", lhs)     # a declaration's type
    return lhs, m.group("op"), m.group("rhs").strip()


def norm(v):
    v = re.sub(r"(\d)f\b", r"\1", str(v).strip().lower())
    return {"none": "nothing", "null": "nothing"}.get(v, v)


def matches(a, text, kind):
    word = re.compile(r"(?<!\w)%s(?!\w)" % re.escape(a.target))
    if a.verb == "sets" and a.by is not None:
        m = re.search(r"(?<!\w)%s\((.*)\)" % re.escape(str(a.by)), text, re.S)
        return m is not None and word.search(m.group(1)) is not None and (a.to is None or str(a.to) in m.group(1))
    if a.verb == "sets":
        lhs, op, rhs = lhs_names(text)
        return lhs is not None and (lhs == a.target or lhs.endswith("." + a.target)) and (a.to is None or (op == "=" and norm(rhs) == norm(a.to)))
    if a.verb == "calls":
        return re.search(r"(?<![\w])%s\s*(?:<[^>]*>)?\(" % re.escape(a.target), text) is not None and all(x in text for x in a.args)
    if a.verb == "branches":
        return kind == "branch" and word.search(text) is not None
    if a.verb == "reads":
        lhs, op, rhs = lhs_names(text)
        return word.search(rhs if rhs is not None else text) is not None
    return False


def text_mutants(a, text):
    out = [("the target renamed", re.sub(r"(?<!\w)%s(?!\w)" % re.escape(a.target), a.target + "Planted", text))]
    if a.by is not None:
        out.append(("another call", re.sub(r"(?<!\w)%s\(" % re.escape(str(a.by)), "PlantedCall(", text)))
    elif a.verb == "sets" and a.to is not None:
        lhs, op, _ = lhs_names(text)
        out.append(("the value changed", "%s %s planted_value" % (lhs, op)))
    for x in a.args:
        out.append(("argument %s dropped" % x, text.replace(x, "planted")))
    return out


def wanted(a):
    if a.by is not None:
        return "%s(%s%s)" % (a.by, a.target, "" if a.to is None else ", %s" % a.to), "call"
    return {"sets": ("%s = %s" % (a.target, "x" if a.to is None else a.to), "set"),
            "calls": ("%s(%s)" % (a.target, ", ".join(a.args)), "call"),
            "branches": ("if (%s)" % a.target, "branch"),
            "reads": ("Use(%s)" % a.target, "call")}[a.verb]


def code_mutants(a, st):
    """the matched statement changed in BEHAVIOUR, as source text replacing st.text - in the order they are tried"""
    t = st.text
    out = []
    if st.kind == "branch" or re.search(r"\b(?:if|while)\s*\(", t):
        m = re.search(r"(\b(?:if|while)\s*)\(", t)
        if m:
            depth, j = 0, m.end() - 1
            for j in range(m.end() - 1, len(t)):
                depth += t[j] == "("
                depth -= t[j] == ")"
                if depth == 0:
                    break
            out.append(("the condition negated", t[:m.end()] + "!(" + t[m.end():j] + ")" + t[j:]))
            # VACUITY (prior art: Beer et al.; Kupferman): the guard forced either way - a check that passes with its guard always
            # true or always false never looked at the guard. `out` variables keep their assignment: the condition still runs
            if not re.search(r"\bout\s", t[m.end():j]):
                out.append(("the condition made true", t[:m.end()] + "true" + t[j:]))
                out.append(("the condition made false", t[:m.end()] + "false" + t[j:]))
    if a.verb in ("sets", "reads") or st.kind == "set":
        lhs, op, rhs = lhs_names(t)
        if lhs is not None and rhs:
            decl = t.split("=")[0].strip()
            # a compound assignment flips its operator: `Money -= default` does not compile (CS8310, MEASURED 10-29)
            if op in ("+=", "-="):
                out.append(("the operator flipped", t.replace(op, "-=" if op == "+=" else "+=", 1)))
            elif not re.match(r"var\s", decl):
                out.append(("the value replaced by default", t[:len(t) - len(t.lstrip())] + t.strip()[:t.strip().rfind(rhs)] + "default"))
    # the assignment a branch makes: `if (... out int n) cost.Count = n` - negating that condition leaves n unassigned (CS0165,
    # MEASURED 10-29), so its body's value is what changes
    if st.kind == "branch":
        m = re.search(r"\)\s*([\w.\[\]]+)\s*=(?![=>])\s*([^;=][^;]*)$", t)
        if m:
            out.append(("the branch's assignment replaced by default", t[:m.start(2)] + "default"))
    if st.kind == "call" and a.verb in ("calls", "sets"):
        out.append(("the call removed", "/* planted: removed */"))
    if st.kind == "return":
        out.append(("the result replaced by default", "return default"))
    if st.kind == "expr":
        out.append(("the value replaced by default", "default"))
    return out


# ── EVERY mutant of a statement, not the first that compiles (10-29) ────────
# The first upgrade broke each effect ONE way and stopped at the first test to notice. A test that notices the condition flipped
# can still miss `<=` written as `<` - the boundary is where a quantity of 1 sells once or twice. So a statement is broken every
# way these operators can (the classic mutation operators), each mutant judged alone, and ONE survivor is a hole.

SPACED_OP = re.compile(r"(?<=\s)(<=|>=|==|!=|&&|\|\||<|>|\+|-|\*|/)(?=\s)")
SWAP = {"<=": "<", ">=": ">", "<": "<=", ">": ">=", "==": "!=", "!=": "==", "&&": "||", "||": "&&", "+": "-", "-": "+", "*": "/", "/": "*"}
OP_KIND = {"<=": "boundary", ">=": "boundary", "<": "boundary", ">": "boundary", "==": "equality", "!=": "equality",
           "&&": "logic", "||": "logic", "+": "arithmetic", "-": "arithmetic", "*": "arithmetic", "/": "arithmetic"}
INT_LIT = re.compile(r"(?<![\w.])(\d+)(?![\w.])")
BOOL_LIT = re.compile(r"(?<![\w.])(true|false)(?![\w.])")
PER_STATEMENT = 24


class _NoEffect(object):
    """a branch no effect names is mutated by its shape alone"""
    verb, target, by, to, args = None, "", None, None, []


def _ternary(b):
    """(i, j, k) of the first top-level ` ? ` .. ` : ` .. end in blanked text b: `?` at i, `:` at j, the false arm ending at k"""
    depth, i = 0, None
    for p, ch in enumerate(b):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if i is not None and depth < d0:
                return (i, j, p) if j is not None else None
        elif b.startswith(" ? ", p) and i is None:
            i, d0, j = p, depth, None
        elif i is not None and b.startswith(" : ", p) and depth == d0 and j is None:
            j = p
        elif i is not None and j is not None and depth == d0 and ch in ",;":
            return i, j, p
    return (i, j, len(b)) if i is not None and j is not None else None


def _calls(b):
    """[(name, [(start, end) of each top-level argument])] for every call in blanked text b"""
    out = []
    for m in re.finditer(r"(?<![\w.])((?:[A-Za-z_]\w*\.)*[A-Za-z_]\w*)\s*\(", b):
        name = m.group(1).split(".")[-1]
        if name in ("if", "while", "for", "foreach", "switch", "using", "lock", "catch", "return", "new", "nameof", "typeof", "default", "sizeof"):
            continue
        depth, s0, args = 0, m.end(), []
        for p in range(m.end() - 1, len(b)):
            ch = b[p]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
                if depth == 0:
                    if b[s0:p].strip():
                        args.append((s0, p))
                    break
            elif ch == "," and depth == 1:
                args.append((s0, p))
                s0 = p + 1
        if args:
            out.append((name, args))
    return out


DECISION = re.compile(r"\b(?:if|while)\s*\(| \? |(?<=\s)(?:<=|>=|==|!=|&&|\|\||<|>)(?=\s)|^\s*(?:return|case|else)\b")


def behaviour_mutants(st, a=None):
    """[(label, new text)] - every behaviour change of one statement: the effect-aware ones of code_mutants, then each binary
    operator swapped where it stands (boundary, equality, logic, arithmetic), each integer made one more, each bool flipped and
    a branch's one-statement body emptied. Labels are stable, so a mutant keeps its name from run to run."""
    t, b = st.text, blank_out(st.text)
    out = list(code_mutants(a or _NoEffect, st))
    seen = {}
    for m in SPACED_OP.finditer(b):
        op = m.group(1)
        if OP_KIND[op] == "arithmetic" and '"' in t:
            continue                                    # `"a" + b` is a string: its minus does not compile, so it is no verdict
        seen[op] = seen.get(op, 0) + 1
        out.append(("%s %s to %s (%d)" % (OP_KIND[op], op, SWAP[op], seen[op]), t[:m.start()] + SWAP[op] + t[m.end():]))
    for i, m in enumerate(INT_LIT.finditer(b)):
        if i < 2:
            # the second of two equal numbers gets its place in the label, or the two mutants share one key and one is lost
            same = sum(1 for x in INT_LIT.finditer(b[:m.start()]) if x.group(1) == m.group(1))
            out.append(("the number %s made %d%s" % (m.group(1), int(m.group(1)) + 1, "" if not same else " (%d)" % (same + 1)),
                        t[:m.start()] + str(int(m.group(1)) + 1) + t[m.end():]))
    for i, m in enumerate(BOOL_LIT.finditer(b)):
        if i < 2:
            flip = "false" if m.group(1) == "true" else "true"
            out.append(("%s made %s" % (m.group(1), flip), t[:m.start()] + flip + t[m.end():]))
    # REVIEW_1 #4-#6 (10-29): a decision need not be an `if` - a ternary's two outcomes swapped, an else-arm removed, a call's
    # arguments each made `default` (a payment of the wrong amount has the right shape), a returned or assigned text emptied
    q = _ternary(b)
    if q:
        i, j, k = q
        out.append(("the ternary's outcomes swapped", t[:i + 3] + t[j + 3:k].strip() + " : " + t[i + 3:j].strip() + t[k:]))
    if re.match(r"\s*else\b(?!\s+if\b)", b) and b.strip() != "else":
        out.append(("the else-arm removed", t[:re.match(r"\s*else\b", t).end()]))
    for name, args in _calls(b):
        for n, (s0, s1) in enumerate(args[:3], 1):
            if t[s0:s1].strip() and t[s0:s1].strip() != "default" and not re.match(r"\s*(?:out|ref|in)\s", t[s0:s1]):
                out.append(("%s's argument %d made default" % (name, n), t[:s0] + "default" + t[s1:]))
    for m in re.finditer(r'"[^"\\]*(?:\\.[^"\\]*)*"', t):
        if m.group(0) != '""' and re.search(r"\breturn\b|=(?!=)|:", t[:m.start()]):
            out.append(("the text %s emptied" % m.group(0)[:24], t[:m.start()] + '""' + t[m.end():]))
            break
    m = re.match(r"\s*(?:else\s+)?(?:if|while)\s*\(", b)
    if m:
        depth, j = 0, m.end() - 1
        for j in range(m.end() - 1, len(b)):
            depth += b[j] == "("
            depth -= b[j] == ")"
            if depth == 0:
                break
        # the body goes and the statement's own ';' (outside its text) becomes the empty body: `if (c) { }` left that ';' standing
        # alone, and a following `else` could not start a statement (CS8641 on 6 mutants, MEASURED 10-29)
        if t[j + 1:].strip() and not t[j + 1:].strip().startswith("{"):
            out.append(("the branch's body removed", t[:j + 1]))
    # a statement that only acts (a call, an assignment, `p.Rank++`) is removed outright - never a declaration, which would
    # leave its name undeclared (no verdict), and never what the branch / return operators already break
    if st.kind in ("call", "set", "other") and not re.match(r"\s*(?:var|[\w<>\[\],.?]+)\s+[A-Za-z_]\w*\s*(?:=|$)", b)             and not re.match(r"\s*(?:else|case|default|break|continue|throw|yield|for|foreach|using|lock)", b) and b.strip().endswith((")", "++", "--")) or             st.kind == "set" and not re.match(r"\s*(?:var|[\w<>\[\],.?]+)\s+[A-Za-z_]\w*\s*=", b):
        out.append(("the statement removed", "/* removed */"))
    uniq, texts = [], {t}
    for what, new in out:
        if new not in texts:
            texts.add(new)
            uniq.append((what, new))
    # one of each operator KIND before a second of any (REVIEW_2 #4: a cap of 12 in source order cut the newest operators off the
    # richest statements - LinesOnSale got no body-removed mutant), then the rest, up to the cap
    kind = lambda w: re.sub(r"\s*\(\d+\)$", "", re.sub(r"^(boundary|equality|logic|arithmetic) .*", r"", re.sub(r"'s argument \d+", "'s argument", w)))
    first, rest, seen = [], [], set()
    for what, new in uniq:
        (rest if kind(what) in seen else first).append((what, new))
        seen.add(kind(what))
    return (first + rest)[:PER_STATEMENT]


class MethodDecl(object):
    """Where a method sits in its file - enough to COPY it and call the copy: the declaration's span, its name, its parameters as
    a call passes them, whether it returns a value. `why_not` names what cannot be copied this way (a constructor, an iterator,
    an async method, a ref return, an operator, an explicit interface member): its mutants compile one at a time instead."""

    def __init__(self, code, place, f, name_at, generics, paren_open, paren_close, body_at, kind):
        b, src = code.blank[f], code.files[f]
        self.place, self.file, self.kind, self.body_at = place, f, kind, body_at
        self.is_property = paren_open == paren_close
        self.name, _, self.which = place.rsplit(".", 1)[1].partition("~")
        self.name_at, self.name_end, self.generics = name_at, name_at + len(self.name), generics
        p = max(b.rfind(";", 0, name_at), b.rfind("{", 0, name_at), b.rfind("}", 0, name_at)) + 1
        while True:                                      # past whitespace, [attributes] and #preprocessor lines
            q = len(b) - len(b[p:].lstrip()) if b[p:].strip() else p
            if b.startswith("[", q):
                depth = 0
                for p in range(q, name_at):
                    depth += b[p] == "["
                    depth -= b[p] == "]"
                    if depth == 0:
                        break
                p += 1
            elif b.startswith("#", q):
                p = b.find("\n", q) + 1
            else:
                p = q
                break
        self.start, self.head = p, src[p:name_at]
        self.end = (match_brace(b, body_at) + 1) if kind == "{" else (b.find(";", body_at) + 1)
        self.is_void = re.search(r"\bvoid\s*$", b[p:name_at]) is not None
        args = []
        part, depth = "", 0
        for c in b[paren_open + 1:paren_close] + ",":
            if c in "([<{":
                depth += 1
            elif c in ")]>}":
                depth -= 1
            if c == "," and depth == 0:
                if part.strip():
                    decl = part.split("=")[0]
                    mods = [w for w in re.findall(r"[A-Za-z_]\w*", decl) if w in ("ref", "out", "in")]
                    args.append(("%s " % mods[0] if mods else "") + re.findall(r"[A-Za-z_]\w*", decl)[-1])
                part = ""
            else:
                part += c
        self.args = ", ".join(args)
        h = b[p:name_at]
        body = b[body_at:self.end]
        bare = re.sub(r"\b(?:public|protected|internal|private|static|unsafe|extern)\b", "", h).strip()
        self.why_not = ("a property with accessors" if self.is_property and kind == "{" else
                        "a constructor" if self.name == place.rsplit(".", 1)[0].split(".")[-1] or not bare else
                        "an operator" if re.search(r"\boperator\b", h) or self.name.startswith("operator") else
                        "an async method" if re.search(r"\basync\b", h) else
                        "a ref return" if re.search(r"\bref\s", h) else
                        "an explicit interface member" if b[name_at - 1:name_at] == "." else
                        "an iterator" if re.search(r"\byield\b", body) else None)

    def copy(self, src, n, st_start, st_end, new_text):
        """the method again, private, named <Name>__m<n>, with one statement replaced; n None is the unchanged control <Name>__ctl"""
        head = re.sub(r"\b(?:public|protected|internal|private|override|virtual|sealed|new|abstract)\b\s*", "", self.head)
        rest = src[self.name_end:self.end] if n is None else src[self.name_end:st_start] + new_text + src[st_end:self.end]
        return "\n    " + head + self.name + ("__ctl" + self.which if n is None else "__m%d" % n) + rest + "\n"

    def dispatch(self, n):
        call = "%s__m%d" % (self.name, n) if self.is_property else "%s__m%d%s(%s)" % (self.name, n, self.generics, self.args)
        return ("if (global::ClosedLoopMutant.Id == %d) { %s; return; } " % (n, call)) if self.is_void else \
               ("if (global::ClosedLoopMutant.Id == %d) return %s; " % (n, call))


class Mutant(object):
    """One behaviour change of one statement. `key` names it for good - `Type.Method#sK: <operator>` - so a verdict follows it
    from run to run; `of` is everything it tests: an effect's assertion ids, a branch cell's key. Its verdict: killed (a test
    failed, or the run hung), survived (every test passed), or None with why (did not compile, no test covers its file)."""

    def __init__(self, key, place, k, st, what, text):
        self.key, self.place, self.k, self.st, self.what, self.text = key, place, k, st, what, text
        self.of, self.n, self.verdict = set(), 0, {}

    def record(self, code):
        return dict(self.verdict, place=self.place, what=self.what, of=sorted(self.of), file=shown(self.st.file),
                    line=code.files[self.st.file].count("\n", 0, self.st.start) + 1, statement=re.sub(r"\s+", " ", self.st.text)[:120],
                    mutant=re.sub(r"\s+", " ", self.text)[:120])


SWITCH_CS = ("\ninternal static class ClosedLoopMutant { public static readonly int Id = int.TryParse(System.Environment.GetEnvironmentVariable(\"CLOSED_LOOP_MUTANT\"), "
             "out var n) ? n : 0; }   // closed_loop.py mutant schemata: never in the live project\n")


class MutantSchema(object):
    """EVERY mutant in ONE compile (mutant schemata). Each mutant's method is copied with its change made, named <Method>__mN,
    and the original dispatches to the copy when CLOSED_LOOP_MUTANT=N - so one compile of Assembly-CSharp (~45 s, MEASURED 10-29) serves
    every mutant, and each costs one test run (~3 s) instead of a compile of its own. A copy that does not compile is dropped and
    named, never a verdict, and the rest rebuilt. Written only to an overlay; the live project is never touched."""

    def __init__(self, code, mutants):
        self.C, self.mutants = code, [m for m in mutants if m.n]
        self.ranges = {}                                 # file -> [(start, end, mutant)] in the schema's text

    def files(self, drop=()):
        """{file: schema text} with every mutant but `drop` built in"""
        by = {}
        for m in self.mutants:
            if m.key not in drop:
                by.setdefault(m.st.file, []).append(m)
        out, self.ranges = {}, {}
        for i, (f, ms) in enumerate(sorted(by.items())):
            src = self.C.files[f]
            edits = []                                   # (at, end, text, mutant or None)
            for place in sorted({m.place for m in ms}):
                d = self.C.decl(place)
                mine = [m for m in ms if m.place == place]
                # each dispatch is its mutant's own edit, so a compile error in it is that mutant's and nobody else's
                if d.kind == "{":
                    for m in mine:
                        edits.append((d.body_at + 1, d.body_at + 1, d.dispatch(m.n), m))
                else:
                    # an arrow body's body_at is the '>' of its '=>' (MEASURED 10-29: the rewrite left `={` and nothing compiled)
                    expr = src[d.body_at + 1:d.end - 1].strip()
                    edits.append((d.body_at - 1, d.end, "{ get { " if d.is_property else "{ ", None))
                    for m in mine:
                        edits.append((d.end, d.end, d.dispatch(m.n), m))
                    edits.append((d.end, d.end, ("return %s; } }" % expr if d.is_property else "%s; }" % expr if d.is_void else "return %s; }" % expr), None))
                for m in mine:
                    edits.append((d.end, d.end, d.copy(src, m.n, m.st.start, m.st.end, m.text), m))
                # the CONTROL copy, unchanged: a mutant whose IL equals it is the same program (L19, Trivial Compiler Equivalence)
                edits.append((d.end, d.end, d.copy(src, None, d.end, d.end, ""), None))
            if i == 0:
                edits.append((len(src), len(src), SWITCH_CS, None))
            text, ranges, pos = [], [], 0
            for at, end, new, m in sorted(edits, key=lambda e: (e[0], e[1])):
                text.append(src[pos:at])
                base = sum(len(x) for x in text)
                text.append(new)
                if m is not None:
                    ranges.append((base, base + len(new), m))
                pos = end
            text.append(src[pos:])
            out[f] = "".join(text)
            self.ranges[f] = ranges
        return out

    def owner(self, f, line, text, col=1):
        """the mutant whose copy or dispatch holds this line and column of the schema text, or None"""
        at = sum(len(x) + 1 for x in text.split("\n")[:line - 1]) + col - 1
        for a, b, m in self.ranges.get(f, []):
            if a <= at < b:
                return m
        return None


class MutationRun(object):
    """The mutants of a model, run: one schema compile (rebuilt without each copy that does not compile), the tests on the code
    as it is first - a test that fails with no mutant makes every kill meaningless, so that stops the run - then every mutant
    alone, CLOSED_LOOP_MUTANT=N, `dotnet test`, a hang counted as a kill. Its own work dir, never the shared headless one."""

    # one work folder PER RUN (MEASURED 10-29: two runs at once shared one folder, and the second cleared the first's worker copies
    # while its tests ran) - named by the process, its worker copies removed when the run ends
    OUT = os.path.join(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()), "Temp", "closed_loop_mutants", "run%d" % os.getpid())

    def __init__(self, code, tests, workers=2):
        self.C, self.tests, self.workers = code, tests, workers
        self.log = []
        if OVERLAY:                     # a sandbox builds in its own work dir: two arms at once must not share one
            self.OUT = os.path.normpath(OVERLAY) + "_mutants"

    def _headless(self, overlay):
        files = []
        for t in self.tests:
            files += ["--test", src_path(t["test"])]
        r = subprocess.run([sys.executable, "-X", "utf8", HEADLESS, "--overlay", overlay, "--out", self.OUT] + files, cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        return r.returncode, r.stdout + r.stderr

    def run(self, mutants):
        schema = MutantSchema(self.C, mutants)
        drop = {}
        for round_ in range(8):
            texts = schema.files(drop)
            ov = tempfile.mkdtemp(prefix="cs_schema_")
            try:
                if OVERLAY:                              # the sandbox first (its tests, its sheets), the schema over it
                    shutil.copytree(OVERLAY, ov, dirs_exist_ok=True, ignore=shutil.ignore_patterns("model", "kg", "chains", "*.md"))
                for f, text in texts.items():
                    dst = os.path.join(ov, self._rel(f))
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    io.open(dst, "w", encoding="utf-8", newline="").write(text)
                rc, out = self._build(ov)
                errs = re.findall(r"^\s*(.+?\.cs)\((\d+),(\d+)\): error (CS\d+): (.*)$", out, re.M)
                self.log.append("schema round %d: %d mutant(s), exit %d, %d compile error(s)" % (round_ + 1, len(schema.mutants) - len(drop), rc, len(errs)))
                if not errs and ("does not compile" not in out):
                    break
                newly, outside = 0, None
                for path, line, col, code_, msg in errs:
                    f = next((g for g in texts if os.path.normpath(path).endswith(self._rel(g))), None)
                    m = schema.owner(f, int(line), texts[f], int(col)) if f else None
                    if m is None:
                        outside = (f, path, line, code_, msg)
                        continue
                    if m.key not in drop:
                        drop[m.key] = "does not compile: %s %s" % (code_, msg.strip()[:90])
                        newly += 1
                if not newly:
                    if errs and outside:
                        f, path, line, code_, msg = outside
                        dbg = os.path.join(self.OUT, "schema_failed_" + os.path.basename(f or path))
                        io.open(dbg, "w", encoding="utf-8", newline="").write(texts[f] if f else "")
                        raise RuntimeError("(the schema is kept at %s) " % dbg +"the schema does not compile outside any mutant's copy: %s(%s): %s %s" % (path, line, code_, msg))
                    raise RuntimeError("the schema does not compile and no error names a copy:\n" + out[-1500:])
            finally:
                shutil.rmtree(ov, ignore_errors=True)
        else:
            raise RuntimeError("the schema still does not compile after 8 rounds")
        base = self._base(out)
        if base[0] is not False:
            raise RuntimeError("the tests do not pass on the code as it is (%s) - no mutant's verdict would mean anything" % base[2])
        self.passing = base[3]
        for m in mutants:
            if m.key in drop:
                m.verdict = {"killed": None, "why": drop[m.key]}
        live = [m for m in schema.mutants if m.key not in drop]
        # L19 EQUIVALENCE FIRST (prior art: Papadakis et al. 2015): a copy whose IL is its method's control copy's is the same
        # program - no test can kill it and it is no hole; it is never run. A mutant not PROVEN so stays a mutant.
        il = self._il()
        for m in live:
            t, name = m.place.rsplit(".", 1)
            name, _, which = name.partition("~")
            mine = il.get((t.split(".")[-1], "%s__m%d" % (name, m.n))) or il.get((t.split(".")[-1], "get_%s__m%d" % (name, m.n)))
            ctl = il.get((t.split(".")[-1], name + "__ctl" + which)) or il.get((t.split(".")[-1], "get_%s__ctl" % name))
            if mine and ctl and mine == ctl:
                m.verdict = {"killed": None, "equivalent": "TCE", "why": "equivalent: its compiled IL is the original's (Trivial Compiler Equivalence)"}
        self.log.append("L19: %d mutant(s) compile to the original's IL (equivalent, not run)" % sum(1 for m in live if m.verdict.get("equivalent")))
        live = [m for m in live if not m.verdict]
        # each worker runs its own COPY of the built tests: HeadlessBoot rewrites assembly-requests.log beside the dll at start,
        # so two runs in one folder would collide on it
        built, dll_name = self._bin()
        import concurrent.futures as cf
        import threading
        lock = threading.Lock()

        def work(w):
            d = os.path.join(self.OUT, "w%d" % w)
            shutil.rmtree(d, ignore_errors=True)
            shutil.copytree(built, d)
            dll = os.path.join(d, dll_name)
            for m in live[w::self.workers]:
                failed, said = self._one(dll, m.n)
                with lock:
                    m.verdict = {"killed": failed, "by": said if failed else [], "why": ("failed: " + ", ".join(said[:6])) if failed is True
                                 else said if failed is None else "every test passed"}
        try:
            with cf.ThreadPoolExecutor(max_workers=self.workers) as ex:
                list(ex.map(work, range(self.workers)))
        finally:
            for w in range(self.workers):
                shutil.rmtree(os.path.join(self.OUT, "w%d" % w), ignore_errors=True)
        return mutants

    def _il(self):
        """{(type, method): sha of its IL} for every mutant and control copy in the built assembly (il_bodies/, run by dotnet)"""
        reader = il_reader()
        if not reader:
            self.log.append("L19 NOT RUN: no dotnet on PATH to build the IL reader - every equivalent stays a survivor")
            return {}
        out = {}
        for dll in self._code_dlls():
            r = subprocess.run(reader + [dll], capture_output=True, text=True, encoding="utf-8", errors="replace")
            if r.returncode != 0:
                raise RuntimeError("il_bodies failed on %s (exit %d): %s" % (dll, r.returncode, (r.stdout + r.stderr)[-400:]))
            for line in r.stdout.splitlines():
                m = re.match(r"(\S+)::(\S+) ([0-9a-f]{40})$", line.strip())
                if m:
                    out[(m.group(1), m.group(2))] = m.group(3)
        return out

    # ── the four steps a runner differs in: where a file sits in the copy, the build, the tests on the code as it is, the built tests
    def _rel(self, f):
        return proj_rel(f)

    def _build(self, overlay):
        return self._headless(overlay)

    def _base(self, out):
        return self._verdict(out)

    def _bin(self):
        return os.path.join(self.OUT, "Tests", "bin", "Debug", "net10.0"), "HeadlessNUnit.dll"

    def _code_dlls(self):
        return [os.path.join(self.OUT, "AC.dll")]

    @staticmethod
    def _verdict(out):
        """(failed?, failed names, why, passed names) of one dotnet test output"""
        # one name per test: `TestX(Classic,Great,100033)` and `TestX(a, b)` are both TestX, or L16 reads a named check as a test that never ran
        failed = sorted(set(x.split("(")[0] for x in re.findall(r"^\s+Failed (\S+)", out, re.M)))
        passed = sorted(set(x.split("(")[0] for x in re.findall(r"^\s+Passed (\S+)", out, re.M)))
        total = re.search(r"Total tests:\s*(\d+)", out)
        if failed:
            return True, failed, "failed", passed
        if total and int(total.group(1)) > 0:
            return False, [], "every test passed", passed
        return None, [], "no test verdict", passed

    def _one(self, dll, n):
        env = dict(os.environ, CLOSED_LOOP_MUTANT=str(n))
        env.setdefault("TAMI_PROJECT", UNITY_PROJECT)
        try:
            r = subprocess.run(["dotnet", "test", dll, "--nologo", "--logger", "console;verbosity=normal",
                                "--blame-hang-timeout", "60s", "--blame-hang-dump-type", "none"], cwd=os.path.dirname(dll), env=env, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=300)
        except subprocess.TimeoutExpired:
            return True, ["(the run hung past 300 s)"]
        out = r.stdout + r.stderr
        failed, names, why, _ = self._verdict(out)
        # whole words: a bare `hang` matched "change" in any output, so a run with no verdict could read as a kill (REVIEW_1 #9)
        if failed is None and re.search(r"\bhang\b|\bhung\b|Test Run Aborted|\baborted\b", out, re.I):
            return True, ["(the test host hung or aborted: an endless loop)"]
        if failed is None:
            return None, "the runner gave no verdict (exit %d): %s" % (r.returncode, out.strip()[-160:])
        return failed, names



class DotnetMutationRun(MutationRun):
    """The same mutants for ANY .NET project (a developer's, not this Unity one): the model's test names a test project
    (meta.loop.tests: {runner: dotnet, test: <tests .csproj>, root: <folder holding the projects>, covers: [...]}). The root is
    copied once to a work folder, the schema written over the copy, `dotnet build` builds it, `dotnet test` runs the code as it is,
    then each mutant runs alone against the built test assembly. The developer's own folders are never written."""

    def __init__(self, code, tests, workers=2):
        MutationRun.__init__(self, code, tests, workers)
        t = tests[0]
        self.root = src_path(t.get("root") or os.path.dirname(os.path.dirname(t["test"])))
        self.csproj = src_path(t["test"])
        self.work = os.path.join(self.OUT, "dotnet_src")

    def _rel(self, f):
        return os.path.relpath(f, self.root)

    def _build(self, overlay):
        shutil.rmtree(self.work, ignore_errors=True)
        shutil.copytree(self.root, self.work, ignore=shutil.ignore_patterns("bin", "obj", ".git", ".vs"), copy_function=shutil.copyfile)
        shutil.copytree(overlay, self.work, dirs_exist_ok=True)
        proj = os.path.join(self.work, os.path.relpath(self.csproj, self.root))
        r = subprocess.run(["dotnet", "build", proj, "-nologo", "-v", "q", "-c", "Debug"], cwd=self.work, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        out = r.stdout + r.stderr
        if r.returncode != 0 and not re.search(r"error CS\d+", out):
            raise RuntimeError("dotnet build failed for a reason no compile error names:\n" + out[-1200:])
        if r.returncode != 0:
            out += "\ndoes not compile"
        return r.returncode, out

    def _bin(self):
        proj = os.path.join(self.work, os.path.relpath(self.csproj, self.root))
        name = os.path.splitext(os.path.basename(proj))[0] + ".dll"
        hits = sorted(glob.glob(os.path.join(os.path.dirname(proj), "bin", "Debug", "*", name)))
        if not hits:
            raise RuntimeError("the built test assembly %s is not under %s/bin/Debug" % (name, os.path.dirname(proj)))
        return os.path.dirname(hits[-1]), name

    def _base(self, out):
        d, name = self._bin()
        r = subprocess.run(["dotnet", "test", os.path.join(d, name), "--nologo", "--logger", "console;verbosity=normal",
                            "--blame-hang-timeout", "60s", "--blame-hang-dump-type", "none"], cwd=d,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        out_ = r.stdout + r.stderr
        verdict = self._verdict(out_)
        if verdict[0] is None and re.search(r"\bhang\b|\bhung\b|Test Run Aborted|\baborted\b", out_, re.I):
            return True, ["(the test host hung or aborted: an endless loop)"], "hung", []
        return verdict

    def alone(self, f, mutated):
        """(compiled, failed, said) of ONE mutated file built and tested on its own - for a mutant the schema cannot hold"""
        ov = tempfile.mkdtemp(prefix="cs_alone_")
        try:
            dst = os.path.join(ov, self._rel(f))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            io.open(dst, "w", encoding="utf-8", newline="").write(mutated)
            rc, out = self._build(ov)
            if rc != 0:
                return False, None, "the mutant does not compile"
            failed, names, why, _ = self._base(out)
            return True, failed, ("failed: " + ", ".join(names[:6])) if failed else why
        finally:
            shutil.rmtree(ov, ignore_errors=True)

    def _code_dlls(self):
        d, name = self._bin()
        return [p for p in glob.glob(os.path.join(d, "*.dll")) if os.path.basename(p) != name and "ClosedLoopMutant" in
                io.open(p, "rb").read().decode("latin-1")]

class Runner(object):
    """runs the model's tests against an overlay holding one mutated file - never the live project"""

    def __init__(self, tests, unity_project=UNITY_PROJECT):
        self.tests, self.project = tests, unity_project
        self.runs = 0

    def covering(self, path):
        p = os.path.normpath(path)
        hit = []
        for t in self.tests:
            for c in ucm.as_list(t.get("covers")):
                cp = src_path(c)
                if p == cp or p.startswith(cp + os.sep):
                    hit.append(t)
                    break
        return hit

    def run(self, test, path, mutated_source):
        """(compiled, failed, said): the test run against the mutant"""
        if test.get("runner") != "headless_nunit":
            return None, None, "no runner %r" % test.get("runner")
        rel = proj_rel(path)
        if rel.startswith(".."):
            return None, None, "%s is outside the Unity project: headless_nunit cannot overlay it" % path
        ov = tempfile.mkdtemp(prefix="cs_kill_")
        try:
            if OVERLAY:
                shutil.copytree(OVERLAY, ov, dirs_exist_ok=True, ignore=shutil.ignore_patterns("model", "kg", "chains", "*.md"))
            dst = os.path.join(ov, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            io.open(dst, "w", encoding="utf-8", newline="").write(mutated_source)
            t = src_path(test["test"])
            self.runs += 1
            r = subprocess.run([sys.executable, "-X", "utf8", HEADLESS, "--overlay", ov, "--test", t], cwd=ROOT,
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
            out = r.stdout + r.stderr
            if "does not compile" in out or re.search(r"csc Assembly-CSharp(?:-Editor)?: exit=[1-9]", out):
                return False, None, "the mutant does not compile"
            failed = re.findall(r"^\s+Failed (\S+)", out, re.M)
            if r.returncode == 0 and not failed:
                return True, False, "every test passed"
            if failed:
                return True, True, "failed: " + ", ".join(sorted(set(failed))[:6])
            return None, None, "the runner stopped (exit %d) without a test verdict" % r.returncode
        finally:
            shutil.rmtree(ov, ignore_errors=True)


class CsEffectCheck(object):
    """Each requirement's EFFECT held to the C#. SHAPE: a statement at its place reads like it, and its plant flips that. BEHAVIOUR:
    every mutant of every statement it matched, and of every branch of the code the model's tests cover, run in one MutationRun.
    An effect is KILLED when every compiled mutant of its statements fails a test; a branch is PINNED the same way. One survivor
    the model does not name EQUIVALENT (meta.loop.equivalent: [{mutant: <key>, why}]) is a hole: a test is missing or blind."""

    def __init__(self, code, runner=None, equivalent=None, workers=2):
        self.C, self.R, self.workers = code, runner, workers
        self.equivalent = dict(equivalent or {})
        self.mutants, self.log, self.passing = {}, [], []

    def verdict(self, a):
        """the shape verdict; `_hit` (dropped before it is written) holds the matched statements for the behaviour pass"""
        if a.verb is None or not a.at:
            return {"holds": None, "why": "malformed: one verb and a place, U18"}
        try:
            sts = self.C.method(str(a.at))
        except KeyError as e:
            return {"holds": None, "why": "its place names nothing: %s" % e}
        hit = [s for s in sts if matches(a, s.text, s.kind)]
        v = {"holds": bool(hit), "matched": [re.sub(r"\s+", " ", s.text)[:120] for s in hit], "read": len(sts),
             "fp": sha("\n".join(s.text for s in (hit or sts))), "_hit": [(sts.index(s), s) for s in hit]}
        if hit:
            still = ["%s (%s)" % (re.sub(r"\s+", " ", s.text)[:60], what) for s in hit for what, m in text_mutants(a, s.text) if matches(a, m, s.kind)]
            v["plant"] = "flips" if not still else "does not flip: " + "; ".join(still)
            v["why"] = "%s at %s" % (a.verb, a.at)
        else:
            want, kind = wanted(a)
            v["plant"] = "flips" if matches(a, want, kind) else "does not flip: the statement it asks for does not satisfy it"
            v["why"] = "no statement in %s %s %s%s (read %d)" % (a.at, a.verb, a.target, "" if a.to is None else " to %s" % a.to, len(sts))
        return v

    def collect(self, hits, branch_files):
        """{key: Mutant}: every mutant of each effect's matched statements and of every branch in the covered files, one per key"""
        muts = {}

        sids = {}

        def sid(place, k):
            """a statement's name by its TEXT, not its position (REVIEW_1 #9): an edit above it moves no key, and an equivalence
            declared for it lapses the moment its own text changes; a second identical statement in one method is #2"""
            if place not in sids:
                seen, names = collections.Counter(), []
                for s in self.C.method(place):
                    h = sha(re.sub(r"\s+", " ", s.text.strip()))[:6]
                    seen[h] += 1
                    names.append(h + ("" if seen[h] == 1 else "#%d" % seen[h]))
                sids[place] = names
            return sids[place][k]

        def add(place, k, st, a, of):
            for what, new in behaviour_mutants(st, a):
                key = "%s@%s: %s" % (place, sid(place, k), what)
                muts.setdefault(key, Mutant(key, place, k, st, what, new)).of.add(of)
        for aid, (a, sts) in sorted(hits.items()):
            for k, st in sts:
                add(str(a.at), k, st, a, aid)
        decisions = self.C.branches(branch_files)
        for key, place, st in decisions:
            add(place, int(key.rsplit("#b", 1)[1]), st, None, key)
        # VACUITY of a whole method: it returns at once (its first statement prefixed by `return default;`, the rest left in place
        # unreachable, so nothing it declares goes missing). A test that passes on an empty method checks nothing in it; the
        # mutant tests every decision and effect of the method at once.
        by_place = {}
        for key, place, st in decisions:
            by_place.setdefault(place, []).append(key)
        for aid, (a, sts) in hits.items():
            by_place.setdefault(str(a.at), []).append(aid)
        for place, ofs in sorted(by_place.items()):
            try:
                d, sts = self.C.decl(place), self.C.method(place)
            except KeyError:
                continue
            if d.why_not or d.kind != "{" or not sts:
                continue
            st0 = sts[0]
            new = ("return; " if d.is_void else "return default; ") + st0.text
            key = "%s@%s: the method returns at once" % (place, sid(place, 0))
            m = muts.setdefault(key, Mutant(key, place, 0, st0, "the method returns at once", new))
            m.of.update(ofs)
        return muts

    def behave(self, muts):
        """every mutant judged: a file no test covers says so; a method that cannot be copied compiles its mutants alone"""
        if self.R is None:
            for m in muts.values():
                m.verdict = {"killed": None, "why": "no tests named in meta.loop.tests"}
            return
        schema, alone = [], []
        for m in muts.values():
            if not self.R.covering(m.st.file):
                m.verdict = {"killed": None, "why": "no test covers %s (meta.loop.tests covers)" % shown(m.st.file)}
                continue
            d = self.C.decl(m.place)
            (alone if d.why_not else schema).append(m)
            if d.why_not:
                m.why_alone = d.why_not
        for n, m in enumerate(schema + alone, 1):
            m.n = n
        if schema:
            dotnet = any(t.get("runner") == "dotnet" for t in self.R.tests)
            run = (DotnetMutationRun if dotnet else MutationRun)(self.C, self.R.tests, self.workers)
            run.run(schema)
            self.log += run.log
            self.passing = run.passing
        dot = DotnetMutationRun(self.C, self.R.tests, self.workers) if any(t.get("runner") == "dotnet" for t in self.R.tests) else None
        for m in alone:                                  # the slow road: one compile per mutant (~45 s)
            src = self.C.files[m.st.file]
            mutated = src[:m.st.start] + m.text + src[m.st.end:]
            said = "no test ran"
            for t in self.R.covering(m.st.file):
                compiled, failed, said = dot.alone(m.st.file, mutated) if dot else self.R.run(t, m.st.file, mutated)
                if compiled:
                    m.verdict = {"killed": bool(failed), "by": re.findall(r"[\w.]+", said.split(":", 1)[1]) if failed else [],
                                 "why": said + " (compiled alone: %s)" % m.why_alone}
                    break
            else:
                m.verdict = {"killed": None, "why": "%s (compiled alone: %s)" % (said, m.why_alone)}

    def summary(self, ms):
        """{killed, mutants, ran, killed_n, survivors, equivalent, by, why} over one effect's or one branch's mutants"""
        if not ms:
            return {"killed": None, "why": "no behaviour mutant can be built for a statement of this shape"}
        ran = [m for m in ms if m.verdict.get("killed") is not None]
        tce = [m for m in ms if m.verdict.get("equivalent") == "TCE"]
        if not ran:
            return {"killed": None, "mutants": len(ms), "tce": len(tce),
                    "why": ("its only mutants compile to its own IL (equivalent): nothing a test could catch" if tce and len(tce) == len(ms)
                            else ms[0].verdict.get("why"))}
        surv = [m for m in ran if m.verdict["killed"] is False and m.key not in self.equivalent]
        eq = [m for m in ran if m.verdict["killed"] is False and m.key in self.equivalent] + tce
        by = sorted({t for m in ran if m.verdict["killed"] for t in m.verdict.get("by") or []})
        out = {"killed": not surv, "mutants": len(ms), "ran": len(ran), "killed_n": sum(1 for m in ran if m.verdict["killed"]),
               "survivors": [m.key for m in surv], "by": by,
               "equivalent": {m.key: self.equivalent.get(m.key) or "the same compiled IL (TCE)" for m in eq}}
        out["why"] = ("%d of %d mutant(s) killed%s" % (out["killed_n"], len(ran), "" if not eq else ", %d equivalent (the same IL, or the model says why)" % len(eq))
                      if not surv else "%d mutant(s) survived every test: %s" % (len(surv), "; ".join(
                          "%s -> %s" % (m.what, re.sub(r"\s+", " ", m.text)[:70]) for m in surv[:3])))
        if len(ran) < len(ms):
            out["why"] += "; %d did not compile or ran no test" % (len(ms) - len(ran))
        return out

    def names(self, texts):
        out = {}
        for t in texts:
            for m in SHAPED.finditer(t or ""):
                n = m.group(1) or m.group(2)
                out[n] = "code" if n in self.C.words else None
        return out

    def run(self, model_dir, out_path, kill=True, only=None):
        model = ucm.UseCaseModel.load(model_dir)
        assertions, says, named, hits = {}, {}, {}, {}
        for r in model.requirements.values():
            for a in r.effect:
                if only and r.id != only:
                    continue
                v = self.verdict(a)
                if v.get("holds"):
                    hits[a.id] = (a, v["_hit"])
                v.pop("_hit", None)
                assertions[a.id] = dict(v, at=a.at, text=a.text())
            says[r.id] = sorted(self.names([r.text]))
            named.update(self.names([r.text]))
            for a in r.effect:
                for n in (a.target, a.by):
                    if n:
                        named[str(n)] = "code" if str(n) in self.C.words else None
        for t in model.terms.values():
            for b in t.binds:
                named[b] = "code" if b in self.C.words else None
        covered = sorted({f for f in self.C.files if self.R is not None and self.R.covering(f)})
        branches = {key: (place, st) for key, place, st in self.C.branches(covered)}
        doc = {"format": "effect-verdicts/1", "adapter": "cs_effect_check", "code": sha("".join(sorted(self.C.files.values()))),
               "assertions": assertions, "names": named, "says": says, "not_walked": self.C.not_walked(covered)}
        if kill:
            self.mutants = self.collect(hits, covered)
            print("cs_effect_check: %d mutant(s) of %d effect(s) and %d branch(es) in %d covered file(s)" % (
                len(self.mutants), len(hits), len(branches), len(covered)))
            self.behave(self.mutants)
            for aid in assertions:
                if aid in hits:
                    assertions[aid]["kill"] = self.summary([m for m in self.mutants.values() if aid in m.of])
            doc["branches"] = {}
            for key, (place, st) in sorted(branches.items()):
                doc["branches"][key] = dict(self.summary([m for m in self.mutants.values() if key in m.of]), place=place,
                                            text=re.sub(r"\s+", " ", st.text)[:120])
            doc["mutants"] = {k: m.record(self.C) for k, m in sorted(self.mutants.items())}
            doc["tests"] = self.passing
            doc["log"] = self.log
            doc["equivalent_declared"] = dict(self.equivalent)
        else:
            for aid in hits:
                assertions[aid]["kill"] = {"killed": None, "why": "not run (--no-kill)"}
        for aid, v in sorted(assertions.items()):
            print("  %-12s %-60s %s%s" % (aid, v["text"][:60], {True: "HOLDS", False: "FAILS", None: "UNREAD"}[v["holds"]],
                                          "" if not v.get("kill") else "  kill: %s (%s)" % (v["kill"].get("killed"), v["kill"].get("why", "")[:70])))
        io.open(out_path, "w", encoding="utf-8").write(json.dumps(doc, indent=1, sort_keys=True))
        return doc


# ── the selftest: every verdict must be able to come out the other way ────────────────────────────

SELF_CS = """
public class Wallet
{
    public int Money { get; private set; }
    // a comment with Pay( and { braces } that must not be read
    public bool CanPay(int price) => price >= 0 && Money >= price;
    public void Pay(int price)
    {
        if (!CanPay(price)) throw new System.InvalidOperationException("cannot pay");
        Money -= price;
        Log("paid {0}", price);
    }
    void Log(string f, int n) { }
}
"""


def selftest():
    ok = True
    tmp = tempfile.mkdtemp(prefix="cs_effect_selftest_")
    p = os.path.join(tmp, "Wallet.cs")
    io.open(p, "w", encoding="utf-8").write(SELF_CS)
    C = CsCode([p])
    chk = CsEffectCheck(C)

    def A(d):
        return ucm.Assertion("R", 1, d)
    cases = [
        ("sets Money at Wallet.Pay holds, its plant flips", A({"sets": "Money", "at": "Wallet.Pay"}), True, "flips"),
        ("calls Log with price holds", A({"calls": "Log", "args": ["price"], "at": "Wallet.Pay"}), True, "flips"),
        ("branches CanPay holds", A({"branches": "CanPay", "at": "Wallet.Pay"}), True, "flips"),
        ("reads Money at Wallet.CanPay (expression body) holds", A({"reads": "Money", "at": "Wallet.CanPay"}), True, "flips"),
        ("sets Credit (absent) fails, its asked-for statement satisfies it", A({"sets": "Credit", "at": "Wallet.Pay"}), False, "flips"),
        ("calls Pay( inside the comment is not read", A({"calls": "Pay", "at": "Wallet.Pay"}), False, "flips"),
        ("a place that names nothing is unread", A({"sets": "Money", "at": "Wallet.Refund"}), None, None),
    ]
    for what, a, holds, plant in cases:
        v = chk.verdict(a)
        good = v.get("holds") is holds and (plant is None or v.get("plant") == plant)
        ok &= good
        print("selftest: SHAPE %-58s -> holds %s, plant %s %s" % (what, v.get("holds"), v.get("plant"), "PASS" if good else "FAIL"))
    st = [s for s in C.method("Wallet.Pay") if s.kind == "branch"][0]
    muts = dict(code_mutants(A({"branches": "CanPay", "at": "Wallet.Pay"}), st))
    good = "the condition negated" in muts and muts["the condition negated"].startswith("if (!(!CanPay(price)))")
    ok &= good
    print("selftest: KILL  a branch's mutant negates its condition: %r %s" % (muts.get("the condition negated"), "PASS" if good else "FAIL"))
    st = [s for s in C.method("Wallet.Pay") if "Money" in s.text][0]
    muts = dict(code_mutants(A({"sets": "Money", "at": "Wallet.Pay"}), st))
    good = muts.get("the operator flipped") == "Money += price" and "the value replaced by default" not in muts
    ok &= good
    print("selftest: KILL  a compound assignment's mutant flips its operator (never -= default, CS8310): %r %s" % (muts.get("the operator flipped"), "PASS" if good else "FAIL"))
    # EVERY mutant (10-29): a boundary is broken apart from the condition, a number and a one-statement body too
    st = Statement(p, 0, 0, "if (stock <= 0) return Result.SoldOut", "branch")
    labels = [w for w, _ in behaviour_mutants(st)]
    want = ["the condition negated", "boundary <= to < (1)", "the number 0 made 1", "the branch's body removed"]
    good = all(w in labels for w in want)
    ok &= good
    print("selftest: ALL   one branch broken %d ways, the boundary among them: %s %s" % (len(labels), labels, "PASS" if good else "FAIL"))
    st = Statement(p, 0, 0, 'name = "Gold: " + count', "set")
    good = not any(w.startswith("arithmetic") for w, _ in behaviour_mutants(st))
    ok &= good
    print("selftest: ALL   a string's + is never made a - (it would not compile) %s" % ("PASS" if good else "FAIL"))
    # the schema: the method copied with one change, the original dispatching to it, the switch written once
    d = C.decl("Wallet.Pay")
    pay = [s for s in C.method("Wallet.Pay") if "Money" in s.text][0]
    m = Mutant("Wallet.Pay#s1: the operator flipped", "Wallet.Pay", 1, pay, "the operator flipped", "Money += price")
    m.n = 7
    text = MutantSchema(C, [m]).files()[p]
    good = (d.why_not is None and d.is_void and d.args == "price" and "void Pay__m7(int price)" in text
            and "Money += price" in text and "if (global::ClosedLoopMutant.Id == 7) { Pay__m7(price); return; }" in text
            and text.count("class ClosedLoopMutant") == 1 and text.count("Money -= price") == 2      # the original and its control
            and "void Pay__ctl(int price)" in text)
    ok &= good
    print("selftest: SCHEMA Wallet.Pay copied as Pay__m7 with += and dispatched on ClosedLoopMutant.Id == 7 %s" % ("PASS" if good else "FAIL"))
    cp = C.decl("Wallet.CanPay")
    m2 = Mutant("Wallet.CanPay#s0: boundary >= to > (1)", "Wallet.CanPay", 0, C.method("Wallet.CanPay")[0], "boundary", "price >= 0 && Money > price")
    m2.n = 8
    text = MutantSchema(C, [m2]).files()[p]
    good = (not cp.is_void and "{ if (global::ClosedLoopMutant.Id == 8) return CanPay__m8(price); return price >= 0 && Money >= price; }" in text
            and "bool CanPay__m8(int price) => price >= 0 && Money > price;" in text
            and "CanPay(int price) { if (global::ClosedLoopMutant.Id == 8)" in text and "={" not in text)
    ok &= good
    print("selftest: SCHEMA an expression body becomes a block that dispatches, its copy keeps the arrow %s" % ("PASS" if good else "FAIL"))
    # T7, 10-30: real code broke five assumptions the shops never tested. A plant for each
    over = os.path.join(tmp, "Over.cs")
    io.open(over, "w", encoding="utf-8").write("""
public class Over {
    int n;
    public int Count => n + 1;
    public int Auto { get; set; }
    public int Held { get { if (n > 0) return n; return 0; } }
    public int Pick(int a) { return a + 1; }
    public int Pick(string s) { return s.Length + 2; }
    public T Gen<T>(T x) where T : struct { return x; }
    public static bool operator <(Over a, Over b) => a.n < b.n;
    public static bool operator >(Over a, Over b) => a.n > b.n;
    public static bool operator <=(Over a, Over b) => a.n <= b.n;
    public static bool operator >=(Over a, Over b) => a.n >= b.n;
    System.Func<int, int> f = x => x + 1;
    System.Text.RegularExpressions.Regex r = new System.Text.RegularExpressions.Regex("a");
    public int this[int i] => i;
    struct In { public int P; public In(int p) { P = p; } public int End => P + 1; }
}
""")
    O = CsCode([over])
    pl = O.places([os.path.normpath(over)])
    want = {"Over.Count", "Over.Held", "Over.Pick~1", "Over.Pick~2", "Over.Gen", "Over.operator<", "Over.operator>", "In.In", "In.End"}
    good = want <= set(pl) and not {"Over.Auto", "Over.f", "Over.x", "Over.Pick", "Over.r"} & set(pl)
    ok &= good
    print("selftest: PLACES overloads, operators, a generic with `where` and property bodies are all walked; an auto-property and a lambda are not %s" % ("PASS" if good else "FAIL: %s" % pl))
    try:
        O.method("Over.Pick")
        good = False
    except KeyError as e:
        good = "~N" in str(e)
    good = good and "s.Length" in O.method("Over.Pick~2")[0].text and "a + 1" in O.method("Over.Pick~1")[0].text
    try:
        O.method("Over.Pick~3")
        good = False
    except KeyError:
        pass
    ok &= good
    print("selftest: OVERLOAD a bare name is refused and says how to name one; ~N picks it; a ~N too high is refused %s" % ("PASS" if good else "FAIL"))
    nw = O.not_walked([os.path.normpath(over)])
    good = any("Over.r" in x for x in nw) and any("this[]" in x for x in nw) and not any("operator" in x for x in nw)
    ok &= good
    print("selftest: NOTWALKED what the walk cannot reach is named, so a report never claims it %s" % ("PASS" if good else "FAIL: %s" % nw))
    good = O.decl("Over.In").why_not == "a constructor" and O.decl("In.In").why_not == "a constructor"
    ok &= good
    print("selftest: NESTED a nested type's constructor, found under its outer type's name too, is a constructor (BP-5) %s" % ("PASS" if good else "FAIL"))
    good = O.decl("Over.operator<").why_not == "an operator" and O.decl("Over.Held").why_not == "a property with accessors" and O.decl("Over.Count").why_not is None
    ok &= good
    print("selftest: ALONE an operator and accessors compile alone; an expression-bodied property goes in the schema %s" % ("PASS" if good else "FAIL"))
    cnt = O.method("Over.Count")[0]
    mc = Mutant("Over.Count#s0: the number 1 made 2", "Over.Count", 0, cnt, "number", "n + 2")
    mc.n = 9
    txt = MutantSchema(O, [mc]).files()[os.path.normpath(over)]
    good = ("{ get { if (global::ClosedLoopMutant.Id == 9) return Count__m9; return n + 1; } }" in txt and "int Count__m9 => n + 2;" in txt and "int Count__ctl => n + 1;" in txt)
    ok &= good
    print("selftest: PROPERTY an expression-bodied property's schema keeps a getter, and its copies stay properties %s" % ("PASS" if good else "FAIL"))

    root = os.path.join(tmp, "root")                      # BP-4: a Perforce workspace is read-only until checked out
    os.makedirs(root)
    ro = os.path.join(root, "A.cs")
    io.open(ro, "w", encoding="utf-8").write("// original")
    os.chmod(ro, stat.S_IREAD)
    proj = os.path.join(root, "T.csproj")
    io.open(proj, "w", encoding="utf-8").write("<Project/>")
    real_run = subprocess.run
    calls = []

    class Fake(object):
        returncode, stdout, stderr = 0, "", ""

    def fake(argv, **kw):
        calls.append(argv)
        r = Fake()
        r.stdout = fake.out
        return r
    fake.out = ""
    subprocess.run = fake
    try:
        run = DotnetMutationRun(C, [{"test": proj, "root": root}], 1)
        run.work = os.path.join(tmp, "work")
        ov = os.path.join(tmp, "ov")
        os.makedirs(ov)
        io.open(os.path.join(ov, "A.cs"), "w", encoding="utf-8").write("// schema")
        rc, _ = run._build(ov)
        good = rc == 0 and io.open(os.path.join(run.work, "A.cs"), encoding="utf-8").read() == "// schema"
        ok &= good
        print("selftest: READONLY a read-only tree is copied writable, so the schema can go over it (BP-4) %s" % ("PASS" if good else "FAIL"))

        run._bin = lambda: (tmp, "T.dll")                 # BP-7: a mutant tested alone can loop forever
        del calls[:]
        fake.out = "Test Run Aborted."
        v = run._base("")
        good = v[0] is True and "--blame-hang-timeout" in calls[0] and "none" in calls[0]
        fake.out = "  Passed A\nTotal tests: 1\n"
        good = good and run._base("")[0] is False
        ok &= good
        print("selftest: HANG a mutant tested alone has the hang limit, and a hang is a kill; a plain pass is not (BP-7) %s" % ("PASS" if good else "FAIL"))
    finally:
        subprocess.run = real_run

    v = DotnetMutationRun._verdict("  Failed TestX(Classic,Great,100033) [1 ms]\n  Failed TestX(a, b) [1 ms]\n  Passed TestY(1) [1 ms]\nTotal tests: 3\n")
    good = v[0] is True and v[1] == ["TestX"] and v[3] == ["TestY"]
    ok &= good
    print("selftest: PARAMS a parameterized test keeps one name, whatever its arguments, so L16 finds it %s" % ("PASS" if good else "FAIL: %s" % (v,)))

    class Stop(Exception):
        pass

    class Cascade(DotnetMutationRun):                     # BP-6: one bad copy's error lands outside it, and comes first
        rounds = 0

        def __init__(self):
            MutationRun.__init__(self, C, [{"test": proj, "root": root}], 1)
            self.root, self.work, self.OUT = root, os.path.join(tmp, "work2"), tmp

        def _build(self, overlay):
            Cascade.rounds += 1
            if Cascade.rounds > 1:
                return 0, "Build succeeded"
            return 1, Cascade.errors

        def _base(self, out):
            raise Stop()

        def _rel(self, f):
            return os.path.basename(f)
    pay = [x for x in C.method("Wallet.Pay") if "Money" in x.text][0]
    bad = Mutant("Wallet.Pay#s9: bad", "Wallet.Pay", 0, pay, "x", "Money -=")
    bad.n = 5
    sch = MutantSchema(C, [bad])
    text = sch.files()[os.path.normpath(p)]
    line = text[:text.index("Pay__m5(int price)")].count("\n") + 1
    Cascade.errors = "%s(1,1): error CS0106: outside\n%s(%d,5): error CS1002: inside\n" % (p, p, line)
    try:
        Cascade().run([bad])
        good = False
    except Stop:
        good = True
    except RuntimeError:
        good = False
    Cascade.rounds, Cascade.errors = 0, "%s(1,1): error CS0106: outside\n" % p
    try:
        Cascade().run([bad])
        good = False
    except RuntimeError as e:
        good = good and "outside any mutant" in str(e)
    except Stop:
        good = False
    ok &= good
    print("selftest: CASCADE a bad copy is dropped though an error outside it came first; a round naming no copy still stops the run (BP-6) %s" % ("PASS" if good else "FAIL"))
    try:
        os.chmod(ro, stat.S_IWRITE)
    except OSError:
        pass
    shutil.rmtree(tmp, ignore_errors=True)
    print("selftest: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--no-kill", action="store_true", help="shape verdicts only: no mutant is compiled or run")
    ap.add_argument("--only", help="one requirement's assertions")
    ap.add_argument("--workers", type=int, default=2, help="mutants run at once, each in its own copy of the built tests")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.folder or not a.out:
        ap.error("<model folder> --out <verdicts.json>, or --selftest")
    model = ucm.UseCaseModel.load(a.folder)
    loop = model.loop or {}
    roots = ucm.as_list((loop.get("code") or {}).get("roots"))
    if not roots:
        print("cs_effect_check: meta.loop.code.roots names no code - nothing to hold the effects to", file=sys.stderr)
        return 2
    runner = Runner([t for t in ucm.as_list(loop.get("tests")) if isinstance(t, dict)]) if loop.get("tests") else None
    eq = {e["mutant"]: e.get("why") for e in ucm.as_list(loop.get("equivalent")) if isinstance(e, dict) and e.get("mutant")}
    chk = CsEffectCheck(CsCode(roots), runner, eq, workers=a.workers)
    doc = chk.run(a.folder, a.out, kill=not a.no_kill, only=a.only)
    va, ms = doc["assertions"], doc.get("mutants") or {}
    print("cs_effect_check: %d assertion(s): %d hold, %d fail, %d unread; killed %d, survived %d, not killable %d -> %s" % (
        len(va), sum(1 for v in va.values() if v.get("holds") is True), sum(1 for v in va.values() if v.get("holds") is False),
        sum(1 for v in va.values() if v.get("holds") is None),
        sum(1 for v in va.values() if (v.get("kill") or {}).get("killed") is True),
        sum(1 for v in va.values() if (v.get("kill") or {}).get("killed") is False),
        sum(1 for v in va.values() if v.get("holds") and (v.get("kill") or {}).get("killed") is None), a.out))
    print("cs_effect_check: %d mutant(s): %d killed, %d survived, %d no verdict; branches %d, %d pinned" % (
        len(ms), sum(1 for m in ms.values() if m.get("killed") is True), sum(1 for m in ms.values() if m.get("killed") is False),
        sum(1 for m in ms.values() if m.get("killed") is None), len(doc.get("branches") or {}),
        sum(1 for b in (doc.get("branches") or {}).values() if b.get("killed") is True)))
    for line in chk.log:
        print("  " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
