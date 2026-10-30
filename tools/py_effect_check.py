#!/usr/bin/env python3
"""py_effect_check.py - the closed loop's code adapter for PYTHON: each requirement's effect held to Python code, and to its tests.

WHY (2026-10-29): the tool's own requirements are kept as use case descriptions, and the tool runs on itself and improves in this
loop. The tool is Python; its first adapter (cs_effect_check.py) reads C#. This is the same contract
over Python's own syntax tree, so the loop can be run on the loop:

  SHAPE    an assertion holds when a statement at its place (`module.function` or `module.Class.method`) does what it says:
           `sets X` (an assignment to X), `calls X` (a call of X), `branches X` (an if / while / comprehension test naming X),
           `reads X` (X used) - read from the syntax tree, never from text.
  KILL     every statement of every covered function broken every way below, each break run alone against the model's tests in a
           COPY of the code's folder (never the live tool), a break the tests do not notice a survivor. Operators: the condition
           negated / made True / made False, each comparison swapped (< <=, == !=, in / not in, is / is not), and <-> or, a `not`
           dropped, a number made one more, a bool flipped, a text emptied, a returned value made None, a call statement removed,
           each call argument made None, an assignment's value made None, and the function returning at once.
  EQUAL    a mutant whose function compiles to the same bytecode as the original is the same program (L19, as TCE is for C#).
  TESTS    a model test is a command - here each tool's own `--selftest` - and its failing lines (`selftest: ... FAIL`) name
           what caught a break; a run that exits non-zero with no FAIL line is caught by the command as a whole.

    python -X utf8 py_effect_check.py <model folder> --out <verdicts.json> [--workers 3] [--no-kill]
    python -X utf8 py_effect_check.py --selftest

Output: effect-verdicts/1, the same document cs_effect_check.py writes, so closed_loop.py and usecase_kg.py read it unchanged.
"""
import argparse
import ast
import collections
import concurrent.futures as cf
import copy
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
# the project the model's paths are relative to: this repo, or a developer's own (CLOSED_LOOP_ROOT, set by the kit's installer)
ROOT = os.path.abspath(os.environ.get("CLOSED_LOOP_ROOT") or os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import usecase_model as ucm  # noqa: E402
import cs_effect_check as cs  # noqa: E402  the Statement record and the reporting helpers are shared, not copied

SWAP_CMP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
            ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is}
PER_STATEMENT = 24


def sha(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class PyStatement(cs.Statement):
    """cs.Statement with where it sits in the syntax tree (line, column, node type), so a break finds it again"""
    __slots__ = ("node",)


class PyCode(object):
    """The Python files under the model's code roots: every function and method as a place (`module.function`,
    `module.Class.method`), each with its statements in source order - the same surface CsCode gives the loop."""

    def __init__(self, roots, scope=None):
        self.files, self.trees, self.index = {}, {}, {}
        # meta.loop.code.scope: place prefixes (`usecase_kg.LoopGate.`) - the functions whose every statement is a cell and a break;
        # a place outside it still resolves (an effect or a chain may cite it), it is just not broken
        self.scope = [str(x) for x in (scope or [])]
        for r in roots:
            p = cs.src_path(r)
            paths = [p] if p.endswith(".py") else [os.path.join(d, f) for d, _, fs in os.walk(p) for f in fs if f.endswith(".py")]
            for f in sorted(paths):
                if "__pycache__" in f:
                    continue
                src = io.open(f, encoding="utf-8").read()
                self.files[os.path.normpath(f)] = src
                tree = ast.parse(src)
                self.trees[os.path.normpath(f)] = tree
                mod = os.path.splitext(os.path.basename(f))[0]
                for node in tree.body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.index["%s.%s" % (mod, node.name)] = (os.path.normpath(f), node)
                    elif isinstance(node, ast.ClassDef):
                        for sub in node.body:
                            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                self.index["%s.%s.%s" % (mod, node.name, sub.name)] = (os.path.normpath(f), sub)
        if not self.files:
            raise RuntimeError("the model's code roots hold no .py file: %s" % roots)
        self.words = set()
        for src in self.files.values():
            self.words.update(re.findall(r"[A-Za-z_]\w*", src))

    def _stmts(self, fn):
        out = []
        for node in ast.walk(fn):
            if node is not fn and isinstance(node, ast.stmt) and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))                     and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                out.append(node)                    # a docstring is no statement of behaviour
        return sorted(out, key=lambda n: (n.lineno, n.col_offset))

    def method(self, place):
        """the statements of a place, as cs.Statement records (file, start line, end line, source text, kind); KeyError if none"""
        if place not in self.index:
            raise KeyError("no function %s in the code roots" % place)
        f, fn = self.index[place]
        src = self.files[f]
        out = []
        for n in self._stmts(fn):
            seg = ast.get_source_segment(src, n) or ""
            first = seg.split("\n", 1)[0]
            kind = ("branch" if isinstance(n, (ast.If, ast.While, ast.For, ast.Try, ast.With)) else "return" if isinstance(n, ast.Return)
                    else "set" if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)) else "call" if isinstance(n, ast.Expr) else "other")
            st = PyStatement(f, n.lineno, n.end_lineno, first if kind == "branch" else seg, kind)
            st.node = (n.lineno, n.col_offset, type(n).__name__)
            out.append(st)
        return out

    def places(self, files):
        return [p for p, (f, _) in sorted(self.index.items(), key=lambda kv: (kv[1][0], kv[1][1].lineno))
                if f in set(files) and (not self.scope or any(p == x or p.startswith(x) for x in self.scope))]

    def branches(self, files):
        """[(key, place, statement)] for every statement of every function in these files (key `place#bK`)"""
        out = []
        for place in self.places(files):
            for k, st in enumerate(self.method(place)):
                out.append(("%s#b%d" % (place, k), place, st))
        return out


def _find(tree, loc):
    lineno, col, typ = loc
    for n in ast.walk(tree):
        if isinstance(n, ast.stmt) and n.lineno == lineno and n.col_offset == col and type(n).__name__ == typ:
            return n
    raise KeyError("statement %s not found" % (loc,))


def operators(stmt):
    """[(label, transform(stmt_copy) -> stmt or list)] - every break of one statement, labels stable from run to run"""
    out = []
    test = getattr(stmt, "test", None) if isinstance(stmt, (ast.If, ast.While)) else None
    if test is not None:
        out.append(("the condition negated", lambda s: setattr(s, "test", ast.UnaryOp(ast.Not(), s.test)) or s))
        out.append(("the condition made true", lambda s: setattr(s, "test", ast.Constant(True)) or s))
        out.append(("the condition made false", lambda s: setattr(s, "test", ast.Constant(False)) or s))
    scope = [stmt.test] if test is not None else [stmt] if not isinstance(stmt, (ast.For, ast.Try, ast.With)) else \
        ([stmt.iter] if isinstance(stmt, ast.For) else [])
    seen = collections.Counter()
    nodes = [n for s in scope for n in ast.walk(s) if not isinstance(n, ast.stmt) or n is s]

    def at(i, fn):
        def t(s):
            target = [n for sc in ([s.test] if test is not None else [s] if not isinstance(s, (ast.For, ast.Try, ast.With)) else
                                    ([s.iter] if isinstance(s, ast.For) else [])) for n in ast.walk(sc) if not isinstance(n, ast.stmt) or n is sc]
            fn(target[i])
            return s
        return t
    for i, n in enumerate(nodes):
        if isinstance(n, ast.Compare):
            for j, op in enumerate(n.ops):
                if type(op) in SWAP_CMP:
                    seen["cmp"] += 1
                    new = SWAP_CMP[type(op)]
                    out.append(("comparison %s to %s (%d)" % (type(op).__name__, new.__name__, seen["cmp"]),
                                at(i, lambda c, j=j, new=new: c.ops.__setitem__(j, new()))))
        elif isinstance(n, ast.BoolOp):
            seen["logic"] += 1
            out.append(("logic %s to %s (%d)" % (type(n.op).__name__.lower(), "or" if isinstance(n.op, ast.And) else "and", seen["logic"]),
                        at(i, lambda b: setattr(b, "op", ast.Or() if isinstance(b.op, ast.And) else ast.And()))))
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            seen["not"] += 1
            out.append(("a not dropped (%d)" % seen["not"], at(i, lambda u: setattr(u, "op", ast.UAdd()) or None)))
        elif isinstance(n, ast.Constant) and isinstance(n.value, bool):
            seen["bool"] += 1
            out.append(("%s made %s (%d)" % (n.value, not n.value, seen["bool"]), at(i, lambda c: setattr(c, "value", not c.value))))
        elif isinstance(n, ast.Constant) and isinstance(n.value, int) and seen["num"] < 2:
            seen["num"] += 1
            out.append(("the number %d made %d (%d)" % (n.value, n.value + 1, seen["num"]), at(i, lambda c: setattr(c, "value", c.value + 1))))
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value and seen["str"] < 1 and not isinstance(stmt, ast.Expr):
            seen["str"] += 1
            out.append(("a text emptied", at(i, lambda c: setattr(c, "value", ""))))
        elif isinstance(n, ast.Call) and seen["arg"] < 3:
            for k in range(min(2, len(n.args))):
                seen["arg"] += 1
                out.append(("an argument made None (%d)" % seen["arg"], at(i, lambda c, k=k: c.args.__setitem__(k, ast.Constant(None)))))
    if isinstance(stmt, ast.Return) and stmt.value is not None:
        out.append(("the result made None", lambda s: setattr(s, "value", ast.Constant(None)) or s))
    if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is not None:
        out.append(("the value made None", lambda s: setattr(s, "value", ast.Constant(None)) or s))
    if isinstance(stmt, ast.AugAssign):
        out.append(("the statement removed", lambda s: ast.Pass()))
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        out.append(("the call removed", lambda s: ast.Pass()))
    if isinstance(stmt, ast.If) and stmt.body:
        out.append(("the branch's body removed", lambda s: setattr(s, "body", [ast.Pass()]) or s))
    return out[:PER_STATEMENT]


class PyMutant(object):
    """One break of one Python statement, named for good by its place, its statement's text and its operator"""

    def __init__(self, key, place, k, st, what, op, node):
        self.key, self.place, self.k, self.st, self.what, self.op, self.node = key, place, k, st, what, op, node
        self.of, self.verdict = set(), {}

    @property
    def text(self):
        """the statement as this break makes it - what a report shows beside a survivor, the name cs.Mutant gives it (10-29: the
        loop's first pass on itself ran 1585 breaks, then died in the summary on a survivor's missing `text`)"""
        new = self.op(copy.deepcopy(self.node))
        nodes = new if isinstance(new, list) else [self.node if new is None else new]
        return "; ".join(ast.unparse(n) for n in nodes)

    def source(self, code):
        """the module's source with this one break made (ast.unparse: the copy loses comments, which no test reads)"""
        tree = copy.deepcopy(code.trees[self.st.file])
        node = _find(tree, self.st.node)
        for parent in ast.walk(tree):
            for field, value in ast.iter_fields(parent):
                if isinstance(value, list) and node in value:
                    i = value.index(node)
                    new = self.op(node)
                    value[i] = new if new is not None else node
                    ast.fix_missing_locations(tree)
                    return ast.unparse(tree)
        raise KeyError("no parent holds %s" % (self.st.node,))

    def record(self):
        return dict(self.verdict, place=self.place, what=self.what, of=sorted(self.of), file=cs.shown(self.st.file),
                    line=self.st.start, statement=re.sub(r"\s+", " ", self.st.text)[:120], mutant=self.what)


def failed_line(line):
    """a self-test line FAILS when its LAST verdict word is FAIL, MISSED or DISAGREES. Any such word anywhere was read as a failure
    until P5-D (10-29): a passing line describing "PASS/FAIL lines" made the self model's baseline fail on unchanged code"""
    words = re.findall(r"\b(PASS|FAIL|MISSED|DISAGREES)\b", line)
    return bool(words) and words[-1] != "PASS"


def code_print(co):
    """a code object's fingerprint WITH every nested code object's (a lambda's, a generator's, a comprehension's): only the outer
    bytecode was compared until P5-D (10-29), so a break inside a lambda was 'equivalent' and never run, even when it raised"""
    return (co.co_code, tuple(code_print(c) if hasattr(c, "co_code") else repr(c) for c in co.co_consts))


class PyRunner(object):
    """A model test as a command (meta.loop.tests: {runner: py_selftest, test: "<tool.py> --selftest", covers: [...]}), run in a
    COPY of the code's folder so a break never reaches the live tool; its FAIL lines name what caught it"""

    def __init__(self, tests):
        self.tests = tests

    def covering(self, path):
        out = []
        for t in self.tests:
            for c in ucm.as_list(t.get("covers")):
                cp = cs.src_path(c)
                if os.path.normpath(path) == cp or os.path.normpath(path).startswith(cp + os.sep):
                    out.append(t)
                    break
        return out

    @staticmethod
    def passed_in(folder, test, timeout=300):
        """the names of the checks a test script reports PASSING, as run_in names a failing one (no verdict word)"""
        cmd = test["test"].split()
        script = os.path.join(folder, os.path.basename(cmd[0]))
        r = subprocess.run([sys.executable, "-X", "utf8", script] + cmd[1:], cwd=ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return [re.sub(r"\s+PASS\b.*$", "", re.sub(r"\s+", " ", l.split("selftest:", 1)[1]).strip())[:90]
                for l in r.stdout.splitlines() if l.startswith("selftest:") and not failed_line(l) and re.search(r"\bPASS\b", l)
                and l.strip() != "selftest: PASS"]

    @staticmethod
    def run_in(folder, test, timeout=300):
        cmd = test["test"].split()
        script = os.path.join(folder, os.path.basename(cmd[0]))
        try:
            r = subprocess.run([sys.executable, "-X", "utf8", script] + cmd[1:], cwd=ROOT, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired:
            return True, ["(%s hung past %d s)" % (os.path.basename(cmd[0]), timeout)]
        # a check is named WITHOUT its verdict word, so the name matches its `# CHECKS:` line (closed_loop.py_checks) for L16
        failed = [re.sub(r"\s+(FAIL|MISSED|DISAGREES)\b.*$", "", re.sub(r"\s+", " ", l.split("selftest:", 1)[1]).strip())[:90]
                  for l in r.stdout.splitlines() if l.startswith("selftest:") and failed_line(l) and "selftest: FAIL" not in l]
        if r.returncode != 0 or failed:
            return True, failed or ["%s exited %d" % (os.path.basename(cmd[0]), r.returncode)]
        return False, []


class PyEffectCheck(object):
    """Each requirement's effect held to the Python, then every statement of the covered functions broken and run alone"""

    def __init__(self, code, runner=None, equivalent=None, workers=3):
        self.C, self.R, self.workers = code, runner, workers
        self.equivalent = dict(equivalent or {})
        self.mutants, self.log, self.passing = {}, [], []

    def matches(self, a, st):
        f, fn = None, None
        node = None
        try:
            node = _find(self.C.trees[st.file], st.node)
        except KeyError:
            return False
        names = {getattr(n, "id", None) or getattr(n, "attr", None) for n in ast.walk(node) if isinstance(n, (ast.Name, ast.Attribute))}
        if a.verb == "sets":
            targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
            tn = {getattr(n, "id", None) or getattr(n, "attr", None) for t in targets if t is not None for n in ast.walk(t)}
            return a.target in tn
        if a.verb == "calls":
            return any(isinstance(n, ast.Call) and (getattr(n.func, "id", None) == a.target or getattr(n.func, "attr", None) == a.target)
                       for n in ast.walk(node))
        if a.verb == "branches":
            test = getattr(node, "test", None)
            return isinstance(node, (ast.If, ast.While)) and test is not None and a.target in {
                getattr(n, "id", None) or getattr(n, "attr", None) for n in ast.walk(test)}
        if a.verb == "reads":
            return a.target in names
        return False

    def verdict(self, a):
        if a.verb is None or not a.at:
            return {"holds": None, "why": "malformed: one verb and a place, U18"}
        try:
            sts = self.C.method(str(a.at))
        except KeyError as e:
            return {"holds": None, "why": "its place names nothing: %s" % e}
        hit = [s for s in sts if self.matches(a, s)]
        v = {"holds": bool(hit), "matched": [re.sub(r"\s+", " ", s.text)[:120] for s in hit], "read": len(sts),
             "fp": sha("\n".join(s.text for s in (hit or sts))), "_hit": [(sts.index(s), s) for s in hit],
             "plant": "flips", "why": "%s at %s" % (a.verb, a.at) if hit else "no statement in %s %s %s" % (a.at, a.verb, a.target)}
        return v

    def collect(self, hits, files):
        muts, sids = {}, {}

        def sid(place, k):
            if place not in sids:
                seen, names = collections.Counter(), []
                for s in self.C.method(place):
                    h = sha(re.sub(r"\s+", " ", s.text.strip()))[:6]
                    seen[h] += 1
                    names.append(h + ("" if seen[h] == 1 else "#%d" % seen[h]))
                sids[place] = names
            return sids[place][k]

        def add(place, k, st, of):
            node = _find(self.C.trees[st.file], st.node)
            for what, op in operators(node):
                key = "%s@%s: %s" % (place, sid(place, k), what)
                muts.setdefault(key, PyMutant(key, place, k, st, what, op, node)).of.add(of)
        for aid, (a, sts) in sorted(hits.items()):
            for k, st in sts:
                add(str(a.at), k, st, aid)
        decisions = self.C.branches(files)
        for key, place, st in decisions:
            add(place, int(key.rsplit("#b", 1)[1]), st, key)
        for place in self.C.places(files):
            sts = self.C.method(place)
            if sts:
                f, fn = self.C.index[place]
                key = "%s@%s: the function returns at once" % (place, sid(place, 0))
                st0 = sts[0]

                def early(s, fn_loc=(fn.lineno, fn.col_offset, type(fn).__name__)):
                    return [ast.Return(ast.Constant(None)), s]
                m = muts.setdefault(key, PyMutant(key, place, 0, st0, "the function returns at once", early,
                                                  _find(self.C.trees[st0.file], st0.node)))
                m.of.update(k for k, p, _ in decisions if p == place)
        return muts

    def _bytecode(self, src, place):
        """the compiled bytecode of one function in a module's source, or None"""
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return None
        parts = place.split(".")[1:]
        node = tree
        for p in parts:
            node = next((n for n in getattr(node, "body", []) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == p), None)
            if node is None:
                return None
        mod = ast.Module(body=[node], type_ignores=[])
        try:
            co = compile(mod, "<m>", "exec")
        except SyntaxError:
            return None
        fco = next((c for c in co.co_consts if hasattr(c, "co_code")), None)
        return None if fco is None else code_print(fco)

    def behave(self, muts):
        if self.R is None:
            for m in muts.values():
                m.verdict = {"killed": None, "why": "no tests named in meta.loop.tests"}
            return
        base = {}
        live = []
        for m in muts.values():
            tests = self.R.covering(m.st.file)
            if not tests:
                m.verdict = {"killed": None, "why": "no test covers %s (meta.loop.tests covers)" % cs.shown(m.st.file)}
                continue
            try:
                src = m.source(self.C)
            except (KeyError, ValueError, TypeError, AttributeError) as e:
                m.verdict = {"killed": None, "why": "the break could not be built: %s" % e}
                continue
            try:
                compile(src, m.st.file, "exec")
            except SyntaxError as e:
                m.verdict = {"killed": None, "why": "does not compile: %s" % e}
                continue
            if m.st.file not in base:
                base[m.st.file] = ast.unparse(self.C.trees[m.st.file])
            if self._bytecode(src, m.place) is not None and self._bytecode(src, m.place) == self._bytecode(base[m.st.file], m.place):
                m.verdict = {"killed": None, "equivalent": "TCE", "why": "equivalent: its function compiles to the original's bytecode"}
                continue
            m.src = src
            live.append(m)
        # the tests on the code as it is, once: a test that fails with no break makes every kill meaningless
        folder = tempfile.mkdtemp(prefix="py_base_")
        try:
            for f in self.C.files:
                shutil.copy2(f, os.path.join(folder, os.path.basename(f)))
            for f in os.listdir(HERE):
                if f.endswith(".py") and not os.path.exists(os.path.join(folder, f)):
                    shutil.copy2(os.path.join(HERE, f), os.path.join(folder, f))
            self._tests_into(folder)
            ran = []
            for t in self.R.tests:
                failed, said = PyRunner.run_in(folder, t)
                if failed:
                    raise RuntimeError("the tests do not pass on the code as it is (%s: %s) - no break's verdict would mean anything" % (t["test"], said[:2]))
                ran += PyRunner.passed_in(folder, t)
            # the tests that RAN are each script's checks, named as the runner names them (10-29: only the scripts' commands were kept,
            # so every Python requirement's named check read as "never ran" in L16)
            self.passing = [t["test"] for t in self.R.tests] + ran
        finally:
            shutil.rmtree(folder, ignore_errors=True)

        def one(m):
            d = tempfile.mkdtemp(prefix="py_mut_")
            try:
                for f in os.listdir(HERE):
                    if f.endswith(".py"):
                        shutil.copy2(os.path.join(HERE, f), os.path.join(d, f))
                for f in self.C.files:
                    shutil.copy2(f, os.path.join(d, os.path.basename(f)))
                self._tests_into(d)
                io.open(os.path.join(d, os.path.basename(m.st.file)), "w", encoding="utf-8").write(m.src)
                for t in self.R.covering(m.st.file):
                    failed, said = PyRunner.run_in(d, t)
                    if failed:
                        return {"killed": True, "by": said, "why": "failed: " + "; ".join(said[:3])}
                return {"killed": False, "by": [], "why": "every test passed"}
            finally:
                shutil.rmtree(d, ignore_errors=True)
        with cf.ThreadPoolExecutor(max_workers=self.workers) as ex:
            for m, v in zip(live, ex.map(one, live)):
                m.verdict = v
        self.log.append("%d break(s) run, %d proven equivalent by bytecode" % (len(live), sum(1 for m in muts.values() if m.verdict.get("equivalent"))))

    def _tests_into(self, folder):
        """each test's script beside the copied code, unless the code already holds it (a tool's own selftest)"""
        for t in self.R.tests:
            src = cs.src_path(t["test"].split()[0])
            dst = os.path.join(folder, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

    summary = cs.CsEffectCheck.summary
    names = cs.CsEffectCheck.names

    def checkpoint_load(self, ck, stamp):
        """the verdicts a checkpoint holds for THIS code and these tests, put on the matching breaks; how many"""
        if not os.path.exists(ck):
            return 0
        old = json.load(io.open(ck, encoding="utf-8"))
        if old.get("stamp") != stamp:
            return 0
        n = 0
        for k, v in (old.get("verdicts") or {}).items():
            if k in self.mutants and v:
                self.mutants[k].verdict = v
                n += 1
        return n

    SAVE_EVERY = 60

    def checkpoint_save(self, ck, stamp):
        """every verdict so far, written whole or not at all (a temp file, then a rename): a reader never sees half a file"""
        tmp = ck + ".tmp"
        io.open(tmp, "w", encoding="utf-8").write(json.dumps({"stamp": stamp, "verdicts": {k: dict(m.verdict) for k, m in list(self.mutants.items())}},
                                                            sort_keys=True))
        os.replace(tmp, ck)

    def run(self, model_dir, out_path, kill=True, only=None):
        model = ucm.UseCaseModel.load(model_dir)
        assertions, says, named, hits = {}, {}, {}, {}
        for r in model.requirements.values():
            for a in r.effect:
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
        covered = sorted({f for f in self.C.files if self.R is not None and self.R.covering(f)})
        branches = {key: (place, st) for key, place, st in self.C.branches(covered)}
        doc = {"format": "effect-verdicts/1", "adapter": "py_effect_check", "code": sha("".join(sorted(self.C.files.values()))),
               "assertions": assertions, "names": named, "says": says}
        if kill:
            self.mutants = self.collect(hits, covered)
            print("py_effect_check: %d break(s) of %d effect(s) and %d statement(s) in %d covered file(s)" % (
                len(self.mutants), len(hits), len(branches), len(covered)))
            # THE CHECKPOINT (10-29: the first self pass lost 65 minutes of runs to a crash in the summary after them): every
            # verdict is written beside the report the moment the runs end, and a later run of the SAME code and the same tests
            # reuses the verdicts it holds and runs only the breaks it lacks
            ck = out_path + ".mutants.json"
            # the tests' SCRIPTS hashed as well as their commands: an edited test must never reuse the verdicts of the old one
            # and the ADAPTER'S OWN SOURCE: how a break is made and judged is part of every verdict (10-29: verdicts named under the
            # old runner were reused after the runner changed, and L16 compared new check names with old killer names)
            stamp = {"code": doc["code"], "tests": sha(json.dumps(self.R.tests, sort_keys=True, default=str) + "".join(
                io.open(os.path.join(ROOT, t["test"].split()[0]), encoding="utf-8").read() for t in self.R.tests)) if self.R is not None else None,
                     "engine": sha(io.open(os.path.abspath(__file__), encoding="utf-8").read())}
            reused = self.checkpoint_load(ck, stamp)
            if reused:
                print("py_effect_check: %d verdict(s) reused from the checkpoint (same code, same tests)" % reused)
            # saved AS THE RUNS GO, every SAVE_EVERY seconds, not only at the end: a pass that dies an hour in keeps what it ran
            stop = threading.Event()

            def saver():
                while not stop.wait(self.SAVE_EVERY):
                    self.checkpoint_save(ck, stamp)
            keeper = threading.Thread(target=saver, daemon=True)
            keeper.start()
            try:
                self.behave({k: m for k, m in self.mutants.items() if not m.verdict})
            finally:
                stop.set()
                keeper.join()
            self.checkpoint_save(ck, stamp)
            for aid in assertions:
                if aid in hits:
                    assertions[aid]["kill"] = self.summary([m for m in self.mutants.values() if aid in m.of])
            doc["branches"] = {key: dict(self.summary([m for m in self.mutants.values() if key in m.of]), place=place,
                                         text=re.sub(r"\s+", " ", st.text)[:120]) for key, (place, st) in sorted(branches.items())}
            doc["mutants"] = {k: m.record() for k, m in sorted(self.mutants.items())}
            doc["tests"] = self.passing
            doc["log"] = self.log
            doc["equivalent_declared"] = dict(self.equivalent)
        io.open(out_path, "w", encoding="utf-8").write(json.dumps(doc, indent=1, sort_keys=True))
        return doc


SELF_PY = '''
def pay(money, price):
    """pays when it can"""
    if price < 0 or price > money:
        return money
    money -= price
    log("paid", price)
    return money


def log(what, n):
    pass
'''
SELF_TEST = '''import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wallet
ok = wallet.pay(10, 3) == 7 and wallet.pay(10, 11) == 10 and wallet.pay(10, -1) == 10 and wallet.pay(10, 10) == 0
print("selftest: pay %s" % ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
'''


def selftest():
    """the adapter on a planted module: its shape verdicts, its breaks, a kill, a survivor and an equivalent"""
    ok = True
    t = tempfile.mkdtemp(prefix="py_effect_selftest_")
    p = os.path.join(t, "wallet.py")
    io.open(p, "w", encoding="utf-8").write(SELF_PY)
    io.open(os.path.join(t, "wallet_test.py"), "w", encoding="utf-8").write(SELF_TEST)
    try:
        C = PyCode([p])
        chk = PyEffectCheck(C)
        A = lambda d: ucm.Assertion("R", 1, d)
        cases = [("branches price at wallet.pay holds", A({"branches": "price", "at": "wallet.pay"}), True),
                 ("calls log at wallet.pay holds", A({"calls": "log", "at": "wallet.pay"}), True),
                 ("sets money at wallet.pay holds", A({"sets": "money", "at": "wallet.pay"}), True),
                 ("calls refund at wallet.pay fails", A({"calls": "refund", "at": "wallet.pay"}), False),
                 ("a place that names nothing is unread", A({"sets": "money", "at": "wallet.refund"}), None)]
        for what, a, holds in cases:
            v = chk.verdict(a)
            good = v.get("holds") is holds
            ok &= good
            print("selftest: SHAPE %-44s -> %s %s" % (what, v.get("holds"), "PASS" if good else "FAIL"))
        stmt = _find(C.trees[os.path.normpath(p)], C.method("wallet.pay")[0].node)
        labels = [w for w, _ in operators(stmt)]
        good = all(x in labels for x in ("the condition negated", "comparison Lt to LtE (1)", "comparison Gt to GtE (2)", "logic or to and (1)"))
        ok &= good
        print("selftest: BREAK the guard broken %d ways: %s %s" % (len(labels), labels[:5], "PASS" if good else "FAIL"))
        runner = PyRunner([{"runner": "py_selftest", "test": os.path.join(t, "wallet_test.py"), "covers": [p]}])
        chk = PyEffectCheck(C, runner, workers=2)
        muts = chk.collect({}, [os.path.normpath(p)])
        HERE_save = HERE
        chk.behave(muts)
        v = {m.what: m.verdict.get("killed") for m in muts.values() if m.place == "wallet.pay"}
        good = v.get("the condition negated") is True and v.get("the call removed") is False and v.get("comparison Gt to GtE (2)") is True
        ok &= good
        print("selftest: KILL  the guard's negation caught, the log call's removal survives (no test reads it) %s" % ("PASS" if good else "FAIL"))
        surv = [m for m in muts.values() if m.place == "wallet.pay" and m.verdict.get("killed") is False]
        shown = json.dumps(chk.summary(surv))
        good = bool(surv) and all(m.what in shown for m in surv[:3]) and "->" in shown
        ok &= good
        print("selftest: SURVIVOR the summary names each survivor with the statement its break makes (the first self pass died "
              "here) %s" % ("PASS" if good else "FAIL"))
        ck, stamp = os.path.join(t, "v.json.mutants.json"), {"code": "c1", "tests": "t1"}
        chk.mutants = muts
        chk.checkpoint_save(ck, stamp)
        again = PyEffectCheck(C, runner, workers=2)
        again.mutants = again.collect({}, [os.path.normpath(p)])
        same = again.checkpoint_load(ck, stamp)
        other = PyEffectCheck(C, runner, workers=2)
        other.mutants = other.collect({}, [os.path.normpath(p)])
        changed = other.checkpoint_load(ck, {"code": "c1", "tests": "t2"})
        half = PyEffectCheck(C, runner, workers=2)
        half.mutants = half.collect({}, [os.path.normpath(p)])
        for i, (k, m) in enumerate(sorted(half.mutants.items())):
            m.verdict = dict(muts[k].verdict) if i % 2 == 0 else {}
        half.checkpoint_save(ck, stamp)
        resumed = PyEffectCheck(C, runner, workers=2)
        resumed.mutants = resumed.collect({}, [os.path.normpath(p)])
        part = resumed.checkpoint_load(ck, stamp)
        good = same == len(muts) and changed == 0 and all(not m.verdict for m in other.mutants.values()) \
            and part == (len(muts) + 1) // 2 and not os.path.exists(ck + ".tmp")
        ok &= good
        print("selftest: CHECKPOINT a rerun of the same code and tests reuses every verdict; changed tests reuse none; a run that "
              "stopped halfway resumes from the half it saved %s" % ("PASS" if good else "FAIL"))
    finally:
        shutil.rmtree(t, ignore_errors=True)
    good = not failed_line("selftest: a test's PASS/FAIL lines are read PASS") and failed_line("selftest: x FAIL got [1]") \
        and failed_line("selftest: y MISSED") and not failed_line("selftest: nothing judged here")
    a1 = compile("def f(xs):\n    return sorted(xs, key=lambda x: x + 1)\n", "<a>", "exec").co_consts
    a2 = compile("def f(xs):\n    return sorted(xs, key=lambda x: x + 2)\n", "<a>", "exec").co_consts
    fa = next(c for c in a1 if hasattr(c, "co_code"))
    fb = next(c for c in a2 if hasattr(c, "co_code"))
    good = good and code_print(fa) != code_print(fb) and code_print(fa) == code_print(fa)
    ok &= good
    print("selftest: RUNNER a line's LAST verdict word decides it (a pass describing PASS/FAIL passes); a break inside a lambda changes "
          "the fingerprint, so it is never 'equivalent' by bytecode %s" % ("PASS" if good else "FAIL"))
    print("selftest: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--no-kill", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.folder or not a.out:
        ap.error("<model folder> --out <verdicts.json>, or --selftest")
    model = ucm.UseCaseModel.load(a.folder)
    loop = model.loop or {}
    eq = {e["mutant"]: e.get("why") for e in ucm.as_list(loop.get("equivalent")) if isinstance(e, dict) and e.get("mutant")}
    runner = PyRunner([t for t in ucm.as_list(loop.get("tests")) if isinstance(t, dict)]) if loop.get("tests") else None
    doc = PyEffectCheck(PyCode(ucm.as_list((loop.get("code") or {}).get("roots"))), runner, eq, a.workers).run(a.folder, a.out, kill=not a.no_kill)
    ms = doc.get("mutants") or {}
    print("py_effect_check: %d break(s): %d killed, %d survived, %d no verdict -> %s" % (
        len(ms), sum(1 for m in ms.values() if m.get("killed") is True), sum(1 for m in ms.values() if m.get("killed") is False),
        sum(1 for m in ms.values() if m.get("killed") is None), a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
