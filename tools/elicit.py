#!/usr/bin/env python3
"""elicit.py - get the rules out of the person who holds them, widely and twice, before any code is written.

    python -X utf8 elicit.py start  <folder> --acts "borrow, return" --things "a loan, a late fee" --phases "the visit" [--producer "<command with {q}>"]
    python -X utf8 elicit.py grid   <folder>                         # the squares nobody has asked about yet, each with its question in plain words
    python -X utf8 elicit.py ask    <folder> "<square>: <question>" ...   # or --file batch.txt, one per line. Asks them together and prints the replies
    python -X utf8 elicit.py answer <folder> <n> "<their words>"     # with no --producer: the person replied to question n
    python -X utf8 elicit.py rules  <folder> --file rules.txt        # the rules as you read them, each with its witnesses (the format is below)
    python -X utf8 elicit.py skip   <folder> <square> "<why it does not apply>"
    python -X utf8 elicit.py check  <folder> [--tests test_file.py ...]   # what is still open, most important first. Exit 1 while anything is, 0 when nothing is
    python -X utf8 elicit.py export <folder> --rules RULES.md        # one plain sentence per line, the witnesses in a comment
    python -X utf8 elicit.py breaks --code shop.py --test "shop_test.py" [--workers 8] [--minutes 8] [--out breaks.txt]   # one pass: every break of the code, which the tests miss
    python -X utf8 elicit.py --selftest

WHY THIS EXISTS. A closed loop holds code to its requirements. It holds a wrong requirement exactly as tightly as a right one, and its tests are
written from the same reading as the code, so it cannot tell them apart: on a held trial a one-operator break that made a wrong rule right was
caught by the agent's own tests as a regression. So the requirements have to be right BEFORE the loop closes, and nothing later will catch it if they
are not. Three things went wrong there, in this order of cost, and each has a class below:
  - An answer was believed on one hearing. Producers contradict themselves and misspeak; an agent usually held a right answer on the same point as the
    wrong one and never compared them. -> a Rule stands on two Witnesses of different Forms, and Against witnesses keep it open.
  - A short answer was taken as final: a "not specified" that a second wording would have answered, a refused question never put again. -> Reply and Readiness.
  - Nobody asked: what a rule's exceptions are, what an act does to whose turn it is, whether a turn may be passed. -> the Grid, whose squares come from the
    acts and things the text and the interface name, not from a reading of the game.
And asking one question per command costs about twice the tokens of asking a batch. -> ProducerRoute asks together.

The closing comes last and once: `breaks` is one pass over the finished code. A break nobody caught is a question about a rule, or it is nothing. Chasing
every last one before the rules are right cost a long time on the trial, and no accuracy.

CLASS MODEL. Evidence for each is in its docstring.
  Form            how a question is put: open, closed, scenario or choice
  Reply           what came back: spoken, silent or unusable
  Asked           one question on the record, with its number, its square, its form and its reply
  Grid            the squares: every act, thing and phase crossed with the stems asked of its kind
  Rule            one sentence as the agent reads it, with the numbers of the answers for it and against it; its state is computed, never declared
  Ledger          the folder: asked.jsonl, grid.json, rules.txt; a session that stops resumes with nothing repeated
  ProducerRoute   how a question reaches the person: a command run per question, several at a time, or by hand
  Readiness       what is still open, in order: a route failure stops everything, then conflicts, one-witness rules, unanswered and silent squares, unasked squares, untested rules
  BreakRun        the one pass at the end, with no model: every break of the code, which the tests miss, grouped by function

rules.txt, one rule per line, `|` between the fields; a blank line or a # line is ignored:
    R1 | for 12 40 | Only a member may borrow.
    R2 | for 13 41 | against 77 | A member may hold at most three tools.
    R3 | silent 14 52 | MY READING: a loan with no due date is refused.
"""
import argparse
import collections
import concurrent.futures as cf
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time


class Form(object):
    """How a question is put. No form is trusted alone. On a small trial against a guarded producer a choice between two readings was right most often, an open question next, and a concrete
    situation about half the time; a false sentence read back was confirmed as readily as a true one, so reading a rule back proves little. On the held trial's logs an answer opening with
    "No" was wrong about twice as often as one opening with "Yes", and rules of arithmetic drew wrong answers in every form. So the open form comes first, a choice is the second hearing, and
    a rule stands on two hearings in different forms: the point of the second is to disagree when the first was wrong."""
    OPEN, CLOSED, SCENARIO, CHOICE = "open", "closed", "scenario", "choice"
    _CLOSED = re.compile(r"^(?:[^,?]{0,90},\s*)?(is|are|does|do|did|may|can|must|should|will|would|has|have|could)\b", re.I)
    _CHOICE = re.compile(r"\(a\).*\(b\)|\bwhich of these\b", re.I)
    _SCENE = re.compile(r"\b(\d+\b.*\b\d+|player \d|seat \d|team [a-z]\b|holds? the\b|has \d+\b)|^[A-Z][^.?!]*\.\s", re.S)

    @classmethod
    def of(cls, q):
        q = q.strip()
        if cls._CHOICE.search(q):
            return cls.CHOICE
        last = re.split(r"(?<=[.:;])\s+", q)[-1]
        if len(re.split(r"(?<=[.:;])\s+", q)) > 1 or cls._SCENE.search(q):
            return cls.SCENARIO
        return cls.CLOSED if cls._CLOSED.match(last) else cls.OPEN


class Reply(object):
    """What came back. SILENT: the person said the rules do not say; on the held trial nearly a third of those were false, and a second wording often got the rule. UNUSABLE: a
    refusal or a request to ask differently; nothing was learned and the question stands. Both keep a square open until it is put again in other words."""
    SPOKEN, SILENT, UNUSABLE = "spoken", "silent", "unusable"
    _SILENT = re.compile(r"^\W*(not specified|unspecified|unknown|i do not know|i don't know|don't know|no idea|the rules? (are|is|do|does) (silent|not say))\b", re.I)
    _UNUSABLE = re.compile(r"ask again|ask about one thing|ask me about|too much for one|one question|no usable|cannot be reached|in other words", re.I)

    @classmethod
    def kind(cls, text):
        t = (text or "").strip()
        if not t or cls._UNUSABLE.search(t):
            return cls.UNUSABLE
        return cls.SILENT if cls._SILENT.search(t) else cls.SPOKEN


class Asked(object):
    """One question on the record. `square` ties it to the grid or to a rule's own label; `reply` is None until it is answered."""

    def __init__(self, n, square, q, reply=None):
        self.n, self.square, self.q, self.reply = n, square, q, reply
        self.form = Form.of(q)

    @property
    def kind(self):
        return None if self.reply is None else Reply.kind(self.reply)

    def record(self):
        return {"n": self.n, "square": self.square, "q": self.q, "form": self.form, "reply": self.reply}


class Grid(object):
    """The squares. Each thing the text or the interface names, crossed with what must be known about its kind. It is generated, so it cannot favour a game, and it covers the
    questions an agent's own walk tends to leave out: the exceptions to a rule, whether a turn may be passed, what an act does to the play around it, what happens when a claim
    has nothing behind it. A stem that does not apply is skipped with a reason, not left blank. The stems speak plainly: a person does not know the interface's names, and a name
    only the interface uses comes back as silence, so put each question in the person's own words. Each stem asks one thing: on a dry run against a guarded producer, a stem that asked two
    or said "what else" was turned away unheard."""
    ACT = (("who", "Who may {x}?"), ("when", "At what moments may {x} be done?"), ("forbidden", "When is {x} not allowed?"), ("after", "What happens right after {x}?"),
           ("next", "Who acts next after {x}?"), ("pass", "May a player or side end its turn without {x}?"), ("waiting", "What is blocked while {x} waits on someone?"),
           ("exceptions", "Is there any moment or state of the game, such as near the end or near a winning score, where the rule for {x} is different?"))
    THING = (("counts", "What is the least that counts as {x}?"), ("owner", "Whom does {x} belong to?"), ("compare", "When there are two of {x}, how is the better one decided?"),
             ("tie", "What happens when two of {x} are exactly equal?"), ("none", "What happens when nobody has {x}?"), ("claimed", "What happens when someone claims {x} and does not have it?"),
             ("decided", "How is {x} finally settled?"), ("worth", "What changes what {x} is worth?"))
    PHASE = (("starts", "What starts {x}?"), ("ends", "What ends {x}?"), ("early", "How can {x} end early?"), ("next", "What happens straight after {x}?"), ("limits", "What may nobody do during {x}?"))

    def __init__(self, acts, things, phases):
        self.acts, self.things, self.phases = acts, things, phases

    def squares(self):
        """{square: question}, in the order to ask: phases first because they frame the rest, then acts, then things"""
        out = collections.OrderedDict()
        for names, stems in ((self.phases, self.PHASE), (self.acts, self.ACT), (self.things, self.THING)):
            for x in names:
                for stem, text in stems:
                    out["%s.%s" % (re.sub(r"\W+", "_", x.strip()).strip("_"), stem)] = text.format(x=x.strip())
        return out

    @staticmethod
    def acts_in(text):
        """the method names an interface document gives in backticks, in the order it gives them, without constructors and readers: a person can be asked about an act, not about a getter"""
        seen = []
        for name in re.findall(r"`(?:\w+\.)?([a-z_]\w*)\(", text):
            if not name.startswith(("from_", "get_", "_")) and name not in ("state", "status") and name not in seen:
                seen.append(name)
        return seen


class Rule(object):
    """One sentence as the agent reads it, and the answers it stands on. The state is computed from the record, so a rule cannot be called settled by saying so.
    Two witnesses of different forms, because a wrong answer and a right one on the same point usually sat in the same log on the held trial, uncompared. An answer against keeps it in CONFLICT until the answers for it outnumber those against by two: a third and fourth hearing, not a choice by preference."""
    STANDS, READING, ONE, SAME_FORM, CONFLICT = "stands", "reading", "one witness", "same form", "conflict"

    def __init__(self, rid, sentence, witnesses=(), against=(), silent=()):
        self.id, self.sentence = rid, sentence
        self.witnesses, self.against, self.silent = list(witnesses), list(against), list(silent)

    def state(self, asked):
        """asked: {n: Asked}"""
        if self.silent and not self.witnesses:
            forms = {asked[n].form for n in self.silent if n in asked}
            told = [n for n in self.silent if n in asked and asked[n].kind == Reply.SILENT]
            return self.READING if len(told) >= 2 and (len(forms) >= 2 or len({asked[n].q for n in told}) >= 2) else self.ONE
        if self.against and len(self.witnesses) < len(self.against) + 2:
            return self.CONFLICT
        spoken = [asked[n] for n in self.witnesses if n in asked and asked[n].kind == Reply.SPOKEN]
        if len(spoken) < 2:
            return self.ONE
        forms = {a.form for a in spoken}
        if len(forms) < 2:
            return self.SAME_FORM
        return self.STANDS

    @staticmethod
    def parse(line):
        """`R1 | for 12 40 | against 77 | The sentence.` -> a Rule; refuses a line it cannot read rather than dropping it"""
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3 or not parts[0] or not parts[-1]:
            raise ValueError("a rule line is `id | for 1 2 [| against 3] | sentence`, or `id | silent 1 2 | sentence`: %r" % line[:90])
        r = Rule(parts[0], parts[-1])
        for field in parts[1:-1]:
            word, _, nums = field.partition(" ")
            if word not in ("for", "against", "silent") or not re.fullmatch(r"[\d\s]+", nums or "x"):
                raise ValueError("%s: a field is `for 1 2`, `against 3` or `silent 4 5`, not %r" % (parts[0], field))
            {"for": r.witnesses, "against": r.against, "silent": r.silent}[word].extend(int(x) for x in nums.split())
        return r


class Ledger(object):
    """The folder. asked.jsonl is append-only with its last record for a number winning, so a stopped session resumes and a reply added later is kept; rules.txt is the agent's own file,
    read whole each time, so the record never disagrees with what the agent last wrote."""

    def __init__(self, folder):
        self.folder = folder
        self.path = os.path.join(folder, "asked.jsonl")
        self.grid_path = os.path.join(folder, "grid.json")
        self.rules_path = os.path.join(folder, "rules.txt")
        self.asked = collections.OrderedDict()
        if os.path.exists(self.path):
            for line in io.open(self.path, encoding="utf-8"):
                if line.strip():
                    d = json.loads(line)
                    self.asked[d["n"]] = Asked(d["n"], d["square"], d["q"], d.get("reply"))

    def write_grid(self, doc):
        os.makedirs(self.folder, exist_ok=True)
        json.dump(doc, io.open(self.grid_path, "w", encoding="utf-8"), indent=1)

    def grid_doc(self):
        if not os.path.exists(self.grid_path):
            raise SystemExit("no grid in %s: run `elicit.py start` first" % self.folder)
        return json.load(io.open(self.grid_path, encoding="utf-8"))

    def grid(self):
        d = self.grid_doc()
        return Grid(d["acts"], d["things"], d["phases"])

    def next_n(self):
        return max(self.asked or [0]) + 1

    def same_words(self, q):
        """an earlier question in the same words: a person may repeat themselves, and a program will, so it teaches nothing"""
        key = re.sub(r"\W+", " ", q).strip().lower()
        for a in self.asked.values():
            if re.sub(r"\W+", " ", a.q).strip().lower() == key:
                return a
        return None

    def put(self, a):
        self.asked[a.n] = a
        with io.open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(a.record(), ensure_ascii=False) + "\n")

    def rules(self):
        if not os.path.exists(self.rules_path):
            return []
        out = []
        for line in io.open(self.rules_path, encoding="utf-8"):
            if line.strip() and not line.lstrip().startswith("#"):
                out.append(Rule.parse(line))
        return out


class ProducerRoute(object):
    """How a question reaches the person who holds the rules. A command run once per question with {q} filled in, several at a time: asking a batch in one step cost about half the tokens per
    question of asking one per step on the held trial. A command that fails is a broken route, not an answer: nothing is recorded and the run says so."""

    def __init__(self, command, workers=6, timeout=600):
        self.command, self.workers, self.timeout = command, workers, timeout

    def one(self, q):
        parts = [p.replace("{q}", q) for p in shlex.split(self.command, posix=True)]
        r = subprocess.run(parts, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=self.timeout)
        if r.returncode != 0:
            raise RuntimeError("the route failed with exit %d: %s" % (r.returncode, (r.stderr or r.stdout).strip()[:200]))
        return r.stdout.strip()

    def many(self, qs):
        with cf.ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(self.one, qs))


class Readiness(object):
    """What is still open, most important first. Each line says what to do about it, so the next step is never a judgement about the tool. Exit 1 while anything is open."""

    def __init__(self, ledger, tests=()):
        self.L, self.tests = ledger, list(tests)

    def tested_rules(self):
        """rule ids named on `# CHECKS:` lines. A test file that cannot be read is an error, never read as having no checks"""
        named = set()
        for t in self.tests:
            for line in io.open(t, encoding="utf-8"):
                m = re.match(r"\s*#\s*CHECKS:\s*(.+)", line)
                if m:
                    named.update(re.findall(r"[A-Za-z][\w-]*\d\b", m.group(1)))
        return named

    def lines(self):
        L = self.L
        doc = L.grid_doc()
        out = collections.OrderedDict((k, []) for k in ("conflict", "one witness", "same form", "bad rule", "unanswered", "silent once", "unasked", "untested"))
        rules = L.rules()
        for r in rules:
            for n in r.witnesses + r.against + r.silent:
                if n not in L.asked:
                    out["bad rule"].append("%s cites question %d, which is not on the record" % (r.id, n))
                elif L.asked[n].kind == Reply.UNUSABLE and n not in r.silent:
                    out["bad rule"].append("%s stands on question %d, whose reply taught nothing" % (r.id, n))
                elif n in r.silent and L.asked[n].kind != Reply.SILENT:
                    out["bad rule"].append("%s calls question %d silent, and it was answered: %s" % (r.id, n, (L.asked[n].reply or "")[:70]))
            st = r.state(L.asked)
            if st == Rule.CONFLICT:
                out["conflict"].append("%s has answers against it (%s): ask again in a new form, an open one first, and record each hearing" % (r.id, " ".join(map(str, r.against))))
            elif st == Rule.ONE:
                out["one witness"].append("%s stands on one hearing: ask once more in a different form, an open one if the first was yes or no" % r.id)
            elif st == Rule.SAME_FORM:
                out["same form"].append("%s has its hearings in one form: put it another way, open if they were yes or no" % r.id)
        squares = Grid(doc["acts"], doc["things"], doc["phases"]).squares()
        skipped = doc.get("skipped") or {}
        by = collections.defaultdict(list)
        for a in L.asked.values():
            by[a.square].append(a)
        for sq, words in squares.items():
            if sq in skipped:
                continue
            got = by.get(sq, [])
            kinds = [a.kind for a in got]
            if Reply.SPOKEN in kinds:
                continue
            if not got:
                out["unasked"].append("%s: %s" % (sq, words))
            elif kinds.count(Reply.SILENT) == 1 and Reply.UNUSABLE not in kinds:
                out["silent once"].append("%s was answered 'not specified' once (question %d): put it again in other words, as a situation, before calling the rules silent" % (sq, got[-1].n))
            elif Reply.UNUSABLE in kinds or None in kinds:
                out["unanswered"].append("%s has no answer yet (question %d): narrow it to one thing and ask again" % (sq, got[-1].n))
        if self.tests:
            named = self.tested_rules()
            for r in rules:
                if r.id not in named:
                    out["untested"].append("%s has no test naming it on a # CHECKS: line" % r.id)
        return out


BELOW_NORMAL = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)     # a pass runs hundreds of processes; the person at this machine comes first


class BreakRun(object):
    """The one pass at the end, with no model and no graph: every break of the code, run against the tests, and the ones they miss, by function. A survivor is read as a question: does it
    name a rule nobody asked about? If so, ask. If not, leave it and say so. A clean pass says the tests hold the code to the rules as the agent read them, and nothing about whether the
    rules are right. Python only: `closed_loop.py run` is the whole loop, in C# as well."""

    def __init__(self, code, test, workers=8, minutes=8.0, timeout=120):
        self.code, self.test, self.workers, self.minutes, self.timeout = os.path.abspath(code), test, workers, minutes, timeout

    def run(self):
        import py_effect_check as pe      # shipped beside this file; loaded only here so the asking needs nothing but the standard library
        os.environ.setdefault("CLOSED_LOOP_ROOT", os.path.dirname(self.code))
        code = pe.PyCode([self.code])
        muts = pe.PyEffectCheck(code).collect({}, list(code.files))
        here = os.path.dirname(self.code)
        cmd = shlex.split(self.test, posix=True)
        base = self._folder(here)
        try:
            base_failed = subprocess.run([sys.executable, "-X", "utf8"] + cmd, cwd=base, capture_output=True, timeout=self.timeout, creationflags=BELOW_NORMAL).returncode != 0
        finally:
            shutil.rmtree(base, ignore_errors=True)
        if base_failed:
            raise SystemExit("the tests fail on the code as it is: `%s`. A break's verdict would mean nothing. Fix that first." % self.test)
        # a test that imports the real file by an absolute path, or a script that exits 0 whatever happens, would make every break look missed: a module that cannot load must fail them
        probe = self._folder(here)
        try:
            io.open(os.path.join(probe, os.path.basename(self.code)), "w", encoding="utf-8").write("raise RuntimeError('a copy that cannot load')\n")
            if subprocess.run([sys.executable, "-X", "utf8"] + cmd, cwd=probe, capture_output=True, timeout=self.timeout, creationflags=BELOW_NORMAL).returncode == 0:
                raise SystemExit("the tests passed with %s unable to load, so they do not read the copy or do not exit non-zero on a failure. Import it from the test's own folder, by no absolute path, and exit 1 when a check fails." % os.path.basename(self.code))
        finally:
            shutil.rmtree(probe, ignore_errors=True)
        deadline = time.time() + self.minutes * 60
        survivors, caught, broken, unrun = [], 0, [], 0

        def one(item):
            key, m = item
            if time.time() > deadline:
                return key, "unrun"
            try:
                src = m.source(code)
                compile(src, m.st.file, "exec")
            except (KeyError, ValueError, TypeError, AttributeError, SyntaxError):
                return key, "unbuilt"
            d = self._folder(here)
            try:
                io.open(os.path.join(d, os.path.basename(self.code)), "w", encoding="utf-8").write(src)
                try:
                    r = subprocess.run([sys.executable, "-X", "utf8"] + cmd, cwd=d, capture_output=True, timeout=self.timeout, creationflags=BELOW_NORMAL)
                except subprocess.TimeoutExpired:
                    return key, "caught"
                return key, "caught" if r.returncode != 0 else "survived"
            finally:
                shutil.rmtree(d, ignore_errors=True)
        with cf.ThreadPoolExecutor(max_workers=self.workers) as ex:
            for key, v in ex.map(one, list(muts.items())):
                if v == "caught":
                    caught += 1
                elif v == "survived":
                    survivors.append(key)
                elif v == "unrun":
                    unrun += 1
                else:
                    broken.append(key)
        return {"breaks": len(muts), "caught": caught, "survivors": sorted(survivors), "unbuilt": broken, "unrun": unrun}

    def _folder(self, here):
        """a copy of the code's folder without the record and without caches, so a break never reaches the real files"""
        d = tempfile.mkdtemp(prefix="breaks_")
        for f in os.listdir(here):
            p = os.path.join(here, f)
            if os.path.isfile(p) and f.endswith(".py"):
                shutil.copy2(p, os.path.join(d, f))
        return d


def cmd_start(a):
    L = Ledger(a.folder)
    if os.path.exists(L.grid_path):
        raise SystemExit("%s already holds a grid, with what was skipped and asked in it. Carry on with `grid` and `check`, or start in a new folder." % a.folder)
    acts = [x.strip() for x in (a.acts or "").split(",") if x.strip()]
    if a.interface:
        acts = acts or Grid.acts_in(io.open(a.interface, encoding="utf-8").read())
    if not acts and not a.things:
        raise SystemExit("name the acts (--acts or --interface) or the things (--things): a grid of nothing would read as done")
    doc = {"acts": acts, "things": [x.strip() for x in (a.things or "").split(",") if x.strip()], "phases": [x.strip() for x in (a.phases or "").split(",") if x.strip()],
           "producer": a.producer, "skipped": {}}
    L.write_grid(doc)
    n = len(Grid(doc["acts"], doc["things"], doc["phases"]).squares())
    print("elicit: %d squares from %d acts, %d things, %d phases. `elicit.py grid %s` lists them." % (n, len(doc["acts"]), len(doc["things"]), len(doc["phases"]), a.folder))
    return 0


def cmd_grid(a):
    L = Ledger(a.folder)
    open_ = Readiness(L).lines()["unasked"]
    print("\n".join(open_) if open_ else "every square has been asked")
    print("# %d unasked. Put them in the person's words, not the interface's; a batch is cheaper than one at a time." % len(open_))
    return 0


def parse_batch(lines):
    out = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([\w.-]+):\s+(.+)$", line)
        if not m:
            raise SystemExit("a question is `<square or rule id>: <question>`, so its answer can be found again: %r" % line[:90])
        out.append((m.group(1), m.group(2)))
    return out


def cmd_ask(a):
    L = Ledger(a.folder)
    doc = L.grid_doc()
    lines = list(a.questions)
    if a.file:
        lines += io.open(a.file, encoding="utf-8").read().splitlines()
    batch = parse_batch(lines)
    if not batch:
        raise SystemExit("nothing to ask")
    fresh = []
    for sq, q in batch:
        same = L.same_words(q)
        if same is not None:
            print("%d %s (asked before in the same words; put it another way) -> %s" % (same.n, same.form, same.reply))
        else:
            fresh.append((sq, q))
    if not fresh:
        return 0
    if doc.get("producer"):
        replies = ProducerRoute(doc["producer"], workers=a.workers).many([q for _, q in fresh])
    else:
        replies = [None] * len(fresh)
    for (sq, q), reply in zip(fresh, replies):
        x = Asked(L.next_n(), sq, q, reply)
        L.put(x)
        print("%d %s %s -> %s" % (x.n, x.form, x.square, reply if reply is not None else "(put this to the person, then: elicit.py answer %s %d \"their words\")" % (a.folder, x.n)))
    return 0


def cmd_answer(a):
    L = Ledger(a.folder)
    if a.n not in L.asked:
        raise SystemExit("question %d is not on the record" % a.n)
    x = L.asked[a.n]
    x.reply = a.reply
    L.put(x)
    print("%d %s -> %s" % (x.n, x.kind, a.reply))
    return 0


def cmd_rules(a):
    L = Ledger(a.folder)
    text = io.open(a.file, encoding="utf-8").read()
    for line in text.splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            Rule.parse(line)       # every line readable, or nothing is taken
    io.open(L.rules_path, "w", encoding="utf-8").write(text)
    rules = L.rules()
    states = collections.Counter(r.state(L.asked) for r in rules)
    print("elicit: %d rules: %s" % (len(rules), ", ".join("%d %s" % (v, k) for k, v in states.most_common())))
    return 0


def cmd_skip(a):
    L = Ledger(a.folder)
    doc = L.grid_doc()
    if a.square not in Grid(doc["acts"], doc["things"], doc["phases"]).squares():
        raise SystemExit("%s is not a square of this grid" % a.square)
    if len(a.why.split()) < 3:
        raise SystemExit("a square left out needs its reason in words: %r says nothing" % a.why)
    doc["skipped"][a.square] = a.why
    L.write_grid(doc)
    print("skipped %s" % a.square)
    return 0


def cmd_check(a):
    open_ = Readiness(Ledger(a.folder), a.tests or ()).lines()
    total = sum(len(v) for v in open_.values())
    for k, v in open_.items():
        if v:
            print("%s (%d):" % (k.upper(), len(v)))
            for line in v[:a.show]:
                print("  " + line)
            if len(v) > a.show:
                print("  ... and %d more" % (len(v) - a.show))
    rules = Ledger(a.folder).rules()
    asked = Ledger(a.folder).asked
    print("elicit: %d asked, %d rules, %d open." % (len(asked), len(rules), total))
    return 1 if total else 0


def cmd_export(a):
    L = Ledger(a.folder)
    rules = L.rules()
    with io.open(a.rules, "w", encoding="utf-8") as f:
        for r in rules:
            f.write("%s\n" % r.sentence)
    states = collections.Counter(r.state(L.asked) for r in rules)
    print("elicit: wrote %d sentences to %s (%s)" % (len(rules), a.rules, ", ".join("%d %s" % (v, k) for k, v in states.most_common())))
    return 0


def cmd_breaks(a):
    res = BreakRun(a.code, a.test, workers=a.workers, minutes=a.minutes).run()
    by = collections.defaultdict(list)
    for key in res["survivors"]:
        by[key.split("@")[0]].append(key.split(": ", 1)[-1])
    lines = ["%d breaks: %d caught, %d missed, %d could not be built, %d not run in the time" % (res["breaks"], res["caught"], len(res["survivors"]), len(res["unbuilt"]), res["unrun"])]
    for place in sorted(by):
        lines.append("%s: %s" % (place, "; ".join(by[place][:6]) + (" and %d more" % (len(by[place]) - 6) if len(by[place]) > 6 else "")))
    text = "\n".join(lines)
    print(text)
    if a.out:
        io.open(a.out, "w", encoding="utf-8").write(text + "\n")
    return 0


def selftest():
    """Every refusal planted beside a control that passes, against a fake producer and a small real program with tests. Exit 1 if any plant does not fire."""
    fails = []

    def check(name, ok):
        print("selftest: %s %s" % (name, "PASS" if ok else "FAIL"))
        if not ok:
            fails.append(name)
    run = lambda *argv: main_parsed(list(argv))
    d = tempfile.mkdtemp(prefix="elicit_")
    try:
        check("form: open", Form.of("What ends the visit?") == Form.OPEN)
        check("form: closed", Form.of("May a player pass?") == Form.CLOSED)
        check("form: closed behind a phrase", Form.of("In the last round, may a player pass?") == Form.CLOSED)
        check("form: a list inside an open question stays open", Form.of("What happens if a, b or c is played?") == Form.OPEN)
        check("form: scenario", Form.of("Member 2 holds the key. May it open?") == Form.SCENARIO)
        check("form: one sentence with named players is a scenario", Form.of("What happens to player 2 and player 3?") == Form.SCENARIO)
        check("form: choice", Form.of("Which of these is the rule: (a) one or (b) two") == Form.CHOICE)
        check("reply: silent", Reply.kind("not specified") == Reply.SILENT)
        check("reply: unusable", Reply.kind("That is too much for one answer. Ask about one thing.") == Reply.UNUSABLE)
        check("reply: spoken", Reply.kind("Yes") == Reply.SPOKEN and Reply.kind("No") == Reply.SPOKEN)
        # a fake producer is a real command: it reads the question and answers from a table
        fake = os.path.join(d, "fake.py")
        io.open(fake, "w", encoding="utf-8").write(
            "import sys\nq=sys.argv[1].lower()\nprint('not specified' if 'silent' in q else 'Ask about one thing.' if 'broad' in q else 'No' if q.startswith('may') else 'The clerk lends.')\n")
        boom = os.path.join(d, "boom.py")
        io.open(boom, "w", encoding="utf-8").write("import sys\nsys.exit(3)\n")
        folder = os.path.join(d, "e")
        route = '"%s" -X utf8 "%s" "{q}"' % (sys.executable.replace("\\", "/"), fake.replace("\\", "/"))
        check("start: nothing to ask refused", _refused(lambda: run("start", folder)))
        check("start", run("start", folder, "--acts", "lend", "--things", "a loan", "--phases", "the visit", "--producer", route) == 0)
        check("start: a second start over a record refused", _refused(lambda: run("start", folder, "--acts", "lend")))
        L = Ledger(folder)
        n_squares = len(L.grid().squares())
        check("grid has every stem of every name", n_squares == len(Grid.ACT) + len(Grid.THING) + len(Grid.PHASE))
        check("check: exit 1 while squares are unasked", run("check", folder) == 1)
        check("ask: a line with no square refused", _refused(lambda: run("ask", folder, "What is this?")))
        check("ask", run("ask", folder, "lend.who: Who lends?", "lend.when: May the clerk lend twice?", "lend.pass: silent one", "lend.forbidden: too broad") == 0)
        L = Ledger(folder)
        check("the record holds each reply and form", [a.kind for a in L.asked.values()] == [Reply.SPOKEN, Reply.SPOKEN, Reply.SILENT, Reply.UNUSABLE] and L.asked[2].form == Form.CLOSED)
        check("same words are not asked twice", run("ask", folder, "lend.who: who lends") == 0 and len(Ledger(folder).asked) == 4)
        lines = Readiness(Ledger(folder)).lines()
        check("a silence asked once stays open", any("lend.pass" in x for x in lines["silent once"]))
        check("a refused square stays open", any("lend.forbidden" in x for x in lines["unanswered"]))
        run("ask", folder, "lend.pass: silent two, as a situation")
        lines = Readiness(Ledger(folder)).lines()
        check("two silences, not reported as once", not any("lend.pass" in x for x in lines["silent once"]))
        # the rule states
        A = {1: Asked(1, "x", "Who lends?", "The clerk."), 2: Asked(2, "x", "May the clerk lend twice?", "No"), 3: Asked(3, "x", "Does the clerk lend?", "Yes"), 4: Asked(4, "x", "Member 1 holds it. Who lends?", "Seat 1."),
             5: Asked(5, "x", "Who signs?", "not specified"), 6: Asked(6, "x", "Member 2 asks. Who signs?", "not specified")}
        check("one witness", Rule("R", "s", [1]).state(A) == Rule.ONE)
        check("two of one form", Rule("R", "s", [2, 3]).state(A) == Rule.SAME_FORM)
        check("two forms stand", Rule("R", "s", [1, 4]).state(A) == Rule.STANDS)
        check("two forms, closed and open, stand", Rule("R", "s", [1, 2]).state(A) == Rule.STANDS)
        check("an answer against keeps it open", Rule("R", "s", [1, 4], against=[2]).state(A) == Rule.CONFLICT)
        check("outnumbered by two, it stands", Rule("R", "s", [1, 3, 4], against=[2]).state(A) == Rule.STANDS)
        check("a reading needs two silences", Rule("R", "s", silent=[5]).state(A) == Rule.ONE and Rule("R", "s", silent=[5, 6]).state(A) == Rule.READING)
        check("a rule line with nothing in the middle refused", _refused(lambda: Rule.parse("R1 | a sentence")))
        check("a bad field refused", _refused(lambda: Rule.parse("R1 | perhaps 3 | a sentence")))
        rules = os.path.join(d, "rules.txt")
        io.open(rules, "w", encoding="utf-8").write("R1 | for 1 2 | The clerk lends.\nR9 | for 1 99 | A rule that cites a question never asked.\n")
        check("rules load", run("rules", folder, "--file", rules) == 0)
        lines = Readiness(Ledger(folder)).lines()
        check("a witness not on the record is named", any("99" in x for x in lines["bad rule"]))
        io.open(rules, "w", encoding="utf-8").write("R1 | for 1 2 | The clerk lends.\nR2 | silent 1 | A rule called silent though it was answered.\n")
        run("rules", folder, "--file", rules)
        lines = Readiness(Ledger(folder)).lines()
        check("a false silence is named", any("was answered" in x for x in lines["bad rule"]))
        io.open(rules, "w", encoding="utf-8").write("not a rule line\n")
        before = io.open(Ledger(folder).rules_path, encoding="utf-8").read()
        check("an unreadable rules file refused and nothing taken", _refused(lambda: run("rules", folder, "--file", rules)) and io.open(Ledger(folder).rules_path, encoding="utf-8").read() == before)
        check("skip needs a reason", _refused(lambda: run("skip", folder, "the_visit.starts", "no")))
        check("skip an unknown square refused", _refused(lambda: run("skip", folder, "nothing.here", "this has a reason")))
        # all squares answered and rules standing: check goes to 0, and a test file that names no rule reopens it
        f2 = os.path.join(d, "f2")
        run("start", f2, "--acts", "lend", "--producer", route)
        for i, (sq, _) in enumerate(Grid(["lend"], [], []).squares().items()):
            run("ask", f2, "%s: Tell me what happens for case %d of it?" % (sq, i))
        io.open(rules, "w", encoding="utf-8").write("R1 | for 1 2 | The clerk lends.\n")
        run("rules", f2, "--file", rules)
        q = Ledger(f2).asked
        if Rule("R1", "s", [1, 2]).state(q) != Rule.STANDS:
            run("ask", f2, "lend.who: Member 2 holds it. Who lends, then?")
            io.open(rules, "w", encoding="utf-8").write("R1 | for 1 %d | The clerk lends.\n" % max(Ledger(f2).asked))
            run("rules", f2, "--file", rules)
        check("check: exit 0 with nothing open", run("check", f2) == 0)
        t = os.path.join(d, "t.py")
        io.open(t, "w", encoding="utf-8").write("# a test with no CHECKS line\n")
        check("check: a rule no test names reopens it", run("check", f2, "--tests", t) == 1)
        io.open(t, "w", encoding="utf-8").write("# CHECKS: R1\n")
        check("check: a test that names it closes it", run("check", f2, "--tests", t) == 0)
        check("check: a test file that cannot be read fails, it is not read as empty", _raises(lambda: run("check", f2, "--tests", os.path.join(d, "nope.py"))))
        # a broken route records nothing
        f3 = os.path.join(d, "f3")
        run("start", f3, "--acts", "lend", "--producer", '"%s" "%s"' % (sys.executable.replace("\\", "/"), boom.replace("\\", "/")))
        check("a failing route raises and records nothing", _raises(lambda: run("ask", f3, "lend.who: Who lends?")) and not Ledger(f3).asked)
        # by hand: no route
        f4 = os.path.join(d, "f4")
        run("start", f4, "--acts", "lend")
        run("ask", f4, "lend.who: Who lends?")
        check("with no route, a question waits for its reply", Ledger(f4).asked[1].reply is None and any("lend.who" in x for x in Readiness(Ledger(f4)).lines()["unanswered"]))
        run("answer", f4, "1", "The clerk lends.")
        check("the reply is kept across a restart", Ledger(f4).asked[1].kind == Reply.SPOKEN)
        check("acts read from an interface, without readers", Grid.acts_in("`cut(seat, n)`: x. `state()`: y. `Game.from_deal(h)`: z. `play(seat)`.") == ["cut", "play"])
        # export
        out = os.path.join(d, "RULES.md")
        run("export", f2, "--rules", out)
        check("export writes the sentences", "The clerk lends." in io.open(out, encoding="utf-8").read())
        # the pass at the end, on a real small program with tests. One test misses a statement; a survivor must be reported, and a clean program must report none
        prog = os.path.join(d, "prog")
        os.makedirs(prog)
        io.open(os.path.join(prog, "shop.py"), "w", encoding="utf-8").write("def pay(money, price):\n    if price > money:\n        return money\n    return money - price\n")
        io.open(os.path.join(prog, "shop_test.py"), "w", encoding="utf-8").write("import shop\nimport sys\nsys.exit(0 if shop.pay(5, 2) == 3 else 1)\n")
        res = BreakRun(os.path.join(prog, "shop.py"), "shop_test.py", workers=2, minutes=2).run()
        check("breaks: the refused branch is reported as missed", any("pay" in k for k in res["survivors"]) and res["caught"] > 0)
        io.open(os.path.join(prog, "shop_test.py"), "w", encoding="utf-8").write("import shop\nimport sys\nsys.exit(0 if shop.pay(5, 2) == 3 and shop.pay(5, 9) == 5 and shop.pay(5, 5) == 0 else 1)\n")
        res2 = BreakRun(os.path.join(prog, "shop.py"), "shop_test.py", workers=2, minutes=2).run()
        check("breaks: better tests miss less", len(res2["survivors"]) < len(res["survivors"]))
        io.open(os.path.join(prog, "shop_test.py"), "w", encoding="utf-8").write("import sys\nsys.path.insert(0, %r)\nimport shop\nsys.exit(0)\n" % prog.replace("\\", "/"))
        check("breaks: tests that read the real file by absolute path are refused", _refused(lambda: BreakRun(os.path.join(prog, "shop.py"), "shop_test.py", workers=2, minutes=2).run()))
        io.open(os.path.join(prog, "shop_test.py"), "w", encoding="utf-8").write("import sys\nsys.exit(1)\n")
        check("breaks: tests that already fail are refused, not read as catching everything", _refused(lambda: BreakRun(os.path.join(prog, "shop.py"), "shop_test.py", workers=2, minutes=2).run()))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("selftest: %s" % ("PASS" if not fails else "FAIL %d: %s" % (len(fails), ", ".join(fails))))
    return 1 if fails else 0


def _refused(fn):
    """True when fn stops with a message (SystemExit or a ValueError a reader gets)"""
    try:
        fn()
    except (SystemExit, ValueError):
        return True
    return False


def _raises(fn):
    try:
        fn()
    except (RuntimeError, OSError):
        return True
    return False


def main_parsed(argv):
    ap = argparse.ArgumentParser(description="Get the rules from the person who holds them, widely and twice, before any code.")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("start")
    p.add_argument("folder")
    p.add_argument("--acts")
    p.add_argument("--things")
    p.add_argument("--phases")
    p.add_argument("--interface")
    p.add_argument("--producer")
    p = sub.add_parser("grid")
    p.add_argument("folder")
    p = sub.add_parser("ask")
    p.add_argument("folder")
    p.add_argument("questions", nargs="*")
    p.add_argument("--file")
    p.add_argument("--workers", type=int, default=6)
    p = sub.add_parser("answer")
    p.add_argument("folder")
    p.add_argument("n", type=int)
    p.add_argument("reply")
    p = sub.add_parser("rules")
    p.add_argument("folder")
    p.add_argument("--file", required=True)
    p = sub.add_parser("skip")
    p.add_argument("folder")
    p.add_argument("square")
    p.add_argument("why")
    p = sub.add_parser("check")
    p.add_argument("folder")
    p.add_argument("--tests", nargs="*")
    p.add_argument("--show", type=int, default=25)
    p = sub.add_parser("export")
    p.add_argument("folder")
    p.add_argument("--rules", required=True)
    p = sub.add_parser("breaks")
    p.add_argument("--code", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--minutes", type=float, default=8.0)
    p.add_argument("--out")
    a = ap.parse_args(argv)
    fn = {"start": cmd_start, "grid": cmd_grid, "ask": cmd_ask, "answer": cmd_answer, "rules": cmd_rules, "skip": cmd_skip, "check": cmd_check, "export": cmd_export, "breaks": cmd_breaks}.get(a.cmd)
    if fn is None:
        ap.print_help()
        return 2
    return fn(a)


def main():
    if "--selftest" in sys.argv:
        return selftest()
    return main_parsed(sys.argv[1:])


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.exit(main())
