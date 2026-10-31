#!/usr/bin/env python3
"""prove.py - run every proof a set of tools carries, together, and say in one plain result what passes, what fails, what is unknown and what was not run, and why.

    python -X utf8 closed_loop.py prove                 every selftest and the examples of the installed kit, in parallel, then the result
    python -X utf8 closed_loop.py prove --quick         the front door only (about a minute)
    python -X utf8 prove.py --folder DIR --list         what would run, and what would not and why, without running anything
    python -X utf8 prove.py --selftest

WHY. A proof that has only ever passed says nothing, and a pile of proofs that each print their own verdict says even less: a reader has to open fifty
outputs to learn whether the whole holds. A tool can also "pass" in three ways that prove nothing: it can hang (nobody waits for ever), it can exit 0
having checked nothing, and it can exit 0 while its own lines say FAIL. This runs each proof in its own process, at below-normal priority because a person
is at the machine, several at once, kills what hangs and says so, and refuses to call any of those three a pass.

The class model (each class is one concept; its docstring says what it is and where its evidence lives):

  Checks     the lines of a tool's `--selftest` output that are verdicts (end in PASS, FAIL or ok), counted: the floor of how many checks it made
  Command    one process to run: argv, where, what exit and text mean "as intended"
  Item       one thing that can be proved: what it proves in words, how it could fail, the Commands that run it, and, if it is not to be run, why not
  Outcome    what running it did: exit, seconds, the output, whether it hung, whether the runner itself broke
  Judge      Outcome -> Verdict, by fixed rules (below); the only place a PASS is given
  Verdict    PASS, FAIL, HANGS, EMPTY, DISAGREES or NOT RUN, and the words that say why
  Runner     the parallel pool: below-normal priority, a timeout per item and a wall budget for the whole, process trees killed, a resource lock for items that share a folder
  Report     the plain result: passes, fails, unknown, not run and why; one closing line
  Folder     a folder of tools as a Catalogue: every tool that carries a selftest, every example that holds a model, the front door (an epic and its draft)
  Plants     the selftest: each way of passing falsely, planted, beside a control that passes

THE RULES THAT GIVE A PASS (Judge). A PASS needs all of: it finished before its timeout; every command exited as intended; for a selftest, at least one
check line and none ending in FAIL; the text its command must print is there; and any re-check the Item carries (the numbers its record states, still
read from the file that holds them) agrees. Each way short of that has its own verdict: HANGS (killed at the timeout: unknown, it gave no answer), EMPTY
(exit 0 and no check line and no count of checks: unknown, it may have checked nothing), DISAGREES (exit 0 but a line ends in FAIL, or the re-check differs from the record:
it contradicts itself, which is a failure), FAIL (a wrong exit or missing text, or a crash), NOT RUN (a reason, always stated: it needs a piece this machine lacks, it is too
long for this tier, its command is not one that can be run).

THE CLOSING LINE. PROVED only when nothing failed, nothing is unknown, and, where the catalogue marks proofs as essential (the examples: the whole loop on a
model), at least one ran and passed. NOT PROVED otherwise; NOT PROVED HERE when the only thing missing is an example that could not run on this machine (exit 3). Never
PROVED on zero items. A project can hand the same engine its own Items (see Item): the Judge, the Runner and the Report do not know where an Item came from.
"""
import argparse
import collections
import concurrent.futures
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BELOW_NORMAL = 0x4000 if os.name == "nt" else 0
PASS, FAIL, HANGS, EMPTY, DISAGREES, NOT_RUN = "PASS", "FAIL", "HANGS", "EMPTY", "DISAGREES", "NOT RUN"
BUCKET = {PASS: "passes", FAIL: "fails", DISAGREES: "fails", HANGS: "unknown", EMPTY: "unknown", NOT_RUN: "not_run"}
OPTIONAL = {"torch": "PyTorch (pip install torch)", "sklearn": "scikit-learn (pip install scikit-learn)"}   # pieces a few tools need and Python + PyYAML alone do not give
FRONT_DOOR = ("usecase_model.py", "grow.py", "ask.py")             # what `--quick` proves: an epic in, the first question out
SECTIONS = (("passes", "PASSES"), ("fails", "FAILS"), ("unknown", "UNKNOWN (it gave no answer, or may have checked nothing)"), ("not_run", "NOT RUN, and why"))


class Checks(object):
    """The lines of a tool's `--selftest` output that are verdicts. A line is a check if it ends in PASS, FAIL or ok. A closing summary ("selftest: PASS",
    "PASS (12 of 12)") is not counted as a check, but its `(x of y)` is kept as what the tool says it made. The count is a FLOOR on the checks: one line can stand for
    several. Evidence: 564 such lines across the 42 tools of the project this came from whose selftest ran."""

    LINE = re.compile(r"(PASS|FAIL|ok)\s*$")
    SUMMARY = re.compile(r"^(?:[\w .-]*selftest:?\s*)?(PASS|FAIL)(?:\s*\((\d+) of (\d+)\))?\s*$")

    @classmethod
    def read(cls, stdout):
        """(lines that are checks, a count the tool printed or None, lines ending in FAIL)"""
        lines, declared, failing = 0, None, []
        for raw in stdout.splitlines():
            t = raw.strip()
            m = cls.SUMMARY.match(t)
            if m:
                if m.group(3):
                    declared = int(m.group(3))
                if m.group(1) == "FAIL" or (m.group(2) and m.group(2) != m.group(3)):      # "PASS (93 of 94)" says pass and says one did not
                    failing.append(t)
                continue
            if cls.LINE.search(t):
                lines += 1
                if t.endswith("FAIL"):
                    failing.append(t)
        return lines, declared, failing


class Command(object):
    """One process: what to run, where, and what means "as intended". `expect_exit` is the set of exit codes the author intends; `expect_text` must appear in the output"""

    def __init__(self, argv, cwd, expect_exit=(0,), expect_text="", env=None):
        self.argv, self.cwd, self.expect_exit, self.expect_text, self.env = list(argv), cwd, tuple(expect_exit), expect_text, env or {}


class Item(object):
    """One thing that can be proved. `proves` says what, in words; `could_fail` says how (each way a person could name); `commands` run in order and stop at the first that is
    not as intended; `kind` is selftest (needs check lines) or run (judged by exit, text and re-check); `essential` marks the proofs whose pass is what PROVED means (the
    examples); `lock` serialises items that write the same folder; `recheck` is called after the run and returns the ways the result differs from what a record states;
    `prepare` is called first (copy an example, place a draft); `not_run` is the reason this is to be listed and not run; `seconds` is the last measured time (for ordering)"""

    def __init__(self, key, title, commands=(), proves="", could_fail=(), kind="selftest", timeout=600, essential=False, lock=None, recheck=None, prepare=None, not_run="", seconds=None, ids=()):
        self.key, self.title, self.commands, self.proves, self.could_fail = key, title, list(commands), proves, list(could_fail)
        self.kind, self.timeout, self.essential, self.lock, self.recheck, self.prepare = kind, timeout, essential, lock, recheck, prepare
        self.not_run, self.seconds, self.ids = not_run, seconds, list(ids)
        self.facts = []                    # what the run learned that a reader should have, set by prepare and recheck (e.g. "the command rewrote 1 of the 3 files it is judged on")


class Outcome(object):
    """What running an Item did. `error` is set when the runner itself broke (a missing folder, a command that would not start): that is a result, never a skip"""

    def __init__(self):
        self.exit = None
        self.seconds = 0.0
        self.out = ""
        self.hung = False
        self.error = ""
        self.stopped_at = None            # the Command (index) that was not as intended
        self.alone = None                 # the Outcome of a second run, alone, when the first ran beside others and did not pass


class Verdict(object):
    """PASS, FAIL, HANGS, EMPTY, DISAGREES or NOT RUN and the words that say why. `checks` is the count of check lines; `declared` the count the tool printed"""

    def __init__(self, word, why="", checks=0, declared=None, note=""):
        self.word, self.why, self.checks, self.declared, self.note = word, why, checks, declared, note

    @property
    def bucket(self):
        return BUCKET[self.word]


def kill_tree(pid):
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
    else:
        import signal
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            pass


def spawn(cmd, timeout, base_env=None):
    """run one Command; returns (exit, output, hung). A process that outlives `timeout` is killed with its children and its output so far is kept"""
    env = dict(base_env if base_env is not None else os.environ)
    env.update(cmd.env)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    p = subprocess.Popen(cmd.argv, cwd=cmd.cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                         creationflags=BELOW_NORMAL, start_new_session=(os.name != "nt"))
    try:
        out, _ = p.communicate(timeout=timeout)
        return p.returncode, out or "", False
    except subprocess.TimeoutExpired:
        kill_tree(p.pid)
        try:
            out, _ = p.communicate(timeout=15)
        except Exception:
            out = ""
        return None, out or "", True


class Judge(object):
    """Outcome -> Verdict. The only place a PASS is given. `optional` maps a module a tool may import to the words naming it; a crash that says that module is missing, on a
    machine where it is missing, is NOT RUN (needs it) and not FAIL. Evidence: `--selftest` plants every verdict."""

    def __init__(self, optional=None):
        self.optional = OPTIONAL if optional is None else optional

    def missing_piece(self, out):
        for mod, words in self.optional.items():
            if re.search(r"No module named '%s(?:\.\w+)*'" % re.escape(mod), out) and importlib.util.find_spec(mod) is None:
                return words
        return None

    def judge(self, item, o):
        if o.error:
            return Verdict(FAIL, "the runner could not run it: %s" % o.error)
        if o.hung:
            return Verdict(HANGS, "did not finish in %d s; killed, not waited on" % item.timeout)
        cmd = item.commands[o.stopped_at] if o.stopped_at is not None else None
        text = o.out
        checks, declared, failing = Checks.read(text)
        if cmd is not None:
            missing = self.missing_piece(text)
            if missing:
                return Verdict(NOT_RUN, "needs %s, which is not installed here" % missing)
            if re.search(r"unrecognized arguments:.*--selftest", text) or re.search(r"unknown option.*--selftest", text, re.I):
                return Verdict(NOT_RUN, "its source mentions a selftest but it takes no --selftest, so there is nothing to run")
            if o.exit not in cmd.expect_exit:
                last = ([l for l in text.strip().splitlines() if l.strip()] or ["(no output)"])[-1].strip()[:140]
                return Verdict(FAIL, "exit %s, wanted %s: %s" % (o.exit, "/".join(map(str, cmd.expect_exit)), last), checks, declared)
            return Verdict(FAIL, "the text %r is not in its output" % cmd.expect_text, checks, declared)
        if "Traceback (most recent call last)" in text and item.kind == "run" and checks == 0 and not any(c.expect_text for c in item.commands):
            last = ([l for l in text.strip().splitlines() if l.strip()] or ["(no output)"])[-1].strip()[:160]
            return Verdict(FAIL, "it crashed (a traceback) and nothing it was meant to print is there: %s" % last, checks, declared)
        if failing and item.kind == "selftest":                 # in a selftest every such line is a check; in another command a line ending in FAIL can be data (a plant whose link goes FAIL)
            return Verdict(DISAGREES, "exit as intended, but %d of its own line(s) end in FAIL: %s" % (len(failing), failing[0][:100]), checks, declared)
        if item.kind == "selftest" and checks == 0 and not declared:
            return Verdict(EMPTY, "exit 0 and not one check line and no count of checks: it may have checked nothing", checks, declared)
        if item.recheck:
            try:
                diffs = item.recheck(item, o)
            except Exception as ex:
                return Verdict(FAIL, "the re-check of its numbers broke: %s" % ex, checks, declared)
            if diffs:
                return Verdict(DISAGREES, "exit as intended, but the record differs from what it left: %s" % "; ".join(diffs[:3]), checks, declared)
        note = ""
        if o.alone is not None:
            note = "failed when run beside others (%s) and passed alone" % o.alone
        return Verdict(PASS, "", checks, declared, note)


class Runner(object):
    """The parallel pool. Each Item runs its Commands in order in its own process, at below-normal priority. Items are started longest first; two items with the same `lock`
    never run at once. `budget` is the wall seconds for the whole: an item that has not started when it is spent is NOT RUN, and none is started late. An item that did not
    pass while others were running is run once more, alone, and both results are kept (a proof that passes only alone interferes with its neighbours: that is stated)."""

    def __init__(self, jobs=4, budget=None, judge=None, log=None):
        self.jobs, self.budget, self.judge, self.log = jobs, budget, judge or Judge(), log or (lambda s: None)
        self.t0 = time.time()
        self.locks = collections.defaultdict(threading.Lock)
        self.guard = threading.Lock()

    def execute(self, item):
        o = Outcome()
        t = time.time()
        try:
            if item.prepare:
                item.prepare(item)
            outs = []
            for i, cmd in enumerate(item.commands):
                left = item.timeout - (time.time() - t)
                rc, out, hung = spawn(cmd, max(left, 1))
                outs.append(out)
                o.exit, o.hung = rc, hung
                if hung:
                    break
                if rc not in cmd.expect_exit or (cmd.expect_text and cmd.expect_text not in out):
                    o.stopped_at = i
                    break
            o.out = "\n".join(outs)
        except Exception as ex:
            o.error = "%s: %s" % (type(ex).__name__, str(ex)[:200])
        o.seconds = round(time.time() - t, 1)
        return o

    def one(self, item):
        if self.budget is not None and time.time() - self.t0 > self.budget:
            return item, None, Verdict(NOT_RUN, "the time budget (%d s) was spent before this started" % self.budget)
        lock = self.locks[item.lock] if item.lock else None
        if lock:
            lock.acquire()
        try:
            o = self.execute(item)
        finally:
            if lock:
                lock.release()
        v = self.judge.judge(item, o)
        with self.guard:
            self.log("  %-9s %6.1f s  %s" % (v.word, o.seconds, item.key))
        return item, o, v

    def run(self, items):
        """[(item, outcome or None, verdict)] in the order given. Every item is in the answer exactly once"""
        runnable = [i for i in items if not i.not_run]
        order = sorted(runnable, key=lambda i: -(i.seconds or 0))
        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, self.jobs)) as pool:
            for item, o, v in pool.map(self.one, order):
                results[item.key] = (item, o, v)
        # an item that failed (a wrong exit, or no check) beside others gets one run alone (not one that contradicts itself, which would only run again over the files it already changed); the verdict stands on the alone run and says the first failed
        for item, o, v in list(results.values()):
            if o is not None and v.word in (FAIL, EMPTY) and self.jobs > 1 and o.seconds < 120:
                o2 = self.execute(item)
                v2 = self.judge.judge(item, o2)
                if v2.word == PASS:
                    o2.alone = "%s: %s" % (v.word, v.why[:90])
                    v2 = self.judge.judge(item, o2)
                results[item.key] = (item, o2, v2)
                self.log("  %-9s %6.1f s  %s (again, alone)" % (v2.word, o2.seconds, item.key))
        out = []
        for item in items:
            if item.not_run:
                out.append((item, None, Verdict(NOT_RUN, item.not_run)))
            else:
                out.append(results[item.key])
        return out


class Report(object):
    """The plain result. `results` is [(Item, Outcome or None, Verdict)]. Nothing is dropped: every item is in exactly one bucket, and the counts add to the total"""

    def __init__(self, title, results, wall, jobs, tier=""):
        self.title, self.results, self.wall, self.jobs, self.tier = title, results, wall, jobs, tier
        self.notes = []                    # sentences for the reader that are not about one item

    def buckets(self):
        b = collections.OrderedDict((k, []) for k, _ in SECTIONS)
        for r in self.results:
            b[r[2].bucket].append(r)
        return b

    def counts(self):
        return {k: len(v) for k, v in self.buckets().items()}

    def work_seconds(self):
        """the seconds every item that ran took, added: what the run would have taken one after another under the same load"""
        return sum(o.seconds for _, o, _ in self.results if o is not None)

    def closing(self):
        c = self.counts()
        total = sum(c.values())
        essential = [r for r in self.results if r[0].essential]
        proved_essential = [r for r in essential if r[2].word == PASS]
        if total == 0 or c["passes"] == 0:
            return "NOT PROVED: nothing ran and passed (%d items)." % total
        if c["fails"] or c["unknown"]:
            return "NOT PROVED: %d failed, %d gave no answer or may have checked nothing; %d passed, %d not run." % (c["fails"], c["unknown"], c["passes"], c["not_run"])
        if essential and not proved_essential:
            return "NOT PROVED HERE: %d passed, but none of the examples that run the whole loop ran and passed on this machine (named below)." % c["passes"]
        return "PROVED: %d passed, none failed, none unknown; %d not run (each named below)." % (c["passes"], c["not_run"])

    @property
    def proved(self):
        return self.closing().startswith("PROVED:")

    @property
    def exit_code(self):
        """0 proved; 3 nothing failed and nothing is unknown but no example that runs the whole loop could run here (a machine that lacks a piece); 1 otherwise"""
        return 0 if self.proved else 3 if self.closing().startswith("NOT PROVED HERE") else 1

    def line(self, item, o, v, full):
        secs = ("%6.1f s" % o.seconds) if o is not None else "       -"
        head = "  %s  %s  %s" % (item.key, secs, item.title)
        if v.word == PASS:
            chk = ((" %d checks%s" % (v.checks, (" (the tool says %d)" % v.declared) if v.declared else "")) if v.checks else
                   (" %d checks counted by the tool, none listed" % v.declared if v.declared else " no check listed")) if item.kind == "selftest" else ""
            head += "  [%s%s]" % ("passed", chk) + (("  NOTE: " + v.note) if v.note else "")
        else:
            head += "  [%s: %s]" % (v.word, v.why)
        out = [head]
        out += ["      %s" % f for f in item.facts]
        if full:
            if item.proves:
                out.append("      proves: %s" % item.proves)
            if item.could_fail:
                out.append("      could fail by: %s" % "; ".join(item.could_fail))
        return out

    def text(self, full=True):
        c = self.counts()
        total = sum(c.values())
        work = self.work_seconds()
        lines = ["%s" % self.title, "", "%d items: %d passed, %d failed, %d unknown, %d not run. %d at once: %.0f s for %.0f s of work, %.1f times faster than one after another%s." % (
            total, c["passes"], c["fails"], c["unknown"], c["not_run"], self.jobs, self.wall, work, (work / self.wall) if self.wall else 0.0, (", tier " + self.tier) if self.tier else ""),
            self.closing()] + self.notes + [""]
        for key, head in SECTIONS:
            rows = self.buckets()[key]
            lines.append("%s (%d)" % (head, len(rows)))
            if not rows:
                lines.append("  none")
            for item, o, v in sorted(rows, key=lambda r: r[0].key):
                lines += self.line(item, o, v, full or key != "passes")
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def data(self):
        return {"title": self.title, "tier": self.tier, "notes": self.notes, "wall_seconds": round(self.wall, 1), "work_seconds": round(self.work_seconds(), 1), "jobs": self.jobs, "counts": self.counts(), "closing": self.closing(),
                "items": [{"key": i.key, "title": i.title, "ids": i.ids, "kind": i.kind, "verdict": v.word, "why": v.why, "note": v.note, "checks": v.checks, "declared": v.declared,
                           "seconds": (o.seconds if o is not None else None), "exit": (o.exit if o is not None else None),
                           "proves": i.proves, "could_fail": i.could_fail, "tail": ((o.out.strip().splitlines() or [""])[-3:] if o is not None else [])} for i, o, v in self.results]}


class Folder(object):
    """A folder of tools as a Catalogue of Items. A tool is provable if its source mentions `--selftest` or `def selftest`. A folder `example*` holding a `model` is an
    example: the whole loop is run on a COPY of it (so parallel runs and the shipped files stay as they were) and it must print HOLDS. A folder holding `*_epic.md` with a
    `*_draft.yml` is the front door: the draft stands in for the agent that would write it, `grow` must give the first question and `ask` must record one answer and ask
    the next. Nothing is listed by hand: a new tool or example is found by the same rules."""

    def __init__(self, folder, python=None, optional=None):
        self.folder, self.python = os.path.abspath(folder), python or sys.executable
        self.optional = OPTIONAL if optional is None else optional
        self.scratch = tempfile.mkdtemp(prefix="prove_")

    def tools(self):
        out = []
        for f in sorted(os.listdir(self.folder)):
            p = os.path.join(self.folder, f)
            if f.endswith(".py") and os.path.isfile(p):
                src = io.open(p, encoding="utf-8", errors="replace").read()
                out.append((f, "--selftest" in src or "def selftest" in src, self.about(src)))
        return out

    @staticmethod
    def about(src):
        m = re.match(r'\s*(?:#!.*\n)?\s*(?:"""|\'\'\')\s*([^\n]*)', src)
        text = re.sub(r"[\"']{3}\s*$", "", (m.group(1) if m else "").strip()).strip()
        return re.sub(r"^[\w.]+\.py\s+-\s+", "", text)[:160]

    def selftest_item(self, f, about):
        p = os.path.join(self.folder, f)
        return Item("selftest:" + f, "the selftest of %s" % f, [Command([self.python, "-X", "utf8", "-u", p, "--selftest"], self.folder)],
                    proves=("each check it prints: %s" % about) if about else "each check it prints", kind="selftest", timeout=900,
                    could_fail=["a check line ends in FAIL", "it exits other than 0", "it prints no check at all", "it does not finish"])

    def examples_root(self):
        """the folder that holds the examples: beside the tools in an install, or one level up when the tools are the `tools` folder of a clone. None if neither does"""
        for cand in ((self.folder, os.path.dirname(self.folder)) if os.path.basename(self.folder) == "tools" else (self.folder,)):
            if any(n.startswith("example") and os.path.isdir(os.path.join(cand, n)) for n in os.listdir(cand)):
                return cand
        return None

    def example_items(self):
        out = []
        root = self.examples_root()
        if root is None:
            return [Item("example:none", "the whole loop on an example", essential=True, proves="the loop run on a model, every statement of its code broken and each break run against the tests",
                         not_run="no example folder beside these tools, so the whole loop was not run here (an install puts the examples beside the tools)")]
        for name in sorted(os.listdir(root)):
            model = os.path.join(root, name, "model")
            if name.startswith("example") and os.path.isdir(model):
                text = " ".join(io.open(os.path.join(model, f), encoding="utf-8", errors="replace").read() for f in os.listdir(model) if f.endswith((".yaml", ".yml")))
                needs_dotnet = bool(re.search(r"runner:\s*dotnet|language:\s*csharp", text))
                key = "example:" + name
                if needs_dotnet and not shutil.which("dotnet"):
                    out.append(Item(key, "the whole loop on %s" % name, proves="every statement broken, each break run against the example's tests, the loop must HOLD",
                                    not_run="needs the .NET SDK, which is not installed here (C# models and this example cannot run; Python models still can)", essential=True))
                    continue
                out.append(self.example_item(key, name, root))
        return out

    def example_item(self, key, name, root):
        work = os.path.join(self.scratch, name.replace("/", "_"))

        def prepare(item, name=name, work=work):
            shutil.rmtree(work, ignore_errors=True)
            shutil.copytree(os.path.join(root, name), os.path.join(work, name), ignore=shutil.ignore_patterns("bin", "obj", "kg", "stats.jsonl", "CLOSED_LOOP.md", "__pycache__", "*.holes.json", "*.mutants.json"))
        cmd = Command([self.python, "-X", "utf8", os.path.join(self.folder, "closed_loop.py"), "run", name + "/model", "--out", name + "/CLOSED_LOOP.md", "--note", "prove",
                       "--workers", str(example_workers())], work,
                      expect_exit=(0,), expect_text="closed_loop: HOLDS", env={"CLOSED_LOOP_ROOT": work})
        return Item(key, "the whole loop on %s" % name, [cmd], kind="run", essential=True, timeout=1800, prepare=prepare,
                    proves="every statement of the example's code broken every way the tool knows, each break run against the example's own tests, and the loop must read HOLDS",
                    could_fail=["a break no test catches, so the loop reads that a hole of ours stands", "the example's tests fail on unbroken code", "the tools crash on the model", "it does not finish"])

    def front_door_item(self):
        out = []
        root = self.examples_root()
        for name in sorted(os.listdir(root)) if root else []:
            d = os.path.join(root, name)
            if not (name.startswith("example") and os.path.isdir(d)):
                continue
            epics = [f for f in os.listdir(d) if f.endswith("_epic.md") and os.path.isfile(os.path.join(d, f[:-len("_epic.md")] + "_draft.yml"))]
            for epic in epics[:1]:
                stem = epic[:-len("_epic.md")]
                work = os.path.join(self.scratch, "door_" + name)
                model = os.path.join(work, "model")

                def prepare(item, d=d, stem=stem, work=work, model=model):
                    shutil.rmtree(work, ignore_errors=True)
                    os.makedirs(model)
                    shutil.copy2(os.path.join(d, stem + "_draft.yml"), os.path.join(model, "draft.yml"))
                cl = os.path.join(self.folder, "closed_loop.py")
                out.append(Item("door:" + name, "the front door on %s: an epic in, the first question out" % name,
                                [Command([self.python, "-X", "utf8", cl, "grow", os.path.join(d, epic), "--out", model], work, expect_exit=(0,), expect_text="THE FIRST QUESTION"),
                                 Command([self.python, "-X", "utf8", cl, "ask", model, "--confirm"], work, expect_exit=(1,), expect_text="recorded answer 1")],
                                kind="run", essential=True, timeout=300, prepare=prepare,
                                proves="the shipped epic and its draft give a model and a first question, then one answer is recorded and the next question asked; the draft stands in for the agent that would write it",
                                could_fail=["the tool refuses the shipped draft", "no first question is printed", "an answer is not recorded", "the next question is not asked"]))
        return out

    def items(self, quick=False):
        out = []
        for f, has, about in self.tools():
            if has and (not quick or f in FRONT_DOOR):
                out.append(self.selftest_item(f, about))
            elif not has and not quick and f not in ("prove.py",):
                out.append(Item("selftest:" + f, "%s has no selftest" % f, not_run="it carries no selftest, so nothing here can prove it", kind="selftest"))
        door = self.front_door_item()
        if quick and not door:
            door = [Item("door:none", "the front door: an epic in, the first question out", essential=True, proves="the shipped epic and its draft give a model and a first question",
                         not_run="no example epic and draft beside these tools, so the front door was not run here")]
        out += door
        if not quick:
            out += self.example_items()
        return out

    def close(self):
        shutil.rmtree(self.scratch, ignore_errors=True)


def example_workers():
    """the mutants of an example run at once: the longest single proof of the install is the C# example, so it gets more of the cores than the default 2 once the machine has them
    (MEASURED 10-31, one run each, on a busy machine: the C# example's 58 mutants read HOLDS in 192 s with 4 workers and 226 s with the default 2; a modest gain, kept because it costs nothing)"""
    return max(2, min(4, (os.cpu_count() or 4) // 4))


def default_jobs():
    return max(2, min(8, (os.cpu_count() or 4) // 2))


def run_items(title, items, jobs, budget=None, tier="", log=print, judge=None):
    t0 = time.time()
    runner = Runner(jobs=jobs, budget=budget, judge=judge, log=log)
    results = runner.run(items)
    return Report(title, results, time.time() - t0, jobs, tier)


PRIORITY_PARENT = r"""import sys
sys.path.insert(0, %r)
import prove
CODE = "import ctypes" + chr(10) + "k = ctypes.windll.kernel32" + chr(10) + "k.GetCurrentProcess.restype = ctypes.c_void_p" + chr(10) + "k.GetPriorityClass.argtypes = [ctypes.c_void_p]" + chr(10) + "print(hex(k.GetPriorityClass(k.GetCurrentProcess())))" + chr(10) + "print('a PASS')" + chr(10)
it = prove.Item("pri", "pri", [prove.Command([sys.executable, "-c", CODE], ".")], kind="selftest")
r = prove.run_items("pri", [it], jobs=1, log=lambda s: None)
print("CHILD", r.results[0][1].out.split()[0])
"""


class Plants(object):
    """The selftest: each way a proof can pass without proving, planted in a tiny fixture and seen to be refused, beside a control that passes. The fixtures are
    `python -c` programs, so nothing outside this file is needed"""

    def __init__(self):
        self.ok = True

    def check(self, name, good):
        self.ok = self.ok and bool(good)
        print("selftest: %s %s" % (name, "PASS" if good else "FAIL"))

    @staticmethod
    def item(key, code, kind="selftest", timeout=30, expect_exit=(0,), expect_text="", **kw):
        return Item(key, key, [Command([sys.executable, "-c", code], tempfile.gettempdir(), expect_exit=expect_exit, expect_text=expect_text)], kind=kind, timeout=timeout, **kw)

    def run(self):
        good_code = "print('a one PASS');print('a two ok');print('selftest: PASS')"
        items = [self.item("control", good_code),
                 self.item("hangs", "import time;print('x ok');time.sleep(120)", timeout=3),
                 self.item("empty", "print('hello, and no check')"),
                 self.item("contradicts", "print('a PASS');print('b FAIL');print('selftest: PASS')"),
                 self.item("wrong_exit", "import sys;print('a PASS');sys.exit(2)"),
                 self.item("fail_is_data", "print('link 2 -> FAIL');print('closed_loop: HOLDS')", kind="run", expect_text="HOLDS"),
                 self.item("summary_fail", "print('a PASS');print('selftest: FAIL')"),
                 self.item("count_only", "print('tool selftest: PASS (62 of 62)')"),
                 self.item("count_short", "print('tool selftest: PASS (61 of 62)')"),
                 self.item("text_missing", "print('a PASS')", kind="run", expect_text="HOLDS"),
                 self.item("run_ok", "print('closed_loop: HOLDS')", kind="run", expect_text="HOLDS"),
                 self.item("recheck", good_code, recheck=lambda i, o: ["the record says 3 and the file holds 4"]),
                 self.item("recheck_ok", good_code, recheck=lambda i, o: []),
                 self.item("skipped", good_code, not_run="needs a thing nobody has")]
        broken = Item("cannot_start", "cannot start", [Command([sys.executable, "-c", "print(1)"], os.path.join(tempfile.gettempdir(), "no_such_folder_prove"))])
        t0 = time.time()
        rep = run_items("plants", items + [broken], jobs=4, log=lambda s: None)
        took = time.time() - t0
        v = {i.key: r for i, o, r in rep.results}
        self.check("a control that checks something and exits 0 PASSES", v["control"].word == PASS and v["control"].checks == 2)
        self.check("a selftest that hangs is HANGS, killed at its timeout, not waited on (the whole run took %.0f s, its sleep was 120)" % took, v["hangs"].word == HANGS and took < 60)
        self.check("a hang is UNKNOWN, never a pass and never a quiet skip", v["hangs"].bucket == "unknown")
        self.check("exit 0 with no check line is EMPTY (unknown), not a pass", v["empty"].word == EMPTY and v["empty"].bucket == "unknown")
        self.check("exit 0 while a line ends in FAIL is DISAGREES (a failure)", v["contradicts"].word == DISAGREES and v["contradicts"].bucket == "fails")
        self.check("exit 0 while the summary says FAIL is DISAGREES", v["summary_fail"].word == DISAGREES)
        self.check("in a command that is not a selftest, a line ending in FAIL is data (a plant whose link goes FAIL), not a failed check", v["fail_is_data"].word == PASS)
        self.check("a tool that prints no check lines but states a count of checks made (62 of 62) PASSES, and the count is kept", v["count_only"].word == PASS and v["count_only"].declared == 62)
        self.check("a summary that says PASS and counts fewer passed than made (61 of 62) is DISAGREES", v["count_short"].word == DISAGREES)
        self.check("a wrong exit is FAIL", v["wrong_exit"].word == FAIL and "exit 2" in v["wrong_exit"].why)
        self.check("text the command must print, missing, is FAIL", v["text_missing"].word == FAIL and "HOLDS" in v["text_missing"].why)
        self.check("the same run with the text there PASSES", v["run_ok"].word == PASS)
        self.check("a pass whose recorded numbers no longer agree is DISAGREES, with the difference said", v["recheck"].word == DISAGREES and "4" in v["recheck"].why)
        self.check("the same with a re-check that agrees PASSES", v["recheck_ok"].word == PASS)
        self.check("an item with a reason not to run is NOT RUN and the reason is kept", v["skipped"].word == NOT_RUN and "nobody has" in v["skipped"].why)
        self.check("a command that cannot start is a FAIL that says so, not a lost item", v["cannot_start"].word == FAIL and "could not run" in v["cannot_start"].why)
        self.check("every item is in exactly one bucket and the counts add to the total", sum(rep.counts().values()) == len(items) + 1 and len(rep.results) == len(items) + 1)
        self.check("with failures and unknowns the closing line says NOT PROVED", rep.closing().startswith("NOT PROVED") and not rep.proved)
        clean = run_items("clean", [self.item("a", good_code), self.item("b", good_code), self.item("n", good_code, not_run="needs a piece")], jobs=2, log=lambda s: None)
        self.check("all passes, one not run: PROVED, and the closing line counts the one not run", clean.proved and "1 not run" in clean.closing())
        none = run_items("none", [], jobs=2, log=lambda s: None)
        self.check("zero items is NOT PROVED, never PROVED", not none.proved)
        ess = run_items("ess", [self.item("s", good_code), self.item("e", good_code, kind="run", essential=True, not_run="needs the .NET SDK")], jobs=2, log=lambda s: None)
        self.check("passes but no essential proof ran: NOT PROVED HERE", ess.closing().startswith("NOT PROVED HERE"))
        ess2 = run_items("ess2", [self.item("s", good_code), self.item("e", "print('closed_loop: HOLDS')", kind="run", essential=True, expect_text="HOLDS")], jobs=2, log=lambda s: None)
        self.check("with the essential proof passed too: PROVED", ess2.proved)
        t0 = time.time()
        four = run_items("par", [self.item("p%d" % i, "import time;time.sleep(3);print('a PASS')") for i in range(4)], jobs=4, log=lambda s: None)
        par = time.time() - t0
        self.check("four items of 3 s run at once (%.1f s, not 12)" % par, four.counts()["passes"] == 4 and par < 9.5)
        lock = run_items("lock", [self.item("l%d" % i, "import time;time.sleep(1.5);print('a PASS')", lock="same") for i in range(3)], jobs=3, log=lambda s: None)
        self.check("items sharing a lock run one at a time (%.1f s for three of 1.5 s)" % lock.wall, lock.wall >= 4.0)
        late = run_items("budget", [self.item("w%d" % i, "import time;time.sleep(2);print('a PASS')") for i in range(6)], jobs=1, budget=3, log=lambda s: None)
        self.check("when the wall budget is spent, items not started are NOT RUN and say the budget did it", late.counts()["not_run"] >= 2 and any("budget" in v.why for _, _, v in late.results if v.word == NOT_RUN))
        if os.name == "nt":
            # the parent is started at NORMAL priority on purpose: a child that merely inherits its parent's priority (a run under a below-normal parent does) must not pass for one that was set
            helper = os.path.join(tempfile.mkdtemp(prefix="prove_pri_"), "parent.py")
            io.open(helper, "w", encoding="utf-8", newline=chr(10)).write(PRIORITY_PARENT % os.path.dirname(os.path.abspath(__file__)))
            out = subprocess.run([sys.executable, helper], capture_output=True, text=True, creationflags=0x20).stdout
            shutil.rmtree(os.path.dirname(helper), ignore_errors=True)
            self.check("the child runs at below-normal priority (0x4000) even when its parent is at normal priority", "CHILD 0x4000" in out)
        flagless = Item("flagless", "flagless", [Command([sys.executable, "-c", "import sys;sys.stderr.write('x: error: unrecognized arguments: --selftest');sys.exit(2)"], tempfile.gettempdir())], kind="selftest")
        r = run_items("flagless", [flagless], jobs=1, log=lambda s: None)
        self.check("a tool whose source mentions a selftest but takes no such flag is NOT RUN with that said, not a FAIL", r.results[0][2].word == NOT_RUN and "takes no --selftest" in r.results[0][2].why)
        miss = Item("miss", "miss", [Command([sys.executable, "-c", "import sys;print('ModuleNotFoundError: No module named \\'nosuchpiece_xyz\\'');sys.exit(1)"], tempfile.gettempdir())], kind="selftest")
        r = run_items("miss", [miss], jobs=1, log=lambda s: None, judge=Judge({"nosuchpiece_xyz": "A Fake Piece"}))
        self.check("a crash for a missing optional piece that really is missing is NOT RUN (needs it), named", r.results[0][2].word == NOT_RUN and "Fake Piece" in r.results[0][2].why)
        r = run_items("miss2", [miss], jobs=1, log=lambda s: None, judge=Judge({"json": "A Real Piece"}))
        self.check("the same crash naming a piece that IS installed stays a FAIL", r.results[0][2].word == FAIL)
        flaky_marker = os.path.join(tempfile.gettempdir(), "prove_flaky_%d" % os.getpid())
        flaky = self.item("flaky", "import os,sys;m=%r\nif not os.path.exists(m):\n open(m,'w').write('x');print('a FAIL');sys.exit(1)\nprint('a PASS')" % flaky_marker)
        r = run_items("flaky", [flaky, self.item("pad", good_code)], jobs=2, log=lambda s: None)
        d = [x for x in r.results if x[0].key == "flaky"][0]
        self.check("an item that fails beside others and passes alone is a PASS that says it failed beside others", d[2].word == PASS and "alone" in d[2].note)
        try:
            os.remove(flaky_marker)
        except OSError:
            pass
        self.folder_plants()
        print("selftest: %s" % ("PASS" if self.ok else "FAIL"))
        return 0 if self.ok else 1

    def folder_plants(self):
        t = tempfile.mkdtemp(prefix="prove_folder_")
        try:
            def w(rel, text):
                p = os.path.join(t, rel)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                io.open(p, "w", encoding="utf-8", newline="\n").write(text)
            w("good.py", '"""good.py - a tool that proves itself"""\nimport sys\nif "--selftest" in sys.argv:\n    print("selftest: a PASS"); print("selftest: PASS")\n')
            w("quiet.py", "import sys\nif '--selftest' in sys.argv:\n    print('done')\n")
            w("plain.py", "print('no selftest here')\n")
            w("example_x/model/meta.yaml", "meta:\n  loop:\n    code: {language: csharp}\n")
            w("example_y/y_epic.md", "A thing.\n")
            w("example_y/y_draft.yml", "x: 1\n")
            f = Folder(t)
            keys = {i.key: i for i in f.items()}
            self.check("a folder's tools that carry a selftest are found by their source, and the ones that do not are listed as not run", "selftest:good.py" in keys and "selftest:quiet.py" in keys
                       and keys["selftest:plain.py"].not_run and not keys["selftest:good.py"].not_run)
            self.check("a folder's about-line is the first line of the tool's docstring", keys["selftest:good.py"].proves.endswith("a tool that proves itself"))
            self.check("an example whose model is C# is NOT RUN when no .NET SDK is on the path, and says so", ("example:example_x" in keys) and (bool(shutil.which("dotnet")) != bool(keys["example:example_x"].not_run)))
            self.check("a folder with an epic and a draft gets a front-door item, marked essential", "door:example_y" in keys and keys["door:example_y"].essential)
            ex = keys.get("example:example_x")
            if ex is not None and not ex.not_run:
                self.check("an example runs its mutants with the worker count this machine can spare", "--workers" in ex.commands[0].argv and ex.commands[0].argv[-1] == str(example_workers()))
            quick = {i.key for i in f.items(quick=True)}
            self.check("--quick keeps the front door's tools and the front door, and drops the rest", quick == {"door:example_y"})
            rep = run_items("folder", [keys["selftest:good.py"], keys["selftest:quiet.py"], keys["selftest:plain.py"]], jobs=2, log=lambda s: None)
            w_ = {i.key: v.word for i, o, v in rep.results}
            self.check("run on the folder: the tool that proves itself PASSES, the quiet one is EMPTY, the one without a selftest is NOT RUN", w_ == {"selftest:good.py": PASS, "selftest:quiet.py": EMPTY, "selftest:plain.py": NOT_RUN})
            f.close()
            # a clone: the tools are in `tools`, the examples one level up; an install with no examples at all must not read as proved
            good_tool = "\"\"\"good.py - a tool that proves itself\"\"\"" + chr(10) + "import sys" + chr(10) + "if '--selftest' in sys.argv:" + chr(10) + "    print('selftest: a PASS'); print('selftest: PASS')" + chr(10)
            w("clone/tools/good.py", good_tool)
            w("clone/example_y/y_epic.md", "A thing." + chr(10))
            w("clone/example_y/y_draft.yml", "x: 1" + chr(10))
            w("bare/tools/good.py", good_tool)
            c = Folder(os.path.join(t, "clone", "tools"))
            self.check("in a clone, the examples one level above the `tools` folder are found", "door:example_y" in {i.key for i in c.items()})
            c.close()
            b = Folder(os.path.join(t, "bare", "tools"))
            bk = {i.key: i for i in b.items()}
            self.check("with no examples anywhere, the whole-loop item is NOT RUN and essential, so the result cannot read PROVED", "example:none" in bk and bk["example:none"].essential and bk["example:none"].not_run)
            rep = run_items("bare", [bk["selftest:good.py"], bk["example:none"]], jobs=1, log=lambda s: None)
            self.check("and a passing selftest beside it reads NOT PROVED HERE (exit 3)", rep.closing().startswith("NOT PROVED HERE") and rep.exit_code == 3)
            self.check("--quick with no front door at all is the same: NOT RUN, essential", any(i.key == "door:none" and i.essential for i in b.items(quick=True)))
            b.close()
        finally:
            shutil.rmtree(t, ignore_errors=True)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", default=HERE, help="the folder of tools to prove (default: the one this file is in)")
    ap.add_argument("--quick", action="store_true", help="the front door only: its tools' selftests and the shipped epic, about a minute")
    ap.add_argument("--jobs", type=int, default=default_jobs(), help="items at once (default %d)" % default_jobs())
    ap.add_argument("--timeout", type=int, default=0, help="seconds an item may take before it is killed and reported (default: the item's own)")
    ap.add_argument("--budget", type=int, default=None, help="wall seconds for the whole run; items not started by then are NOT RUN")
    ap.add_argument("--list", action="store_true", help="say what would run and what would not, and why, without running anything")
    ap.add_argument("--only", default="", help="run only the items whose key contains this text")
    ap.add_argument("--out", help="also write the result to this file")
    ap.add_argument("--json", help="also write the result as JSON to this file")
    ap.add_argument("--brief", action="store_true", help="what each passing item proves is left out of the printed result")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return Plants().run()
    folder = Folder(a.folder)
    try:
        items = [i for i in folder.items(quick=a.quick) if a.only in i.key]
        if a.timeout:
            for i in items:
                i.timeout = a.timeout
        if a.list:
            for i in items:
                print("%-34s %s" % (i.key, ("NOT RUN: " + i.not_run) if i.not_run else "runs"))
            return 0
        print("prove: %d items, %d at once, below-normal priority%s" % (len(items), a.jobs, ", quick" if a.quick else ""), flush=True)
        rep = run_items("Proof of the tools in %s" % folder.folder.replace("\\", "/").rsplit("/", 1)[-1], items, a.jobs, a.budget, "quick" if a.quick else "all", log=lambda s: print(s, flush=True))
    finally:
        folder.close()
    text = rep.text(full=not a.brief)
    print("\n" + text)
    if a.out:
        io.open(a.out, "w", encoding="utf-8", newline="\n").write(text)
    if a.json:
        io.open(a.json, "w", encoding="utf-8", newline="\n").write(json.dumps(rep.data(), indent=1) + "\n")
    return rep.exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
