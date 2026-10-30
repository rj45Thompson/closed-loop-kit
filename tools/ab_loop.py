#!/usr/bin/env python3
"""ab_loop.py - the closed loop ON against OFF, prompt by prompt: two agents get the same prompt, one works with the loop and one
without, each in its own sandbox; the outcome is measured, a grader that cannot tell which is which sorts each one's failures into
piles, and a board counts the piles.

WHY (2026-10-29): for every prompt, another agent runs too, to see which does better with the closed loop on or off. Results are
not always comparable, so failures are sorted into piles and points are gauged to be fair across them. The loop's own report cannot answer "does it help" - it grades the work by its own criteria. This harness
grades the OUTCOME both arms produce, by measures neither arm controls, and keeps "different but fine" as its own pile instead of
forcing two different good answers into one number.

EVERY PROMPT, WITHOUT A HOOK. Claude Code already writes every prompt to the session transcripts
(~/.claude/projects/<project>/*.jsonl); `harvest` reads them, so nothing is installed and no agent starts on its own. The list
stays in ~/.claude/ab (never in a repo: the transcripts hold the user's personal work too). Pairs run when someone starts them.

THE PIECES (each a class below):
  Harvest   the prompts typed into the transcripts, one per message, filtered to the ones a model's words name
  Sandbox   one arm's copy: the model's code and tests, the sheets the tests read, the model and its chains (DIR/Assets/..., DIR/model)
  Brief     the arm's instructions: the same prompt and the same rules; ON adds the closed loop, OFF forbids it
  Outcome   what an arm produced, measured: the tests pass, the code compiles, how many mutants of the methods it changed survive,
            how big the change is and whether it stayed in its sandbox - measured by this harness, never reported by the arm
  Acceptance the pair's HIDDEN acceptance test: written before either sandbox exists, seen by neither arm, run on both
  Grading   the blind grader's brief (the two diffs as X and Y, the requirements, the rubric) and its answer, a pile count per arm
  Board     every graded pair, and the piles summed per arm

    python -X utf8 ab_loop.py harvest [--match word,word] [--limit 40]
    python -X utf8 ab_loop.py accept <run dir> <test.cs>        the hidden acceptance test, BEFORE either sandbox
    python -X utf8 ab_loop.py sandbox <model folder> <dir>
    python -X utf8 ab_loop.py brief <dir> --arm on|off --prompt "<text>"
    python -X utf8 ab_loop.py outcome <dir> [--mutants]
    python -X utf8 ab_loop.py grade <run dir>            writes the grader's brief (X/Y assigned at random, the key kept apart)
    python -X utf8 ab_loop.py record <run dir> <grade.json>
    python -X utf8 ab_loop.py board
    python -X utf8 ab_loop.py --selftest
"""
import argparse
import difflib
import glob
import hashlib
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# the project the model's paths are relative to: this repo, or a developer's own (CLOSED_LOOP_ROOT, set by the kit's installer)
ROOT = os.path.abspath(os.environ.get("CLOSED_LOOP_ROOT") or os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import usecase_model as ucm  # noqa: E402
import yaml  # noqa: E402
import cs_effect_check as cs  # noqa: E402

AB = os.path.join(os.path.expanduser("~"), ".claude", "ab")
# Claude Code keeps a project's transcripts in a folder named by the project's path (its ':' and separators made '-')
TRANSCRIPTS = os.path.join(os.path.expanduser("~"), ".claude", "projects", re.sub(r"[:\\/]", "-", ROOT))
HEADLESS = os.path.join(ROOT, ".opus-tools", "headless_nunit", "headless_nunit.py")
PILES = [  # (pile, what goes in it) - a failure counts once, in the first pile it fits
    ("wrong-behaviour", "the change makes the game do what a requirement says it must not, or not do what it must (cite the requirement)"),
    ("missed-requirement", "a requirement the prompt touches that the change leaves unmet or unchanged (cite it)"),
    ("invented-value", "a number, name or rule written as fact that no source gives (iron law 1: READ or ABSENT, never guessed)"),
    ("broke-a-gate", "the outcome shows failing tests, a compile error, or a gate the change turned red"),
    ("weak-test", "a test that would pass if the new code were broken (the outcome's surviving mutants are the evidence)"),
    ("unfinished", "work the arm said was done that is not, or a step it stopped before"),
    ("scope-creep", "a change nobody asked for"),
]
FINE = "different-but-fine"     # two good answers that differ: counted apart, never as a failure


class Harvest(object):
    """Every prompt the user typed in this project's sessions, read from the transcripts Claude Code keeps - one row per message, with
    its session and time; tool results, reminders and slash-command wrappers are not prompts."""

    NOT_A_PROMPT = re.compile(r"^\s*(<command-|<local-command|<system-reminder|<task-notification|\[Request interrupted|Caveat:)")

    @staticmethod
    def prompts(project_dir=TRANSCRIPTS):
        out, seen = [], set()           # a continued or forked session repeats its parent's rows: one prompt, once
        for path in sorted(glob.glob(os.path.join(project_dir, "*.jsonl"))):
            for line in io.open(path, encoding="utf-8", errors="replace"):
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("type") != "user" or row.get("isMeta") or row.get("isSidechain"):
                    continue
                msg = row.get("message") or {}
                content = msg.get("content")
                texts = [content] if isinstance(content, str) else [c.get("text") for c in content or [] if isinstance(c, dict) and c.get("type") == "text"]
                for t in texts:
                    t = (t or "").strip()
                    if len(t) >= 20 and not Harvest.NOT_A_PROMPT.match(t) and "This session is being continued" not in t                             and (row.get("timestamp"), t) not in seen:
                        seen.add((row.get("timestamp"), t))
                        out.append({"session": os.path.basename(path)[:-6], "time": row.get("timestamp"), "prompt": t[:4000]})
        return out


class SandboxLayout(object):
    """Where a model's files sit in an arm's sandbox and how its tests run there, read from the model's own loop block (KT-3,
    10-29: the harness had this repo's Unity layout and one shop test file written into its code, so it could not run on anyone
    else's project).
    UNITY - every test runs through headless_nunit (this repo: site.json's unity_project): the Unity project folder's contents at the
      sandbox root, so headless_nunit reads the sandbox as an overlay, with the sheet mirrors the tests read beside them.
    REPO - `runner: dotnet` or `runner: py_selftest` (any other project, the kit's case): every path where it sits under
      CLOSED_LOOP_ROOT, so `dotnet test` or the test script runs inside the sandbox exactly as it would in the repo; a test
      project's whole folder comes along, or a Python test's whole folder (what it imports sits beside it). PK-5, 10-29: two of the
      six tasks are Python, and the harness ran only C# arms."""

    RUNNERS = ("headless_nunit", "dotnet", "py_selftest")

    def __init__(self, model):
        self.tests = [t for t in ucm.as_list(model.loop.get("tests")) if isinstance(t, dict)]
        if not self.tests:
            raise RuntimeError("the model's loop block names no tests: an arm's outcome cannot be measured")
        runners = {t.get("runner", "headless_nunit") for t in self.tests}
        unknown = sorted(runners - set(self.RUNNERS))
        if unknown:
            raise RuntimeError("no arm runner for %s (known: %s)" % (", ".join(unknown), ", ".join(self.RUNNERS)))
        if len(runners) > 1:
            raise RuntimeError("one runner per model, this one names %s" % ", ".join(sorted(runners)))
        self.kind = "unity" if runners == {"headless_nunit"} else "repo"
        self.runner = next(iter(runners))
        if self.kind == "unity" and not cs.UNITY_DIR:
            raise RuntimeError("headless_nunit tests need the Unity project folder (site.json's unity_project)")

    @staticmethod
    def of(sandbox):
        return SandboxLayout(ucm.UseCaseModel.load(os.path.join(sandbox, "model")))

    def dst(self, rel):
        r = rel.replace("\\", "/")
        if self.kind == "repo":
            return r
        head = cs.UNITY_DIR + "/"
        if not r.startswith(head):
            raise RuntimeError("a sandbox is laid out as the Unity project (site.json's unity_project): %s is outside it" % rel)
        return r[len(head):]

    def test_paths(self):
        """what each test entry needs copied: a headless_nunit test file, or a dotnet test project's (or a Python test's) whole folder"""
        return sorted({t["test"] if self.kind == "unity" else os.path.dirname(t["test"].split()[0]).replace("\\", "/") for t in self.tests})

    def reads(self, sandbox):
        """how the closed loop is pointed at this sandbox: the Unity layout is an overlay over the live project; a repo layout IS
        the project, so the loop's root is the sandbox itself (an --overlay there would be read only for Unity paths)"""
        s = sandbox.replace("\\", "/")
        return ("", "--overlay " + s) if self.kind == "unity" else ("CLOSED_LOOP_ROOT=%s " % s, "")

    def hidden_place(self):
        """the folder a hidden acceptance test goes into: beside the model's first test (its file's folder, or its project's)"""
        return os.path.dirname(self.dst(self.tests[0]["test"].split()[0]))


def parse_dotnet(out):
    """(tests run, failed names) from `dotnet test` output: its 'Total: N' summary and its '  Failed <name>' lines"""
    total = sum(int(n) for n in re.findall(r"Total:\s*(\d+)", out))
    return total, sorted(set(re.findall(r"^\s+Failed (\S+)", out, re.M)))


def hidden_class(src, where="the hidden test"):
    """the first class a C# file DECLARES - its comments and strings blanked first (KT-3, 10-29: the hidden test's own comment said
    "filtered to this class when measured", so the filter became FullyQualifiedName~when and ran 0 tests)"""
    m = re.search(r"\bclass\s+(\w+)", cs.blank_out(src))
    if not m:
        raise RuntimeError("%s declares no class to filter on" % where)
    return m.group(1)


SELFTEST_LINE = re.compile(r"^selftest: (.+?) (PASS|FAIL)\s*$")


def parse_selftest(out, rc, script):
    """(checks run, failed names) from a Python test's `selftest: <what> PASS|FAIL` lines (the summary line `selftest: PASS` is
    not a check); a non-zero exit with no FAIL line is one failure named by its script"""
    checks = [(m.group(1), m.group(2)) for m in (SELFTEST_LINE.match(l) for l in out.splitlines()) if m and m.group(1) not in ("", "selftest:")]
    failed = sorted({w[:90] for w, v in checks if v == "FAIL"})
    if rc != 0 and not failed:
        failed = ["%s exited %d" % (os.path.basename(script), rc)]
    return len(checks), failed


def arm_tests(sandbox, layout, suffix="_measure", only=None):
    """(output, tests run, failed names) of the model's tests in the sandbox. `only`: a test file inside the sandbox run alone
    (the hidden acceptance test): headless_nunit compiles just it; dotnet filters to its class."""
    d = os.path.abspath(sandbox)
    if layout.kind == "unity":
        outs, total, failed = [], 0, []
        for t in ([only] if only else [os.path.join(d, layout.dst(t["test"])) for t in layout.tests]):
            o, n, f = headless(d, t, suffix)
            outs.append(o)
            total += n
            failed += f
        return "\n".join(outs), total, sorted(set(failed))
    if layout.runner == "py_selftest":
        outs, total, failed = [], 0, []
        env = dict(os.environ, CLOSED_LOOP_ROOT=d)
        for cmd in ([[only]] if only else [t["test"].split() for t in layout.tests]):
            script = cmd[0] if os.path.isabs(cmd[0]) else os.path.join(d, layout.dst(cmd[0]))
            r = subprocess.run([sys.executable, "-X", "utf8", script] + cmd[1:], capture_output=True, text=True, encoding="utf-8",
                               errors="replace", cwd=d, env=env)
            o = r.stdout + r.stderr
            n, f = parse_selftest(o, r.returncode, script)
            outs.append(o)
            total += n
            failed += f
        return "\n".join(outs), total, sorted(set(failed))
    outs = []
    for t in layout.tests:
        args = ["dotnet", "test", os.path.join(d, layout.dst(t["test"].split()[0])), "--nologo"]
        if only:
            args += ["--filter", "FullyQualifiedName~" + hidden_class(io.open(only, encoding="utf-8-sig").read(), only)]
        r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=d)
        outs.append(r.stdout + r.stderr)
    out = "\n".join(outs)
    total, failed = parse_dotnet(out)
    return out, total, failed


class Sandbox(object):
    # what names the arm in a blind diff (PK-5f, 10-30: P5r-B on quoted its brief's path into people.yaml, P5r-E on named the tool in a
    # comment, and the grader read both): a path into the harness, the brief's file, the tool's own name
    TELLS = (r"[A-Za-z]:[/\\][^\s'\"]*?[/\\]\.claude[/\\]ab[/\\][^\s'\"]*", r"\bbrief_(?:on|off|lite)(?:\.txt)?\b", r"\bclosed_loop(?:\.py)?\b")

    """One arm's copy, laid out as the Unity project so headless_nunit and the closed loop can read it as an overlay: the model's
    code roots and tests under DIR/Assets, every sheet mirror the tests read (Resources/*.csv, Resources/UI/Shops), the model
    under DIR/model and its chains under DIR/chains. A pristine twin (DIR.base) is kept to diff the arm's change against."""

    def __init__(self, path):
        self.path = os.path.abspath(path)

    def make(self, model_dir):
        model = ucm.UseCaseModel.load(model_dir)
        if os.path.exists(self.path):
            raise RuntimeError("the sandbox %s exists - one arm, one fresh sandbox" % self.path)
        layout = SandboxLayout(model)
        rels = [r for r in ucm.as_list((model.loop.get("code") or {}).get("roots"))] + layout.test_paths()
        pattern = shutil.ignore_patterns("*.meta", "bin", "obj", "__pycache__", "kg", "*.sqlite", "*.nt", "*.jsonl", "tick_runs", ".loop_tick*")
        twins = {os.path.normcase(os.path.abspath(p)) for p in (model_dir, os.path.join(model_dir, "..", "chains"))}

        def skip(folder, names):
            return set(pattern(folder, names)) | {n for n in names if os.path.normcase(os.path.abspath(os.path.join(folder, n))) in twins}
        for rel in rels:
            src, dst = os.path.join(ROOT, rel), os.path.join(self.path, layout.dst(rel))
            if os.path.isdir(src):
                # a Python test's folder comes whole: never its graphs, logs or caches (the tool's own folder holds 8 MB of them), and
                # never the model or its chains a second time - they sit at DIR/model and DIR/chains, and a twin inside the code's
                # folder is one an arm could edit by mistake (PK-5, 10-29: the kit's example_py holds its model)
                shutil.copytree(src, dst, ignore=skip, dirs_exist_ok=True)
            else:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
        if layout.kind == "unity":
            res = os.path.join(ROOT, cs.UNITY_DIR, "Assets", "Resources")
            # every text sheet the tests read beside the csvs (AccessorySpec.txt: the OFF arm of R1 saw its absence as an error, 10-29)
            for f in glob.glob(os.path.join(res, "*.csv")) + glob.glob(os.path.join(res, "*.txt")) + glob.glob(os.path.join(res, "UI", "Shops", "*.csv")):
                dst = os.path.join(self.path, "Assets", "Resources", os.path.relpath(f, res))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(f, dst)
        shutil.copytree(model_dir, os.path.join(self.path, "model"), ignore=shutil.ignore_patterns("generated_*"))   # its locks too: the arm starts where the live model stands
        chains = os.path.join(model_dir, "..", "chains")
        if os.path.isdir(chains):
            shutil.copytree(chains, os.path.join(self.path, "chains"))
        shutil.copytree(self.path, self.path + ".base")
        return self

    def diff(self, blind=False):
        """the arm's change as a unified diff against its pristine twin (code, tests, model, chains). `blind` leaves out the
        argument chains: only the loop arm writes one, so showing it would tell the grader which arm is which (R1, 10-29)"""
        out = []
        base = self.path + ".base"
        for top in (self.path, base):
            pass
        files = set()
        for d in (self.path, base):
            for dp, _, fs in os.walk(d):
                for f in fs:
                    rel = os.path.relpath(os.path.join(dp, f), d)
                    # a generated file (generated_branches.yaml: the loop's own output) is no change of the arm's (R2, 10-29: a dead
                    # arm's loop run left one, and the diff counted 1060 lines it never wrote), nor is anything a build writes (bin/,
                    # obj/: KT-3, 10-29 - a dotnet sandbox's diff counted the build's own AssemblyInfo.cs files as 52 lines of change)
                    built = {"bin", "obj"} & set(rel.replace("\\", "/").split("/")[:-1])
                    if rel.endswith((".cs", ".py", ".yaml", ".csv", ".md")) and not rel.startswith(("kg", "Assets" + os.sep + "Resources")) \
                            and not os.path.basename(rel).startswith("generated_") and not built                             and not (blind and rel.startswith("chains")):
                        files.add(rel)
        for rel in sorted(files):
            a, b = os.path.join(base, rel), os.path.join(self.path, rel)
            ta = io.open(a, encoding="utf-8", errors="replace").read().splitlines(True) if os.path.exists(a) else []
            tb = io.open(b, encoding="utf-8", errors="replace").read().splitlines(True) if os.path.exists(b) else []
            if ta != tb:
                out += list(difflib.unified_diff(ta, tb, "a/" + rel.replace(os.sep, "/"), "b/" + rel.replace(os.sep, "/")))
        text = "".join(out)
        if blind:
            # a chain's NAME gives the arm away as surely as the chain: the loop arm's model cited "CH-UNLOCK-1" in a comment (R2,
            # 10-29), and only the loop arm writes chains - so in a blind diff every chain id and chain file name is redacted
            for name in sorted(self.chain_names(), key=len, reverse=True):
                text = re.sub(r"\b%s\b" % re.escape(name), "[a note]", text)
            for tell in Sandbox.TELLS:
                text = re.sub(tell, "[a note]", text)
        return text

    @staticmethod
    def distinctive(name):
        """a chain name worth hiding: one with a digit, a hyphen, an underscore or a capital (CH-UNLOCK-1, fairer_near_miss). A plain
        word is never hidden: P5-A, 10-29, a chain FILE named refund.yaml blanked every "refund" in the loop arm's code and model,
        the blind grader read 17 "[a note]" as placeholders it left, and the grade went against the arm for the harness's edit"""
        return bool(re.search(r"[0-9_A-Z-]", name))

    def chain_names(self):
        """the distinctive ids and file names of every chain in the sandbox or its twin"""
        names = set()
        for d in (os.path.join(self.path, "chains"), os.path.join(self.path + ".base", "chains")):
            for f in glob.glob(os.path.join(d, "*.yaml")):
                names.add(os.path.splitext(os.path.basename(f))[0])
                try:
                    cid = ((yaml.safe_load(io.open(f, encoding="utf-8")) or {}).get("chain") or {}).get("id")
                except yaml.YAMLError:
                    cid = None
                if cid:
                    names.add(str(cid))
        return {n for n in names if Sandbox.distinctive(n)}


class Brief(object):
    """The arm's instructions. Both arms get the same prompt, the same sandbox rules and the same way to test; ON adds the closed
    loop (the skill, the run, an argument chain), OFF is told not to use it. Neither is told it is being compared."""

    # the PROJECT's words come from the run folder (context.txt: what the code is, what the sandbox holds, how to test it, what
    # never to touch), so the harness ships with no project of its own in it; {sandbox} and {arm} are filled in either way
    COMMON = """{context}
Do the task below inside your sandbox ONLY:
  {sandbox}
Never write outside it, and read nothing beside it that this brief does not name: the notes there are for the people running the work. Iron laws: never guess a value (every value is READ from a source or ABSENT with the reason); no fallbacks;
say what you did not verify. When you are done, reply with a short summary of what you changed and anything you could not finish.

THE TASK:
{prompt}
"""
    ON = """
Work WITH the closed loop. Read {skill} first. Before you change anything, run the loop once as your baseline:
  {env}py -X utf8 {tools}/closed_loop.py run {sandbox}/model {overlay} --out {sandbox}/../before.md
Put each new or changed behaviour into the model as a requirement with its effect (model/requirements.yaml); name the requirements
each new test checks on a `/// CHECKS:` line above it; add tests that kill its mutants; then run
  {env}py -X utf8 {tools}/closed_loop.py run {sandbox}/model {overlay} --since {sandbox}/../before.md [--chain <each chain in {sandbox}/chains>]
and close the holes of ours it lists as ADDED - those are your change's; the ones already there are not your task. Write the argument for your change as a chain file in
{sandbox}/chains/ (the skill's chain format: each step a claim and the evidence it rests on) and run it too.
"""
    # a chain FILE is a structured argument with evidence, never the model's private thinking, so the brief says so
    OFF = """
Work the way you normally would. Do NOT use the closed loop: no closed_loop.py, cs_effect_check.py, usecase_kg.py,
usecase_model.py, graph_gnn.py, and no closed-loop / usecase-map / usecase-interrogate skill.
"""

    @staticmethod
    def text(sandbox, arm, prompt, context=None):
        """the arm's brief; `context` is the project's own description (the run folder's context.txt), read by the caller"""
        if context is None:
            p = os.path.join(os.path.dirname(os.path.abspath(sandbox)), "context.txt")
            if not os.path.exists(p):
                raise RuntimeError("no context.txt beside %s: write what the code is, how to test it and what never to touch" % sandbox)
            context = io.open(p, encoding="utf-8").read().strip()
        context = context.replace("{sandbox}", sandbox).replace("{arm}", arm)
        env, overlay = SandboxLayout.of(sandbox).reads(sandbox) if arm == "on" else ("", "")
        return Brief.COMMON.format(context=context, sandbox=sandbox, arm=arm, prompt=prompt) + \
            (Brief.ON if arm == "on" else Brief.OFF).format(sandbox=sandbox, tools=HERE.replace("\\", "/"), env=env, overlay=overlay,
                                                            skill=os.path.join(os.path.expanduser("~"), ".claude", "skills", "closed-loop", "SKILL.md").replace("\\", "/"))


class Outcome(object):
    """What an arm produced, measured by this harness (never the arm's word): the tests pass, the change's size, files outside
    the sandbox the arm touched (the live project's Shops scripts against their state when the run began), and - with
    `mutants` - how many mutants of the methods the arm changed survive its tests (closed_loop.py over the sandbox)."""

    def __init__(self, sandbox):
        self.box = Sandbox(sandbox)

    def measure(self, mutants=False):
        d = self.box.path
        out, total, failed = arm_tests(d, SandboxLayout.of(d))
        diff = self.box.diff()
        res = {"compiles": compiled(out, SandboxLayout.of(d).runner == "py_selftest"), "tests": total,
               "failed": failed, "diff_lines": sum(1 for l in diff.splitlines() if l[:1] in "+-" and not l.startswith(("+++", "---"))),
               "files": sorted({l[6:] for l in diff.splitlines() if l.startswith("+++ b/")})}
        acc = Acceptance.measure(os.path.dirname(d), d)
        if acc is not None:
            res["acceptance"] = acc
        if mutants:
            lay = SandboxLayout.of(d)
            env = dict(os.environ, CLOSED_LOOP_ROOT=d) if lay.kind == "repo" else None
            rr = subprocess.run([sys.executable, "-X", "utf8", os.path.join(HERE, "closed_loop.py"), "run", os.path.join(d, "model")]
                                + (["--overlay", d] if lay.kind == "unity" else []) + ["--out", os.path.join(d, "LOOP_OUTCOME.md")],
                                capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=d if env else ROOT, env=env)
            m = re.search(r"(\d+) made, (\d+) killed, (\d+) SURVIVED", rr.stdout)
            res["mutants"] = {"made": int(m.group(1)), "killed": int(m.group(2)), "survived": int(m.group(3))} if m else {"error": rr.stdout[-300:]}
        return res


def compiled(out, python=False):
    """C#: no compile error. Python: no SyntaxError and nothing it imports missing (a Python arm's "does not compile"). Judged by
    the LANGUAGE that ran: a Python tool's own self-test prints the words "does not compile" in its plants (P5-D, 10-29: the
    toolkit arm read as not compiling with all 28 of its checks passing)"""
    if python:
        return not re.search(r"^(SyntaxError|IndentationError|ImportError|ModuleNotFoundError)\b", out, re.M)
    return "does not compile" not in out and "error CS" not in out


def headless(sandbox, test, suffix="_measure"):
    """one headless NUnit run of `test` with the sandbox as the overlay: (its output, the tests it ran, the names that failed)"""
    d = os.path.abspath(sandbox)
    r = subprocess.run([sys.executable, "-X", "utf8", HEADLESS, "--overlay", d, "--out", d + suffix, "--test", test],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT)
    out = r.stdout + r.stderr
    total = re.search(r"Total tests:\s*(\d+)", out)
    return out, int(total.group(1)) if total else 0, sorted(set(re.findall(r"^\s+Failed (\S+)", out, re.M)))


def _sha(path):
    return hashlib.sha1(io.open(path, "rb").read()).hexdigest()


class Acceptance(object):
    """The pair's HIDDEN acceptance test (REVIEW_2 R2-9, 10-29). R1's two arms each passed their OWN tests, which says only that
    each agreed with itself. This test is written by whoever sets the pair up BEFORE either sandbox exists (set() refuses once one
    does), kept in the run folder with its hash, and copied into an arm's sandbox only while it is measured, then removed: no arm
    sees it and no diff holds it. The prompt names the entry point it calls, so an arm that leaves it out fails to compile it -
    that is a failure of the task, not of the test. Before the arms run, the test is proved able to FAIL (the pristine code) and
    able to PASS (a reference change kept apart from both)."""

    NAME = "AcceptanceHidden.cs"
    NAMES = {".cs": "AcceptanceHidden.cs", ".py": "acceptance_hidden.py"}     # the hidden test keeps its language (PK-5, 10-29)
    # kept OUTSIDE the run folder: a sandbox's parent is one `ls ..` from an arm, so the hidden test never sits there
    HIDDEN = os.path.join(AB, "hidden")

    @staticmethod
    def path(run, ext=None):
        """the run's hidden test: the one set (whichever language), or, with `ext`, where one in that language goes"""
        d = os.path.join(Acceptance.HIDDEN, os.path.basename(os.path.abspath(run)))
        if ext:
            return os.path.join(d, Acceptance.NAMES[ext])
        there = [n for n in Acceptance.NAMES.values() if os.path.exists(os.path.join(d, n))]
        if len(there) > 1:
            raise RuntimeError("%s holds two hidden tests (%s): a run has one" % (d, ", ".join(there)))
        return os.path.join(d, there[0] if there else Acceptance.NAME)

    @staticmethod
    def record(run):
        return os.path.join(os.path.dirname(Acceptance.path(run)), "acceptance.json")

    @staticmethod
    def set(run, test_file):
        for arm in ("on", "off"):
            if os.path.exists(os.path.join(run, arm)):
                raise RuntimeError("the %s sandbox already exists: a hidden acceptance test is written FIRST, before either arm" % arm)
        ext = os.path.splitext(test_file)[1].lower()
        if ext not in Acceptance.NAMES:
            raise RuntimeError("a hidden test is C# or Python (%s), not %s" % (", ".join(sorted(Acceptance.NAMES)), test_file))
        dst = Acceptance.path(run, ext)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        for other in Acceptance.NAMES.values():
            if os.path.exists(os.path.join(os.path.dirname(dst), other)) and other != os.path.basename(dst):
                raise RuntimeError("this run already has a hidden test in another language (%s)" % other)
        shutil.copy2(test_file, dst)
        rec = {"sha": _sha(dst), "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "from": os.path.abspath(test_file)}
        io.open(Acceptance.record(run), "w", encoding="utf-8").write(json.dumps(rec, indent=1))
        return rec

    @staticmethod
    def measure(run, sandbox, runner=None, place=None):
        """None when the pair has no hidden test; else {compiles, total, failed, passed}. `place`: the folder in the sandbox it goes
        into (default: beside the model's first test, SandboxLayout.hidden_place); `runner`: arm_tests (a selftest's stand-in)"""
        src = Acceptance.path(run)
        if not os.path.exists(src):
            return None
        rec = json.load(io.open(Acceptance.record(run), encoding="utf-8"))
        if _sha(src) != rec["sha"]:
            raise RuntimeError("the hidden acceptance test changed after it was set (%s): it is set once, before the arms" % src)
        layout = SandboxLayout.of(sandbox) if place is None or runner is None else None
        dst = os.path.join(sandbox, place if place is not None else layout.hidden_place(), os.path.basename(src))
        if os.path.exists(dst):
            raise RuntimeError("%s is already in the sandbox: the hidden test's name is taken there" % dst)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        # a FRESH timestamp: copied with its own (older than the arm's built dll), MSBuild's incremental check saw no new input and
        # never compiled it - the dotnet sandbox ran 0 hidden tests (KT-3, 10-29)
        shutil.copyfile(src, dst)
        os.utime(dst, None)
        try:
            if runner:
                out, total, failed = runner(sandbox, dst, "_accept")
            elif dst.endswith(".py"):
                # a Python hidden test runs as Python in ANY layout: it may check the model rather than the code (a model-only task,
                # PK-5 10-29); it is told where the sandbox is, never left to find it
                r = subprocess.run([sys.executable, "-X", "utf8", dst], capture_output=True, text=True, encoding="utf-8", errors="replace",
                                   cwd=sandbox, env=dict(os.environ, AB_SANDBOX=os.path.abspath(sandbox), AB_TOOLS=HERE, CLOSED_LOOP_ROOT=ROOT))
                out = r.stdout + r.stderr
                total, failed = parse_selftest(out, r.returncode, dst)
            else:
                out, total, failed = arm_tests(sandbox, layout, "_accept", only=dst)
        finally:
            os.remove(dst)
        ok = compiled(out, dst.endswith(".py"))
        return {"compiles": ok, "total": total, "failed": failed, "passed": ok and total > 0 and not failed}


class Leaks(object):
    """Did an arm ask for what it should never see: the other arm's folder, the operator's notes, the hidden test, the grades. Read from
    the arm's own tool calls, so a grade can be weighed by it. PK-5f, 10-30: 3 of 27 arms, all with the loop, had read `events.txt` (the
    run's notes, which say how the other arm finished), found only because one of them said so in its report."""

    NOTES = re.compile(r"events\.txt|grade_(?:brief|answer|key)|_outcome\.json|suite_(?:on|off)|/ab/hidden|/ab/grading|results\.jsonl")

    @staticmethod
    def scan(transcript, run, arm):
        """[(what, the command around it)] for one arm's transcript"""
        other = "off" if arm == "on" else "on"
        sibling = re.compile(r"(?:/%s|\.\.)/%s(?:\.base|_build|_mutants)?(?:/|\b)" % (re.escape(run), other))
        found = []
        for line in io.open(transcript, encoding="utf-8"):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            message = row.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            for item in content if isinstance(content, list) else []:
                if not (isinstance(item, dict) and item.get("type") == "tool_use"):
                    continue
                asked = dict(item.get("input") or {})
                asked.pop("description", None)     # words about the call, not the call: "turn off" is not the off arm
                text = json.dumps(asked).replace("\\\\", "/").replace("\\", "/")
                for rx, what in ((sibling, "the other arm"), (Leaks.NOTES, "the run's notes or results")):
                    hit = rx.search(text)
                    if hit:
                        found.append((what, text[max(0, hit.start() - 40):hit.end() + 30]))
        return found


class Grading(object):
    """The blind grader's brief and its answer. The two arms' diffs are shown as X and Y, assigned at random; the key is written
    beside the run, never into the brief. The grader puts each failure in ONE pile and says which requirement or line shows it."""

    @staticmethod
    def brief(run):
        on, off = os.path.join(run, "on"), os.path.join(run, "off")
        prompt = io.open(os.path.join(run, "prompt.txt"), encoding="utf-8").read()
        model = ucm.UseCaseModel.load(os.path.join(on + ".base", "model"))
        reqs = "\n".join("  %s (%s): %s" % (r.id, r.kind, r.text) for r in model.requirements.values())
        flip = random.random() < 0.5  # noqa: S311 - a coin, not a secret
        key = {"X": "off" if flip else "on", "Y": "on" if flip else "off"}
        io.open(os.path.join(run, "grade_key.json"), "w", encoding="utf-8").write(json.dumps(key))
        arms = {}
        for label, arm in key.items():
            d = os.path.join(run, arm)
            outcome = json.load(io.open(os.path.join(run, "%s_outcome.json" % arm), encoding="utf-8"))                 if os.path.exists(os.path.join(run, "%s_outcome.json" % arm)) else {}
            left = os.path.join(run, "suite_%s.json" % arm)
            if os.path.exists(left):
                outcome["left_behind"] = json.load(io.open(left, encoding="utf-8")).get("score")
            arms[label] = {"diff": Sandbox(d).diff(blind=True)[:60000], "outcome": Grading.shown(outcome)}
        piles = "\n".join("  %-20s %s" % p for p in PILES) + "\n  %-20s %s" % (FINE, "the two differ here and both are acceptable (not a failure)")
        text = """You grade two answers to one task. You do not know how either was produced; judge only what is in front of you.

THE TASK:
{prompt}

THE REQUIREMENTS the project is held to:
{reqs}

THE PILES (a failure goes in exactly one, the first that fits):
{piles}

ANSWER X - its change and its measured outcome (measured by the harness, not reported by the answer; "acceptance" is a test
written before either answer existed and hidden from both):
{xo}
{xd}

ANSWER Y:
{yo}
{yd}

Reply with ONLY a JSON object: {{"X": {{"<pile>": [<one line per failure, each citing a requirement id or a diff line>]}},
"Y": {{...}}, "different_but_fine": [<lines>], "better": "X" | "Y" | "tie", "why": "<two sentences>"}}. An empty pile is [].
""".format(prompt=prompt, reqs=reqs, piles=piles, xo=json.dumps(arms["X"]["outcome"]), xd=arms["X"]["diff"] or "(no change)",
           yo=json.dumps(arms["Y"]["outcome"]), yd=arms["Y"]["diff"] or "(no change)")
        io.open(os.path.join(run, "grade_brief.txt"), "w", encoding="utf-8").write(text)
        return text

    # what the grader sees of an outcome (REVIEW_2 R2-9): whether it compiles, the HIDDEN acceptance test's result, the change's
    # size and files - never an arm's own test count or failures (each arm wrote its own tests: a count measures its effort, not
    # the task), never loop survivors, never a chain (only the loop arm writes one, see Sandbox.diff)
    # left_behind (PK-5, 10-29): how many plausible WRONG versions of the feature the tests the answer left behind reject - measured
    # by the harness the same way for both, so it says how well the next change is guarded, not how hard an arm worked
    SHOWN = ("compiles", "acceptance", "diff_lines", "files", "left_behind")

    @staticmethod
    def shown(outcome):
        o = {k: outcome[k] for k in Grading.SHOWN if k in outcome}
        o["files"] = [f for f in o.get("files", []) if not f.startswith("chains/")]
        return o

    @staticmethod
    def unblind(run, answer):
        key = json.load(io.open(os.path.join(run, "grade_key.json"), encoding="utf-8"))
        row = {"run": os.path.basename(run), "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "better": key.get(answer.get("better"), "tie"),
               "why": answer.get("why"), "fine": answer.get("different_but_fine") or []}
        for label in ("X", "Y"):
            piles = answer.get(label) or {}
            row[key[label]] = {p: len(piles.get(p) or []) for p, _ in PILES}
            row[key[label] + "_lines"] = piles
        return row


class Board(object):
    """Every graded pair, and each pile summed per arm - the counts side by side, never one score: two different good answers
    are counted as `different-but-fine`, not as a failure of either"""

    PATH = os.path.join(AB, "results.jsonl")

    @staticmethod
    def add(row):
        os.makedirs(AB, exist_ok=True)
        io.open(Board.PATH, "a", encoding="utf-8").write(json.dumps(row) + "\n")

    @staticmethod
    def show(path=None):
        rows = [json.loads(l) for l in io.open(path or Board.PATH, encoding="utf-8")] if os.path.exists(path or Board.PATH) else []
        print("A/B BOARD  %d graded pair(s)  (%s)" % (len(rows), path or Board.PATH))
        print("  %-20s %6s %6s" % ("pile", "ON", "OFF"))
        for p, _ in PILES:
            print("  %-20s %6d %6d" % (p, sum(r["on"][p] for r in rows), sum(r["off"][p] for r in rows)))
        print("  %-20s %6d" % (FINE, sum(len(r["fine"]) for r in rows)))
        print("  better: ON %d, OFF %d, tie %d" % tuple(sum(1 for r in rows if r["better"] == w) for w in ("on", "off", "tie")))
        return rows


def selftest():
    """the harness's own plants: a transcript row of each kind is or is not a prompt; a sandbox's diff sees an edit and only it;
    a grader's answer is unblinded to the right arm; the board sums piles per arm"""
    import tempfile
    ok = True
    t = tempfile.mkdtemp(prefix="ab_selftest_")
    rows = [{"type": "user", "message": {"content": "finish the shops menu and show me screenshots"}, "timestamp": "t1"},
            {"type": "user", "message": {"content": [{"type": "tool_result", "content": "x"}]}},
            {"type": "user", "message": {"content": "<command-name>/clear</command-name> and more words here"}},
            {"type": "user", "isMeta": True, "message": {"content": "a meta row that is long enough to pass"}},
            {"type": "assistant", "message": {"content": "an answer, not a prompt, long enough"}}]
    io.open(os.path.join(t, "s1.jsonl"), "w", encoding="utf-8").write("\n".join(json.dumps(r) for r in rows))
    got = Harvest.prompts(t)
    good = [p["prompt"] for p in got] == ["finish the shops menu and show me screenshots"]
    ok &= good
    print("selftest: HARVEST one prompt from five rows (a tool result, a slash command, a meta row and an answer are not) %s" % ("PASS" if good else "FAIL"))
    box = Sandbox(os.path.join(t, "box"))
    os.makedirs(os.path.join(box.path, "Assets"))
    io.open(os.path.join(box.path, "Assets", "A.cs"), "w", encoding="utf-8").write("class A { int x = 1; }\n")
    shutil.copytree(box.path, box.path + ".base")
    io.open(os.path.join(box.path, "Assets", "A.cs"), "w", encoding="utf-8").write("class A { int x = 2; }\n")
    os.makedirs(os.path.join(box.path, "model"))
    io.open(os.path.join(box.path, "model", "generated_branches.yaml"), "w", encoding="utf-8").write("cells: {}\n")
    os.makedirs(os.path.join(box.path, "Assets", "obj"))
    io.open(os.path.join(box.path, "Assets", "obj", "Built.AssemblyInfo.cs"), "w", encoding="utf-8").write("// a build wrote this\n")
    d = box.diff()
    good = "-class A { int x = 1; }" in d and "+class A { int x = 2; }" in d and d.count("+++") == 1 and "generated_" not in d
    os.makedirs(os.path.join(box.path, "chains"))
    io.open(os.path.join(box.path, "chains", "CH-X-1.yaml"), "w", encoding="utf-8").write("chain: {id: CH-X-1, steps: []}\n")
    io.open(os.path.join(box.path, "Assets", "A.cs"), "w", encoding="utf-8").write("class A { int x = 2; } // see CH-X-1\n")
    good = good and "CH-X-1" in box.diff() and "CH-X-1" not in box.diff(blind=True) and "[a note]" in box.diff(blind=True)
    io.open(os.path.join(box.path, "chains", "refund.yaml"), "w", encoding="utf-8").write("chain: {steps: []}\n")
    io.open(os.path.join(box.path, "Assets", "A.cs"), "w", encoding="utf-8").write("class A { int x = 2; } // a refund, see CH-X-1\n")
    good = good and "a refund" in box.diff(blind=True) and "CH-X-1" not in box.diff(blind=True)
    ok &= good
    print("selftest: SANDBOX the diff holds the one edit, and no generated or built file; a blind diff names no chain id, and never "
          "blanks a plain word a chain file happens to be named %s" % ("PASS" if good else "FAIL"))
    io.open(os.path.join(box.path, "Assets", "A.cs"), "w", encoding="utf-8").write(
        "class A { int x = 2; } // from C:/x/.claude/ab/runs/P5/brief_on.txt, closed_loop.py says so; the loop of a closed door\n")
    seen = box.diff(blind=True)
    good = "closed_loop" not in seen and "brief_on" not in seen and ".claude" not in seen and "the loop of a closed door" in seen and "closed_loop.py" in box.diff()
    ok &= good
    print("selftest: TELLS a blind diff hides a path into the harness, the brief's file and the tool's name, and keeps plain words %s" % ("PASS" if good else "FAIL"))
    log = os.path.join(t, "arm.jsonl")
    calls = [{"command": "cat model/meta.yaml", "description": "turn off the lights"}, {"file_path": "C:\\x\\ab\\runs\\R\\on\\model\\a.yaml"}]
    with io.open(log, "w", encoding="utf-8") as f:
        for c in calls:
            f.write(json.dumps({"message": {"content": [{"type": "tool_use", "input": c}]}}) + "\n")
    good = Leaks.scan(log, "R", "on") == []
    with io.open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps({"message": {"content": [{"type": "tool_use", "input": {"command": "cat ../context.txt ../events.txt"}}]}}) + "\n")
        f.write(json.dumps({"message": {"content": [{"type": "tool_use", "input": {"file_path": "C:\\x\\ab\\runs\\R\\off\\model\\a.yaml"}}]}}) + "\n")
    good = good and [w for w, _ in Leaks.scan(log, "R", "on")] == ["the run's notes or results", "the other arm"]
    ok &= good
    print("selftest: LEAKS an arm's own files and a word in a description are clean; the run's notes and the other arm's folder are found %s" % ("PASS" if good else "FAIL"))
    run = os.path.join(t, "run")
    os.makedirs(run)
    io.open(os.path.join(run, "grade_key.json"), "w", encoding="utf-8").write(json.dumps({"X": "off", "Y": "on"}))
    row = Grading.unblind(run, {"X": {"weak-test": ["a"], "invented-value": ["b", "c"]}, "Y": {}, "better": "Y", "why": "w"})
    good = row["better"] == "on" and row["off"]["invented-value"] == 2 and row["on"]["weak-test"] == 0
    ok &= good
    print("selftest: GRADING the answer unblinded to the right arm %s" % ("PASS" if good else "FAIL"))
    p = os.path.join(t, "results.jsonl")
    io.open(p, "w", encoding="utf-8").write(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    rows = Board.show(p)
    good = len(rows) == 2 and sum(r["off"]["invented-value"] for r in rows) == 4
    ok &= good
    print("selftest: BOARD piles summed per arm %s" % ("PASS" if good else "FAIL"))
    run2 = os.path.join(t, "run2")
    os.makedirs(run2)
    Acceptance.HIDDEN = os.path.join(t, "hidden")      # never the real store
    hidden = os.path.join(t, "Hidden.cs")
    io.open(hidden, "w", encoding="utf-8").write("// a hidden test\n")
    os.utime(hidden, (time.time() - 86400, time.time() - 86400))     # written a day ago: the copy must still be fresh
    Acceptance.set(run2, hidden)
    os.makedirs(os.path.join(run2, "on", "Assets"))
    try:
        Acceptance.set(run2, hidden)
        refused = False
    except RuntimeError:
        refused = True
    seen = []

    def broken(sandbox, test, suffix):
        seen.append(os.path.exists(test) and os.path.getmtime(test) > time.time() - 60)
        raise RuntimeError("the runner broke")
    try:
        Acceptance.measure(run2, os.path.join(run2, "on"), runner=broken, place="T")
    except RuntimeError:
        pass
    left = os.path.exists(os.path.join(run2, "on", "T", Acceptance.NAME))
    io.open(Acceptance.path(run2), "a", encoding="utf-8").write("// edited after the arms began\n")
    try:
        Acceptance.measure(run2, os.path.join(run2, "on"), runner=lambda s, x, y: ("Total tests: 1", 1, []), place="T")
        stale = False
    except RuntimeError:
        stale = True
    good = refused and seen == [True] and not left and stale
    ok &= good
    print("selftest: ACCEPTANCE set only before any sandbox; in a sandbox only while measured, with a fresh timestamp, and gone "
          "after, even when the run breaks; a hidden test edited after it was set is refused %s" % ("PASS" if good else "FAIL"))
    shown = Grading.shown({"compiles": True, "tests": 28, "failed": ["T"], "mutants": {"survived": 3}, "acceptance": {"passed": True},
                           "files": ["chains/c.yaml", "Assets/A.cs"], "diff_lines": 4})
    good = set(shown) == {"compiles", "acceptance", "files", "diff_lines"} and shown["files"] == ["Assets/A.cs"]
    ok &= good
    print("selftest: GRADER sees the hidden test's result, never an arm's own test count, failures, survivors or chain %s" % ("PASS" if good else "FAIL"))
    import types
    M = lambda tests: types.SimpleNamespace(loop={"tests": tests})
    lay_u = SandboxLayout(M([{"runner": "headless_nunit", "test": cs.UNITY_DIR + "/Assets/Editor/XTests.cs"}])) if cs.UNITY_DIR else None
    lay_r = SandboxLayout(M([{"runner": "dotnet", "test": "example/tests/W.Tests/W.Tests.csproj", "covers": ["example/src/W"]}]))
    refused = 0
    for bad in ([], [{"runner": "pytest", "test": "t.py"}], [{"runner": "dotnet", "test": "a.csproj"}, {"runner": "headless_nunit", "test": "b.cs"}]):
        try:
            SandboxLayout(M(bad))
        except RuntimeError:
            refused += 1
    good = lay_r.kind == "repo" and lay_r.dst("example/src/W") == "example/src/W" and lay_r.test_paths() == ["example/tests/W.Tests"] \
        and lay_r.hidden_place() == "example/tests/W.Tests" and refused == 3 \
        and (lay_u is None or (lay_u.kind == "unity" and lay_u.dst(cs.UNITY_DIR + "/Assets/Editor/XTests.cs") == "Assets/Editor/XTests.cs"
                               and lay_u.hidden_place() == "Assets/Editor"))
    ok &= good
    print("selftest: LAYOUT headless tests lay out as the Unity project, dotnet ones as the repo; no tests, an unknown runner or "
          "two runners are refused %s" % ("PASS" if good else "FAIL"))
    n, f = parse_dotnet("Failed!  - Failed:     1, Passed:     4, Skipped:     0, Total:     5, Duration: 9 ms - W.Tests.dll (net8.0)\n"
                        "  Failed W.Tests.PayTests.Pay_TakesThePrice [3 ms]\n")
    good = n == 5 and f == ["W.Tests.PayTests.Pay_TakesThePrice"] and \
        hidden_class('// filtered to this class when measured\n/* class Nope */ var s = "class Nor";\npublic class WalletAcceptanceHidden { }') == "WalletAcceptanceHidden"
    ok &= good
    print("selftest: DOTNET a dotnet test summary gives its total and the names that failed; the hidden test's class is read past "
          "its comments and strings %s" % ("PASS" if good else "FAIL"))
    # PYTHON (PK-5, 10-29): a Python test's checks are its `selftest: <what> PASS|FAIL` lines, never its summary line; a crash with
    # no FAIL line is a failure; a SyntaxError or a missing import is "does not compile"; the layout copies the test's folder and
    # points the loop at the sandbox as its root; a hidden test keeps its language and a run holds one
    out = "selftest: SINCE holes apart PASS\nselftest: TICK a lock PASS\nselftest: BUY refuses FAIL\nselftest: FAIL\n"
    n1, f1 = parse_selftest(out, 1, "x.py")
    n2, f2 = parse_selftest("Traceback (most recent call last):\nZeroDivisionError: division by zero\n", 1, "/a/shop_test.py")
    lay_p = SandboxLayout(M([{"runner": "py_selftest", "test": "example_py/shop_test.py", "covers": ["example_py/shop.py"]},
                             {"runner": "py_selftest", "test": "example_py/more_test.py --selftest", "covers": ["example_py/shop.py"]}]))
    env, ov = lay_p.reads("C:\\sb\\on")
    hid = os.path.join(t, "hidden_lang")
    os.makedirs(hid)
    for name in ("h.py", "h.cs", "h.txt"):
        io.open(os.path.join(hid, name), "w", encoding="utf-8").write("# hidden\n")
    saved, Acceptance.HIDDEN = Acceptance.HIDDEN, os.path.join(t, "hidden_store")
    try:
        Acceptance.set(os.path.join(t, "runP"), os.path.join(hid, "h.py"))
        lang = os.path.basename(Acceptance.path(os.path.join(t, "runP"))) == "acceptance_hidden.py"
        two = 0
        for name in ("h.cs", "h.txt"):
            try:
                Acceptance.set(os.path.join(t, "runP"), os.path.join(hid, name))
            except RuntimeError:
                two += 1
    finally:
        Acceptance.HIDDEN = saved
    good = (n1, f1) == (3, ["BUY refuses"]) and (n2, f2) == (0, ["shop_test.py exited 1"]) \
        and not compiled("Traceback\nSyntaxError: invalid syntax\n", True) and not compiled("ModuleNotFoundError: No module named 'shopx'\n", True) \
        and compiled("selftest: x PASS\n", True) and compiled("selftest: a copy that does not compile is refused PASS\n", True) \
        and not compiled("error CS0103: the name 'x' does not exist\n") and lay_p.kind == "repo" and lay_p.test_paths() == ["example_py"] \
        and env == "CLOSED_LOOP_ROOT=C:/sb/on " and ov == "" and lang and two == 2
    ok &= good
    print("selftest: PYTHON a Python test's checks are its PASS/FAIL lines, not its summary; a crash is a failure; a SyntaxError or a "
          "missing import does not compile; its folder is copied and the loop's root is the sandbox; a hidden test keeps its "
          "language, one per run %s" % ("PASS" if good else "FAIL"))
    shutil.rmtree(t, ignore_errors=True)
    print("selftest: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["harvest", "accept", "sandbox", "brief", "outcome", "grade", "record", "board", "leaks"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--match", default="")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--arm", choices=["on", "off"])
    ap.add_argument("--prompt")
    ap.add_argument("--mutants", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "harvest":
        ps = Harvest.prompts()
        words = [w.strip().lower() for w in a.match.split(",") if w.strip()]
        if words:
            ps = [p for p in ps if any(w in p["prompt"].lower() for w in words)]
        os.makedirs(AB, exist_ok=True)
        io.open(os.path.join(AB, "prompts.jsonl"), "w", encoding="utf-8").write("".join(json.dumps(p) + "\n" for p in ps))
        print("ab_loop: %d prompt(s)%s -> %s" % (len(ps), " naming " + ", ".join(words) if words else "", os.path.join(AB, "prompts.jsonl")))
        for p in ps[-a.limit:]:
            print("  %s %s  %s" % ((p["time"] or "")[:16], p["session"][:8], re.sub(r"\s+", " ", p["prompt"])[:150]))
        return 0
    if a.cmd == "accept":
        rec = Acceptance.set(os.path.abspath(a.args[0]), a.args[1])
        print("ab_loop: the hidden acceptance test -> %s (sha %s, set %s, before either sandbox)" % (Acceptance.path(os.path.abspath(a.args[0])), rec["sha"][:12], rec["at"]))
        return 0
    if a.cmd == "sandbox":
        Sandbox(a.args[1]).make(a.args[0])
        print("ab_loop: sandbox %s (and its pristine twin %s.base)" % (a.args[1], a.args[1]))
        return 0
    if a.cmd == "brief":
        print(Brief.text(os.path.abspath(a.args[0]), a.arm, a.prompt))
        return 0
    if a.cmd == "outcome":
        print(json.dumps(Outcome(a.args[0]).measure(a.mutants), indent=1))
        return 0
    if a.cmd == "grade":
        Grading.brief(a.args[0])
        print("ab_loop: the grader's brief -> %s (the key is in grade_key.json beside it - never give it to the grader)" % os.path.join(a.args[0], "grade_brief.txt"))
        return 0
    if a.cmd == "record":
        row = Grading.unblind(a.args[0], json.load(io.open(a.args[1], encoding="utf-8")))
        Board.add(row)
        print(json.dumps({k: v for k, v in row.items() if not k.endswith("_lines")}, indent=1))
        return 0
    if a.cmd == "board":
        Board.show()
        return 0
    if a.cmd == "leaks":
        found = Leaks.scan(a.args[2], a.args[0], a.args[1])
        for what, around in found:
            print("ab_loop: %s: %s" % (what, around))
        print("ab_loop: %s %s asked for what it should not see %d time(s)" % (a.args[0], a.args[1], len(found)))
        return 1 if found else 0
    ap.error("a command, or --selftest")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
