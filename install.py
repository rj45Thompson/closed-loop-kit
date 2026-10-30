#!/usr/bin/env python3
"""install.py - put the closed loop into Claude Code: the skill where Claude finds it, the tools beside it, what they need checked,
every selftest run, and the whole loop run on the example project as the acceptance test.

    cd closed-loop-kit                     (the zip unpacks into this folder)
    python install.py                      install for this user (~/.claude/skills/closed-loop and ~/.claude/closed-loop)
    python install.py --dest DIR           the tools (and the examples, with the report each run writes) somewhere else
    python install.py --skills DIR         the skill somewhere else (Claude Code reads ~/.claude/skills)
    python install.py --check              check only: nothing is copied
    python install.py --uninstall          remove what an install put down (it touches nothing else)

Needs: Python 3.10+ and PyYAML. Optional, each named if absent: PyTorch (the GNN link proposer), scikit-learn (ranking which holes to
show first), the .NET SDK (the C# adapter and the example and the C# equivalence check by IL, built with dotnet itself). Without an
optional piece the rest still runs; what it would have done is said, never faked.
"""
import argparse
import importlib
import os
import shutil
import subprocess
import sys

KIT = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")
TOOLS = ["usecase_model.py", "usecase_kg.py", "closed_loop.py", "cs_effect_check.py", "py_effect_check.py", "graph_gnn.py",
         "ab_loop.py", "grade_panel.py", "loop_stats.py", "improvement.py", "improvement/stat_requirements.yaml", "buzzsaw.py",
         "loop_tick.py", "TICK.md", "run_plan.py", "telemetry_audit.py", "survivor_rank.py", "gnn_tasks.py", "gnn_more.py", "gnn_curve.py", "graph_flythrough.py", "gnn_goals.py", "gnn_goals.yaml",
         "il_bodies/il_bodies.csproj", "il_bodies/Program.cs", "STATS.md"]
SELFTESTS = ["usecase_model.py", "usecase_kg.py", "closed_loop.py", "cs_effect_check.py", "py_effect_check.py", "loop_stats.py",
             "ab_loop.py", "grade_panel.py", "improvement.py", "buzzsaw.py", "loop_tick.py", "telemetry_audit.py", "graph_flythrough.py", "gnn_goals.py"]
# the selftests that need an optional piece. Absent, each is named as skipped and never counted as passed (10-30: on Python and PyYAML
# alone, run_plan.py and survivor_rank.py crashed on the import and a clean install read INSTALLED WITH FAILURES)
NEEDS = {"torch": ["graph_gnn.py", "gnn_more.py", "gnn_tasks.py", "gnn_curve.py", "run_plan.py"], "sklearn": ["survivor_rank.py"]}
NEED_NAMES = {"torch": "PyTorch", "sklearn": "scikit-learn"}


def say(ok, what, why=""):
    print("  %-4s %s%s" % ({True: "ok", False: "FAIL", None: "--"}[ok], what, ("  (" + why + ")") if why else ""))
    return ok


def check_needs():
    """(required ok, {optional: present}) - Python and PyYAML are required; torch and dotnet are optional and named"""
    print("What the tools need:")
    ok = say(sys.version_info >= (3, 10), "Python %d.%d" % sys.version_info[:2], "3.10 or newer is needed")
    try:
        importlib.import_module("yaml")
        ok &= say(True, "PyYAML")
    except ImportError:
        ok &= say(False, "PyYAML", "pip install pyyaml")
    opt = {}
    try:
        importlib.import_module("torch")
        opt["torch"] = say(True, "PyTorch (the GNN link proposer)")
    except ImportError:
        opt["torch"] = say(None, "PyTorch not found", "the GNN proposer is skipped; pip install torch to use it")
    try:
        importlib.import_module("sklearn")
        opt["sklearn"] = say(True, "scikit-learn (ranking which holes to show first)")
    except ImportError:
        opt["sklearn"] = say(None, "scikit-learn not found", "survivor_rank.py is skipped; pip install scikit-learn to use it")
    opt["dotnet"] = shutil.which("dotnet") is not None
    say(True if opt["dotnet"] else None, ".NET SDK (the C# adapter and the example)" if opt["dotnet"] else ".NET SDK not found",
        "" if opt["dotnet"] else "C# models and the example cannot run; Python models still can")
    # the C# equivalence proof by IL is built with dotnet itself (il_bodies/); PowerShell is not needed (KT-9, 10-29)
    return ok, opt


def install(dest, skills):
    os.makedirs(dest, exist_ok=True)
    for f in TOOLS:
        os.makedirs(os.path.dirname(os.path.join(dest, f)), exist_ok=True)
        shutil.copy2(os.path.join(KIT, "tools", f), os.path.join(dest, f))
    for ex in ("example", "example_py"):
        if os.path.exists(os.path.join(dest, ex)):
            shutil.rmtree(os.path.join(dest, ex))
        shutil.copytree(os.path.join(KIT, ex), os.path.join(dest, ex),
                        ignore=shutil.ignore_patterns("bin", "obj", "kg", "stats.jsonl", "CLOSED_LOOP.md", "__pycache__"))
    for doc in ("README.md", "RELEASE_NOTES.md", "LICENSE"):
        if os.path.exists(os.path.join(KIT, doc)):
            shutil.copy2(os.path.join(KIT, doc), os.path.join(dest, doc))
    skill_dir = os.path.join(skills, "closed-loop")
    os.makedirs(skill_dir, exist_ok=True)
    text = open(os.path.join(KIT, "skill", "SKILL.md"), encoding="utf-8").read().replace("{TOOLS}", dest.replace("\\", "/"))
    open(os.path.join(skill_dir, "SKILL.md"), "w", encoding="utf-8", newline="\n").write(text)
    open(os.path.join(dest, ".installed"), "w", encoding="utf-8").write("skill=%s\n" % skill_dir)
    print("Installed: the skill at %s, the tools at %s" % (skill_dir, dest))


def selftests(dest, opt):
    print("Every selftest:")
    ok = True
    names = list(SELFTESTS)
    for need, files in NEEDS.items():
        if opt.get(need):
            names += files
        else:
            for f in files:
                say(None, "%s skipped: it needs %s" % (f, NEED_NAMES[need]))
    for f in names:
        r = subprocess.run([sys.executable, "-X", "utf8", os.path.join(dest, f), "--selftest"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", cwd=dest)
        last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
        ok &= say(r.returncode == 0, f, "" if r.returncode == 0 else last[-120:])
    return ok


# the examples the acceptance test runs: (name, model, the optional piece it needs or None). The Python one needs nothing
# optional, so every machine that installs at all runs the whole loop once (KT-8, 10-29: with no .NET SDK nothing did)
EXAMPLES = [("Python", "example_py", None), ("C#", "example", "dotnet")]


def acceptance(dest, opt):
    """the whole loop on each example this machine can run: every statement broken, each break run against its tests; each must
    HOLD (the owner's one question open). True, False, or None when no example could run"""
    print("The acceptance test - the whole loop on the examples:")
    env = dict(os.environ, CLOSED_LOOP_ROOT=dest)
    results = []
    for name, folder, needs in EXAMPLES:
        if needs and not opt.get(needs):
            say(None, "the %s example skipped: no %s" % (name, {"dotnet": ".NET SDK"}.get(needs, needs)))
            continue
        r = subprocess.run([sys.executable, "-X", "utf8", os.path.join(dest, "closed_loop.py"), "run", folder + "/model", "--out",
                            folder + "/CLOSED_LOOP.md", "--note", "the install's acceptance run"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", cwd=dest, env=env)
        held = "closed_loop: HOLDS" in r.stdout
        made = next((l for l in r.stdout.splitlines() if " made, " in l), "")
        say(held, "the %s example HOLDS" % name if held else "the %s example did not hold" % name, made.strip()[-140:])
        if not held:
            # the HOLES, not the tail: the tail can be a tool's first-run banner (KT-5, 10-29: .NET's welcome text hid the cause)
            holes = [l for l in r.stdout.splitlines() if "OURS:" in l or l.startswith("      ") or "Traceback" in l or "Error" in l]
            print("\n".join(holes[:40]) or (r.stdout[-1500:] + r.stderr[-800:]))
        results.append(held)
    return None if not results else all(results)


def uninstall(dest, skills):
    mark = os.path.join(dest, ".installed")
    if not os.path.exists(mark):
        print("Nothing to remove: %s holds no install of this kit" % dest)
        return 0
    skill_dir = open(mark, encoding="utf-8").read().split("=", 1)[1].strip()
    if os.path.isdir(skill_dir):
        shutil.rmtree(skill_dir)
    shutil.rmtree(dest)
    print("Removed %s and %s" % (dest, skill_dir))
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=os.path.join(HOME, ".claude", "closed-loop"))
    ap.add_argument("--skills", default=os.path.join(HOME, ".claude", "skills"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--uninstall", action="store_true")
    a = ap.parse_args(argv)
    dest, skills = os.path.abspath(a.dest), os.path.abspath(a.skills)
    if a.uninstall:
        return uninstall(dest, skills)
    ok, opt = check_needs()
    if not ok:
        print("Not installed: a required piece is missing (above).")
        return 1
    if a.check:
        return 0
    install(dest, skills)
    ok = selftests(dest, opt)
    held = acceptance(dest, opt)
    ok &= held is not False
    # a skipped acceptance test is not a pass (KT-5, 10-29: with no .NET SDK this printed READY although the whole loop had never
    # run on this machine): the selftests prove each piece, only the acceptance test proves them together
    if not ok:
        print("\nINSTALLED WITH FAILURES (above): the skill is in place, but do not trust its verdicts until each is fixed.")
    elif held is None:
        print("\nINSTALLED, NOT PROVEN HERE: every selftest passed, but the whole loop never ran on this machine (the acceptance test "
              "needs what is named above). Install it, then run `python install.py` again.")
    else:
        print("\nREADY. In Claude Code, ask for the closed loop on a model (the skill is `closed-loop`); start from README.md.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
