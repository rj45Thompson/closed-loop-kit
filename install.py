#!/usr/bin/env python3
"""install.py - put the closed loop into Claude Code: the skill where Claude finds it, the tools beside it, what they need checked,
then every selftest and the whole loop on the examples proved, several at once, by the kit's own `prove` (tools/prove.py).

    cd closed-loop-kit                     (the zip unpacks into this folder)
    python install.py                      install for this user (~/.claude/skills/closed-loop and ~/.claude/closed-loop)
    python install.py --dest DIR           the tools (and the examples, with the report each run writes) somewhere else
    python install.py --skills DIR         the skill somewhere else (Claude Code reads ~/.claude/skills)
    python install.py --quick              the front door only (grow and ask): about a minute, instead of every selftest and both examples
    python closed_loop.py prove            (in the tools folder, after an install) prove it all again, or `prove --quick`; see its result
    python install.py --check              check only: nothing is copied
    python install.py --uninstall          remove what an install put down (it touches nothing else)
    python install.py --agent NAME         write the skill in the layout another coding agent reads, into THIS project, and touch nothing else.
                                           NAME is codex, opencode or agents (AGENTS.md), cursor (.cursor/rules/closed-loop.mdc), gemini (GEMINI.md),
                                           copilot (.github/copilot-instructions.md) or windsurf (.windsurf/rules/closed-loop.md). No other agent has
                                           been tried: these files are the layouts their documentation names, generated from skill/SKILL.md at build time.
    closed-loop-kit install ...            the same installer, from the pip package (pipx run / uvx), which has no clone

Needs: Python 3.10+ and PyYAML. Optional, each named if absent: PyTorch (the GNN link proposer), scikit-learn (ranking which holes to
show first), the .NET SDK (the C# adapter and the example and the C# equivalence check by IL, built with dotnet itself). Without an
optional piece the rest still runs; what it would have done is said, never faked.
"""
import argparse
import importlib
import importlib.util
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

KIT = os.path.dirname(os.path.abspath(__file__))
# in a clone the tools are KIT/tools; in the pip package this file sits in the tools folder itself
SRC = os.path.join(KIT, "tools") if os.path.isfile(os.path.join(KIT, "tools", "closed_loop.py")) else KIT
HOME = os.path.expanduser("~")
TOOLS = ["usecase_model.py", "doc_nodes.py", "usecase_kg.py", "closed_loop.py", "prove.py", "grow.py", "ask.py", "elicit.py", "source_rules.py", "cs_effect_check.py", "py_effect_check.py", "graph_gnn.py",
         "ab_loop.py", "grade_panel.py", "loop_stats.py", "improvement.py", "improvement/stat_requirements.yaml", "buzzsaw.py",
         "loop_tick.py", "TICK.md", "run_plan.py", "telemetry_audit.py", "survivor_rank.py", "gnn_tasks.py", "gnn_more.py", "gnn_curve.py", "graph_flythrough.py", "gnn_goals.py", "gnn_goals.yaml",
         "il_bodies/il_bodies.csproj", "il_bodies/Program.cs", "STATS.md"]
LF, CRLF = chr(10), chr(13) + chr(10)


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
    # asked for by name and NOT imported: importing PyTorch to see whether it is there took 50 s on a machine that had it (MEASURED 10-31, a clean home on a loaded machine)
    if importlib.util.find_spec("torch") is not None:
        opt["torch"] = say(True, "PyTorch (the GNN link proposer)")
    else:
        opt["torch"] = say(None, "PyTorch not found", "the GNN proposer is skipped; pip install torch to use it")
    if importlib.util.find_spec("sklearn") is not None:
        opt["sklearn"] = say(True, "scikit-learn (ranking which holes to show first)")
    else:
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
        shutil.copy2(os.path.join(SRC, f), os.path.join(dest, f))
    for ex in ("example", "example_py", "example_grow"):
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


def prove(dest, quick):
    """the proof, by the kit's own `prove` (prove.py): every selftest and the examples of the installed tools, several at once at below-normal priority, one result.
    A selftest that needs a piece this machine lacks is NOT RUN and says which; a tool that hangs, or exits 0 having checked nothing, is named and is not a pass.
    Returns prove's exit: 0 proved, 3 nothing failed but no example that runs the whole loop could run here, 1 otherwise. MEASURED 10-31 on a clean home folder:
    the same proofs run one after another took 15 to 17 minutes; the sequential list is gone"""
    print("The proof, %s:" % ("the front door only" if quick else "every selftest and the whole loop on the examples, together"))
    cmd = [sys.executable, "-X", "utf8", os.path.join(dest, "prove.py"), "--folder", dest, "--brief"] + (["--quick"] if quick else [])
    return subprocess.run(cmd, cwd=dest).returncode


def install_agent(name, project):
    """Write the skill in the layout agent `name` reads, into `project`, and touch nothing else (no tools, no skill folder, no proof run). The layouts
    are cut from skill/SKILL.md at build time (agents/, with layouts.json saying where each goes); here only the stamp is checked, so a SKILL.md edited
    after the build is refused and not served as a stale copy. A file the project may own (AGENTS.md, GEMINI.md, copilot-instructions.md) gets a block between
    two marker comments, replaced when present and appended when not, the rest of the file untouched. A file that is ours alone (Cursor, Windsurf) is
    written when absent or already ours, and refused when it is somebody's. Returns 0 written or already current, 1 refused, 2 not possible here."""
    agents = os.path.join(KIT, "agents")
    try:
        doc = json.load(open(os.path.join(agents, "layouts.json"), encoding="utf-8"))
        skill = open(os.path.join(KIT, "skill", "SKILL.md"), encoding="utf-8", newline="").read()
    except (OSError, ValueError) as e:
        print("No agent layouts here (%s). They are written by the kit's build (agent_layouts.py)." % e)
        return 2
    by_name = {a: k for k, v in doc["layouts"].items() for a in v["agents"]}
    if name not in by_name:
        print("Unknown agent %r. Choose one of: %s" % (name, ", ".join(sorted(by_name))))
        return 2
    lay = doc["layouts"][by_name[name]]
    have = hashlib.sha256(skill.replace(CRLF, LF).encode("utf-8")).hexdigest()
    src = open(os.path.join(agents, lay["src"]), encoding="utf-8", newline="").read()
    if doc.get("sha256") != have or ("sha256=" + have) not in src:
        print("Refused: the %s layout was cut from an older skill/SKILL.md than the one beside it. Rebuild the kit (build_kit.py) and install again." % lay["src"])
        return 2
    dest = os.path.join(project, *lay["dest"].split("/"))
    begin = re.compile(r"<!-- closed-loop:begin [^>]*-->")
    end = "<!-- closed-loop:end -->"
    old = open(dest, encoding="utf-8", newline="").read() if os.path.isfile(dest) else None
    nl = CRLF if old and CRLF in old else LF
    mine = src.replace(CRLF, LF)
    if lay["kind"] == "file":
        if old is not None and not begin.search(old):
            print("Refused: %s exists and is not ours (it carries no closed-loop block). Nothing was changed; move it or merge by hand." % lay["dest"])
            return 1
        new = mine.replace(LF, nl)
    elif old is None:
        new = mine.replace(LF, nl)
    else:
        m = begin.search(old)
        e = old.find(end) if m else -1
        if m and e < m.start():
            print("Refused: %s holds a closed-loop begin marker with no end marker after it. Nothing was changed." % lay["dest"])
            return 1
        block_text = mine[mine.index("<!-- closed-loop:begin"):].rstrip(LF).replace(LF, nl)
        if m:
            new = old[:m.start()] + block_text + old[e + len(end):]
        else:
            new = old.rstrip(CRLF) + nl + nl + block_text + nl
    if new == old:
        print("Already current: %s" % dest)
        return 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    open(dest, "w", encoding="utf-8", newline="").write(new)
    print("%s %s for %s. Nothing else was touched." % ("Updated" if old is not None else "Wrote", dest, name))
    return 0


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
    ap.add_argument("--agent", help="write the skill in the layout of another coding agent into this project and do nothing else: codex, opencode, agents, cursor, gemini, copilot or windsurf")
    ap.add_argument("--project", default=".", help="--agent: the project folder (default: the current one)")
    ap.add_argument("--quick", action="store_true", help="only the front door (grow and ask): its selftests and its acceptance run, about a minute; the rest of the loop is not proved")
    a = ap.parse_args(argv)
    dest, skills = os.path.abspath(a.dest), os.path.abspath(a.skills)
    if a.agent:
        return install_agent(a.agent, os.path.abspath(a.project))
    if a.uninstall:
        return uninstall(dest, skills)
    ok, opt = check_needs()
    if not ok:
        print("Not installed: a required piece is missing (above).")
        return 1
    if a.check:
        return 0
    install(dest, skills)
    rc = prove(dest, a.quick)
    if a.quick:
        print("\nINSTALLED, FRONT DOOR ONLY: %s. The other selftests and both examples were not run; run `python install.py` when you want the whole loop proved."
              % ("grow and ask are proved here" if rc == 0 else "THE FRONT DOOR FAILED (above)"))
        return 0 if rc == 0 else 1
    # a skipped acceptance test is not a pass (the whole loop never ran on this machine and an earlier version of this printed READY anyway):
    # the selftests prove each piece, only an example proves them together
    if rc == 3:
        print("\nINSTALLED, NOT PROVEN HERE: every selftest that could run passed, but the whole loop never ran on this machine (the examples "
              "need what is named above). Install it, then run `python install.py` again.")
        return 0
    if rc != 0:
        print("\nINSTALLED WITH FAILURES (above): the skill is in place, but do not trust its verdicts until each is fixed.")
        return 1
    print("\nREADY. In Claude Code, ask for the closed loop on a model (the skill is `closed-loop`); start from README.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
