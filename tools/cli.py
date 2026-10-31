#!/usr/bin/env python3
"""closed-loop-kit - the console command of the pip package, so that the kit runs with no clone:

    uvx --from git+https://github.com/rj45Thompson/closed-loop-kit closed-loop-kit try
    pipx run --spec git+https://github.com/rj45Thompson/closed-loop-kit closed-loop-kit try

    closed-loop-kit try [--keep]      run the shipped Python example through the whole loop on a copy; it must print HOLDS (exit 0, else 1)
    closed-loop-kit grow|ask|run|prove ...   the tools' own commands (closed_loop.py): the front door, the loop, and the proof of the install
    closed-loop-kit install ...       install.py: the skill into ~/.claude, the tools beside it, or `--agent NAME` for another coding agent's file layout
    closed-loop-kit tool NAME ...     any other shipped tool by name (elicit, loop_stats, doc_nodes, ...)
    closed-loop-kit where             the folder the tools are in

Every command runs the tool as `python -X utf8 <tool>`, the form the README documents, so a Windows console cannot turn a report into an encoding error.
Nothing is written outside the folder you name; `try` works in a temporary folder.
"""
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def run_tool(script, args, cwd=None, env=None, stream=None):
    """the tool as a child process, output passed through; returns its exit code. `stream`, a list, also collects the output lines"""
    cmd = [sys.executable, "-X", "utf8", os.path.join(HERE, script)] + list(args)
    e = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    e.update(env or {})
    if stream is None:
        return subprocess.call(cmd, cwd=cwd, env=e)
    p = subprocess.Popen(cmd, cwd=cwd, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    for line in p.stdout:
        stream.append(line)
        sys.stdout.write(line)
        sys.stdout.flush()
    return p.wait()


def try_example(args):
    """the shipped Python example, on a copy, through the whole loop: every statement of its code broken, each break run against its tests. HOLDS or exit 1"""
    src = os.path.join(HERE, "example_py")
    if not os.path.isdir(os.path.join(src, "model")):
        print("closed-loop-kit try: this install holds no example_py/model, so nothing was run")
        return 1
    work = tempfile.mkdtemp(prefix="closed-loop-try-")
    shutil.copytree(src, os.path.join(work, "example_py"),
                    ignore=shutil.ignore_patterns("bin", "obj", "kg", "stats.jsonl", "CLOSED_LOOP.md", "__pycache__", "*.holes.json", "*.mutants.json"))
    out = []
    rc = run_tool("closed_loop.py", ["run", "example_py/model", "--out", "example_py/CLOSED_LOOP.md", "--note", "try", "--workers", "2"],
                  cwd=work, env={"CLOSED_LOOP_ROOT": work}, stream=out)
    held = rc == 0 and any(line.startswith("closed_loop: HOLDS") for line in out)
    report = os.path.join(work, "example_py", "CLOSED_LOOP.md")
    if "--keep" in args:
        print("closed-loop-kit try: the report is %s" % report)
    else:
        shutil.rmtree(work, ignore_errors=True)
    print("closed-loop-kit try: %s" % ("HOLDS - the Python example's requirements are held by its tests, every break caught. Next: closed-loop-kit grow my_epic.md --out my_model" if held
                                       else "DID NOT HOLD (exit %d). The output above says why." % rc))
    return 0 if held else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "try":
        return try_example(rest)
    if cmd == "where":
        print(HERE)
        return 0
    if cmd == "install":
        return run_tool("install.py", rest)
    if cmd == "tool":
        if not rest:
            print("closed-loop-kit tool NAME ...  (a tool in %s)" % HERE)
            return 2
        name = rest[0] if rest[0].endswith(".py") else rest[0] + ".py"
        if os.path.basename(name) != name or not os.path.isfile(os.path.join(HERE, name)):
            print("closed-loop-kit: no tool named %s in %s" % (rest[0], HERE))
            return 2
        return run_tool(name, rest[1:])
    return run_tool("closed_loop.py", argv)


if __name__ == "__main__":
    sys.exit(main())
