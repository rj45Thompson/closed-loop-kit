#!/usr/bin/env python3
"""loop_tick.py - the closed loop's standing tick, the parts that must not be left to a fresh session's reading of prose.

WHY (2026-10-29): the loop should keep running and keep slowly improving. A scheduled task starts a new
session every few hours with no memory of the last one; TICK.md is its runbook, the board (CLOSED_LOOP_TASKS.md) its work,
and this its guard, its launcher and its ledger:

  start    exit 4 STOPPED while the owner's switch file (.loop_tick.STOP) exists; exit 3 BUSY while another tick holds the lock (beaten
           within STALE_AFTER); exit 5 IDLE when the last IDLE_AFTER ticks were all worth "no" and nothing they watch changed
           since (the task fires every few minutes so the next tick starts soon after the last ends, and an idle loop
           must cost almost nothing); otherwise takes the lock and says the MODE: LIGHT when a loop run, a GNN tune or an A/B arm is
           already running, or the machine is busy (no tool may be edited while a run uses it; the machine is shared), FULL
           otherwise. It also lists every launched run that has ENDED and not been harvested, with its exit code (or DIED when the
           run left no exit line), so the next tick reads what the last one started, and every permission prompt refused since the
           last tick (unattended_prompt_guard.py), so a refusal is read rather than only banked
  beat     the tick is alive (between its steps): a lock that stops beating is freed after STALE_AFTER
  launch   starts ONE long run detached from this session (a tick's session ends; the run must not end with it), its output and
           its exit code written to tick_runs/<stamp>.log; refused while another run is going
  harvest  marks a launched run's log as read (its result recorded on the board and in the ledger)
  finish   one row in the ledger (loop_ticks.jsonl: the item, the minutes, what it found, whether it was worth it), the lock released
  show     the ledger's last rows

    python -X utf8 loop_tick.py start
    python -X utf8 loop_tick.py beat
    python -X utf8 loop_tick.py launch -- python -X utf8 -u closed_loop.py run <model> ...
    python -X utf8 loop_tick.py harvest <log>
    python -X utf8 loop_tick.py finish --item KT-6 --minutes 35 --found "..." --worth yes|no|unknown
    python -X utf8 loop_tick.py show [--last 10]
    python -X utf8 loop_tick.py --selftest
"""
import argparse
import io
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = os.path.join(HERE, ".loop_tick.lock")
STOP = os.path.join(HERE, ".loop_tick.STOP")      # the owner's switch: while it exists no tick does anything. Only the owner removes it.
# a HOLD freezes the tools without stopping the ticks: while A/B arms use the tools as their instrument (PK-5, 10-29), a tool edited
# between two of an arm's runs would change what is measured mid-pair. Its text says why and who set it; whoever set it removes it
HOLD = os.path.join(HERE, ".loop_tick.HOLD")
PROMPTS = os.path.join(HERE, "..", ".unattended_prompts.jsonl")      # unattended_prompt_guard.py's record: each prompt nobody could answer
LEDGER = os.path.join(HERE, "loop_ticks.jsonl")
RUNS_LOG = os.path.join(HERE, "tick_runs.jsonl")
RUNS_DIR = os.path.join(HERE, "tick_runs")
STALE_AFTER = 2 * 3600          # a tick has an hour or so and beats between its steps; a lock not beaten for this long is a tick that died
BUSY_CPU = 70.0
IDLE_AFTER = 3                  # this many ledger rows in a row worth "no", and nothing changed since: the next tick does nothing
# an EPOCH, so it brainstorms how to improve at planned intervals: every EPOCH ticks, the first is a
# BRAINSTORM (a fresh agent reads the stats, the ledger and the results and ranks ideas by the stat each would move) and the last a
# BENCHMARK (the fixed measures in TICK.md, never trained on: the epoch improved only if they moved)
EPOCH = 12
# The loop should run again as soon as it has finished, so the task fires every few minutes and a tick that finds work
# still going, or nothing new to work on, ends at once. What counts as new: the board, the tools, the two models, the kit.
WATCH = ("CLOSED_LOOP_TASKS.md", "*.py", "selfmodel/model/*.yaml", "../shops/model/*.yaml", "../closed_loop_kit/*", "../closed_loop_kit/*/*")
# a command line holding one of these is a run no tool may be edited under (its process imports, copies or builds the tools)
RUNS = ("closed_loop.py run", "graph_gnn.py tune", "ab_loop.py outcome", "py_effect_check.py", "cs_effect_check.py", "loop_tick.py _run")


def running(procs=None):
    """the command lines of the runs alive now"""
    if procs is None:
        import psutil
        procs = []
        for p in psutil.process_iter(["cmdline"]):
            try:
                procs.append(" ".join(p.info.get("cmdline") or []))
            except Exception:  # noqa: BLE001 - a process that ended while it was read is simply not there
                continue
    return [c for c in procs if any(r in c.replace("\\", "/") for r in RUNS)]


def _rows(path):
    return [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()] if os.path.exists(path) else []


def ended(runs_log=RUNS_LOG, alive=None):
    """[(log, cmd, outcome)] for every launched run that is over and not harvested: outcome is "EXIT <rc>", or "DIED" when the
    log has no exit line (the run was killed, or the machine went down) - never guessed as a pass"""
    rows = _rows(runs_log)
    done = {r["harvested"] for r in rows if "harvested" in r}
    out = []
    for r in rows:
        if "log" not in r or r["log"] in done:
            continue
        if alive(r) if alive else _alive(r):
            continue
        tail = io.open(r["log"], encoding="utf-8", errors="replace").read().splitlines()[-1:] if os.path.exists(r["log"]) else []
        out.append((r["log"], r["cmd"], tail[0] if tail and tail[0].startswith("EXIT ") else "DIED"))
    return out


def _alive(row):
    import psutil
    try:
        return "_run" in " ".join(psutil.Process(row["pid"]).cmdline()) and row["log"] in " ".join(psutil.Process(row["pid"]).cmdline())
    except Exception:  # noqa: BLE001 - no such process (or not ours): it is over
        return False


def changed_since(t, here=HERE, watch=WATCH):
    """the watched files written after time t (newest first), the tick's own lock, ledger and run logs never among them"""
    import glob
    out = []
    for pat in watch:
        for p in glob.glob(os.path.join(here, pat)):
            if os.path.isfile(p) and os.path.getmtime(p) > t:
                out.append((os.path.getmtime(p), os.path.normpath(p)))
    return [p for _, p in sorted(out, reverse=True)]


def idle(ledger=LEDGER, here=HERE, watch=WATCH):
    """the reason this tick has nothing to do, or None: the last IDLE_AFTER rows all said worth "no" and nothing watched changed
    after the last of them finished"""
    rows = _rows(ledger)[-IDLE_AFTER:]
    if len(rows) < IDLE_AFTER or any(r.get("worth") != "no" for r in rows) or "t" not in rows[-1]:
        return None
    if changed_since(rows[-1]["t"], here, watch):
        return None
    return "the last %d ticks found nothing worth their cost and nothing has changed since %s" % (IDLE_AFTER, rows[-1]["time"])


def refused(prompts, ledger):
    """the permission prompts refused since the last tick ended: work that went another way or was not done"""
    seen = _rows(ledger)
    since = seen[-1].get("time", "") if seen else ""
    return [r for r in _rows(prompts) if r.get("at", "") > since]


def start(lock=LOCK, now=None, procs=None, cpu=None, stop=STOP, runs_log=RUNS_LOG, alive=None, ledger=LEDGER, here=HERE, watch=WATCH,
          hold=HOLD):
    """(exit code, words): 4 STOPPED, 3 BUSY, 5 IDLE, else 0 with the MODE and the ended runs to harvest, the lock taken"""
    now = time.time() if now is None else now
    if os.path.exists(stop):
        return 4, "STOPPED: the owner's switch %s exists; this tick does nothing (only the owner removes it)" % stop
    if os.path.exists(lock):
        held = json.load(io.open(lock, encoding="utf-8"))
        age = now - max(held.get("started", 0), held.get("beat", 0))
        if age < STALE_AFTER:
            return 3, "BUSY: a tick holds the lock (taken %s, last beat %d min ago); this one stops" % (held.get("at"), age // 60)
    runs = running(procs)
    harvests = ended(runs_log, alive)
    why_idle = None if runs or harvests else idle(ledger, here, watch)
    if why_idle:
        return 5, "IDLE: %s; this tick does nothing (a change to the board, a tool, a model or the kit wakes the loop)" % why_idle
    if cpu is None:
        import psutil
        cpu = psutil.cpu_percent(interval=3)
    held = io.open(hold, encoding="utf-8").read().strip() if os.path.exists(hold) else None
    mode = "LIGHT" if runs or held or cpu > BUSY_CPU else "FULL"
    why = ("the tools are on HOLD: " + held[:160]) if held else ("a run is going: " + runs[0][:160]) if runs else         ("the CPU is at %.0f%%" % cpu) if cpu > BUSY_CPU else "nothing else is running"
    io.open(lock, "w", encoding="utf-8").write(json.dumps({"started": now, "at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
                                                        "mode": mode}))
    said = "MODE %s (%s)%s" % (mode, why, " - edit no tool this tick; docs, board, analysis and harvests only" if mode == "LIGHT" else "")
    duty = epoch_duty(len([r for r in _rows(ledger) if "item" in r]))
    if duty:
        said += "\n  " + duty
    for log, cmd, outcome in harvests:
        said += "\n  TO HARVEST: %s  %s  (%s)" % (outcome, log, cmd[:140])
    new = refused(PROMPTS, ledger)
    if new:
        said += "\n  REFUSED PROMPTS since the last tick: %d (work that went another way or was not done; each may need a rule). Last: %s %s" % (
            len(new), new[-1].get("tool"), str(new[-1].get("input"))[:140])
    return 0, said


def runbook(ours, kit):
    """where a tick reads what to do: our runbook here, TICK.md in the kit, where our runbook does not ship"""
    return "LOOP_TICK.md " + ours if os.path.exists(os.path.join(HERE, "LOOP_TICK.md")) else "TICK.md " + kit


def epoch_duty(done):
    """what this tick owes the epoch, from the ticks already in the ledger: the BRAINSTORM opens an epoch, the BENCHMARK closes it"""
    e, at = done // EPOCH + 1, done % EPOCH
    if at == 0:
        return "EPOCH %d STARTS: this tick is the BRAINSTORM (%s)" % (e, runbook("section 4a", "section 3"))
    if at == EPOCH - 1:
        return "EPOCH %d ENDS: this tick is the BENCHMARK (%s)" % (e, runbook("section 4b", "section 5"))
    return None


def beat(lock=LOCK, now=None):
    """the tick is alive: called between its steps, so a lock that stops beating is a tick that died, freed after STALE_AFTER"""
    if not os.path.exists(lock):
        return False
    held = json.load(io.open(lock, encoding="utf-8"))
    held["beat"] = time.time() if now is None else now
    io.open(lock, "w", encoding="utf-8").write(json.dumps(held))
    return True


def launch(cmd, runs_log=RUNS_LOG, runs_dir=RUNS_DIR, procs=None):
    """(exit code, words): starts cmd detached through `_run`, which writes its output and then "EXIT <rc>" to the log"""
    busy = running(procs)
    if busy:
        return 3, "BUSY: a run is going (%s); one long run at a time on a shared machine" % busy[0][:160]
    os.makedirs(runs_dir, exist_ok=True)
    log = os.path.join(runs_dir, time.strftime("%Y%m%d_%H%M%S") + ".log")
    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    p = subprocess.Popen([sys.executable, "-X", "utf8", os.path.abspath(__file__), "_run", log, "--"] + list(cmd),
                         creationflags=flags, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         close_fds=True, cwd=os.getcwd())
    io.open(runs_log, "a", encoding="utf-8").write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": p.pid, "log": log,
                                                              "cmd": " ".join(cmd)}) + "\n")
    return 0, "LAUNCHED pid %d, log %s" % (p.pid, log)


def _run(log, cmd):
    with io.open(log, "w", encoding="utf-8", newline="\n") as f:
        f.write("RUN %s\n" % " ".join(cmd))
        f.flush()
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
        f.write("\nEXIT %d\n" % rc)
    return rc


def harvest(log, runs_log=RUNS_LOG):
    io.open(runs_log, "a", encoding="utf-8").write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "harvested": log}) + "\n")


def finish(item, minutes, found, worth, lock=LOCK, ledger=LEDGER, targets=None):
    """`targets`: the stat this tick meant to move (improvement.py measures the outcome from the morph logs: level 2's VERIFIED)"""
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "t": time.time(), "item": item, "minutes": minutes, "found": found, "worth": worth}
    if targets:
        row["targets"] = targets
    if os.path.exists(lock):
        row["mode"] = json.load(io.open(lock, encoding="utf-8")).get("mode")
        os.remove(lock)
    io.open(ledger, "a", encoding="utf-8").write(json.dumps(row, sort_keys=True) + "\n")
    return row


def selftest():
    import shutil
    global PROMPTS
    import tempfile
    t = tempfile.mkdtemp(prefix="loop_tick_selftest_")
    PROMPTS = os.path.join(t, "no_prompts")           # sealed from the live record
    lock, ledger, stop, rl = (os.path.join(t, n) for n in ("lock", "ledger.jsonl", "STOP", "runs.jsonl"))
    ok = True
    none = lambda r: False  # noqa: E731 - every planted run is over
    w = ("board.md",)
    rc, said = start(lock, now=1000.0, procs=["python closed_loop.py run shops/model"], cpu=5.0, stop=stop, runs_log=rl, alive=none, ledger=ledger, here=t, watch=w, hold=os.path.join(t, "no_hold"))
    good = rc == 0 and said.startswith("MODE LIGHT") and os.path.exists(lock)
    rc2, said2 = start(lock, now=1000.0 + 600, procs=[], cpu=5.0, stop=stop, runs_log=rl, alive=none, ledger=ledger, here=t, watch=w, hold=os.path.join(t, "no_hold"))
    good = good and rc2 == 3 and said2.startswith("BUSY")
    rc3, said3 = start(lock, now=1000.0 + STALE_AFTER + 1, procs=[], cpu=5.0, stop=stop, runs_log=rl, alive=none, ledger=ledger, here=t, watch=w, hold=os.path.join(t, "no_hold"))
    good = good and rc3 == 0 and said3.startswith("MODE FULL")
    os.remove(lock)
    rc4, said4 = start(lock, now=5000.0, procs=["python D:\\x\\graph_gnn.py tune kg.sqlite"], cpu=95.0, stop=stop, runs_log=rl, alive=none, ledger=ledger, here=t, watch=w, hold=os.path.join(t, "no_hold"))
    good = good and rc4 == 0 and "LIGHT" in said4
    row = finish("KT-6", 30, "a fresh agent installed it", "yes", lock, ledger)
    good = good and not os.path.exists(lock) and row["mode"] == "LIGHT" and len(io.open(ledger, encoding="utf-8").readlines()) == 1
    ok &= good
    print("selftest: TICK a live lock stops a second tick, a stale one does not; a run or a busy CPU makes a LIGHT tick; finish "
          "writes one ledger row and frees the lock %s" % ("PASS" if good else "FAIL"))
    io.open(stop, "w").write("owner")
    rc5, said5 = start(lock, now=9000.0, procs=[], cpu=5.0, stop=stop, runs_log=rl, alive=none, ledger=ledger, here=t, watch=w, hold=os.path.join(t, "no_hold"))
    good = rc5 == 4 and said5.startswith("STOPPED") and not os.path.exists(lock)
    os.remove(stop)
    ok &= good
    print("selftest: STOP the owner's switch file stops a tick before it takes the lock %s" % ("PASS" if good else "FAIL"))
    hold = os.path.join(t, "HOLD")
    io.open(hold, "w", encoding="utf-8").write("PK-5 arms running (main session)")
    rc15, said15 = start(lock, now=30000.0, procs=[], cpu=5.0, stop=stop, runs_log=rl, alive=none, ledger=ledger, here=t, watch=w, hold=hold)
    os.remove(lock)
    os.remove(hold)
    rc16, said16 = start(lock, now=30001.0, procs=[], cpu=5.0, stop=stop, runs_log=rl, alive=none, ledger=ledger, here=t, watch=w, hold=hold)
    os.remove(lock)
    good = rc15 == 0 and said15.startswith("MODE LIGHT") and "HOLD: PK-5 arms" in said15 and said16.startswith("MODE FULL")
    ok &= good
    print("selftest: HOLD a hold on the tools makes every tick LIGHT and says why; without it the tick is FULL %s" % ("PASS" if good else "FAIL"))
    plant, seen = os.path.join(t, "prompts.jsonl"), os.path.join(t, "seen.jsonl")
    io.open(plant, "w", encoding="utf-8").write(json.dumps({"at": "9999", "tool": "Bash", "input": "PLANTED"}) + "\n" + json.dumps({"at": "0000", "tool": "Bash", "input": "OLD"}) + "\n")
    io.open(seen, "w", encoding="utf-8").write(json.dumps({"time": "5000"}) + "\n")
    new = refused(plant, seen)
    sealed, PROMPTS = PROMPTS, plant
    rc17, said17 = start(lock, now=31000.0, procs=[], cpu=5.0, stop=stop, runs_log=rl, alive=none, ledger=seen, here=t, watch=w, hold=os.path.join(t, "no_hold"))
    PROMPTS = sealed
    os.remove(lock)
    good = len(new) == 1 and new[0]["input"] == "PLANTED" and rc17 == 0 and "REFUSED PROMPTS since the last tick: 1" in said17 and "PLANTED" in said17
    ok &= good
    print("selftest: REFUSED a prompt refused since the last tick is reported at the next start, an older one is not %s" % ("PASS" if good else "FAIL"))
    duties = [epoch_duty(n) for n in range(2 * EPOCH)]
    good = duties[0].startswith("EPOCH 1 STARTS") and duties[EPOCH - 1].startswith("EPOCH 1 ENDS") and duties[EPOCH].startswith("EPOCH 2 STARTS") \
        and sum(1 for d in duties if d) == 4
    ok &= good
    print("selftest: EPOCH every %d ticks the first is the brainstorm and the last the benchmark, and no other tick owes either %s" % (
        EPOCH, "PASS" if good else "FAIL"))
    # a beating tick keeps its lock past STALE_AFTER from its start; one that stopped beating loses it
    rc8, _ = start(lock, now=20000.0, procs=[], cpu=5.0, stop=stop, runs_log=rl, alive=none, ledger=ledger, here=t, watch=w, hold=os.path.join(t, "no_hold"))
    beat(lock, now=20000.0 + STALE_AFTER - 60)
    rc9, said9 = start(lock, now=20000.0 + STALE_AFTER + 60, procs=[], cpu=5.0, stop=stop, runs_log=rl, alive=none, ledger=ledger, here=t, watch=w, hold=os.path.join(t, "no_hold"))
    rc10, _ = start(lock, now=20000.0 + 2 * STALE_AFTER, procs=[], cpu=5.0, stop=stop, runs_log=rl, alive=none, ledger=ledger, here=t, watch=w, hold=os.path.join(t, "no_hold"))
    good = rc8 == 0 and rc9 == 3 and "BUSY" in said9 and rc10 == 0
    os.remove(lock)
    ok &= good
    print("selftest: BEAT a tick that beats keeps its lock past its start's age; one that stopped beating loses it %s" % ("PASS" if good else "FAIL"))
    # IDLE: three "no" rows and nothing changed since -> the next tick does nothing; a change to a watched file wakes it; a run to
    # harvest wakes it; two "no" rows are not enough
    board = os.path.join(t, "board.md")
    io.open(board, "w").write("x")
    os.utime(board, (1000.0, 1000.0))
    io.open(ledger, "w").write("")
    for i in range(2):
        finish("X-%d" % i, 5, "nothing", "no", lock, ledger)
    rc11, _ = start(lock, now=time.time(), procs=[], cpu=5.0, stop=stop, runs_log=os.path.join(t, "none.jsonl"), alive=none,
                    ledger=ledger, here=t, watch=w, hold=os.path.join(t, "no_hold"))
    finish("X-2", 5, "nothing", "no", lock, ledger)
    rc12, said12 = start(lock, now=time.time(), procs=[], cpu=5.0, stop=stop, runs_log=os.path.join(t, "none.jsonl"), alive=none,
                         ledger=ledger, here=t, watch=w, hold=os.path.join(t, "no_hold"))
    rc13, _ = start(lock, now=time.time(), procs=["py closed_loop.py run m"], cpu=5.0, stop=stop, runs_log=os.path.join(t, "none.jsonl"),
                    alive=none, ledger=ledger, here=t, watch=w, hold=os.path.join(t, "no_hold"))
    if os.path.exists(lock):
        os.remove(lock)
    os.utime(board, (time.time() + 5, time.time() + 5))
    rc14, _ = start(lock, now=time.time(), procs=[], cpu=5.0, stop=stop, runs_log=os.path.join(t, "none.jsonl"), alive=none,
                    ledger=ledger, here=t, watch=w, hold=os.path.join(t, "no_hold"))
    if os.path.exists(lock):
        os.remove(lock)
    good = rc11 == 0 and rc12 == 5 and said12.startswith("IDLE") and rc13 == 0 and rc14 == 0
    ok &= good
    print("selftest: IDLE three ticks worth nothing and no change since: the next does nothing; a run going or a changed board "
          "wakes it; two are not enough %s" % ("PASS" if good else "FAIL"))
    # a real detached run: its exit code is read from its log; one that left no exit line is DIED; a harvested one is not listed
    rc6, said6 = launch([sys.executable, "-c", "print('hello'); raise SystemExit(5)"], runs_log=rl, runs_dir=os.path.join(t, "runs"), procs=[])
    log = said6.split("log ")[-1]
    for _ in range(100):
        if os.path.exists(log) and io.open(log, encoding="utf-8").read().rstrip().endswith("EXIT 5"):
            break
        time.sleep(0.1)
    died = os.path.join(t, "runs", "died.log")
    io.open(died, "w", encoding="utf-8").write("RUN x\nhalf a line")
    io.open(rl, "a", encoding="utf-8").write(json.dumps({"pid": 1, "log": died, "cmd": "x"}) + "\n")
    got = {lg: oc for lg, _, oc in ended(rl, alive=none)}
    good = rc6 == 0 and got.get(log) == "EXIT 5" and got.get(died) == "DIED" and "hello" in io.open(log, encoding="utf-8").read()
    harvest(log, rl)
    good = good and [lg for lg, _, _ in ended(rl, alive=none)] == [died]
    rc7, said7 = launch(["x"], runs_log=rl, runs_dir=os.path.join(t, "runs"), procs=["py closed_loop.py run m"])
    good = good and rc7 == 3
    ok &= good
    print("selftest: LAUNCH a detached run's exit code is read from its log, one with no exit line is DIED, a harvested one is not "
          "listed again, a second run while one is going is refused %s" % ("PASS" if good else "FAIL"))
    shutil.rmtree(t, ignore_errors=True)
    print("selftest: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv):
    if argv[:1] == ["_run"]:
        return _run(argv[1], argv[3:] if argv[2:3] == ["--"] else argv[2:])
    if argv[:1] == ["launch"]:
        cmd = argv[2:] if argv[1:2] == ["--"] else argv[1:]
        if not cmd:
            print("loop_tick: launch -- <command>")
            return 2
        rc, said = launch(cmd)
        print("loop_tick: " + said)
        return rc
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["start", "beat", "finish", "show", "harvest"])
    ap.add_argument("log", nargs="?", help="harvest: the run's log")
    ap.add_argument("--item")
    ap.add_argument("--minutes", type=int)
    ap.add_argument("--found")
    ap.add_argument("--worth", choices=["yes", "no", "unknown"])
    ap.add_argument("--targets", help="the stat this tick meant to move, as the morph log names it (e.g. holes_ours)")
    ap.add_argument("--last", type=int, default=10)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "start":
        rc, said = start()
        print("loop_tick: " + said)
        return rc
    if a.cmd == "beat":
        print("loop_tick: beat" if beat() else "loop_tick: no lock to beat - this tick never ran start")
        return 0
    if a.cmd == "harvest":
        if not a.log:
            ap.error("harvest needs the run's log")
        harvest(a.log)
        print("loop_tick: harvested %s" % a.log)
        return 0
    if a.cmd == "finish":
        if not (a.item and a.minutes is not None and a.found and a.worth):
            ap.error("finish needs --item, --minutes, --found and --worth")
        print("loop_tick: " + json.dumps(finish(a.item, a.minutes, a.found, a.worth, targets=a.targets)))
        return 0
    if a.cmd == "show":
        for r in _rows(LEDGER)[-a.last:]:
            print("  %s  %-10s %3s min  worth %-7s %s" % (r["time"][:16], r["item"], r["minutes"], r["worth"], r["found"][:110]))
        return 0
    ap.error("start | launch | harvest | finish | show | --selftest")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
