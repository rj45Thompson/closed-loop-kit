#!/usr/bin/env python3
"""improvement.py - the loop's IMPROVEMENT graph: the closed loop turned on its own improvement, one level up, closed on itself.

WHY (2026-10-29): a stat should trigger something. Stats get their own type of requirements, a subsystem that determines improvement
rather than requirements for software, and that gives a recursive graph for self-improvement, fitted here into a pipeline.

THE LEVELS, and why the recursion ends:
  1  THE PRODUCT GRAPH (usecase_kg): requirements, code, breaks, tests. The loop holds the code to the requirements.
  2  THE IMPROVEMENT GRAPH (this): the stats, the improvement requirements on them (improvement/stat_requirements.yaml), the actions the
     ticks take (loop_ticks.jsonl, each naming the stat it targets) and their measured OUTCOMES (the stat's value before and after,
     from the morph logs). Its closure rules are level 1's, one level up:
       COVERED   every recorded stat is read by code that acts on it, or by an improvement requirement with a trigger
       ENDED     every trigger that fired on a model's last pass is acted on by a tick that targets its stat after that pass; one
                 still open once ENDED_WITHIN ticks have gone by since the pass is OVERDUE. A tick before the pass does not end it: the
                 pattern still stood after that tick. A tick after it does, until a new pass can say whether it worked
       VERIFIED  every action names the stat it targeted and has a measured outcome
       RETIRED   an action kind that never moved its stat after RETIRE_AFTER tries is retired, not repeated
     A GNN can learn on it exactly as on level 1: link prediction on `moved` (which action moves which stat) - switched by run_plan's
     rule, OFF until there are enough measured outcomes (MIN_OUTCOMES).
  3  is level 2 again: the improvement graph's own health (stats covered, triggers ended, actions verified, actions' hit rate) is
     written as stats with their own improvement requirements, so they are nodes of the same graph. Self-reference closes the tower.

THE PIPELINE (TICK.md): each tick - start, harvest, health (selftests, telemetry_audit), `improvement.py check` (the triggers that
fired are items), choose, do, finish with `--targets <stat>`; each epoch's benchmark - `improvement.py graph` rebuilds level 2 and
checks its closure, and its own stats join the epoch row.

    python -X utf8 improvement.py check [<model dir> ...]     the triggers that fire on each model's latest passes
    python -X utf8 improvement.py graph [<model dir> ...]     level 2 built, closed or not, and its own stats
    python -X utf8 improvement.py --selftest
"""
import argparse
import collections
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REG = os.path.join(HERE, "improvement", "stat_requirements.yaml")
LEDGER = os.path.join(HERE, "loop_ticks.jsonl")
MODELS = (os.path.join(HERE, "..", "shops", "model"), os.path.join(HERE, "selfmodel", "model"))
ENDED_WITHIN, RETIRE_AFTER, MIN_OUTCOMES = 3, 3, 20


def load(reg=REG):
    import yaml
    d = yaml.safe_load(io.open(reg, encoding="utf-8")) or {}
    return d.get("requirements") or [], d.get("banked") or []


def rows_of(model):
    p = os.path.join(os.path.dirname(os.path.abspath(model)), "stats.jsonl")
    return [json.loads(l) for l in io.open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def value(row, req):
    d = req.get("derived")
    if d and "ratio" in d:
        num, den = d["ratio"]
        a, b = row.get(num), sum(row.get(x) or 0 for x in den)
        return None if a is None or not b else a / float(b)
    if d and "key" in d:
        # the stat is a dict of scores and the requirement is about ONE of them (gnn_mrr holds the owner network's score beside the plain rules'):
        # totalling the dict, as scalar() does, moves with every rule's score and says nothing about the network (MEASURED 10-30 by planting a
        # worsening on a copy: the outcome of a tick on the owner score read +0.19 for a +0.23 move)
        v = row.get(req["stat"])
        return v.get(d["key"]) if isinstance(v, dict) else None
    return row.get(req["stat"])


def scalar(v):
    """a value a trend can compare: a number, or a dict's total, or a list's length"""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        return float(sum(x for x in v.values() if isinstance(x, (int, float)))) if all(isinstance(x, (int, float)) for x in v.values()) else float(len(v))
    if isinstance(v, list):
        return float(len(v))
    return None


def named(v):
    """(what a stat names, how to say it): a dict's largest key, a list's first item, else the value. Ties go by name, so a dict that
    only changed its order names the same thing."""
    if isinstance(v, dict) and v:
        # a dict of lists (holes: criterion -> [ours, owner's, listed]) compares on its FIRST number, the one of ours; counting a
        # list's length named L1 as the worst criterion on 10-29, every criterion holding three numbers
        k = max(v, key=lambda k: ((v[k][0] if isinstance(v[k], list) and v[k] and isinstance(v[k][0], (int, float))
                                   else scalar(v[k]) or 0), k))
        return k, "largest: %s (%s)" % (k, v[k])
    if isinstance(v, list) and v:
        top = v[0][0] if isinstance(v[0], list) else v[0]
        return top, "top: %s" % top
    return v, "value: %s" % v


def fires(req, rows):
    """(fired, detail) for one requirement on one model's rows, oldest first; (None, why) when it cannot be judged yet"""
    if not rows:
        return None, "no passes"
    t = req.get("trigger") or {}
    vals = [value(r, req) for r in rows]
    last = vals[-1]
    if "null" in t:
        return (last is None and req["stat"] in rows[-1]), "recorded as null" if last is None else "measured"
    if last is None:
        return None, "not in the last pass"
    if "names" in t:
        # it speaks when what it names CHANGED. It used to return True whenever the stat existed, so it fired on every pass of every
        # model (MEASURED 10-30: 8 of the 12 fired), could never be ended, and lifted its area in the pick for good
        was = next((v for v in reversed(vals[:-1]) if v is not None), None)
        if was is None:
            return None, "needs 2 passes to see a change, has 1"
        now, said = named(last)
        before = named(was)[0]
        return now != before, said if now == before else "%s, was %s" % (said, before)
    if "key_above" in t:
        k = t["key_above"]["key"]
        v = (last or {}).get(k, 0) if isinstance(last, dict) else None
        return (v is not None and v > t["key_above"]["n"]), "%s = %s" % (k, v)
    if "key_zero" in t:
        n, m = t["key_zero"]["passes"], t["key_zero"]["min"]
        recent = [v for v in vals[-n:] if isinstance(v, dict)]
        if len(recent) < n:
            return None, "needs %d passes, has %d" % (n, len(recent))
        keys = sorted(set().union(*[set(v) for v in recent]))
        zero = [k for k in keys if all(not v.get(k) for v in recent) and sum(scalar(v) or 0 for v in recent) >= m]
        return bool(zero), "zero in %d passes: %s" % (n, ", ".join(zero) or "none")
    s = [scalar(v) for v in vals]
    if "above" in t:
        return s[-1] is not None and s[-1] > t["above"], "last %s" % s[-1]
    if "below" in t:
        return s[-1] is not None and s[-1] < t["below"], "last %s" % s[-1]
    for kind in ("rises", "falls", "stalled"):
        if kind in t:
            n = t[kind]
            w = [x for x in s[-(n + 1):] if x is not None]
            if len(w) < n + 1:
                return None, "needs %d passes, has %d" % (n + 1, len(w))
            ok = all(b > a for a, b in zip(w, w[1:])) if kind == "rises" else all(b < a for a, b in zip(w, w[1:])) if kind == "falls" \
                else len(set(w)) == 1
            return ok, "%s over %d: %s" % (kind, n, " -> ".join("%g" % x for x in w))
    return None, "no trigger form this engine knows: %s" % sorted(t)


def read_ledger(path=LEDGER):
    return [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()] if os.path.exists(path) else []


def since_pass(ledger, rows):
    """the finished ticks after this model's last pass: the only ones that can have acted on what that pass fired"""
    last = _epoch(rows[-1].get("time")) if rows else 0.0
    return [t for t in ledger if "item" in t and t.get("t", 0) > last]


def check(models=MODELS, reg=REG, say=print, ledger_path=LEDGER):
    """every trigger that fires on each model's last pass, with its state: answered by the tick that targeted its stat since that pass,
    or open (overdue once ENDED_WITHIN ticks have gone by). An open one is an item; an answered one waits for the next pass to judge it"""
    reqs, _ = load(reg)
    ledger = read_ledger(ledger_path)
    fired = []
    for m in models:
        rows = rows_of(m)
        name = os.path.basename(os.path.dirname(os.path.abspath(m)))
        since = since_pass(ledger, rows)
        for r in reqs:
            f, why = fires(r, rows)
            if f:
                by = next((t["item"] for t in since if str(t.get("targets")) == r["stat"]), None)
                fired.append({"id": r["id"], "stat": r["stat"], "model": name, "detail": why, "action": r.get("action"),
                              "answered_by": by, "overdue": by is None and len(since) >= ENDED_WITHIN})
    for x in fired:
        state = "answered by %s" % x["answered_by"] if x["answered_by"] else "OVERDUE" if x["overdue"] else "OPEN"
        say("  FIRED %-6s %-24s %-10s %-44s %s -> %s" % (x["id"], x["stat"], x["model"], x["detail"][:44], state, json.dumps(x["action"])[:90]))
    still = [x for x in fired if not x["answered_by"]]
    say("improvement: %d trigger(s) fired, %d open%s" % (len(fired), len(still), " - each open one is an item (the tick's runbook)" if still else ""))
    return fired


def outcomes(ledger, models, reg=REG):
    """[(item, stat, before, after, moved)] for each finished tick that named the stat it targeted: the stat's value in the last pass
    BEFORE the tick and the first pass AFTER it, on any model that has both. A stat read as one score of a dict is read as its trigger reads it"""
    out = []
    keyed = {r["stat"]: r for r in load(reg)[0] if (r.get("derived") or {}).get("key")}
    read = lambda row, stat: scalar(value(row, keyed[stat])) if stat in keyed else scalar(row[stat])
    rows = {os.path.basename(os.path.dirname(os.path.abspath(m))): rows_of(m) for m in models}
    for t in ledger:
        stat = t.get("targets")
        if not stat or "t" not in t:
            continue
        for name, rs in rows.items():
            start = t["t"] - 60 * (t.get("minutes") or 0)
            before = [r for r in rs if _epoch(r.get("time")) <= start and stat in r]
            after = [r for r in rs if _epoch(r.get("time")) > t["t"] and stat in r]
            if before and after:
                a, b = read(before[-1], stat), read(after[0], stat)
                out.append({"item": t.get("item"), "stat": stat, "model": name, "before": a, "after": b,
                            "moved": None if a is None or b is None else b - a})
    return out


def _epoch(s):
    try:
        return time.mktime(time.strptime(str(s)[:19], "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        return 0.0


def level2(models=MODELS, reg=REG, ledger_path=LEDGER):
    """the improvement graph as data and nothing else: (nodes, edges, fired triggers, measured outcomes, requirements, banked records, ledger). It writes no log, so a page can read
    the same graph `graph` closes without adding a reading to graph_stats.jsonl (that file is a series of passes, and a page build is not one)"""
    reqs, banked = load(reg)
    ledger = read_ledger(ledger_path)
    nodes, edges = set(), []
    for r in reqs:
        nodes |= {"imp:" + r["id"], "stat:" + r["stat"]}
        edges.append(("imp:" + r["id"], "watches", "stat:" + r["stat"]))
        act = json.dumps(r.get("action"), sort_keys=True)
        nodes.add("action:" + act)
        edges.append(("imp:" + r["id"], "triggers", "action:" + act))
    fired = check(models, reg, say=lambda s: None, ledger_path=ledger_path)
    for f in fired:
        edges.append(("stat:" + f["stat"], "fired", "imp:" + f["id"]))
    outs = outcomes(ledger, models)
    for o in outs:
        nodes |= {"tick:" + str(o["item"]), "stat:" + o["stat"]}
        edges.append(("tick:" + str(o["item"]), "targets", "stat:" + o["stat"]))
        if o["moved"] is not None:
            edges.append(("tick:" + str(o["item"]), "moved %+g" % o["moved"], "stat:" + o["stat"]))
    return nodes, edges, fired, outs, reqs, banked, ledger


def graph(models=MODELS, reg=REG, ledger_path=LEDGER, audit=None, say=print):
    """level 2: nodes and edges, its closure, and its own stats (level 3, the same graph)"""
    nodes, edges, fired, outs, reqs, banked, ledger = level2(models, reg, ledger_path)
    # CLOSURE, level 1's rules one level up
    acted = [t for t in ledger if "item" in t]
    unended = [f for f in fired if not f["answered_by"]]
    unverified = [t for t in acted if not t.get("targets")]
    tries = collections.defaultdict(list)
    for o in outs:
        tries[(str(o["item"]).split("-")[0], o["stat"])].append(o["moved"])
    retire = sorted("%s on %s" % k for k, ms in tries.items() if len(ms) >= RETIRE_AFTER and not any(m for m in ms if m))
    own = {"improvement_requirements": len(reqs), "banked_records": len(banked), "triggers_fired": len(fired),
           "fired_without_action": len(unended), "ticks_without_target": len(unverified), "measured_outcomes": len(outs),
           "outcomes_that_moved": sum(1 for o in outs if o["moved"]), "actions_to_retire": len(retire),
           "learnable": len(outs) >= MIN_OUTCOMES}
    if audit is not None:
        own["stats_unread"] = audit
    say("improvement graph: %d nodes, %d edges" % (len(nodes), len(edges)))
    say("  COVERED   %s" % ("see telemetry_audit (%s unread)" % audit if audit is not None else "run telemetry_audit.py for the count"))
    say("  ENDED     %d fired trigger(s) no tick has acted on since their pass, %d of them past the %d ticks allowed%s" % (
        len(unended), sum(1 for f in unended if f["overdue"]), ENDED_WITHIN,
        "" if not unended else ": " + ", ".join("%s (%s)" % (f["id"], f["model"]) for f in unended[:6])))
    say("  VERIFIED  %d of %d ticks named the stat they targeted; %d outcome(s) measured" % (len(acted) - len(unverified), len(acted), len(outs)))
    say("  RETIRED   %s" % (", ".join(retire) or "nothing: no action kind has failed %d times yet" % RETIRE_AFTER))
    say("  LEARNABLE %s (%d measured outcomes; the improvement GNN needs %d)" % ("yes" if own["learnable"] else "not yet", len(outs), MIN_OUTCOMES))
    out = os.path.join(HERE, "improvement", "graph_stats.jsonl")
    io.open(out, "a", encoding="utf-8").write(json.dumps(dict(own, time=time.strftime("%Y-%m-%dT%H:%M:%S")), sort_keys=True) + "\n")
    return own


def selftest():
    """a trigger fires on its pattern and not without it; one that cannot be judged says why, never a verdict; a tick that named its
    stat gets a measured outcome; a fired trigger with no action is ENDED's gap; the graph's own stats are recorded"""
    import shutil
    import tempfile
    ok = True
    rows = [{"time": "2026-10-29T10:00:00", "a": 1, "b": {"x": 0, "y": 5}, "c": None, "d": 3},
            {"time": "2026-10-29T11:00:00", "a": 2, "b": {"x": 0, "y": 60}, "c": None, "d": 3},
            {"time": "2026-10-29T12:00:00", "a": 3, "b": {"x": 0, "y": 70}, "c": None, "d": 3}]
    cases = [({"stat": "a", "trigger": {"rises": 2}}, True), ({"stat": "a", "trigger": {"falls": 1}}, False),
             ({"stat": "d", "trigger": {"stalled": 2}}, True), ({"stat": "b", "trigger": {"key_zero": {"passes": 3, "min": 100}}}, True),
             ({"stat": "c", "trigger": {"null": True}}, True), ({"stat": "a", "trigger": {"above": 5}}, False),
             ({"stat": "a", "trigger": {"rises": 5}}, None), ({"stat": "zz", "trigger": {"above": 0}}, None)]
    got = [fires(r, rows)[0] for r, _ in cases]
    good = got == [want for _, want in cases]
    ok &= good
    print("selftest: TRIGGER each form fires on its pattern and not otherwise; too few passes or a missing stat is no verdict %s" % (
        "PASS" if good else "FAIL %s" % got))
    # `names`: speaks when what it names changed, on a real shape of each kind; the largest of a dict of lists is by its FIRST number
    names = {"stat": "holes", "trigger": {"names": "largest"}}
    l13, l15 = {"L1": [0, 0, 0], "L13": [27, 0, 43], "L15": [21, 0, 2]}, {"L1": [0, 0, 0], "L13": [20, 0, 43], "L15": [21, 0, 2]}
    moved = fires(names, [{"holes": l15}, {"holes": l13}])
    same = fires(names, [{"holes": l13}, {"holes": dict(l13, L15=[26, 0, 2])}])
    first = fires(names, [{"holes": l13}])
    gap = fires(names, [{"holes": l15}, {}, {"holes": l13}])          # a pass that did not write it is no reading: the last one that did
    tie = fires({"stat": "k", "trigger": {"names": "largest"}}, [{"k": {"a": 1, "b": 1}}, {"k": {"b": 1, "a": 1}}])
    top = {"stat": "t", "trigger": {"names": "top"}}
    val = {"stat": "v", "trigger": {"names": "value"}}
    good = moved[0] is True and "L13" in moved[1] and "was L15" in moved[1] and same[0] is False and "L13" in same[1] and first[0] is None \
        and gap[0] is True and "was L15" in gap[1] and tie[0] is False \
        and fires(top, [{"t": [["x", 9], ["y", 1]]}, {"t": [["y", 9], ["x", 1]]}])[0] is True \
        and fires(top, [{"t": [["x", 9]]}, {"t": [["x", 3]]}])[0] is False \
        and fires(val, [{"v": "r1"}, {"v": "r2"}])[0] is True and fires(val, [{"v": "r1"}, {"v": "r1"}])[0] is False
    ok &= good
    print("selftest: NAMES a `names` trigger fires when what it names changed and says what it was; the same name, a reordered tie or one "
          "pass does not; a dict of lists compares on its first number %s" % ("PASS" if good else "FAIL %s" % [moved, same, first, gap, tie]))
    t = tempfile.mkdtemp(prefix="improvement_selftest_")
    m = os.path.join(t, "m", "model")
    os.makedirs(m)
    io.open(os.path.join(t, "m", "stats.jsonl"), "w", encoding="utf-8").write("".join(json.dumps(r) + "\n" for r in rows))
    reg = os.path.join(t, "reg.yaml")
    io.open(reg, "w", encoding="utf-8").write("requirements:\n  - {id: IMP-T, stat: a, trigger: {rises: 2}, action: {area: tests}, why: t}\nbanked: []\n")
    led = os.path.join(t, "ledger.jsonl")
    io.open(led, "w", encoding="utf-8").write(json.dumps({"item": "X-1", "targets": "a", "minutes": 30,
                                                          "t": _epoch("2026-10-29T10:45:00")}) + "\n")
    outs = outcomes([json.loads(l) for l in io.open(led, encoding="utf-8")], [m])
    good = len(outs) == 1 and outs[0]["before"] == 1.0 and outs[0]["after"] == 2.0 and outs[0]["moved"] == 1.0
    saved = os.path.join(HERE, "improvement", "graph_stats.jsonl")
    keep = io.open(saved, encoding="utf-8").read() if os.path.exists(saved) else None
    own = graph([m], reg, led, say=lambda s: None)
    if keep is None:
        os.remove(saved)
    else:
        io.open(saved, "w", encoding="utf-8").write(keep)
    # X-1 targeted the stat BEFORE the last pass and the pattern stood after it: that is an action that did not end it, so still open
    good = good and own["triggers_fired"] == 1 and own["fired_without_action"] == 1 and own["measured_outcomes"] == 1
    tick = lambda item, when, targets=None: json.dumps(dict({"item": item, "minutes": 5, "t": _epoch(when)}, **({"targets": targets} if targets else {}))) + "\n"
    io.open(led, "w", encoding="utf-8").write(tick("Y-1", "2026-10-29T13:00:00"))
    own2 = graph([m], reg, led, say=lambda s: None)
    good = good and own2["fired_without_action"] == 1 and own2["ticks_without_target"] == 1
    # a tick after the pass that targets the stat answers it, and only that tick does; three after it that do not make it overdue, two do not
    io.open(led, "w", encoding="utf-8").write(tick("X-1", "2026-10-29T10:45:00", "a") + tick("Z-1", "2026-10-29T12:30:00", "a"))
    own3 = graph([m], reg, led, say=lambda s: None)
    answered = check([m], reg, say=lambda s: None, ledger_path=led)
    good = good and own3["fired_without_action"] == 0 and answered[0]["answered_by"] == "Z-1"
    state = []
    for n in (2, 3):
        io.open(led, "w", encoding="utf-8").write("".join(tick("O-%d" % i, "2026-10-29T13:0%d:00" % i, "other") for i in range(n)))
        f = check([m], reg, say=lambda s: None, ledger_path=led)[0]
        state.append((f["answered_by"], f["overdue"]))
    good = good and state == [(None, False), (None, True)]
    if keep is None and os.path.exists(saved):
        os.remove(saved)
    elif keep is not None:
        io.open(saved, "w", encoding="utf-8").write(keep)
    ok &= good
    print("selftest: GRAPH a tick that named its stat gets the outcome it measured; a fired trigger no tick acted on since its pass is ENDED's "
          "gap, overdue after %d ticks; a tick before the pass does not end it, one after does; a tick with no target is VERIFIED's %s" % (
              ENDED_WITHIN, "PASS" if good else "FAIL %s" % [own["fired_without_action"], own2, own3, state]))
    # KEY: a stat that is a dict of scores, with the requirement about one of them: its trigger and its outcome read that one, not the total
    kr = [{"time": "2026-10-29T10:00:00", "s": {"gnn": 0.8, "w": 0.5}}, {"time": "2026-10-29T11:00:00", "s": {"gnn": 0.5, "w": 0.9}},
          {"time": "2026-10-29T12:00:00", "s": {"gnn": 0.7, "w": 0.9}}]
    keyed = {"stat": "s", "derived": {"key": "gnn"}, "trigger": {"falls": 1}}
    k = os.path.join(t, "k", "model")
    os.makedirs(k)
    kreg = os.path.join(t, "kreg.yaml")
    io.open(kreg, "w", encoding="utf-8").write("requirements:\n  - {id: IMP-K, stat: s, derived: {key: gnn}, trigger: {falls: 1}, action: {area: tests}}\nbanked: []\n")
    io.open(led, "w", encoding="utf-8").write(json.dumps({"item": "K-1", "targets": "s", "minutes": 5, "t": _epoch("2026-10-29T10:30:00")}) + "\n")
    flat = fires(dict(keyed, derived=None), kr[:2])[0]                     # the total rose 1.3 -> 1.4: a plain read cannot see the network fall
    got = fires(keyed, kr[:2])[0]
    io.open(os.path.join(t, "k", "stats.jsonl"), "w", encoding="utf-8").write("".join(json.dumps(r) + "\n" for r in kr))
    out = outcomes(read_ledger(led), [k], kreg)
    good = got is True and flat is False and len(out) == 1 and abs(out[0]["moved"] - (0.5 - 0.8)) < 1e-9      # the total went 1.3 -> 1.4, +0.1
    ok &= good
    print("selftest: KEY a stat that is a dict of scores, with the requirement about one of them, fires and is measured on that score, not on the dict's total (the total rises while the score falls) %s" % (
        "PASS" if good else "FAIL %s" % [got, flat, out]))
    shutil.rmtree(t, ignore_errors=True)
    print("selftest: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["check", "graph"])
    ap.add_argument("models", nargs="*")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    models = a.models or MODELS
    if a.cmd == "check":
        check(models)
        return 0
    if a.cmd == "graph":
        graph(models)
        return 0
    ap.error("check | graph | --selftest")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
