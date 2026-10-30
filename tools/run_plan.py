#!/usr/bin/env python3
"""run_plan.py - which parts of the loop run for a model, decided by the measured evidence, never by habit.

WHY (2026-10-29): what is better not run is disabled for the run, and a graph too small for a part does not run it, both decided
from the data. Every learned part has a plain rule it must beat (graph_gnn's rankers, the KILLS rule, the operator prior). The PASS
(exploit) runs a learned part only where the evidence says it wins by more than MARGIN; elsewhere the plain rule does the job. The
epoch BENCHMARK (explore, TICK.md 4b) still measures every part on every model, so the data can turn a part back on.

THE EVIDENCE it reads (each written by the tool that measured it; nothing here is assumed):
  <kg>/gnn_tasks.jsonl       gnn_tasks.py eval: each job's learned score against its best plain rule, and the size of the graph it
                             measured: a score about a graph that has since changed switches nothing on (GraphEvidence)
  <kg>/gnn_curve.jsonl and gnn_curve_*.jsonl beside the tools   gnn_curve.py: the GNN against the plain rules by fullness
  <kg>/survivor_rank.jsonl   survivor_rank.py eval: the learned break order against the operator prior
  ~/.claude/ab/results.jsonl with runs/*/profile.json   ab_loop.py: loop on against off, by the kind of task (PK-6)
THE SIZE FLOOR: across every graph with evidence, the fewest requirements at which the owner GNN has ever won. A model below it runs
the plain rules without measuring the GNN in a pass (the benchmark still does).

    python -X utf8 run_plan.py <model dir> [--write]     prints the plan; --write keeps it at <kg>/run_plan.json
    python -X utf8 run_plan.py --selftest
"""
import argparse
import collections
import glob
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MARGIN = 0.02
MIN_SURVIVORS = 20          # the learned break order is trusted only when it learned from at least this many survivors
OWNER_VARIANTS = ("owner + calls", "owner + caught by", "owner + both")     # gnn_tasks' jobs: the owner GNN hearing an added relation
AB = os.path.join(os.path.expanduser("~"), ".claude", "ab")


class GraphEvidence(object):
    """A gnn_tasks row and the graph it measured. A job's score is evidence about THAT graph: once a pass or a model edit has changed
    the graph the GNN reads, the score speaks for a graph that no longer exists, so no learned part runs on it until `gnn_tasks.py
    eval` measures again (the plain rule meanwhile; the epoch benchmark re-measures anyway). ANY change counts: no tick has measured
    how much change a score survives, so a tolerance here would be a guess. Breadcrumb: the row's `shape` is gnn_tasks.graph_shape, and
    E1-1 (10-30) gave these sizes their reader: they sat beside every score with nothing comparing them."""

    SIZE = ("nodes", "edges", "methods", "owner_links", "requirements")

    def __init__(self, kg, row):
        self.kg, self.row = kg, row

    def moved(self):
        """None when the row describes the graph that is here; else the words saying how it differs (or that it is gone)"""
        if not self.row:
            return None
        name = self.row.get("graph")
        if not name or not os.path.exists(os.path.join(self.kg, name)):
            return "the graph it measured, %s, is not here" % name
        import gnn_tasks       # torch: only when there is a row to compare, and a machine with a row has measured with it
        now = gnn_tasks.graph_shape(os.path.join(self.kg, name))
        was = self.row.get("shape") or {}
        return ", ".join("%s %s, now %s" % (k, was.get(k), now.get(k)) for k in self.SIZE if was.get(k) != now.get(k)) or None


def rows(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]


def all_task_rows(roots):
    out = []
    for r in roots:
        for f in glob.glob(os.path.join(r, "**", "gnn_tasks.jsonl"), recursive=True):
            out += rows(f)
    return out


def curve_verdict(kg, roots):
    """the owner GNN from gnn_curve.py's rows for this graph at full fullness: on only if it beat the best plain rule by more than
    MARGIN at every seed; None when there is no curve for it"""
    name = None
    for f in glob.glob(os.path.join(kg, "*.sqlite")):
        name = os.path.basename(f)
    if not name:
        return None
    found = []
    for r in roots:
        for f in glob.glob(os.path.join(r, "**", "gnn_curve*.jsonl"), recursive=True):
            found += [x for x in rows(f) if x.get("graph") == name and x.get("fullness") == 1.0 and x.get("by", "methods") == "methods"]
    if not found:
        return None
    wins = all(x["gnn"] > x["best_mrr"] + MARGIN for x in found)
    g = sum(x["gnn"] for x in found) / len(found)
    b = sum(x["best_mrr"] for x in found) / len(found)
    return {"on": wins, "why": "the fullness curve, every claim seen, %d seed(s): GNN %.2f against %s %.2f (%s)" % (
        len(found), g, found[-1].get("best"), b, "wins at every seed" if wins else "does not win at every seed: the plain rule runs")}


def size_floor(task_rows, curve_rows=()):
    """the fewest requirements of any graph where the owner GNN won (a gnn_tasks job, or a full-fullness curve row beating its best
    plain rule by more than MARGIN), or None when it never has"""
    won = [r["shape"]["requirements"] for r in task_rows if (r.get("jobs", {}).get("owner as today") or {}).get("wins")]
    won += [r["requirements"] for r in curve_rows if r.get("fullness") == 1.0 and r.get("by", "methods") == "methods"
            and "requirements" in r and r["gnn"] > r["best_mrr"] + MARGIN]
    return min(won) if won else None


def all_curve_rows(roots):
    out = []
    for r in roots:
        for f in glob.glob(os.path.join(r, "**", "gnn_curve*.jsonl"), recursive=True):
            out += rows(f)
    return out


def decide(kg, roots=(HERE,), ab=AB):
    """{part: {on, why}} for the model whose graph folder is `kg`"""
    plan = {}
    own = rows(os.path.join(kg, "gnn_tasks.jsonl"))
    last = own[-1] if own else None
    floor = size_floor(all_task_rows(roots), all_curve_rows(roots))
    reqs = last["shape"]["requirements"] if last else model_requirements(kg)
    moved = GraphEvidence(kg, last).moved()

    def stale(what):
        return {"on": False, "why": "%s: measured on a graph that has changed since (%s): gnn_tasks.py eval measures it again; until then the plain rule" % (what, moved)}

    def job(name, part, what):
        j = (last or {}).get("jobs", {}).get(name)
        if j is None:
            plan[part] = {"on": False, "why": "no evidence for %s on this model yet: the benchmark measures it; until then the plain rule" % what}
        elif moved:
            plan[part] = stale(what)
        elif j.get("wins") is None:
            plan[part] = {"on": False, "why": "not measurable here: %s" % j.get("why")}
        else:
            plan[part] = {"on": bool(j["wins"]), "why": "%s: learned %.2f against %s %.2f (%s)" % (
                what, j["learned"], j["best_plain"], j["plain"], "wins by more than %.2f" % MARGIN if j["wins"] else "does not win: the plain rule runs")}

    # the model's OWN evidence decides first; the size floor speaks only for a model not measured yet
    curve = curve_verdict(kg, roots)
    if (last or {}).get("jobs", {}).get("owner as today") is None and curve is not None:
        plan["gnn owner"] = curve
    elif (last or {}).get("jobs", {}).get("owner as today") is None and floor is not None and reqs is not None and reqs < floor:
        plan["gnn owner"] = {"on": False, "why": "too small: %d requirements, and the owner GNN has only ever won on graphs of %d or more" % (reqs, floor)}
    else:
        job("owner as today", "gnn owner", "the owner GNN")
    # the owner GNN runs in its BEST winning form: if hearing a new relation is what makes it win, it runs hearing that relation
    # (10-29: on the self model the GNN as it is lost, 0.70 to 0.72, and with `calls` won, 0.74 - the plan had switched it off)
    variants = [(n, j) for n, j in ((n, (last or {}).get("jobs", {}).get(n)) for n in OWNER_VARIANTS) if j and j.get("wins")]
    if variants and not moved and not plan.get("gnn owner", {}).get("on"):
        n, j = max(variants, key=lambda v: v[1]["learned"])
        plan["gnn owner"] = {"on": True, "why": "the owner GNN %s: learned %.2f against %s %.2f (wins only hearing the added relation%s)" % (
            n.replace("owner + ", "hearing `").rstrip() + "`", j["learned"], j["best_plain"], j["plain"], "s" if n == "owner + both" else "")}
    for rel, key in (("caught by", "owner + caught by"), ("calls", "owner + calls")):
        j = (last or {}).get("jobs", {}).get(key)
        if not j or j.get("wins") is None:
            plan["relation " + rel] = {"on": False, "why": "no evidence the GNN does better hearing `%s`" % rel}
        elif moved:
            plan["relation " + rel] = stale("hearing `%s`" % rel)
        else:
            gain = j.get("adds_over_today", 0.0)
            plan["relation " + rel] = {"on": bool(j["wins"]) and gain > MARGIN, "why": "hearing `%s` moves the owner GNN by %+.2f%s" % (
                rel, gain, "" if gain > MARGIN else " (not more than %.2f: left out)" % MARGIN)}
    job("tests", "gnn tests", "which tests catch a branch's breaks")
    job("steps", "gnn steps", "which steps carry a requirement")
    job("equivalent", "equivalent ranking", "which survivors are likely equivalent")
    sr = rows(os.path.join(kg, "survivor_rank.jsonl"))
    if sr and sr[-1].get("survived", 0) < MIN_SURVIVORS:
        plan["break order"] = {"on": False, "why": "too few examples: %d survivors among %d breaks (the learned order needs %d or more): the plain order" % (
            sr[-1].get("survived", 0), sr[-1].get("breaks", 0), MIN_SURVIVORS)}
    elif sr:
        r = sr[-1]["recall20"]
        on = r["learned"] > r["operator prior"] + MARGIN and r["learned"] > r["random"] + MARGIN
        plan["break order"] = {"on": on, "why": "the learned order finds %.0f%% of survivors in the first fifth against the operator prior's %.0f%% (%s)" % (
            100 * r["learned"], 100 * r["operator prior"], "learned order" if on else "the operator prior orders the breaks")}
    else:
        plan["break order"] = {"on": False, "why": "no evidence: the breaks run in their plain order"}
    plan["when to use the loop"] = when_to_use(ab)
    return plan


def model_requirements(kg):
    """the requirements the model beside this graph folder holds, read from its requirements.yaml (None when there is none)"""
    p = os.path.join(kg, "..", "model", "requirements.yaml")
    if not os.path.exists(p):
        return None
    import yaml
    return len((yaml.safe_load(io.open(p, encoding="utf-8")) or {}).get("requirements") or [])


def when_to_use(ab=AB):
    """PK-6: per kind of task, how the loop did against off (the A/B board joined to each run's profile) - advice, not a switch"""
    res = rows(os.path.join(ab, "results.jsonl"))
    panel = []
    if os.path.exists(os.path.join(ab, "agreement.json")):      # a pair with a panel of graders counts by its majority (grade_panel.py, PK-5e)
        panel = json.load(io.open(os.path.join(ab, "agreement.json"), encoding="utf-8"))
    majority = {p["run"]: p["majority"] for p in panel}
    by = collections.defaultdict(collections.Counter)
    for r in res:
        prof = os.path.join(ab, "runs", r.get("run", ""), "profile.json")
        kind = json.load(io.open(prof, encoding="utf-8")).get("kind", "unprofiled") if os.path.exists(prof) else "unprofiled"
        verdict = majority.get(r.get("run"), r.get("better", "tie"))
        by[kind]["tie" if verdict == "split" else verdict] += 1
    if not by:
        return {"on": None, "why": "no graded pairs yet"}
    return {"on": None, "why": "; ".join("%s: on better %d, off better %d, tie %d" % (k, c["on"], c["off"], c["tie"]) for k, c in sorted(by.items()))}


def show(plan):
    return "\n".join("  %-22s %-4s %s" % (p, {True: "ON", False: "off", None: "-"}[d["on"]], d["why"]) for p, d in plan.items())


def selftest():
    """a part runs only where its evidence wins by more than the margin; a model below the size floor runs the plain rules without
    being measured; no evidence means off with that reason; the advice joins the A/B board to each run's kind"""
    import shutil
    import tempfile
    t = tempfile.mkdtemp(prefix="run_plan_selftest_")
    big, small = os.path.join(t, "big", "kg"), os.path.join(t, "small", "kg")
    for d in (big, small):
        os.makedirs(d)

    def real_graph(kg, reqs):
        """a real sqlite graph of `reqs` requirements, one claim; returns the row's head as gnn_tasks writes it"""
        import gnn_tasks
        import sqlite3
        db = os.path.join(kg, "g.sqlite")
        con = sqlite3.connect(db)
        con.executescript("CREATE TABLE entity(id TEXT PRIMARY KEY, kind TEXT, label TEXT, sub TEXT);"
                          "CREATE TABLE edge(s TEXT, p INTEGER, o TEXT, fam INTEGER);CREATE TABLE literal(s TEXT, p INTEGER, v TEXT);")
        con.execute("INSERT INTO entity VALUES ('cell:S-BRANCH:T.M#b0', 'cell', 'T.M#b0', 'S-BRANCH')")
        con.execute("INSERT INTO entity VALUES ('code:T.M', 'code', 'T.M', NULL)")
        con.execute("INSERT INTO edge VALUES ('cell:S-BRANCH:T.M#b0', 152, 'code:T.M', 32)")
        for i in range(reqs):
            con.execute("INSERT INTO entity VALUES (?, 'requirement', ?, 'authored')", ("req:R-%d" % i, "R-%d" % i))
        con.execute("INSERT INTO edge VALUES ('cell:S-BRANCH:T.M#b0', 122, 'req:R-0', 32)")
        con.commit()
        con.close()
        return {"graph": "g.sqlite", "shape": gnn_tasks.graph_shape(db)}

    def change_graph(kg, sql):
        import sqlite3
        con = sqlite3.connect(os.path.join(kg, "g.sqlite"))
        con.execute(sql)
        con.commit()
        con.close()
    win = {"wins": True, "learned": 0.77, "best_plain": "effects", "plain": 0.67}
    lose = {"wins": False, "learned": 0.70, "best_plain": "effects", "plain": 0.69}
    io.open(os.path.join(big, "gnn_tasks.jsonl"), "w", encoding="utf-8").write(json.dumps(dict(real_graph(big, 27), jobs={
        "owner as today": win, "owner + caught by": dict(win, adds_over_today=0.05), "owner + calls": dict(win, adds_over_today=0.01),
        "tests": lose, "steps": {"wins": None, "why": "too small"}})) + "\n")
    os.makedirs(os.path.join(t, "small", "model"))
    io.open(os.path.join(t, "small", "model", "requirements.yaml"), "w", encoding="utf-8").write(
        "requirements:\n" + "".join("  - {id: W-%d, kind: authored, text: x}\n" % i for i in range(4)))
    mid = os.path.join(t, "mid", "kg")
    os.makedirs(mid)
    io.open(os.path.join(mid, "gnn_tasks.jsonl"), "w", encoding="utf-8").write(json.dumps(dict(real_graph(mid, 6), jobs={
        "owner as today": win})) + "\n")
    io.open(os.path.join(big, "survivor_rank.jsonl"), "w", encoding="utf-8").write(json.dumps({"breaks": 1022, "survived": 55, "recall20": {
        "learned": 0.64, "operator prior": 0.25, "random": 0.15}}) + "\n")
    io.open(os.path.join(small, "survivor_rank.jsonl"), "w", encoding="utf-8").write(json.dumps({"breaks": 60, "survived": 3, "recall20": {
        "learned": 0.9, "operator prior": 0.2, "random": 0.2}}) + "\n")
    os.makedirs(os.path.join(t, "ab", "runs", "P-1"))
    io.open(os.path.join(t, "ab", "runs", "P-1", "profile.json"), "w", encoding="utf-8").write(json.dumps({"kind": "bug fix"}))
    io.open(os.path.join(t, "ab", "results.jsonl"), "w", encoding="utf-8").write(json.dumps({"run": "P-1", "better": "off"}) + "\n")
    pb = decide(big, roots=(t,), ab=os.path.join(t, "ab"))
    ps = decide(small, roots=(t,), ab=os.path.join(t, "ab"))
    good = pb["gnn owner"]["on"] and pb["relation caught by"]["on"] and not pb["relation calls"]["on"] and not pb["gnn tests"]["on"] \
        and pb["gnn steps"]["on"] is False and "too small" in pb["gnn steps"]["why"] and pb["break order"]["on"] \
        and not pb["equivalent ranking"]["on"] and "no evidence" in pb["equivalent ranking"]["why"]
    pm = decide(mid, roots=(t,), ab=os.path.join(t, "ab"))
    good = good and not ps["gnn owner"]["on"] and "too small: 4 requirements" in ps["gnn owner"]["why"] and "6 or more" in ps["gnn owner"]["why"]
    good = good and pm["gnn owner"]["on"]          # its own winning evidence beats any floor
    good = good and not ps["break order"]["on"] and "too few examples: 3 survivors" in ps["break order"]["why"]
    var = os.path.join(t, "var", "kg")
    os.makedirs(var)
    io.open(os.path.join(var, "gnn_tasks.jsonl"), "w", encoding="utf-8").write(json.dumps(dict(real_graph(var, 20), jobs={
        "owner as today": lose, "owner + calls": dict(win, adds_over_today=0.04)})) + "\n")
    pv = decide(var, roots=(t,), ab=os.path.join(t, "ab"))
    good = good and pv["gnn owner"]["on"] and "calls" in pv["gnn owner"]["why"] and pv["relation calls"]["on"]
    both = os.path.join(t, "both", "kg")
    os.makedirs(both)
    io.open(os.path.join(both, "gnn_tasks.jsonl"), "w", encoding="utf-8").write(json.dumps(dict(real_graph(both, 20), jobs={
        "owner as today": lose, "owner + both": win})) + "\n")
    pboth = decide(both, roots=(t,), ab=os.path.join(t, "ab"))
    good = good and pboth["gnn owner"]["on"] and "both" in pboth["gnn owner"]["why"]
    # THE EVIDENCE IS ABOUT ITS GRAPH: one requirement added, one claim taken away, or the graph gone, and nothing learned runs on it
    change_graph(mid, "INSERT INTO entity VALUES ('req:R-new', 'requirement', 'R-new', 'authored')")
    pm2 = decide(mid, roots=(t,), ab=os.path.join(t, "ab"))
    fresh = not pm2["gnn owner"]["on"] and "changed since" in pm2["gnn owner"]["why"] and "nodes 8, now 9" in pm2["gnn owner"]["why"] \
        and "requirements 6, now 7" in pm2["gnn owner"]["why"]
    change_graph(var, "DELETE FROM edge WHERE p = 122")
    pv2 = decide(var, roots=(t,), ab=os.path.join(t, "ab"))
    fresh = fresh and not pv2["gnn owner"]["on"] and not pv2["relation calls"]["on"] and "owner_links 1, now 0" in pv2["relation calls"]["why"]
    os.remove(os.path.join(both, "g.sqlite"))
    pb2 = decide(both, roots=(t,), ab=os.path.join(t, "ab"))
    fresh = fresh and not pb2["gnn owner"]["on"] and "is not here" in pb2["gnn owner"]["why"]
    print("selftest: EVIDENCE a score about a graph that has changed or gone switches nothing on, and says what moved %s" % ("PASS" if fresh else "FAIL"))
    good = good and fresh
    good = good and "bug fix: on better 0, off better 1" in pb["when to use the loop"]["why"]
    print(show(pb))
    print("selftest: PLAN a part runs only where it wins by more than %.2f; a relation that adds no more than that is left out; below the "
          "size floor the plain rules run unmeasured; no evidence is off with the reason; the advice reads the A/B board by kind %s" % (
              MARGIN, "PASS" if good else "FAIL"))
    json.dump([{"run": "P-1", "majority": "on"}], io.open(os.path.join(t, "ab", "agreement.json"), "w", encoding="utf-8"))
    panel = "on better 1, off better 0" in when_to_use(os.path.join(t, "ab"))["why"]
    print("selftest: PANEL a pair graded by a panel counts by its majority, not its first grade %s" % ("PASS" if panel else "FAIL"))
    good = good and panel
    shutil.rmtree(t, ignore_errors=True)
    print("selftest: %s" % ("PASS" if good else "FAIL"))
    return 0 if good else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.model:
        ap.error("<model dir> | --selftest")
    kg = os.path.join(os.path.abspath(a.model), "..", "kg")
    plan = decide(os.path.normpath(kg), roots=(os.path.join(HERE, ".."),))
    print("run_plan: %s\n%s" % (os.path.normpath(a.model), show(plan)))
    if a.write:
        io.open(os.path.join(kg, "run_plan.json"), "w", encoding="utf-8").write(json.dumps(plan, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
