#!/usr/bin/env python3
"""run_plan.py - which parts of the loop run for a model, decided by the measured evidence, never by habit.

WHY (2026-10-29): what is better not run is disabled for the run, and a graph too small for a part does not run it, both decided
from the data. Every learned part has a plain rule it must beat (graph_gnn's rankers, the KILLS rule, the operator prior). The epoch
BENCHMARK (explore) still measures every part on every model, so the data can turn a part back on.

A LINE IS A SWITCH ONLY IF A PASS ACTS ON IT (PlanLine, measured 2026-10-30: this header said a pass obeyed all seven lines, and no
tool a pass runs named this file, so two lines flipping by seed alone changed nothing). `closed_loop.py run --gnn` reads `gnn owner`
and both `relation` lines before it rebuilds the graph. The other lines are MEASURED ONLY: `on` is None, `wins` holds what the evidence
says, `why` names what would have to read them. telemetry_audit.py fails a switch no pass reads.

THE EVIDENCE it reads (each written by the tool that measured it; nothing here is assumed):
  <kg>/gnn_tasks.jsonl       gnn_tasks.py eval: each job's learned score against its best plain rule, and the size of the graph it
                             measured: a score about a graph that has since changed switches nothing on (GraphEvidence)
  <kg>/gnn_curve.jsonl and gnn_curve_*.jsonl beside the tools   gnn_curve.py: the GNN against the plain rules by fullness
  <kg>/survivor_rank.jsonl   survivor_rank.py eval: the learned break order against the operator prior
  ~/.claude/ab/results.jsonl with runs/*/profile.json   ab_loop.py: loop on against off, by the kind of task (PK-6)
EVERY SEED (board E1-3c). The plan reads every seed measured on the graph as it is, by the network as it is, and a line is on only if it wins by more than the margin at EVERY one of them. One seed
flatters: on the self model the learned break order won at 9 of 10 seeds and lost at seed 0, so reading the last row flipped with the order the rows were written. Not the mean, since a mean lets three
winning seeds carry a losing one that a pass would then act on; this is the rule the fullness curve already used. A repeat of a seed replaces that seed and adds no vote. A row that names no seed cannot
be told from a repeat or from a new seed, so it is read as a reading of its own. The score a line quotes is the mean over what it read, with the count beside it, never one seed's.
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


class PlanLine(object):
    """One line of the plan: the evidence it rests on, and whether a pass acts on it. Without `waits` the line is a SWITCH: its `on` is
    what a pass does, and telemetry_audit.PlanReaders must find a tool of the pass reading it by name, or the audit fails. With `waits`
    it is MEASURED ONLY: a score with nothing behind it to obey, so `on` is None and `wins` says what the evidence says; `waits` names
    the reader it would take. A stat that only feeds a measured line is WATCHED, however often it is read here: a plan that nobody reads
    decides nothing. Breadcrumb: the evidence names are `<field>` of `<record>.jsonl` as telemetry_audit spells them."""

    def __init__(self, name, evidence, waits=None):
        self.name, self.evidence, self.waits = name, evidence, waits

    def rests_on(self, record, field):
        return field in self.evidence.get(record, ())


SHAPE = ("shape.nodes", "shape.edges", "shape.methods", "shape.owner_links", "shape.requirements")     # GraphEvidence.SIZE, as the audit names them
LINES = (
    PlanLine("gnn owner", {"gnn_tasks": ("jobs.owner as today",) + tuple("jobs." + v for v in OWNER_VARIANTS) + SHAPE,
                           "gnn_curve": ("gnn", "best_mrr", "best", "requirements")}),
    PlanLine("relation caught by", {"gnn_tasks": ("jobs.owner + caught by",) + SHAPE}),
    PlanLine("relation calls", {"gnn_tasks": ("jobs.owner + calls",) + SHAPE}),
    PlanLine("gnn tests", {"gnn_tasks": ("jobs.tests",) + SHAPE}, waits="a pass that runs the tests likely to catch a break first"),
    PlanLine("gnn steps", {"gnn_tasks": ("jobs.steps",) + SHAPE}, waits="a prompt that suggests the steps for a new requirement"),
    PlanLine("equivalent ranking", {"gnn_tasks": ("jobs.equivalent",) + SHAPE},
             waits="a report that lists survivors likeliest equivalent first, once its positives include survivors a person reads"),
    PlanLine("break order", {"survivor_rank": ("breaks", "survived", "recall20.learned", "recall20.operator prior", "recall20.random")},
             waits="a fast pass that runs the breaks likeliest to survive first"),
)


class GraphEvidence(object):
    """A gnn_tasks row and the graph it measured. A job's score is evidence about THAT graph: once a pass or a model edit has changed
    the graph the GNN reads, the score speaks for a graph that no longer exists, so no learned part runs on it until `gnn_tasks.py
    eval` measures again (the plain rule meanwhile; the epoch benchmark re-measures anyway). ANY change counts: no tick has measured
    how much change a score survives, so a tolerance here would be a guess. Breadcrumb: the row's `shape` is gnn_tasks.graph_shape, and
    E1-1 (10-30) gave these sizes their reader: they sat beside every score with nothing comparing them. A score by an older network is in the same case: the network it measured no longer exists."""

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
        if self.row.get("network") != gnn_tasks.gg.NETWORK:
            return "measured by network %s, this is network %s" % (self.row.get("network", "1"), gnn_tasks.gg.NETWORK)
        now = gnn_tasks.graph_shape(os.path.join(self.kg, name))
        was = self.row.get("shape") or {}
        return ", ".join("%s %s, now %s" % (k, was.get(k), now.get(k)) for k in self.SIZE if was.get(k) != now.get(k)) or None


def rows(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]


def readings(all_rows, same, seed_of):
    """the rows about the same thing as the last one (`same`: a key of the data it measured), the latest per seed. A row naming no seed is a reading of its own"""
    if not all_rows:
        return []
    mine = same(all_rows[-1])
    by = collections.OrderedDict()
    for i, r in enumerate(all_rows):
        if same(r) == mine:
            s = seed_of(r)
            by[("row", i) if s is None else ("seed", s)] = r
    return list(by.values())


def task_key(r):
    return (r.get("graph"), r.get("network"), tuple((r.get("shape") or {}).get(k) for k in GraphEvidence.SIZE))


def task_seed(r):
    return tuple(r["seeds"]) if r.get("seeds") else None


def merged(reads, name):
    """one job over every reading: it wins only if it wins at each; its scores are means; `won` and `read` count the seeds. None when no reading has it"""
    js = [r["jobs"][name] for r in reads if name in r.get("jobs", {})]
    if not js:
        return None
    said = [j for j in js if j.get("wins") is not None]
    if not said:
        return js[-1]
    mean = lambda f: sum(j.get(f, 0.0) for j in said) / float(len(said))
    return {"wins": all(j["wins"] for j in said), "learned": mean("learned"), "best_plain": said[-1]["best_plain"], "plain": mean("plain"), "adds_over_today": mean("adds_over_today"),
            "won": sum(1 for j in said if j["wins"]), "read": len(said)}


def at_seeds(j):
    return " at %d of %d seeds" % (j["won"], j["read"]) if j.get("read", 1) > 1 else ""


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
        len(found), g, found[-1].get("best"), b, "wins at every seed" if wins else "does not win at every seed: it stays off")}


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
    reads = readings(own, task_key, task_seed)

    def stale(what):
        return {"on": False, "why": "%s: measured on a graph or by a network that has changed since (%s): gnn_tasks.py eval measures it again; until then it stays off" % (what, moved)}

    def job(name, part, what):
        j = merged(reads, name)
        if j is None:
            plan[part] = {"on": False, "why": "no evidence for %s on this model yet: gnn_tasks.py eval measures it; until then it stays off" % what}
        elif moved:
            plan[part] = stale(what)
        elif j.get("wins") is None:
            plan[part] = {"on": False, "why": "not measurable here: %s" % j.get("why")}
        else:
            plan[part] = {"on": bool(j["wins"]), "why": "%s: learned %.2f against %s %.2f (%s%s)" % (
                what, j["learned"], j["best_plain"], j["plain"], "wins by more than %.2f" % MARGIN if j["wins"] else "does not win: it stays off", at_seeds(j))}

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
    variants = [(n, j) for n, j in ((n, merged(reads, n)) for n in OWNER_VARIANTS) if j and j.get("wins")]
    needs = ()
    if variants and not moved and not plan.get("gnn owner", {}).get("on"):
        n, j = max(variants, key=lambda v: v[1]["learned"])
        needs = ("caught by", "calls") if n == "owner + both" else (n.replace("owner + ", ""),)
        plan["gnn owner"] = {"on": True, "why": "the owner GNN %s: learned %.2f against %s %.2f (wins only hearing the added relation%s)" % (
            n.replace("owner + ", "hearing `").rstrip() + "`", j["learned"], j["best_plain"], j["plain"], "s" if n == "owner + both" else "")}
    for rel, key in (("caught by", "owner + caught by"), ("calls", "owner + calls")):
        j = merged(reads, key)
        if not j or j.get("wins") is None:
            plan["relation " + rel] = {"on": False, "why": "no evidence the GNN does better hearing `%s`" % rel}
        elif moved:
            plan["relation " + rel] = stale("hearing `%s`" % rel)
        else:
            gain = j.get("adds_over_today", 0.0)
            plan["relation " + rel] = {"on": bool(j["wins"]) and gain > MARGIN, "why": "hearing `%s` moves the owner GNN by %+.2f%s" % (
                rel, gain, ("" if j["wins"] else " (the GNN still does not win: left out)") if gain > MARGIN else " (not more than %.2f: left out)" % MARGIN)}
    for rel in needs:         # the win above was measured hearing this relation, so the pass must hear it whatever its gain alone
        plan["relation " + rel] = {"on": True, "why": "the owner GNN wins only hearing `%s`" % rel}
    job("tests", "gnn tests", "which tests catch a branch's breaks")
    job("steps", "gnn steps", "which steps carry a requirement")
    job("equivalent", "equivalent ranking", "which survivors are likely equivalent")
    sr = readings(rows(os.path.join(kg, "survivor_rank.jsonl")), lambda r: (r.get("breaks"), r.get("survived")), lambda r: r.get("seed"))
    if sr and sr[-1].get("survived", 0) < MIN_SURVIVORS:
        plan["break order"] = {"on": False, "why": "too few examples: %d survivors among %d breaks (the learned order needs %d or more): the plain order" % (
            sr[-1].get("survived", 0), sr[-1].get("breaks", 0), MIN_SURVIVORS)}
    elif sr:
        rs = [x["recall20"] for x in sr]
        won = sum(1 for r in rs if r["learned"] > r["operator prior"] + MARGIN and r["learned"] > r["random"] + MARGIN)
        on = won == len(rs)
        mean = lambda f: sum(r[f] for r in rs) / float(len(rs))
        ceiling = rs[-1].get("ceiling")           # the most any order can find in the first fifth: scores on two graphs compare only beside it
        plan["break order"] = {"on": on, "why": "the learned order finds %.0f%% of survivors in the first fifth against the operator prior's %.0f%%%s (%s%s)" % (
            100 * mean("learned"), 100 * mean("operator prior"), "" if ceiling is None else ", and no order could find more than %.0f%%" % (100 * ceiling),
            "ahead by more than %.2f" % MARGIN if on else "not ahead by more than %.2f" % MARGIN, " at %d of %d seeds" % (won, len(rs)) if len(rs) > 1 else "")}
    else:
        plan["break order"] = {"on": False, "why": "no evidence: the breaks run in their plain order"}
    for line in LINES:
        if line.waits:      # the evidence stands; there is nothing to switch until something reads it
            said = plan[line.name]
            plan[line.name] = {"on": None, "wins": bool(said["on"]), "why": "%s; measured only, no pass acts on this line: it would take %s" % (said["why"], line.waits)}
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
        import graph_gnn as gg
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
        return {"graph": "g.sqlite", "shape": gnn_tasks.graph_shape(db), "network": gg.NETWORK}

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
    good = pb["gnn owner"]["on"] and pb["relation caught by"]["on"] and not pb["relation calls"]["on"] and not pb["gnn tests"]["wins"] \
        and pb["gnn steps"]["wins"] is False and "too small" in pb["gnn steps"]["why"] and pb["break order"]["wins"] \
        and not pb["equivalent ranking"]["wins"] and "no evidence" in pb["equivalent ranking"]["why"]
    pm = decide(mid, roots=(t,), ab=os.path.join(t, "ab"))
    good = good and not ps["gnn owner"]["on"] and "too small: 4 requirements" in ps["gnn owner"]["why"] and "6 or more" in ps["gnn owner"]["why"]
    good = good and pm["gnn owner"]["on"]          # its own winning evidence beats any floor
    good = good and not ps["break order"]["wins"] and "too few examples: 3 survivors" in ps["break order"]["why"]
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
    # A SWITCH IS WHAT A PASS DOES: a win hearing a relation turns on the relation it was measured hearing, even with no gain written alone
    heard = pboth["relation calls"]["on"] and pboth["relation caught by"]["on"] and pv["relation calls"]["on"] and not pv["relation caught by"]["on"]
    # A MEASURED LINE HAS NOTHING TO SWITCH: on is None even where its evidence wins, and it says what it waits for; a switch is True or False
    lines = {l.name: l for l in LINES}
    measured = pb["break order"]["on"] is None and pb["break order"]["wins"] is True and "no pass acts" in pb["break order"]["why"] \
        and all(pb[n]["on"] is None and lines[n].waits in pb[n]["why"] for n, l in lines.items() if l.waits) \
        and all(pb[n]["on"] in (True, False) and "wins" not in pb[n] for n, l in lines.items() if not l.waits)
    print("selftest: LINES a win hearing a relation turns that relation on; a line no pass reads is measured only, on None, its evidence in wins "
          "and what it waits for in its reason; a switch is on or off %s" % ("PASS" if heard and measured else "FAIL"))
    good = good and heard and measured
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
    good = good and every_seed_selftest(t, real_graph, win, lose)
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


def every_seed_selftest(t, real_graph, win, lose):
    """a plant for each flip of the rule: order does not matter, one loss turns it off, a repeat replaces, an older graph and an older network do not count, and a row naming no seed still votes"""
    ok = True
    heads = {}
    import gnn_tasks
    now = gnn_tasks.gg.NETWORK

    def plan(name, jobs, seeds, network=now):
        kg = os.path.join(t, name, "kg")
        if name not in heads:
            os.makedirs(kg)
            heads[name] = real_graph(kg, 20)
        with io.open(os.path.join(kg, "gnn_tasks.jsonl"), "a", encoding="utf-8") as f:
            for s, j in zip(seeds, jobs):
                f.write(json.dumps(dict(heads[name], seeds=s, jobs={"owner as today": j}, network=network)) + "\n")
        return decide(kg, roots=(os.path.join(t, name),), ab=os.path.join(t, "ab"))["gnn owner"]

    def say(what, good):
        nonlocal ok
        ok = ok and good
        print("selftest: EVERY SEED %s %s" % (what, "PASS" if good else "FAIL"))
    say("two losing seeds then a winning one stay off, in any order", not plan("o1", [lose, lose, win], [[0], [1], [2]])["on"] and not plan("o2", [win, lose, lose], [[0], [1], [2]])["on"])
    said = plan("o3", [win, win, win], [[0], [1], [2]])
    say("wins at every seed is on, and quotes the count and the mean", said["on"] and "at 3 of 3 seeds" in said["why"] and "0.77" in said["why"])
    say("one winning row then a new losing seed flips it off", plan("o4", [win], [[0]])["on"] and not plan("o4", [lose], [[1]])["on"])
    say("a repeat of a seed replaces it and adds no vote: a loss then the same seed's win is on", plan("o5", [lose, win], [[0], [0]])["on"])
    say("rows naming no seed each vote", not plan("o6", [lose, win], [None, None])["on"])
    # an older network: the last row is by this network, so the older row does not count; a last row by an older network switches nothing
    say("a loss by an older network does not count against this one", (plan("o7", [lose], [[0]], network=now - 1), plan("o7", [win], [[1]], network=now))[1]["on"])
    old = plan("o8", [win], [[0]], network=now - 1)
    say("a win by an older network switches nothing on", not old["on"] and "network" in old["why"])

    def order(name, rs):
        kg = os.path.join(t, name, "kg")
        os.makedirs(kg)
        io.open(os.path.join(kg, "survivor_rank.jsonl"), "w", encoding="utf-8").write("\n".join(json.dumps(r) for r in rs) + "\n")
        return decide(kg, roots=(os.path.join(t, name),), ab=os.path.join(t, "ab"))["break order"]

    def row(learned, seed, breaks=1000):
        return {"breaks": breaks, "survived": 55, "seed": seed, "recall20": {"learned": learned, "operator prior": 0.3, "random": 0.2, "ceiling": 0.8}}
    say("break order: the margin is 0.02, 0.015 ahead is off and 0.025 ahead is on", order("b4", [row(0.315, 0)])["wins"] is False and order("b5", [row(0.325, 0)])["wins"] is True)
    say("break order: one losing seed of three is off, and says 2 of 3", "at 2 of 3 seeds" in order("b1", [row(0.6, 0), row(0.1, 1), row(0.6, 2)])["why"] and order("b2", [row(0.6, 0), row(0.1, 1), row(0.6, 2)])["wins"] is False)
    say("break order: a loss on older verdicts does not count, and the ceiling is quoted", (lambda p: p["wins"] is True and "80%" in p["why"])(order("b3", [row(0.1, 0, breaks=900), row(0.6, 0), row(0.6, 1)])))
    return ok


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
