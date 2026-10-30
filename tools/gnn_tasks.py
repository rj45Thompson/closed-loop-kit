#!/usr/bin/env python3
"""gnn_tasks.py - every job the graph can do for the loop, each measured against the plain rule it must beat, each run leaving a row
of EVIDENCE that run_plan.py reads to switch a part on or off.

WHY (2026-10-29): everything a GNN could do goes into the graph, with stats for whatever could improve it. What is better not run
is disabled for the run, and a graph too small for a part does not run it, both decided from the data.

THE JOBS (board section MORE FOR THE GNN). Every one is scored only on what it never saw, against plain rules that do not learn:
  owner       GN-1  which requirement owns a branch (graph_gnn's job), with the new relations `caught by` and `calls` (gnn_more)
  tests       GN-2  which tests will catch a break in this branch: run them first, hand them to the agent changing that code
  steps       GN-4  which use-case steps should carry a requirement (model completion), whole use cases held out
  equivalent  GN-5  of the breaks no test caught, which are likely to change nothing (proven ones are the examples): the owner reads
                    the real holes first. A logistic model over the break's words - not a GNN, and named so
  cochange    GN-6  RECORDED, not yet learnable: which methods' statements changed between two passes (no method-level history exists
                    in the repo; the loop's own fingerprints start it now)
  impact      GN-3  RECORDED, not yet learnable: for each change measured with --since, the methods changed and the requirements whose
                    holes it ADDED - the ground truth an impact predictor will be scored on

    python -X utf8 gnn_tasks.py eval <graph.sqlite> [--verdicts V.json] [--seeds 0] [--jobs owner,tests,steps,equivalent]
    python -X utf8 gnn_tasks.py record <graph.sqlite>            the cochange snapshot (GN-6)
    python -X utf8 gnn_tasks.py impact <graph.sqlite> <before.md> <after.md>   the impact ground truth (GN-3)
    python -X utf8 gnn_tasks.py --selftest
Evidence rows: <graph folder>/gnn_tasks.jsonl (one per eval: the graph's size and pillars, then each job's learned and plain scores).
"""
import argparse
import collections
import hashlib
import io
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import graph_gnn as gg  # noqa: E402
import gnn_more as gm  # noqa: E402
import loop_stats  # noqa: E402

MARGIN = 0.02            # a learned job "wins" only by more than this over the best plain rule (graph_gnn's tune margin)


class TaskView(gm.RichView):
    """the graph re-aimed at another job: `cells` are the job's sources, `reqs` its candidate answers, `links` the true pairs, `place`
    the unit a fold holds out whole. The true pairs of held-out sources are hidden from the message passing, and so is whatever else
    a new source would not have yet (`blind`: e.g. a new branch has no owner either)"""

    def aim(self, cells, targets, links, place, target_ps, blind_ps=(), effects=None):
        self.cells, self.reqs, self.links = sorted(cells), sorted(targets), sorted(links)
        self.place = dict(place)
        self.target_ps, self.blind_ps = set(target_ps), set(blind_ps)
        self.effects_at = effects or collections.defaultdict(collections.Counter)
        self.kill_votes = {}
        return self

    def group(self, p):
        return "claims" if p in self.target_ps else gm.group_of(p)

    def adjacency(self, hidden=()):
        hide = set(hidden)
        out_src = {c for c, _ in hide}
        n = len(self.ids)
        edges = self.edges + [e for name in sorted(self.add) for e in self.derived(name)]
        mats = []
        for g in gg.RELATION_GROUPS:
            a = gg.torch.zeros(n, n)
            for s, o, p in edges:
                if (p in self.target_ps and (s, o) in hide) or (s in out_src and p in self.blind_ps) or p in self.drop or self.group(p) != g:
                    continue
                a[self.index[s], self.index[o]] = a[self.index[o], self.index[s]] = 1.0
            mats.append(a / a.sum(1, keepdim=True).clamp(min=1.0))
        return mats


def popular(self, cell, visible):
    """POPULAR: the answers most often true in the training pairs - the plain rule for a job whose answers repeat (a few tests catch
    most breaks)"""
    c = collections.Counter(r for _, r in visible)
    return sorted(self.v.reqs, key=lambda r: (-c[r], -self.cos(cell, r), r))


def run_eval(view, seeds, epochs, extra=()):
    """{ranker: MRR} averaged over seeds, folded by `place`; `extra` plain rules joined to graph_gnn's for this job only"""
    saved = gg.BASELINES
    gg.BASELINES = tuple(b for b in saved if b != "kills") + tuple(extra)
    gg.Baselines.popular = popular
    try:
        res = [gg.evaluate(view, folds=min(5, len({view.place.get(c) for c, _ in view.links})), seed=s, epochs=epochs, group="method")
               for s in seeds]
        names = gg.BASELINES + ("gnn",)
    finally:
        gg.BASELINES = saved
    return {k: sum(r[k]["mrr"] for r in res) / len(res) for k in names}


def job_tests(db, seeds, epochs):
    """GN-2: for each branch, the tests that caught its breaks; a new method's branches have neither tests nor an owner yet"""
    v = TaskView(db, kills=True, calls=True)
    caught = [(s, o) for s, o, p in v.edges if p == gm.CAUGHT_BY]
    srcs = {c for c, _ in caught}
    tests = sorted({t for _, t in caught})
    owners = collections.defaultdict(set)
    for s, o, p in v.edges:
        if p == gm.CHECKED_BY:
            owners[o].add(s)
    eff = collections.defaultdict(collections.Counter)            # the tests that name the requirements whose effects sit here
    for place, reqs in gm.RichView(db).effects_at.items():
        for r in reqs:
            for t in tests:
                if r in owners.get(t, ()):
                    eff[place][t] += 1
    v.aim(srcs, tests, caught, v.place, {gm.CAUGHT_BY}, blind_ps={122}, effects=eff)
    if len(srcs) < 10 or len(tests) < 3:
        return {"skipped": "too small: %d branches with a catching test, %d tests" % (len(srcs), len(tests))}
    return run_eval(v, seeds, epochs, extra=("popular",))


def job_steps(db, seeds, epochs):
    """GN-4: the requirements a use-case step (or extension) carries; a new use case's steps carry nothing yet"""
    v = TaskView(db)
    import sqlite3
    con = sqlite3.connect(db)
    uc_of = {}
    for s, o in con.execute("SELECT s, o FROM edge WHERE p IN (100, 116)"):
        uc_of[o] = s
    pairs = [(s, o) for s, o in con.execute("SELECT s, o FROM edge WHERE p IN (107, 108)") if s in v.index and o in v.index]
    con.close()
    srcs = {s for s, _ in pairs}
    if len({uc_of.get(s) for s in srcs}) < 3 or len(pairs) < 10:
        return {"skipped": "too small: %d step links in %d use cases" % (len(pairs), len({uc_of.get(s) for s in srcs}))}
    v.aim(srcs, v.reqs, pairs, uc_of, {107, 108})
    return run_eval(v, seeds, epochs)


def job_equivalent(verdicts):
    """GN-5: of the breaks no test caught, the ones proven to change nothing (the compiled code is the same) are the positives"""
    import survivor_rank as sr
    m = json.load(io.open(verdicts, encoding="utf-8"))["mutants"]
    data = []
    for r in m.values():
        why = str(r.get("why") or "")
        if why.startswith("equivalent"):
            data.append((sr.features(r), 1, str(r.get("place"))))
        elif why.startswith("every test passed"):
            data.append((sr.features(r), 0, str(r.get("place"))))
    if sum(y for _, y, _ in data) < 5:
        return {"skipped": "too few proven equivalents (%d)" % sum(y for _, y, _ in data)}
    res, n, pos = sr.by_method(data)
    return {"learned": res[20]["learned"], "operator prior": res[20]["operator prior"], "random": res[20]["random"], "n": n, "positives": pos,
            "measure": "share of the proven equivalents in the first 20% of the ranked survivors"}


def graph_shape(db):
    v = gm.RichView(db)
    return dict({"nodes": len(v.ids), "edges": len(v.edges), "requirements": len(v.reqs), "methods": len({v.place.get(c) for c, _ in v.links}),
                 "owner_links": len(v.links)}, **loop_stats.owner_pillars(db))


def verdict(row, learned="gnn"):
    """a job's outcome in the evidence: does the learned ranker beat every plain rule by more than MARGIN"""
    if "skipped" in row:
        return {"wins": None, "why": row["skipped"]}
    plain = {k: v for k, v in row.items() if k not in (learned, "random", "n", "positives", "measure") and isinstance(v, float)}
    best = max(plain, key=plain.get)
    return {"wins": row[learned] > plain[best] + MARGIN, "learned": round(row[learned], 3), "best_plain": best, "plain": round(plain[best], 3)}


def evaluate(db, verdicts=None, jobs=("owner", "tests", "steps", "equivalent"), seeds=(0,), epochs=300, say=print):
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "graph": os.path.basename(db), "shape": graph_shape(db), "jobs": {}}
    say("gnn_tasks: %s - %s" % (row["graph"], ", ".join("%s %s" % kv for kv in row["shape"].items())))
    if "owner" in jobs:
        variants = gm.evaluate(db, seeds=seeds, epochs=epochs, say=lambda s: say("  owner " + s.strip()))
        for name, mrr in variants.items():
            row["jobs"]["owner " + name] = verdict(mrr)
        base = variants["as today"]["gnn"]
        for name in ("+ caught by", "+ calls", "+ both"):
            row["jobs"]["owner " + name]["adds_over_today"] = round(variants[name]["gnn"] - base, 3)
    for name, fn in (("tests", job_tests), ("steps", job_steps)):
        if name in jobs:
            r = fn(db, seeds, epochs)
            row["jobs"][name] = verdict(r)
            say("  %-10s %s" % (name, json.dumps(row["jobs"][name])))
    if "equivalent" in jobs and verdicts:
        r = job_equivalent(verdicts)
        row["jobs"]["equivalent"] = verdict(r, learned="learned") if "skipped" not in r else {"wins": None, "why": r["skipped"]}
        say("  %-10s %s" % ("equivalent", json.dumps(row["jobs"]["equivalent"])))
    out = os.path.join(os.path.dirname(os.path.abspath(db)), "gnn_tasks.jsonl")
    io.open(out, "a", encoding="utf-8").write(json.dumps(row, sort_keys=True) + "\n")
    say("gnn_tasks: evidence row -> %s" % out)
    return row


def method_prints(db):
    """{method: a fingerprint of its statements as the graph holds them}: the loop's own record of what each method says"""
    v = gm.RichView(db)
    by = collections.defaultdict(list)
    for c in v.cells:
        by[v.place.get(c)].append(v.text.get(c, ""))
    return {m: hashlib.sha1("\n".join(sorted(ts)).encode("utf-8")).hexdigest()[:12] for m, ts in by.items() if m}


def record_cochange(db):
    """GN-6: compare this pass's method fingerprints with the last snapshot; the methods that changed between them changed TOGETHER"""
    kg = os.path.dirname(os.path.abspath(db))
    snap, log = os.path.join(kg, "method_prints.json"), os.path.join(kg, "cochange.jsonl")
    now = method_prints(db)
    if os.path.exists(snap):
        was = json.load(io.open(snap, encoding="utf-8"))
        changed = sorted(m for m in set(now) | set(was) if now.get(m) != was.get(m))
        io.open(log, "a", encoding="utf-8").write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "changed": changed}) + "\n")
    else:
        changed = None
    io.open(snap, "w", encoding="utf-8").write(json.dumps(now, sort_keys=True, indent=0))
    return changed


def record_impact(db, before, after):
    """GN-3: the ground truth an impact predictor will be scored on - the methods changed (cochange) and the requirements whose holes
    the change ADDED (closed_loop --since)"""
    import closed_loop as cl
    b = json.load(io.open(before + ".holes.json", encoding="utf-8"))["ours"]
    a = json.load(io.open(after + ".holes.json", encoding="utf-8"))["ours"]
    added, closed, _ = cl.hole_delta(b, a)
    reqs = sorted({h.split(" ", 1)[1] for h in added if " req:" in " " + h})
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "added": added, "closed": closed, "requirements_hit": reqs,
           "changed": record_cochange(db)}
    io.open(os.path.join(os.path.dirname(os.path.abspath(db)), "impact.jsonl"), "a", encoding="utf-8").write(json.dumps(row) + "\n")
    return row


def selftest():
    """each job is aimed at its own pairs and hides them for held-out sources; the verdict needs more than MARGIN; a graph too small
    for a job is skipped with its reason, never scored; the cochange record names exactly the methods whose statements changed"""
    import tempfile
    import shutil
    import sqlite3
    ok = True
    d = tempfile.mkdtemp(prefix="gnn_tasks_selftest_")
    p = os.path.join(d, "g.sqlite")
    con = sqlite3.connect(p)
    con.executescript("CREATE TABLE entity(id TEXT PRIMARY KEY, kind TEXT, label TEXT, sub TEXT);"
                      "CREATE TABLE edge(s TEXT, p INTEGER, o TEXT, fam INTEGER);"
                      "CREATE TABLE literal(s TEXT, p INTEGER, v TEXT);")
    for m in range(6):
        con.execute("INSERT INTO entity VALUES (?, 'code', ?, NULL)", ("code:T.M%d" % m, "T.M%d" % m))
        for b in range(2):
            c = "cell:S-BRANCH:T.M%d#b%d" % (m, b)
            con.execute("INSERT INTO entity VALUES (?, 'cell', ?, 'S-BRANCH')", (c, c[14:]))
            con.execute("INSERT INTO literal VALUES (?, 133, ?)", (c, "x = %d" % (m * 10 + b)))
            con.execute("INSERT INTO edge VALUES (?, 152, ?, 32)", (c, "code:T.M%d" % m))
    con.execute("INSERT INTO entity VALUES ('req:R-1', 'requirement', 'R-1', 'authored')")
    con.commit()
    con.close()
    v = TaskView(p).aim({"cell:S-BRANCH:T.M0#b0", "cell:S-BRANCH:T.M1#b0"}, ["req:R-1"],
                        [("cell:S-BRANCH:T.M0#b0", "req:R-1"), ("cell:S-BRANCH:T.M1#b0", "req:R-1")],
                        {"cell:S-BRANCH:T.M0#b0": "code:T.M0", "cell:S-BRANCH:T.M1#b0": "code:T.M1"}, {999})
    v.edges += [("cell:S-BRANCH:T.M0#b0", "req:R-1", 999), ("cell:S-BRANCH:T.M1#b0", "req:R-1", 999)]
    full = v.adjacency()[gg.RELATION_GROUPS.index("claims")]
    part = v.adjacency([("cell:S-BRANCH:T.M0#b0", "req:R-1")])[gg.RELATION_GROUPS.index("claims")]
    good = int((full > 0).sum()) == 4 and int((part > 0).sum()) == 2
    good = good and verdict({"gnn": 0.71, "effects": 0.70, "random": 0.1})["wins"] is False \
        and verdict({"gnn": 0.75, "effects": 0.70, "random": 0.1})["wins"] is True and verdict({"skipped": "too small"})["wins"] is None
    small = job_steps(p, (0,), 1)
    good = good and "skipped" in small
    ok &= good
    print("selftest: AIM a job hides its own true pairs for a held-out source and nothing else; a win needs more than the margin; a "
          "graph too small for a job is skipped with its reason %s" % ("PASS" if good else "FAIL"))
    first = record_cochange(p)
    con = sqlite3.connect(p)
    con.execute("UPDATE literal SET v = 'x = 999' WHERE s = 'cell:S-BRANCH:T.M3#b1'")
    con.commit()
    con.close()
    second = record_cochange(p)
    good = first is None and second == ["code:T.M3"]
    ok &= good
    print("selftest: COCHANGE the first snapshot records nothing; the next names exactly the method whose statements changed %s" % ("PASS" if good else "FAIL"))
    shutil.rmtree(d, ignore_errors=True)
    print("selftest: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["eval", "record", "impact"])
    ap.add_argument("db", nargs="?")
    ap.add_argument("more", nargs="*")
    ap.add_argument("--verdicts")
    ap.add_argument("--jobs", default="owner,tests,steps,equivalent")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.db:
        ap.error("eval | record | impact <graph.sqlite> ..., or --selftest")
    gg.torch.set_num_threads(a.threads)
    if a.cmd == "eval":
        evaluate(a.db, a.verdicts, tuple(a.jobs.split(",")), tuple(int(x) for x in a.seeds.split(",")), a.epochs)
    elif a.cmd == "record":
        print("gnn_tasks: changed since the last snapshot: %s" % record_cochange(a.db))
    elif a.cmd == "impact":
        print("gnn_tasks: %s" % json.dumps(record_impact(a.db, a.more[0], a.more[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
