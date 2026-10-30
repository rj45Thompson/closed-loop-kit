#!/usr/bin/env python3
"""gnn_more.py - what to ADD to the graph so the GNN learns something it cannot today, measured before anything is kept.

WHY (2026-10-29): what else could a GNN be used for, and what could be added to the node graph so that a GNN could test it?
Looking at the shops graph: its largest relation, `killed by` (5,664 breaks -> the tests that caught them), and `checked by`
(requirement -> the tests that name it) are LEFT OUT of what the GNN hears (graph_gnn.LEFT_OUT: "a new branch has none") - yet a
pass runs every covered branch's breaks BEFORE an owner is proposed, so new code does have them. And code has no edge to code:
the 50 methods are islands (a chain step citing two methods that call each other cannot be joined, P5-E 10-29). Two additions:

  CAUGHT BY   cell -> the tests that caught its breaks (cell -mutated as-> mutant -killed by-> test), and the tests' `checked by`
              edges from the requirements they name: new code reaches a requirement through the tests that already guard it
  CALLS       method -> method, read off the statements the graph already holds (a call to a method name that is one code place's
              alone); only between covered methods

and one plain rule they make possible, which the GNN must also beat: KILLS - rank a cell's requirements by how many of its breaks
were caught by tests that name them (the novelty review's "kill vote", NV-2).

Measured the way graph_gnn eval is: whole methods held out, every baseline and the GNN on the same folds; each addition alone and
together. Nothing here is written to the graph; a relation is added to usecase_kg only after it wins here.

    python -X utf8 gnn_more.py eval <graph.sqlite> [more ...] [--seeds 0,1] [--threads 2]
    python -X utf8 gnn_more.py --selftest
"""
import argparse
import collections
import os
import re
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import graph_gnn as gg  # noqa: E402

CAUGHT_BY, CALLS, CHECKED_BY = 902, 903, 149
_group_of = gg.group_of


def group_of(p):
    """the new relations are CODE evidence (what tests and calls say about the code), never claims"""
    return "code" if p in (CAUGHT_BY, CALLS, CHECKED_BY) else _group_of(p)


class RichView(gg.GraphView):
    """the graph as graph_gnn reads it, plus the relations asked for. `kill_votes[cell]`: the requirements whose tests caught the
    cell's breaks, by how many breaks - kept for every view, so the KILLS rule exists whether or not the GNN hears the edges"""

    def __init__(self, db_path, kills=False, calls=False):
        gg.GraphView.__init__(self, db_path)
        db = sqlite3.connect(db_path)
        mut_of = collections.defaultdict(list)
        for s, o in db.execute("SELECT s, o FROM edge WHERE p = 147"):
            mut_of[s].append(o)
        caught = collections.defaultdict(set)
        for s, o in db.execute("SELECT s, o FROM edge WHERE p = 148"):
            caught[s].add(o)
        checks = collections.defaultdict(set)
        for s, o in db.execute("SELECT s, o FROM edge WHERE p = ?", (CHECKED_BY,)):
            checks[o].add(s)
        self.kill_votes = {}
        for c in self.cells:
            v = collections.Counter()
            for m in mut_of.get(c, ()):
                for t in caught.get(m, ()):
                    for r in checks.get(t, ()):
                        v[r] += 1
            self.kill_votes[c] = v
        self.added = collections.Counter()
        if kills:
            tests = sorted({t for ts in caught.values() for t in ts} | {t for t in checks})
            for t in tests:
                if t not in self.index:
                    self.index[t] = len(self.ids)
                    self.ids.append(t)
                    self.kind[t] = "test"
                    self.text[t] = " ".join(re.findall(r"[A-Z][a-z]+|[a-z]+", t[5:]))
            for c in self.cells:
                for t in sorted({t for m in mut_of.get(c, ()) for t in caught.get(m, ())}):
                    self.edges.append((c, t, CAUGHT_BY))
                    self.added["caught by"] += 1
            for t, rs in checks.items():
                for r in rs:
                    if r in self.index and t in self.index:
                        self.edges.append((r, t, CHECKED_BY))
                        self.added["checked by"] += 1
        if calls:
            for a, b in sorted(self.call_edges()):
                self.edges.append((a, b, CALLS))
                self.added["calls"] += 1
        db.close()

    def call_edges(self):
        """(caller, callee) code places: a cell of the caller's whose statement calls a name that is ONE code place's method alone"""
        codes = [e for e in self.ids if self.kind.get(e) == "code" and "." in e[5:]]
        by_name = collections.defaultdict(list)
        for e in codes:
            by_name[e[5:].rsplit(".", 1)[1]].append(e)
        unique = {n: es[0] for n, es in by_name.items() if len(es) == 1 and len(n) >= 4}
        out = set()
        for c in self.cells:
            caller = self.place.get(c)
            # any case: C# methods are capitalised, Python functions are not (10-29: the self model got no call edge at all)
            for n in set(re.findall(r"\b([A-Za-z_]\w{3,})\s*\(", self.text.get(c, ""))):
                callee = unique.get(n)
                if callee and caller and callee != caller and caller in self.index:
                    out.add((caller, callee))
        return out


def kills(self, cell, visible):
    """KILLS: the requirements whose tests caught the cell's breaks, most first, then the effects rule - no learning"""
    v = self.v.kill_votes.get(cell, {})
    at = self.v.effects_at.get(self.v.place.get(cell), {})
    return sorted(self.v.reqs, key=lambda r: (-v.get(r, 0), -at.get(r, 0), -self.cos(cell, r), r))


def evaluate(db, seeds=(0, 1), folds=5, epochs=300, say=print):
    gg.group_of = group_of
    gg.Baselines.kills = kills
    if "kills" not in gg.BASELINES:
        gg.BASELINES = gg.BASELINES + ("kills",)
    rows = {}
    for name, kw in (("as today", {}), ("+ caught by", {"kills": True}), ("+ calls", {"calls": True}), ("+ both", {"kills": True, "calls": True})):
        v = RichView(db, **kw)
        res = [gg.evaluate(v, folds=folds, seed=s, epochs=epochs, group="method") for s in seeds]
        mrr = {k: sum(r[k]["mrr"] for r in res) / len(res) for k in gg.BASELINES + ("gnn",)}
        rows[name] = mrr
        best = max(gg.BASELINES, key=lambda b: mrr[b])
        say("  %-12s added %-34s GNN MRR %.2f; plain rules: %s; best plain %s %.2f -> %s" % (
            name, ", ".join("%s %d" % kv for kv in sorted(v.added.items())) or "nothing", mrr["gnn"],
            ", ".join("%s %.2f" % (b, mrr[b]) for b in gg.BASELINES if b != "random"), best, mrr[best],
            "the GNN BEATS it" if mrr["gnn"] > mrr[best] else "the GNN does not beat it"))
    return rows


def selftest():
    """a planted graph where owners are learnable ONLY through the tests that catch the breaks: the KILLS rule must find them and
    the words must not; the call edges join only distinct covered methods with unique names; nothing is added unless asked"""
    import tempfile
    d = tempfile.mkdtemp(prefix="gnn_more_selftest_")
    p = os.path.join(d, "g.sqlite")
    db = sqlite3.connect(p)
    db.executescript("CREATE TABLE entity(id TEXT PRIMARY KEY, kind TEXT, label TEXT, sub TEXT);"
                     "CREATE TABLE edge(s TEXT, p INTEGER, o TEXT, fam INTEGER);"
                     "CREATE TABLE literal(s TEXT, p INTEGER, v TEXT);")
    reqs = ["req:R-%d" % i for i in range(4)]
    for r in reqs:
        db.execute("INSERT INTO entity VALUES (?, 'requirement', ?, 'authored')", (r, r[4:]))
        db.execute("INSERT INTO entity VALUES (?, 'test', ?, NULL)", ("test:Checks" + r[4:].replace("-", ""), "t"))
        db.execute("INSERT INTO edge VALUES (?, 149, ?, 0)", (r, "test:Checks" + r[4:].replace("-", "")))
    for m in range(12):
        place, owner = "code:T.Method%d" % m, reqs[m % 4]
        db.execute("INSERT INTO entity VALUES (?, 'code', ?, NULL)", (place, place[5:]))
        for b in range(3):
            c = "cell:S-BRANCH:T.Method%d#b%d" % (m, b)
            db.execute("INSERT INTO entity VALUES (?, 'cell', ?, 'S-BRANCH')", (c, c[14:]))
            callee = "Method%d" % ((m + 1) % 12)
            db.execute("INSERT INTO literal VALUES (?, 133, ?)", (c, "x = %s(y)" % callee))
            db.execute("INSERT INTO edge VALUES (?, 152, ?, 32)", (c, place))
            db.execute("INSERT INTO edge VALUES (?, 122, ?, 32)", (c, owner))
            mu = "mut:%s#%d" % (place[5:], b)
            db.execute("INSERT INTO entity VALUES (?, 'mutant', 'm', NULL)", (mu,))
            db.execute("INSERT INTO edge VALUES (?, 147, ?, 0)", (c, mu))
            db.execute("INSERT INTO edge VALUES (?, 148, ?, 0)", (mu, "test:Checks" + owner[4:].replace("-", "")))
    db.commit()
    db.close()
    ok = True
    plain = RichView(p)
    rich = RichView(p, kills=True, calls=True)
    gg.Baselines.kills = kills
    base = gg.Baselines(plain)
    hits = sum(1 for c in plain.cells if base.kills(c, [])[0] in {r for cc, r in plain.links if cc == c})
    words = sum(1 for c in plain.cells if base.words(c, [])[0] in {r for cc, r in plain.links if cc == c})
    good = hits == len(plain.cells) and words < len(plain.cells) / 2 and not plain.added
    ok &= good
    print("selftest: KILLS owners reachable only through the tests that caught the breaks are found by the kill rule, not by the "
          "words, and nothing is added unless asked %s" % ("PASS" if good else "FAIL"))
    calls = rich.call_edges()
    good = len(calls) == 12 and all(a != b for a, b in calls) and rich.added["caught by"] == 36 and rich.added["checked by"] == 4 \
        and all(rich.kind[t] == "test" for t in rich.ids if t.startswith("test:"))
    ok &= good
    print("selftest: ADDED each method calls the next (12 call edges, none to itself), 36 caught-by and 4 checked-by edges, the tests "
          "heard as test nodes %s" % ("PASS" if good else "FAIL"))
    print("selftest: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["eval"])
    ap.add_argument("db", nargs="*")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd != "eval" or not a.db:
        ap.error("eval <graph.sqlite> [...] | --selftest")
    gg.torch.set_num_threads(a.threads)
    for db in a.db:
        print("gnn_more: %s (whole methods held out, MRR averaged over seeds %s)" % (os.path.basename(db), a.seeds))
        evaluate(db, tuple(int(x) for x in a.seeds.split(",")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
