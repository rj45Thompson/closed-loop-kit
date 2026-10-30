#!/usr/bin/env python3
"""gnn_curve.py - how full must the graph be before the GNN beats the rankers that do not learn?

WHY (2026-10-29): how full must the graph be before the GNN works? Every stat needs a baseline, or nobody can say that using it is
better than not using it at all. graph_gnn.py eval answers "does it beat them on THIS
graph, as it is"; this answers the question underneath: AT WHAT FULLNESS does it start to, so a project knows whether its graph is
there yet.

FULLNESS is the share of the graph's known `maps to` links (which requirement owns which branch) that the rankers may see: a young
project has claimed few of its methods, a mature one most. At each fullness f the SAME held-out methods (folded by METHOD: new code,
whose siblings own nothing yet) are ranked by the GNN and by every baseline (graph_gnn.Baselines: random, words, method, effects,
combined), each trained on the first f of the other methods' links in one fixed shuffle (so f=0.5 holds everything f=0.25 did);
the links not kept are removed from the graph and the training alike, as if never claimed.

THE ANSWER for a graph is its CROSSOVER: the least f from which the GNN's MRR beats the best baseline's at every seed, at every
larger f too. None = it never earns its place on this graph (and a graph too small to fold says so, never a number).

    python -X utf8 gnn_curve.py <graph.sqlite> [more ...] [--fractions 0.1,0.25,0.5,0.75,1] [--seeds 0,1]
                                                [--folds 5] [--epochs 300] [--threads 2]
    python -X utf8 gnn_curve.py --selftest

Each row goes to <graph folder>/gnn_curve.jsonl (one per graph, fraction and seed), beside the graph it measured.

SEEDED BY REQUIREMENT (--by pillar|spread|random): the loop needs as many requirements seeded as possible before it is fully useful,
which ones matters because some are pillars, and the open question is the smallest set that could expand into the full closed
loop. The fullness unit becomes a REQUIREMENT: only the first f of
the requirements, in an ORDER, have their claims seen; every requirement still exists (its words, its effects), so the rankers must
find the owners of new code - including owners whose claims were never seen - from the seed alone. Orders: PILLAR (the requirements
that own the most branches first), SPREAD (those whose branches span the most methods first), RANDOM. If a pillar order reaches the
full graph's MRR with fewer requirements than random, the pillars carry the graph, and the least seed that gets there is the
smallest set this graph expands from. Each order is computed from the training methods only (never the held-out ones).
"""
import argparse
import collections
import io
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import graph_gnn as gg  # noqa: E402

FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)


def split(view, folds, seed):
    """[(test methods, the other methods in one fixed shuffle)] per fold - the shuffle is what makes fullness nested"""
    units = sorted({view.place.get(c) for c, _ in view.links})
    if len(units) < folds:
        raise ValueError("%d method(s) with claimed branches, fewer than %d folds: too small to fold, no curve" % (len(units), folds))
    rnd = random.Random(seed)
    rnd.shuffle(units)
    out = []
    for k in range(folds):
        test = set(units[k::folds])
        rest = [u for u in units if u not in test]
        random.Random(seed * 1000 + k).shuffle(rest)
        out.append((test, rest))
    return out


def kept(rest, f):
    """the first f of the other methods (at least one): nested, so a fuller graph holds every link a thinner one did"""
    return set(rest[:max(1, int(round(f * len(rest))))])


ORDERS = ("pillar", "spread", "random")


def req_order(view, rest, by, rnd):
    """the requirements that own a branch in the training methods, in the seed ORDER: pillar = most branches owned first, spread =
    most methods spanned first (ties by id), random = one seeded shuffle"""
    owned, spans = collections.Counter(), collections.defaultdict(set)
    rest = set(rest)
    for c, r in view.links:
        if view.place.get(c) in rest:
            owned[r] += 1
            spans[r].add(view.place.get(c))
    reqs = sorted(owned)
    if by == "pillar":
        return sorted(reqs, key=lambda r: (-owned[r], r))
    if by == "spread":
        return sorted(reqs, key=lambda r: (-len(spans[r]), -owned[r], r))
    if by == "random":
        rnd.shuffle(reqs)
        return reqs
    raise ValueError("no seed order %r (known: methods, %s)" % (by, ", ".join(ORDERS)))


def measure(view, fractions=FRACTIONS, seed=0, folds=5, epochs=300, fit=None, by="methods"):
    """{f: {ranker: MRR}} for one seed; `fit` stands in for the GNN in the selftest (it records what each fit was given). `by`:
    methods (the first f of the claimed METHODS seen) or a requirement order (the first f of the REQUIREMENTS' claims seen)"""
    truth = collections.defaultdict(set)
    for c, r in view.links:
        truth[c].add(r)
    ranks = collections.defaultdict(lambda: collections.defaultdict(list))
    for k, (test, rest) in enumerate(split(view, folds, seed)):
        tcells = sorted({c for c, _ in view.links if view.place.get(c) in test})
        order = None if by == "methods" else req_order(view, rest, by, random.Random(seed * 1000 + k + 7))
        for f in fractions:
            if order is None:
                keep = kept(rest, f)
                visible = [(c, r) for c, r in view.links if view.place.get(c) in keep]
            else:
                seedset = set(order[:max(1, int(round(f * len(order))))])
                visible = [(c, r) for c, r in view.links if view.place.get(c) in set(rest) and r in seedset]
            seen = set(visible)
            hidden = [l for l in view.links if l not in seen]                              # the test's links AND the unclaimed rest
            if fit:
                gnn = fit(view, seed + k, epochs, visible, hidden, tcells)
            else:
                gnn = gg.LinkProposer(view, seed + k, epochs).fit(visible, hidden)
            base = gg.Baselines(view, seed + k)
            for c in tcells:
                ranks[f]["gnn"].append((gnn.rank(c), truth[c]))
                for name in gg.BASELINES:
                    ranks[f][name].append((getattr(base, name)(c, visible), truth[c]))
    return {f: {name: gg.metrics(r)["mrr"] for name, r in by.items()} for f, by in ranks.items()}


def crossover(by_seed):
    """the least fullness from which the GNN beats the best baseline at every seed and every larger fullness; None if never.
    by_seed: {seed: {f: {ranker: MRR}}}"""
    fs = sorted(next(iter(by_seed.values())))
    wins = {f: all(m[f]["gnn"] > max(m[f][b] for b in gg.BASELINES) for m in by_seed.values()) for f in fs}
    for i, f in enumerate(fs):
        if all(wins[g] for g in fs[i:]):
            return f
    return None


def curve(db, fractions=FRACTIONS, seeds=(0, 1), folds=5, epochs=300, say=print, by="methods"):
    view = gg.GraphView(db)
    cells = {c for c, _ in view.links}
    shape = {"nodes": len(view.ids), "edges": len(view.edges), "requirements": len(view.reqs), "linked_cells": len(cells),
             "methods": len({view.place.get(c) for c in cells}), "links": len(view.links)}
    say("gnn_curve: %s - %d nodes, %d edges, %d requirements, %d claimed branches in %d methods; seeded by %s" % (
        os.path.basename(db), shape["nodes"], shape["edges"], shape["requirements"], shape["linked_cells"], shape["methods"],
        "METHOD" if by == "methods" else "REQUIREMENT, %s first" % by.upper() if by != "random" else "REQUIREMENT, in random order"))
    by_seed, out = {}, os.path.join(os.path.dirname(os.path.abspath(db)), "gnn_curve.jsonl")
    for s in seeds:
        t0 = time.time()
        by_seed[s] = measure(view, fractions, s, folds, epochs, by=by)
        for f in sorted(by_seed[s]):
            m = by_seed[s][f]
            best = max(gg.BASELINES, key=lambda b: m[b])
            row = dict(shape, time=time.strftime("%Y-%m-%dT%H:%M:%S"), graph=os.path.basename(db), fullness=f, seed=s, folds=folds, by=by,
                       epochs=epochs, gnn=round(m["gnn"], 4), best=best, best_mrr=round(m[best], 4),
                       baselines={b: round(m[b], 4) for b in gg.BASELINES})
            io.open(out, "a", encoding="utf-8").write(json.dumps(row, sort_keys=True) + "\n")
        say("  seed %d done in %.0f s" % (s, time.time() - t0))
    for f in sorted(fractions):
        g = [by_seed[s][f]["gnn"] for s in seeds]
        b = [max(by_seed[s][f][x] for x in gg.BASELINES) for s in seeds]
        names = collections.Counter(max(gg.BASELINES, key=lambda x: by_seed[s][f][x]) for s in seeds)
        say("  fullness %4.0f%%: GNN MRR %.2f against the best that does not learn %.2f (%s)  %s" % (
            100 * f, sum(g) / len(g), sum(b) / len(b), names.most_common(1)[0][0],
            "BEATS at every seed" if all(x > y for x, y in zip(g, b)) else "does not beat at every seed"))
    x = crossover(by_seed)
    say("  CROSSOVER: %s" % ("from %.0f%% of the claims on, the GNN beats every ranker that does not learn" % (100 * x) if x is not None
                             else "NONE - on this graph the GNN never beats them at every seed; its proposals stay hidden"))
    return x, by_seed


def selftest():
    """the plumbing, not the GNN's quality (graph_gnn.py --selftest proves that): nested fullness, no test link ever seen, the
    same test cells at every fullness, the crossover's rule, and a graph too small to fold refused"""
    ok = True

    class V(object):          # a graph's links as GraphView holds them: 10 methods, 3 branches each
        def __init__(self, n=10):
            self.links = [("cell:S-BRANCH:T.M%d#b%d" % (m, b), "req:R-%d" % (m % 3)) for m in range(n) for b in range(3)]
            self.place = {c: "code:T.M%s" % c.split("M")[1].split("#")[0] for c, _ in self.links}
            self.reqs = ["req:R-%d" % i for i in range(3)]

    seen = collections.defaultdict(list)

    class Fake(object):
        def __init__(self, view):
            self.v = view

        def rank(self, c):
            return list(self.v.reqs)

    def fit(view, seed, epochs, visible, hidden, tcells):
        seen[seed].append((set(visible), set(tcells), set(hidden)))
        return Fake(view)

    v = V()
    real = gg.Baselines
    gg.Baselines = lambda view, seed: type("B", (), {n: (lambda self, c, vis, view=view: list(view.reqs)) for n in gg.BASELINES})()
    try:
        measure(v, (0.25, 0.5, 1.0), seed=0, folds=5, epochs=1, fit=fit)
    finally:
        gg.Baselines = real
    nested = all(a[0] <= b[0] for runs in seen.values() for a, b in zip(runs, runs[1:]))
    no_leak = all(not ({c for c, _ in vis} & tc) for runs in seen.values() for vis, tc, _ in runs)
    same_test = all(len({frozenset(tc) for _, tc, _ in runs}) == 1 for runs in seen.values())
    hidden_ok = all(len(vis) + len(hid) == len(v.links) for runs in seen.values() for vis, _, hid in runs)
    grows = all(len(runs[-1][0]) > len(runs[0][0]) for runs in seen.values())
    good = nested and no_leak and same_test and hidden_ok and grows and len(seen) == 5
    ok &= good
    print("selftest: FULLNESS nested (a fuller graph holds every link a thinner one did), no held-out link ever seen, the same "
          "held-out branches at every fullness, every unkept link hidden %s" % ("PASS" if good else "FAIL"))

    def m(gnn, base):
        return {"gnn": gnn, **{b: (base if b == "combined" else 0.0) for b in gg.BASELINES}}
    s1 = {0.25: m(0.5, 0.6), 0.5: m(0.7, 0.6), 1.0: m(0.8, 0.6)}
    s2 = {0.25: m(0.5, 0.6), 0.5: m(0.65, 0.6), 1.0: m(0.8, 0.6)}
    dip = {0.25: m(0.7, 0.6), 0.5: m(0.5, 0.6), 1.0: m(0.8, 0.6)}
    never = {0.25: m(0.5, 0.6), 0.5: m(0.5, 0.6), 1.0: m(0.55, 0.6)}
    one_seed_loses = {0.25: m(0.5, 0.6), 0.5: m(0.59, 0.6), 1.0: m(0.8, 0.6)}
    good = crossover({0: s1, 1: s2}) == 0.5 and crossover({0: dip}) == 1.0 and crossover({0: never}) is None \
        and crossover({0: s1, 1: one_seed_loses}) == 1.0
    ok &= good
    print("selftest: CROSSOVER the least fullness from which it wins at every seed and every fuller graph: a dip moves it up, a "
          "seed that loses holds it back, never winning is NONE %s" % ("PASS" if good else "FAIL"))
    try:
        split(V(3), 5, 0)
        good = False
    except ValueError:
        good = True
    ok &= good
    print("selftest: SMALL a graph with fewer claimed methods than folds is refused, never given a number %s" % ("PASS" if good else "FAIL"))
    seen.clear()
    v2 = V()
    v2.links += [("cell:S-BRANCH:T.M%d#b%d" % (m, b), "req:R-2") for m in range(10) for b in (7, 8, 9)]   # R-2 leads in every fold, and is not the lowest id
    v2.place.update({c: "code:T.M%s" % c.split("M")[1].split("#")[0] for c, _ in v2.links})
    gg.Baselines = lambda view, seed: type("B", (), {n: (lambda self, c, vis, view=view: list(view.reqs)) for n in gg.BASELINES})()
    try:
        measure(v2, (0.34, 1.0), seed=0, folds=5, epochs=1, fit=fit, by="pillar")
    finally:
        gg.Baselines = real
    first = [runs[0][0] for runs in seen.values()]
    good = all({r for _, r in vis} == {"req:R-2"} for vis in first) \
        and all(not ({c for c, _ in vis} & tc) for runs in seen.values() for vis, tc, _ in runs) \
        and all(a[0] <= b[0] for runs in seen.values() for a, b in zip(runs, runs[1:])) \
        and req_order(v2, ["code:T.M%d" % m for m in range(10)], "pillar", random.Random(0))[0] == "req:R-2"
    try:
        req_order(v2, [], "nonsense", random.Random(0))
        good = False
    except ValueError:
        pass
    ok &= good
    print("selftest: SEED by requirement: the least seed holds the pillar's claims alone, a fuller seed holds every claim a thinner "
          "one did, no held-out link is ever seen, an unknown order is refused %s" % ("PASS" if good else "FAIL"))
    print("selftest: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db", nargs="*")
    ap.add_argument("--fractions", default=",".join(str(f) for f in FRACTIONS))
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--threads", type=int, default=2, help="torch threads: the machine is shared")
    ap.add_argument("--by", default="methods", help="methods, or a requirement seed order: %s (comma-separated for several)" % ", ".join(ORDERS))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.db:
        ap.error("name at least one graph (.sqlite), or --selftest")
    gg.torch.set_num_threads(a.threads)
    fr = tuple(float(x) for x in a.fractions.split(","))
    seeds = tuple(int(x) for x in a.seeds.split(","))
    for db in a.db:
        try:
            for by in a.by.split(","):
                curve(db, fr, seeds, a.folds, a.epochs, by=by)
        except ValueError as e:
            print("gnn_curve: %s - %s" % (os.path.basename(db), e))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
