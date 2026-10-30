#!/usr/bin/env python3
"""buzzsaw.py - BUZZSAW testing: every assumption the requirements rest on, attacked with the answer that would change what
happens later, and what rests on it measured in the graph.

WHY (2026-10-29): take every assumption, find the range of answers that could change future events, and attack it across that whole
range, which is hard because the future is not known. The code's breaks (cs_effect_check, py_effect_check) attack the CODE. This attacks
the MODEL'S ASSUMPTIONS: the requirements nobody wrote (inferred: the owner's mays), the conditions taken as given from outside the
model (entry), and the reasons a surviving break was called the same program (declared equivalents). Each assumption's answer is
flipped, and everything that rests on it in the knowledge graph is its BLAST RADIUS: the later steps of every use case that carries
it, the requirements those steps carry, their effects, the code those effects sit at, that code's branches and breaks, and the
tests whose meaning would change. The widest radius goes to the owner first.

WHAT IT CANNOT DO, AND SAYS: it does not know which answer is right (only the owner does), and it counts STRUCTURE, not
probability - a wide radius means much depends on the assumption, not that it is likely wrong. Its graph is the record
usecase_kg.py writes (the store closed_loop.py builds on every run); a relation missing from it is a radius it cannot see.

THE PIECES:
  Assumption   one thing the model takes as given: its kind, its node in the graph, its words, the flipped answer
  BlastRadius  what rests on one assumption, by kind (steps, requirements, effects, code, branches, breaks, tests), and its score
  Buzzsaw      every assumption of a graph, each with its radius, ranked widest first

    python -X utf8 buzzsaw.py <graph.sqlite> [--verdicts verdicts.json] [--top 10]
    python -X utf8 buzzsaw.py --selftest
"""
import argparse
import collections
import io
import json
import os
import sqlite3
import sys

# the graph's relations (usecase_kg.PROPS): the ones a radius walks
HAS_STEP, NEXT_STEP, BINDS, INFERS, BRANCHES_AT, REJOINS, CHILD_OF, INCLUDES = 100, 101, 107, 108, 117, 119, 120, 121
MAPS, ASSERTS, NAMES, SETS, BRANCHES_ON, MUTATED_AS, KILLED_BY, CHECKED_BY, IS_AT = 122, 134, 139, 140, 141, 147, 148, 149, 152
TEXT, ENTRY, LEVEL = 133, 143, 131


class Assumption(object):
    """One thing the model takes as given. `kind`: an inferred requirement (the flip: it does not hold), an entry condition (the
    flip: it is not true when the use case starts), or a declared equivalent (the flip: the break is a real difference)."""

    FLIPS = {"inferred requirement": "it does not hold", "entry condition": "it is not true when its effect reads it",
             "declared equivalent": "the break is a real difference after all"}

    def __init__(self, kind, node, text):
        self.kind, self.node, self.text = kind, node, text

    @property
    def flip(self):
        return self.FLIPS[self.kind]


class BlastRadius(object):
    """Everything that rests on one node in the graph, by kind. For a requirement: the steps and extensions that carry it, every
    step after them (the next steps, the step an extension rejoins, the use cases a step includes or decomposes into, and their
    steps), the requirements all those carry, their effects, the code the effects sit at, the branch cells at that code, their
    breaks, and the tests that catch those breaks or are named as the requirements' checks. The SCORE counts steps, requirements,
    effects, code places and tests - the things a person would have to look at again; breaks and branches are listed, not scored
    (one rich method would otherwise outweigh everything)."""

    KINDS = ("steps", "requirements", "effects", "code", "branches", "breaks", "tests")
    SCORED = ("steps", "requirements", "effects", "code", "tests")

    def __init__(self):
        self.parts = {k: set() for k in self.KINDS}

    @property
    def score(self):
        return sum(len(self.parts[k]) for k in self.SCORED)

    def line(self):
        return ", ".join("%d %s" % (len(self.parts[k]), k) for k in self.KINDS if self.parts[k]) or "nothing rests on it"


class Buzzsaw(object):
    """Every assumption of one graph with its blast radius, widest first"""

    def __init__(self, db_path, verdicts=None):
        db = sqlite3.connect(db_path)
        self.kind = dict(db.execute("SELECT id, kind FROM entity"))
        self.sub = dict(db.execute("SELECT id, sub FROM entity"))
        self.label = dict(db.execute("SELECT id, label FROM entity"))
        self.out, self.into = collections.defaultdict(set), collections.defaultdict(set)
        for s, p, o in db.execute("SELECT s, p, o FROM edge"):
            self.out[(s, p)].add(o)
            self.into[(o, p)].add(s)
        self.text = {s: v for s, v in db.execute("SELECT s, v FROM literal WHERE p = ?", (TEXT,))}
        self.entry = {s: v for s, v in db.execute("SELECT s, v FROM literal WHERE p = ?", (ENTRY,))}
        self.levels = {s: v for s, v in db.execute("SELECT s, v FROM literal WHERE p = ?", (LEVEL,))}
        db.close()
        self.declared = dict((json.load(io.open(verdicts, encoding="utf-8")).get("equivalent_declared") or {})) if verdicts else {}

    def assumptions(self):
        out = [Assumption("inferred requirement", n, self.text.get(n) or self.label.get(n) or n)
               for n, k in sorted(self.kind.items()) if k == "requirement" and self.sub.get(n) == "inferred"]
        out += [Assumption("entry condition", n, "%s: %s" % (n.split(":", 1)[1], why)) for n, why in sorted(self.entry.items())]
        out += [Assumption("declared equivalent", "mut:" + k, "%s: %s" % (k, why)) for k, why in sorted(self.declared.items())
                if "mut:" + k in self.kind]
        return out

    def _steps_after(self, carriers):
        """every step that comes after (or inside) the carrying steps"""
        seen, todo = set(), list(carriers)
        while todo:
            n = todo.pop()
            if n in seen:
                continue
            seen.add(n)
            nxt = set(self.out[(n, NEXT_STEP)]) | set(self.out[(n, REJOINS)])
            for uc in set(self.out[(n, INCLUDES)]) | set(self.into[(n, CHILD_OF)]):
                nxt |= self.out[(uc, HAS_STEP)]
            todo += [x for x in nxt if x not in seen]
        return seen

    LEVELS = ("fish", "sea", "kite")

    def level(self, carrier):
        """the goal level of the use case a step or extension belongs to (the use case's `level` literal)"""
        step = next(iter(self.out[(carrier, BRANCHES_AT)]), carrier)       # an extension belongs where it branches
        uc = next(iter(self.into[(step, HAS_STEP)]), None)
        return self.levels.get(uc)

    def radius_of_requirements(self, reqs):
        r = BlastRadius()
        carriers = {s for q in reqs for p in (BINDS, INFERS) for s in self.into[(q, p)]}
        # the MOST SPECIFIC carriers only: a summary (kite) step that carries the same requirement as a user-goal step is a summary
        # of that work, and walking on from it reaches every use case below the epic - MEASURED 10-29 on the shops: every one of
        # 41 assumptions then reached 33 steps and all 41 tests, and the ranking told none of them apart
        for lv in self.LEVELS:
            at = {c for c in carriers if self.level(c) == lv}
            if at:
                carriers = at
                break
        steps = self._steps_after(carriers)
        r.parts["steps"] = {s for s in steps if self.kind.get(s) in ("step", "extension")}
        r.parts["requirements"] = set(reqs) | {q for s in steps for p in (BINDS, INFERS) for q in self.out[(s, p)]}
        r.parts["effects"] = {a for q in r.parts["requirements"] for a in self.out[(q, ASSERTS)]}
        r.parts["code"] = {c for a in r.parts["effects"] for c in self.out[(a, IS_AT)] if self.kind.get(c) == "code"}
        r.parts["branches"] = {b for c in r.parts["code"] for b in self.into[(c, IS_AT)] if self.kind.get(b) == "cell"}
        r.parts["breaks"] = {m for b in r.parts["branches"] for m in self.out[(b, MUTATED_AS)]}
        r.parts["tests"] = {t for m in r.parts["breaks"] for t in self.out[(m, KILLED_BY)]} | \
                           {t for q in r.parts["requirements"] for t in self.out[(q, CHECKED_BY)]}
        return r

    def radius(self, a):
        if a.kind == "inferred requirement":
            return self.radius_of_requirements({a.node})
        if a.kind == "entry condition":
            effects = {e for p in (NAMES, SETS, BRANCHES_ON) for e in self.into[(a.node, p)]}
            return self.radius_of_requirements({q for e in effects for q in self.into[(e, ASSERTS)]})
        # a declared equivalent: the break's branch cells, the requirements that claim them
        cells = self.into[(a.node, MUTATED_AS)]
        return self.radius_of_requirements({q for c in cells for q in self.out[(c, MAPS)]})

    def ranked(self):
        rows = [(a, self.radius(a)) for a in self.assumptions()]
        return sorted(rows, key=lambda t: (-t[1].score, t[0].kind, t[0].node))

    def stats(self):
        """for the morph log: how many assumptions of each kind, the owner's widest question, and the entry conditions nothing
        reads (a stale entry: the code that read it moved or went - MEASURED 10-29 on the shops: CanOpen, IsRecipe, n)"""
        rows = self.ranked()
        mays = [(a, r) for a, r in rows if a.kind == "inferred requirement"]
        return {"assumptions": len(rows), "assumptions_by_kind": dict(collections.Counter(a.kind for a, _ in rows)),
                "may_radius_max": mays[0][1].score if mays else 0, "may_widest": mays[0][0].node if mays else None,
                "entry_unread": sum(1 for a, r in rows if a.kind == "entry condition" and r.score == 0)}


def selftest():
    """a planted graph: an inferred requirement carried by step 1 of a three-step use case must reach the later steps, what they
    carry, its effect's code, the breaks of that code and the test that catches them; one carried only by the LAST step reaches
    less; an entry condition reaches the requirements whose effects read it; a requirement nobody carries reaches only itself"""
    import shutil
    import tempfile
    d = tempfile.mkdtemp(prefix="buzzsaw_selftest_")
    p = os.path.join(d, "g.sqlite")
    db = sqlite3.connect(p)
    db.executescript("CREATE TABLE entity(id TEXT PRIMARY KEY, kind TEXT, label TEXT, sub TEXT);"
                     "CREATE TABLE edge(s TEXT, p INTEGER, o TEXT, fam INTEGER);"
                     "CREATE TABLE literal(s TEXT, p INTEGER, v TEXT);")
    E = lambda i, k, sub=None: db.execute("INSERT INTO entity VALUES (?, ?, ?, ?)", (i, k, i, sub))
    L = lambda s, p, o: db.execute("INSERT INTO edge VALUES (?, ?, ?, 0)", (s, p, o))
    E("uc:U", "use case")
    for n in (1, 2, 3):
        E("step:U#%d" % n, "step")
        L("uc:U", HAS_STEP, "step:U#%d" % n)
    L("step:U#1", NEXT_STEP, "step:U#2")
    L("step:U#2", NEXT_STEP, "step:U#3")
    for q, sub in (("req:IR-1", "inferred"), ("req:R-2", "authored"), ("req:IR-3", "inferred"), ("req:IR-4", "inferred")):
        E(q, "requirement", sub)
    L("step:U#1", INFERS, "req:IR-1")
    L("step:U#2", BINDS, "req:R-2")
    L("step:U#3", INFERS, "req:IR-3")
    E("assert:R-2#1", "assertion")
    E("code:T.M", "code")
    E("cell:S:T.M#b0", "cell")
    E("mut:T.M@1: x", "mutant")
    E("test:T1", "test")
    E("name:Flag", "name")
    L("req:R-2", ASSERTS, "assert:R-2#1")
    L("assert:R-2#1", IS_AT, "code:T.M")
    L("cell:S:T.M#b0", IS_AT, "code:T.M")
    L("cell:S:T.M#b0", MUTATED_AS, "mut:T.M@1: x")
    L("mut:T.M@1: x", KILLED_BY, "test:T1")
    L("assert:R-2#1", BRANCHES_ON, "name:Flag")
    db.execute("INSERT INTO literal VALUES ('name:Flag', ?, 'set by the sheet, outside the model')", (ENTRY,))
    db.commit()
    db.close()
    ok = True
    try:
        b = Buzzsaw(p)
        got = {a.node: r for a, r in b.ranked()}
        r1, r3, r4, rf = got["req:IR-1"], got["req:IR-3"], got["req:IR-4"], got["name:Flag"]
        good = r1.parts["steps"] == {"step:U#1", "step:U#2", "step:U#3"} and {"req:R-2", "req:IR-3"} <= r1.parts["requirements"] \
            and r1.parts["code"] == {"code:T.M"} and r1.parts["breaks"] == {"mut:T.M@1: x"} and r1.parts["tests"] == {"test:T1"}
        ok &= good
        print("selftest: RADIUS an assumption at step 1 reaches the later steps, what they carry, its code's breaks and the test "
              "that catches them %s (%s)" % ("PASS" if good else "FAIL", r1.line()))
        good = r1.score > r3.score > r4.score and r4.parts["requirements"] == {"req:IR-4"} and not r4.parts["steps"]
        ok &= good
        print("selftest: RANK carried earlier reaches more than carried last; carried by nothing reaches only itself %s" % ("PASS" if good else "FAIL"))
        good = "req:R-2" in rf.parts["requirements"] and rf.parts["code"] == {"code:T.M"}
        ok &= good
        print("selftest: ENTRY an entry condition reaches the requirements whose effects read it %s" % ("PASS" if good else "FAIL"))
        good = [a.node for a, _ in b.ranked()][0] == "req:IR-1"
        ok &= good
        print("selftest: FIRST the widest radius is asked first %s" % ("PASS" if good else "FAIL"))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("selftest: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db", nargs="?")
    ap.add_argument("--verdicts")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.db:
        ap.error("<graph.sqlite> | --selftest")
    verdicts = a.verdicts or os.path.join(os.path.dirname(os.path.abspath(a.db)), "verdicts.json")
    b = Buzzsaw(a.db, verdicts if os.path.exists(verdicts) else None)
    rows = b.ranked()
    by = collections.Counter(x.kind for x, _ in rows)
    print("buzzsaw: %d assumption(s) in %s (%s)" % (len(rows), a.db, ", ".join("%d %s" % (n, k) for k, n in sorted(by.items()))))
    heads = {"inferred requirement": "THE OWNER'S QUESTIONS (inferred requirements), widest first: ask these in this order",
             "entry condition": "ENTRY CONDITIONS taken from outside the model; one nothing reads is stale",
             "declared equivalent": "DECLARED EQUIVALENTS (ours): the widest are the first to re-prove"}
    for kind in ("inferred requirement", "entry condition", "declared equivalent"):
        mine = [(x, r) for x, r in rows if x.kind == kind]
        if not mine:
            continue
        print(heads[kind])
        for x, r in mine[:a.top]:
            print("  %3d  %s - if %s: %s" % (r.score, x.node, x.flip, r.line()))
            print("       %s" % x.text[:150])
    s = b.stats()
    print("buzzsaw: the owner's widest question %s (radius %d); %d entry condition(s) nothing reads" % (
        s["may_widest"], s["may_radius_max"], s["entry_unread"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
