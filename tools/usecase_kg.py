#!/usr/bin/env python3
"""usecase_kg.py - a use case model as a KNOWLEDGE GRAPH: one SQLite store, a store-declared completeness schema, and the gap
queries a graph can answer that the rules over YAML cannot.

WHY (2026-10-27): the knowledge graph the project already keeps can detect gaps that the rules over YAML cannot, provided the
graph is stable, efficient and in a sound format: a production-quality node graph store, not a JSON file.

JOINED, NOT A FIFTH STORE. ANCHOR_API.md (the anchor project) records a use-case graph that was deleted the day it was written for
being "a fourth store with its own rules". This one is joined to what exists:
  - built from the SAME loader as the rules checker (usecase_model.py's UseCaseModel), so there is one parse;
  - the Anchor store's relation numbers are reused where the meaning matches (100 has step, 101 next step, 102 performed
    by, 107 binds requirement, 108 infers requirement, 109 goal, 110 trigger, 114 stated by); new relations start at 116;
  - the completeness schema is DECLARED BY THE STORE (ANCHOR API invariant I7) in its schema_rule table, and the gap query is a
    generic walk over it - adding a rule is adding a row, not code;
  - entity ids come from identity (kind + the model's own id), never from a counter (invariant I12), so a rebuild is stable;
  - SQLite, chosen by measurement (a survey of four graph stores, its K3): the fastest of the four installed systems on every
    query shape, and Node 25's built-in node:sqlite reads the same file, so a JavaScript engine can join it.
  - its review format is N-Triples (one W3C triple per line, sorted): two builds of one model are byte-identical.

THE ANCHOR COUNT. Every edge carries the evidence FAMILY that asserts it (1 story = the product owner's words, 2 framework = a
published ergonomics framework, 4 language = the C# language, 8 brainstorm = a persona's question, 16 model = this model's own
structure, 32 code = exists in the code). A requirement's k is the number of DISTINCT families other than the model's own
structure behind it - the anchor law: the author's own structure never raises k (ANCHOR API I2). k = 0 for a requirement that is
not inferred is a hole; k = 1 is listed: it rests on one family, and the question is which independent family would confirm it.

THE CLOSED LOOP (2026-10-28). Closed-loop requirements replace code as what an AI acts on: it writes the code from them and is
held to them, so they must leave no room for a mistake that is not already in the requirements themselves. LOOP lists the
criteria a node structure must meet for that to hold, each one named by the mistake it stops; LoopGate measures them all. A
requirement's EFFECT (usecase_model.Assertion: what the code sets, calls, branches on or reads, and where) is the keystone: the
code adapter holds each assertion to the code and plants it the other way, and
its verdicts.json comes in here. The lock (<model>/loop.lock) is each requirement's fingerprint as last verified.

    python -X utf8 usecase_kg.py build <model folder> --db <store.sqlite> [--nt <graph.nt>]
    python -X utf8 usecase_kg.py gaps --db <store.sqlite>        exit 1 while any failing gap stands
    python -X utf8 usecase_kg.py loop <model folder> [--verdicts verdicts.json] [--db F] [--ours] [--accept "<why>"]
                                          every criterion; exit 1 while a hole stands (--ours: while a hole of OURS stands -
                                          the owner's questions are still counted and printed)
    python -X utf8 usecase_kg.py --selftest                      agreement with usecase_model.py's plants + the graph's own
"""
import argparse
import collections
import copy
import hashlib
import io
import json
import os
import re
import sqlite3
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import usecase_model as ucm  # noqa: E402  the one loader; the graph never parses the model itself

FAMILIES = {1: "story", 2: "framework", 4: "language", 8: "brainstorm", 16: "model", 32: "code"}
MODEL = 16
BASIS_FAMILY = {"prose": 1, "framework": 2, "language": 4, "brainstorm": 8, "code": 32}
SOURCE_FAMILY = {"prose": 1, "quote": 1, "standing": 1, "framework": 2}
PROPS = [  # (id, label, aliases) - 100..115 as the Anchor store numbers them, 116+ new
    (100, "has step", "contains step"), (101, "next step", "then"), (102, "performed by", "actor, primary actor"),
    (107, "binds requirement", "carries"), (108, "infers requirement", "carries an inferred requirement"),
    (109, "goal", "success guarantee"), (110, "trigger", "starts when"), (114, "stated by", "authored in"),
    (116, "has extension", "alternate pathway"), (117, "branches at", "extension of step"), (118, "outcome", "ends"),
    (119, "rejoins", "returns to step"), (120, "child of", "parent"), (121, "includes", "uses"), (122, "maps to", "cell to requirement"),
    (123, "excluded because", "reason"), (124, "in space", "space"), (125, "measure", "checked by"), (126, "asked by", "persona"),
    (127, "answered by", "inverse use case"), (128, "gulf of execution", "how the person acts"),
    (129, "gulf of evaluation", "how the person sees the result"), (130, "supporting actor", "supported by"),
    (131, "level", "goal level"), (132, "requirement kind", "authored, derived or inferred"), (133, "text", "says"),
    # the closed loop (10-28): a requirement's effect as nodes, its verdicts against the code, and the words it is written in
    (134, "asserts", "effect"), (135, "at", "in the code at"), (136, "verdict", "holds, fails or unread"),
    (137, "code read", "fingerprint of the statements a verdict rests on"), (138, "binds", "means in the code"),
    (139, "names", "says the word"), (140, "sets", "produces"), (141, "branches on", "consumes"), (142, "plant", "seen to flip"),
    (143, "entry", "set outside the model, because"), (144, "resolves", "found in"), (145, "no effect because", "absent"),
    (146, "because", "why the verdict"),
    # the loop's evidence as nodes (10-29; the graph is the record): every mutant of the code, the test that
    # caught it, the code place each node sits at, the tests a requirement names as its check, and the argument chains agents write
    (147, "mutated as", "broken as"), (148, "killed by", "caught by test"), (149, "checked by", "its own test"),
    (150, "reasons", "chain step"), (151, "cites", "stands on"), (152, "is at", "code place"),
]
MEASURED = (136, 142, 146)   # verdicts are measurements of the content, not content: a fingerprint never reads them
MEASURED_EDGES = (147, 148)  # nor the mutants a run made of it, or the tests that caught them: a new operator never stales a lock
CODE = 32
# The completeness schema, declared by the store: (rule, kind, filter, edge props, literal props, direction, teach)
SCHEMA = [
    ("S-UC-ACTOR", "use case", None, "102", "", "out", "name its primary actor"),
    ("S-UC-TRIGGER", "use case", None, "", "110", "out", "say what starts it"),
    ("S-UC-GOAL", "use case", None, "", "109", "out", "say what is true when it succeeds"),
    ("S-UC-STEPS", "use case", None, "100", "", "out", "give it a main scenario"),
    ("S-UC-GULF-EXEC", "use case", "kite,sea", "128", "", "out", "say how the person acts, with a requirement"),
    ("S-UC-GULF-EVAL", "use case", "kite,sea", "129", "", "out", "say how the person sees the result, with a requirement"),
    ("S-KITE-STEP-CHILD", "step", "kite", "120,121", "", "in-or-out", "decompose the summary step into a use case"),
    ("S-STEP-CARRIES", "step", None, "107,108,121", "", "out", "carry a requirement, or include a use case that does"),
    ("S-EXT-OUTCOME", "extension", None, "", "118", "out", "say where the pathway ends: rejoin N, end success or end minimal"),
    ("S-EXT-CARRIES", "extension", None, "107,108", "", "out", "carry a requirement on this pathway"),
    ("S-REQ-CARRIED", "requirement", None, "107,108", "", "in", "carry it on a step or an extension"),
    ("S-REQ-MEASURE", "requirement", None, "", "125", "out", "say how it is checked"),
    ("S-CELL-MAPPED", "cell", None, "122", "123", "out", "map it to a requirement, or exclude it with the reason"),
    ("S-Q-ANSWERED", "question", None, "127", "", "out", "invert it into the use case that answers it"),
    ("S-PERSON-PRIMARY", "actor", "person", "102", "", "in", "make them primary actor of a use case"),
]
FAILING_GRAPH = ("G-REF", "G-UNSOURCED", "G-REACH")


class Gap(object):
    """One hole the graph found: the rule (a schema row or a graph rule), the entity, and the teach line that closes it. An
    `owner` hole is a question only the requirements' owner can close; every other failing hole is the model author's."""

    def __init__(self, rule, entity, message, failing=True, owner=False):
        self.rule, self.entity, self.message, self.failing, self.owner = rule, entity, message, failing, owner

    def __str__(self):
        return "[%s] %s: %s" % (self.rule, self.entity, self.message)


class KnowledgeGraph(object):
    """The store: entities with stable ids, typed edges carrying their evidence family, literals, and the declared schema."""

    def __init__(self, path):
        self.path = path
        self.db = sqlite3.connect(path)

    # ------------------------------------------------------------------ build

    @classmethod
    def build(cls, model, path, verdicts=None):
        """the store from the model; `verdicts` (logic_map.py's verdicts.json) adds each assertion's verdict against the code"""
        if os.path.exists(path):
            os.remove(path)
        kg = cls(path)
        db = kg.db
        db.executescript("""
            CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE family(bit INTEGER PRIMARY KEY, name TEXT NOT NULL);
            CREATE TABLE prop(id INTEGER PRIMARY KEY, label TEXT NOT NULL, aliases TEXT);
            CREATE TABLE entity(id TEXT PRIMARY KEY, kind TEXT NOT NULL, label TEXT, sub TEXT);
            CREATE TABLE edge(s TEXT NOT NULL, p INTEGER NOT NULL, o TEXT NOT NULL, fam INTEGER NOT NULL,
                              PRIMARY KEY (s, p, o, fam));
            CREATE TABLE literal(s TEXT NOT NULL, p INTEGER NOT NULL, v TEXT NOT NULL, PRIMARY KEY (s, p, v));
            CREATE TABLE schema_rule(rule TEXT PRIMARY KEY, kind TEXT, filter TEXT, edge_props TEXT, literal_props TEXT,
                                     direction TEXT, teach TEXT);
            CREATE INDEX edge_po ON edge(p, o);
        """)
        db.executemany("INSERT INTO family VALUES (?, ?)", sorted(FAMILIES.items()))
        db.executemany("INSERT INTO prop VALUES (?, ?, ?)", PROPS)
        db.executemany("INSERT INTO schema_rule VALUES (?, ?, ?, ?, ?, ?, ?)", SCHEMA)
        ents, edges, lits = {}, set(), set()

        def ent(eid, kind, label, sub=None):
            ents.setdefault(eid, (eid, kind, label, sub))

        def edge(s, p, o, fam=MODEL):
            edges.add((s, p, o, fam))

        def lit(s, p, v):
            if v is not None and str(v).strip():
                lits.add((s, p, str(v)))

        for a in model.actors.values():
            ent("actor:%s" % a.id, "actor", a.name, a.kind)
        for s in model.sources.values():
            ent("src:%s" % s.id, "source", s.id, s.kind)
        for sp in model.spaces.values():
            ent("space:%s" % sp.id, "space", sp.title or sp.id, sp.basis)
            fam = BASIS_FAMILY.get(sp.basis, MODEL)
            try:
                expected = sp.expected_keys(model)
            except (ValueError, OSError):
                expected = None
            keys = list(sp.cells) + [k for k in (expected or []) if k not in sp.cells]
            for k in keys:
                cid = "cell:%s:%s" % (sp.id, k)
                ent(cid, "cell", k, sp.id)
                edge(cid, 124, "space:%s" % sp.id)
                c = sp.cells.get(k)
                if c is None:
                    continue                      # expected and missing: an entity with nothing - the schema asks for it
                for r in c.maps:
                    edge(cid, 122, "req:%s" % r, fam)
                lit(cid, 123, None if ucm.blank(c.excluded) else c.excluded)
                lit(cid, 133, c.text)
        for r in model.requirements.values():
            rid = "req:%s" % r.id
            ent(rid, "requirement", r.id, r.kind)
            lit(rid, 133, r.text)
            lit(rid, 125, None if ucm.blank(r.measure) else r.measure)
            lit(rid, 132, r.kind)
            for s in r.sources:
                src = model.sources.get(s)
                edge(rid, 114, "src:%s" % s, SOURCE_FAMILY.get(src.kind, MODEL) if src else MODEL)
        # the closed loop: each assertion of an effect is a node; a `sets` produces its name, a `branches` consumes it
        V = verdicts or {}
        va, vn = V.get("assertions") or {}, V.get("names") or {}

        def name(n):
            ent("name:%s" % n, "name", n)
            return "name:%s" % n
        for r in model.requirements.values():
            rid = "req:%s" % r.id
            lit(rid, 145, None if ucm.blank(r.absent) else r.absent)
            for a in r.effect:
                aid = "assert:%s" % a.id
                ent(aid, "assertion", a.text(), a.verb)
                edge(rid, 134, aid)
                lit(aid, 133, a.text())
                lit(aid, 135, a.at)
                if a.verb in ("sets", "branches") and a.target:
                    edge(aid, 140 if a.verb == "sets" else 141, name(a.target))
                v = va.get(a.id)
                if v is not None:
                    lit(aid, 136, {True: "holds", False: "fails"}.get(v.get("holds"), "unread"))
                    lit(aid, 142, v.get("plant"))
                    lit(aid, 137, v.get("fp"))
                    lit(aid, 146, v.get("why"))
                if a.at and V.get("mutants") is not None:       # a C# place: the assertion sits AT a code node
                    ent("code:%s" % a.at, "code", str(a.at))
                    edge(aid, 152, "code:%s" % a.at, CODE)
            for n in (V.get("says") or {}).get(r.id, []):
                edge(rid, 139, name(n))
            for t in r.checks:
                ent("test:%s" % t.rsplit(".", 1)[-1], "test", t)
                edge(rid, 149, "test:%s" % t.rsplit(".", 1)[-1])
        # every mutant of the code as a node: what it broke (an assertion, a branch cell), where, its verdict, the tests that caught
        # it. A verdict is a measurement (MEASURED), so a mutant never moves a requirement's fingerprint.
        for t in V.get("tests") or []:
            ent("test:%s" % t, "test", t)
        for key, b in (V.get("branches") or {}).items():
            ent("code:%s" % b["place"], "code", b["place"])
            cell = "cell:S-BRANCH:%s" % key
            if cell in ents:                                # the generated branch space's cell; never made here, or it is unmapped
                edge(cell, 152, "code:%s" % b["place"], CODE)
        for key, m in (V.get("mutants") or {}).items():
            mid = "mut:%s" % key
            ent(mid, "mutant", m.get("what"), m.get("place"))
            lit(mid, 133, m.get("mutant"))
            lit(mid, 136, {True: "killed", False: "survived"}.get(m.get("killed"), "no verdict"))
            lit(mid, 146, m.get("why"))
            ent("code:%s" % m["place"], "code", m["place"])
            edge(mid, 152, "code:%s" % m["place"], CODE)
            for of in m.get("of") or []:
                src_ = "assert:%s" % of if "#b" not in of else "cell:S-BRANCH:%s" % of
                if src_ in ents:
                    edge(src_, 147, mid, CODE)
            for t in m.get("by") or []:
                if not t.startswith("("):
                    ent("test:%s" % t, "test", t)
                    edge(mid, 148, "test:%s" % t, CODE)
        for t in model.terms.values():
            tid = "term:%s" % t.id
            ent(tid, "term", t.id)
            lit(tid, 133, t.why)
            for b in t.binds:
                edge(tid, 138, name(b))
            for r in model.requirements.values():
                if t.id and re.search(r"(?i)(?<!\w)%s(?!\w)" % re.escape(t.id), r.text or ""):
                    edge("req:%s" % r.id, 139, tid)
        for n, where in vn.items():
            lit(name(n), 144, where or "unresolved")
        for e in ucm.as_list(model.loop.get("entry")):
            if isinstance(e, dict) and e.get("name"):
                lit(name(e["name"]), 143, e.get("why"))
        for uc in model.usecases.values():
            uid = "uc:%s" % uc.id
            ent(uid, "use case", uc.name, uc.level)
            lit(uid, 131, uc.level)
            lit(uid, 110, None if ucm.blank(uc.raw.get("trigger")) else uc.raw.get("trigger"))
            lit(uid, 109, None if ucm.blank(uc.raw.get("success")) else uc.raw.get("success"))
            if uc.actor:
                edge(uid, 102, "actor:%s" % uc.actor)
            for a in uc.supporting:
                edge(uid, 130, "actor:%s" % a)
            p = uc.parent
            if p.get("uc"):
                target = ("step:%s#%s" % (p["uc"], p["step"])) if p.get("step") is not None else ("ext:%s#%s" % (p["uc"], p.get("ext")))
                edge(uid, 120, target)
            for g, prop in (("exec", 128), ("eval", 129)):
                for r in ucm.as_list((uc.gulf.get(g) or {}).get("reqs")):
                    edge(uid, prop, "req:%s" % r)
            prev = None
            for s in uc.steps:
                sid = "step:%s#%d" % (uc.id, s.n)
                ent(sid, "step", "%s step %d" % (uc.id, s.n), uc.level)
                edge(uid, 100, sid)
                lit(sid, 133, s.text)
                if prev:
                    edge(prev, 101, sid)
                prev = sid
                for r in s.reqs:
                    req = model.requirements.get(r)
                    edge(sid, 108 if req is not None and req.kind == "inferred" else 107, "req:%s" % r)
                for i in s.includes:
                    edge(sid, 121, "uc:%s" % i)
            for e in uc.ext:
                xid = "ext:%s#%s" % (uc.id, e.id)
                ent(xid, "extension", "%s %s" % (uc.id, e.id), uc.level)
                edge(uid, 116, xid)
                if isinstance(e.at, int):
                    edge(xid, 117, "step:%s#%d" % (uc.id, e.at))
                m = ucm.OUT_RE.match(str(e.out or ""))
                lit(xid, 118, e.out if m else None)
                if m and m.group(1):
                    edge(xid, 119, "step:%s#%s" % (uc.id, m.group(1)))
                lit(xid, 133, e.when)
                for r in e.reqs:
                    req = model.requirements.get(r)
                    edge(xid, 108 if req is not None and req.kind == "inferred" else 107, "req:%s" % r)
                for i in e.includes:
                    edge(xid, 121, "uc:%s" % i)
        for q in model.questions:
            qid = "q:%s" % q.id
            ent(qid, "question", q.text, None)
            if q.persona:
                edge(qid, 126, "actor:%s" % q.persona, 8)
            if q.inverse:
                edge(qid, 127, "uc:%s" % q.inverse, 8)
        db.executemany("INSERT INTO entity VALUES (?, ?, ?, ?)", sorted(ents.values()))
        db.executemany("INSERT INTO edge VALUES (?, ?, ?, ?)", sorted(edges))
        db.executemany("INSERT INTO literal VALUES (?, ?, ?)", sorted(lits))
        db.executemany("INSERT INTO meta VALUES (?, ?)", [("model", os.path.abspath(model.root).replace("\\", "/")),
                                                         ("format", "usecase_kg/1"),
                                                         ("loop", "closed" if model.closed_loop else "not closed"),
                                                         ("verdicts", V.get("trace") or "none")])
        db.commit()
        return kg

    FOLLOW = (100, 116, 107, 108, 121, 134)   # a node's fingerprint takes in what it contains: steps, pathways, requirements, effects

    def fingerprints(self):
        """entity -> a hash of what it says and of everything it holds: a requirement's text, kind and measure with each assertion
        of its effect and the code that assertion read; a step its text and the requirements it carries; a use case its steps
        and pathways. A change anywhere below a node moves the node's fingerprint, so a verdict taken before it is STALE.
        Verdicts are measurements of the content, not content, and never enter it (MEASURED)."""
        lits, outs = collections.defaultdict(list), collections.defaultdict(list)
        for s, p, v in self.db.execute("SELECT s, p, v FROM literal"):
            if p not in MEASURED:
                lits[s].append("%d=%s" % (p, v))
        for s, p, o in self.db.execute("SELECT s, p, o FROM edge"):
            if p not in MEASURED_EDGES:
                outs[s].append((p, o))
        memo = {}

        def fp(e, path):
            if e not in memo:
                below = ["%d>%s=%s" % (p, o, fp(o, path | {e}) if p in self.FOLLOW and o not in path else "") for p, o in outs[e]]
                memo[e] = hashlib.sha1("\n".join([e] + sorted(lits[e]) + sorted(below)).encode("utf-8")).hexdigest()[:16]
            return memo[e]
        return {e: fp(e, frozenset()) for (e,) in self.db.execute("SELECT id FROM entity ORDER BY id")}

    # ------------------------------------------------------------------ the gaps

    def schema_gaps(self):
        """Walk every schema row the store declares: each entity of that kind (and filter) lacking every listed relation."""
        out = []
        rows = self.db.execute("SELECT rule, kind, filter, edge_props, literal_props, direction, teach FROM schema_rule "
                               "ORDER BY rule").fetchall()
        for rule, kind, filt, eprops, lprops, direction, teach in rows:
            ep = [int(x) for x in (eprops or "").split(",") if x]
            lp = [int(x) for x in (lprops or "").split(",") if x]
            sql = "SELECT e.id FROM entity e WHERE e.kind = ?"
            args = [kind]
            if filt:
                vals = filt.split(",")
                sql += " AND e.sub IN (%s)" % ",".join("?" * len(vals))
                args += vals
            conds = []
            if ep and direction in ("out", "in-or-out"):
                conds.append("EXISTS (SELECT 1 FROM edge x WHERE x.s = e.id AND x.p IN (%s))" % ",".join(map(str, ep)))
            if ep and direction in ("in", "in-or-out"):
                conds.append("EXISTS (SELECT 1 FROM edge x WHERE x.o = e.id AND x.p IN (%s))" % ",".join(map(str, ep)))
            if lp:
                conds.append("EXISTS (SELECT 1 FROM literal l WHERE l.s = e.id AND l.p IN (%s))" % ",".join(map(str, lp)))
            sql += " AND NOT (%s) ORDER BY e.id" % " OR ".join(conds)
            for (eid,) in self.db.execute(sql, args):
                out.append(Gap(rule, eid, teach))
        return out

    def anchor_k(self):
        """requirement -> (k, families): distinct families behind it, the model's own structure excluded (ANCHOR API I2)"""
        # Only evidence FOR THIS requirement counts: a cell that maps to it, a source that states it. A persona's question
        # supports the use case that answers it, not each requirement that use case carries - counting it here inflated k
        # (the first selftest: a requirement stripped to one framework cell still read k = 2 through its use case's question).
        rows = self.db.execute("""
            SELECT r.id, x.fam FROM entity r JOIN edge x ON x.o = r.id AND x.p = 122 WHERE r.kind = 'requirement'
            UNION SELECT r.id, x.fam FROM entity r JOIN edge x ON x.s = r.id AND x.p = 114 WHERE r.kind = 'requirement'
        """).fetchall()
        fams = collections.defaultdict(set)
        for rid, fam in rows:
            if fam != MODEL:
                fams[rid].add(fam)
        out = {}
        for (rid, sub) in self.db.execute("SELECT id, sub FROM entity WHERE kind = 'requirement' ORDER BY id"):
            out[rid] = (len(fams[rid]), sorted(FAMILIES[f] for f in fams[rid]), sub)
        return out

    def reachable_requirements(self):
        """every requirement carried by a step or extension of a use case that descends from a kite"""
        return {r for (r,) in self.db.execute("""
            WITH RECURSIVE tree(uc) AS (
                SELECT e.id FROM entity e WHERE e.kind = 'use case' AND e.sub = 'kite'
                UNION
                SELECT c.s FROM edge c JOIN edge own ON own.o = c.o AND own.p IN (100, 116) JOIN tree t ON t.uc = own.s
                WHERE c.p = 120
                UNION
                SELECT inc.o FROM edge inc JOIN edge own ON own.o = inc.s AND own.p IN (100, 116) JOIN tree t ON t.uc = own.s
                WHERE inc.p = 121
            )
            SELECT DISTINCT x.o FROM tree t JOIN edge own ON own.s = t.uc AND own.p IN (100, 116)
            JOIN edge x ON x.s = own.o AND x.p IN (107, 108)
        """)}

    def graph_gaps(self):
        out = []
        known = {r for (r,) in self.db.execute("SELECT id FROM entity")}
        for s, p, o in self.db.execute("SELECT s, p, o FROM edge ORDER BY s, p, o"):
            if o not in known:
                out.append(Gap("G-REF", s, "relation %d points at %s, which is not in the graph" % (p, o)))
        ks = self.anchor_k()
        for rid, (k, fams, kind) in ks.items():
            if k == 0 and kind != "inferred":
                out.append(Gap("G-UNSOURCED", rid, "no evidence family stands behind it (k = 0): map a source sentence or a "
                                                  "framework cell to it, or mark it inferred"))
                continue
            # the KIND of family a requirement claims must be among its families (the same contract as usecase_model U6)
            if kind == "authored" and "story" not in fams:
                out.append(Gap("G-UNSOURCED", rid, "authored, but no story family (the product owner's words, a source sentence) states it"))
            # a CODE cell derives a requirement as a framework's does (usecase_model U6, 10-28). The graph learned it a day late:
            # its selftest failed from 08:58 to the closed-loop pass, unrun by any gate (L2 of the loop found it, DEBUG.md D37)
            elif kind == "derived" and not set(fams) & {"framework", "language", "brainstorm", "code"}:
                out.append(Gap("G-UNSOURCED", rid, "derived, but no framework, language, brainstorm or code cell maps to it"))
            if k == 1 and kind != "inferred":
                out.append(Gap("G-ANCHOR", rid, "rests on one family only (%s): which independent family would confirm it?"
                               % fams[0], failing=False))
        reach = self.reachable_requirements()
        for (rid,) in self.db.execute("""SELECT DISTINCT x.o FROM edge x WHERE x.p IN (107, 108) ORDER BY x.o"""):
            if rid in known and rid not in reach:
                out.append(Gap("G-REACH", rid, "carried only by use cases no epic reaches"))
        sig = collections.defaultdict(list)
        cells = collections.defaultdict(set)
        carriers = collections.defaultdict(set)
        for c, r in self.db.execute("SELECT s, o FROM edge WHERE p = 122"):
            cells[r].add(c)
        for s, r in self.db.execute("SELECT s, o FROM edge WHERE p IN (107, 108)"):
            carriers[r].add(s)
        for rid in ks:
            if cells[rid] and carriers[rid]:
                sig[(frozenset(cells[rid]), frozenset(carriers[rid]))].append(rid)
        for group in sig.values():
            if len(group) > 1:
                out.append(Gap("G-DUP", " ".join(sorted(group)), "mapped from the same cells and carried by the same "
                                                                  "pathways: one requirement, or two that need telling apart?",
                               failing=False))
        return out

    def chains(self):
        """the chaining, counted: source sentences whose requirement is carried somewhere an epic reaches"""
        reach = self.reachable_requirements()
        sentence_cells = self.db.execute("""SELECT c.id FROM entity c JOIN edge sp ON sp.s = c.id AND sp.p = 124
                                            JOIN entity s ON s.id = sp.o AND s.sub = 'prose' WHERE c.kind = 'cell'""").fetchall()
        complete = 0
        for (cid,) in sentence_cells:
            if any(r in reach for (r,) in self.db.execute("SELECT o FROM edge WHERE s = ? AND p = 122", (cid,))):
                complete += 1
        paths = self.db.execute("SELECT COUNT(*) FROM entity WHERE kind IN ('use case', 'extension')").fetchone()[0]
        return {"source sentences": len(sentence_cells), "reaching an epic": complete, "pathways": paths}

    def add_chain(self, model, chain):
        """an argument chain as nodes: the chain, each step (its words), and an edge from the step to everything it cites -
        a requirement, an owner question, or a code place. A cite that names nothing in the graph is a node too, so G-REF
        refuses it: a claim that stands on something the model does not hold is a hole, not a footnote."""
        cid = "chain:%s" % chain.get("id")
        rows = [(cid, "chain", chain.get("question"), chain.get("by"))]
        edges, lits = [], [(cid, 133, str(chain.get("question") or ""))]
        known = {e for (e,) in self.db.execute("SELECT id FROM entity")}
        for i, st in enumerate(ucm.as_list(chain.get("steps")), 1):
            sid = "%s#%d" % (cid, i)
            rows.append((sid, "chain step", "%s step %d" % (chain.get("id"), i), None))
            edges.append((cid, 150, sid, MODEL))
            lits.append((sid, 133, str(st.get("says") or "")))
            for c in ucm.as_list(st.get("cites")):
                c = str(c)
                target = ("req:%s" % c if c in model.requirements else
                          "q:%s" % c if any(q.id == c for q in model.questions) else "code:%s" % c)
                if target not in known and target.startswith("code:"):
                    rows.append((target, "code", c, None))
                    known.add(target)
                edges.append((sid, 151, target, MODEL))
        self.db.executemany("INSERT OR IGNORE INTO entity VALUES (?, ?, ?, ?)", rows)
        self.db.executemany("INSERT OR IGNORE INTO edge VALUES (?, ?, ?, ?)", edges)
        self.db.executemany("INSERT OR IGNORE INTO literal VALUES (?, ?, ?)", [x for x in lits if x[2].strip()])
        self.db.commit()
        return [sid for sid, kind, _, _ in rows if kind == "chain step"]

    def distance(self, a, b, limit=4, skip=(150, 151, 124, 147, 148)):
        """the fewest edges between two nodes, either direction, at most `limit` (None beyond it). Never through the chain's own
        cite edges (every pair a step cites would be two apart through the step), a space's membership edge (REVIEW_1 #8: the
        branch space's one hub put 82% of requirement pairs within 4), or the mutant/test measurements"""
        if a == b:
            return 0
        adj = collections.defaultdict(set)
        for s, p, o in self.db.execute("SELECT s, p, o FROM edge WHERE p NOT IN (%s)" % ",".join(map(str, skip))):
            adj[s].add(o)
            adj[o].add(s)
        seen, frontier = {a}, {a}
        for d in range(1, limit + 1):
            frontier = {n for f in frontier for n in adj[f]} - seen
            if b in frontier:
                return d
            seen |= frontier
        return None

    def counts(self):
        q = lambda sql: self.db.execute(sql).fetchone()[0]
        return {"entities": q("SELECT COUNT(*) FROM entity"), "edges": q("SELECT COUNT(*) FROM edge"),
                "literals": q("SELECT COUNT(*) FROM literal"), "schema rules": q("SELECT COUNT(*) FROM schema_rule")}

    def export_ntriples(self, path):
        """one triple per line, sorted: urn:ucm:<entity id>, urn:ucm:p<id>, literals escaped - byte-identical across rebuilds"""
        def iri(x):
            return "<urn:ucm:%s>" % "".join(c if c.isalnum() or c in "-._~:#" else "%%%02X" % ord(c) for c in x)

        def esc(v):
            return '"%s"' % v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
        lines = set()
        for eid, kind, label, sub in self.db.execute("SELECT id, kind, label, sub FROM entity"):
            lines.add("%s <urn:ucm:kind> %s ." % (iri(eid), esc(kind)))
        for s, p, o, fam in self.db.execute("SELECT s, p, o, fam FROM edge"):
            lines.add("%s <urn:ucm:p%d> %s ." % (iri(s), p, iri(o)))
            lines.add("%s <urn:ucm:p%d-f%d> %s ." % (iri(s), p, fam, iri(o)))
        for s, p, v in self.db.execute("SELECT s, p, v FROM literal"):
            lines.add("%s <urn:ucm:p%d> %s ." % (iri(s), p, esc(v)))
        io.open(path, "w", encoding="utf-8", newline="\n").write("\n".join(sorted(lines)) + "\n")
        return len(lines)


# ---------------------------------------------------------------------- the closed loop

class Criterion(object):
    """One property the requirements' node structure must have for an AI to act on them instead of on prompts, named by the
    mistake it stops. Together they leave a mistake no way into the code unless the requirements themselves contain it.
    `holes(gate)` are the Gaps that still let that mistake through; a `loop` criterion needs a closed-loop model's effects to be
    measured at all, and an open model is told it was not held to it."""

    def __init__(self, cid, name, mistake, method, loop=False):
        self.id, self.name, self.mistake, self.method, self.loop = cid, name, mistake, method, loop

    def holes(self, gate):
        return [Gap("%s-%s" % (self.id, self.name), g.entity, g.message, g.failing, g.owner) for g in getattr(gate, self.method)()]


LOOP = [
    Criterion("L1", "IDENTITY", "a rebuild points an edge at another node, and a verdict follows the wrong thing", "identity"),
    Criterion("L2", "SOURCED", "a requirement nothing outside the model states, or a cell dropped with no reason", "sourced"),
    Criterion("L3", "CORROBORATED", "a requirement only one family states inherits that family's mistakes (listed)", "corroborated"),
    Criterion("L4", "CARRIED", "a requirement no pathway an epic reaches realises: nobody builds it", "carried"),
    Criterion("L5", "ENDED", "a pathway with no end or no actor, a step carrying nothing: behaviour nobody specified", "ended"),
    Criterion("L6", "ANSWERED", "a question nobody answered, settled by a guess", "answered"),
    Criterion("L7", "REALIZED", "code linked to the right requirement that does the wrong thing", "realized", loop=True),
    Criterion("L8", "FALSIFIABLE", "a check that passes whatever the code does", "falsifiable", loop=True),
    Criterion("L9", "JUSTIFIED", "code no requirement asked for: behaviour that slipped in", "justified"),
    Criterion("L10", "DEFINED", "a requirement naming what the code does not have, or a word read two ways", "defined", loop=True),
    Criterion("L11", "CAUSAL", "a step relying on a condition nothing makes true", "causal", loop=True),
    Criterion("L12", "CONSISTENT", "two requirements demanding opposite effects: meeting one breaks the other", "consistent", loop=True),
    Criterion("L13", "CHANGE-SAFE", "a requirement, or the code it reads, changed after it was verified", "change_safe", loop=True),
    # 2026-10-29: L7 and L8 hold the code's SHAPE - a statement at the
    # place reads like the effect, and the matcher can tell when it does not. Neither proves the code BEHAVES: a payment of the
    # wrong amount has the right shape. L14 asks the tests: the statement each effect matched, mutated in behaviour, must make
    # one of them fail (the code adapter's `kill` verdict, cs_effect_check.py for C#).
    Criterion("L14", "KILLED", "code that can be broken with no test noticing: the requirement is checked by nothing that runs", "killed", loop=True),
    # 2026-10-29, the same night: L14 broke an effect ONE way. L15 breaks every BRANCH every way - the
    # intention on each branch, held by a test - and L16 asks that the requirement's OWN check be the test that notices.
    Criterion("L15", "PINNED", "a branch whose behaviour no test holds: break it any way and everything passes", "pinned", loop=True),
    Criterion("L16", "OWN-CHECK", "a requirement whose named check is blind to its code breaking: another test noticed, or none", "own_check", loop=True),
    # the prior art's first pick (research_PRIOR_ART_1029.md): equivalents out before survivors are counted, each with its proof
    Criterion("L19", "EQUIVALENT", "a survivor waved off as 'the same program' with no proof, or a proof a test refutes", "equivalent", loop=True),
]


class LoopGate(object):
    """Every Criterion over one graph: which mistakes the requirements still let through, counted. A hole of OURS (the model's
    author can close it) fails the gate; the OWNER's questions are counted and keep the loop OPEN, never hidden. THE LOCK
    (<model>/loop.lock, a line per requirement: id, fingerprint, why): a requirement whose fingerprint moved since it was accepted
    is STALE, with everything that carries it, until it is verified and accepted again - and `accept` refuses while any other
    hole of ours stands, so a lock can never be taken to turn a red loop green."""

    def __init__(self, kg, model, verdicts=None):
        self.kg, self.model, self.V = kg, model, verdicts or {}
        self.lock_path = os.path.join(model.root, "loop.lock")
        self._gaps = None

    def _rules(self, *rules):
        if self._gaps is None:
            self._gaps = self.kg.schema_gaps() + self.kg.graph_gaps()
        return [g for g in self._gaps if g.rule in rules]

    def _code_cell(self, cell):
        basis = dict(self.kg.db.execute("SELECT id, sub FROM entity WHERE kind = 'space'"))
        return basis.get("space:%s" % cell.split(":")[1]) == "code"

    def _reqs(self):
        return self.model.requirements.values()

    def _owners_only(self, users):
        """True when every requirement among `users` is inferred: the name is code the owner's question proposes"""
        rs = [self.model.requirements[u] for u in users if u in self.model.requirements]
        return bool(rs) and len(rs) == len(users) and all(r.kind == "inferred" for r in rs)

    def triples_apart(self, a, b):
        """how many N-Triples differ between the stores two models build"""
        import tempfile
        d, lines = tempfile.mkdtemp(prefix="loop_identity_"), []
        for i, m in enumerate((a, b)):
            kg = KnowledgeGraph.build(m, os.path.join(d, "%d.sqlite" % i), self.V)
            kg.export_ntriples(os.path.join(d, "%d.nt" % i))
            kg.db.close()
            lines.append(set(io.open(os.path.join(d, "%d.nt" % i), encoding="utf-8").read().splitlines()))
        return len(lines[0] ^ lines[1])

    # ------------------------------------------------------------------ the criteria, one method each

    def identity(self):
        """L1: the same model read in reverse order builds the same triples - every id comes from what a node is"""
        rev = copy.copy(self.model)
        for attr in ("requirements", "usecases", "terms", "spaces", "actors", "sources"):
            d = getattr(self.model, attr)
            setattr(rev, attr, type(d)(reversed(list(d.items()))))
        rev.questions = list(reversed(self.model.questions))
        n = self.triples_apart(self.model, rev)
        return [] if not n else [Gap("", "the store", "%d triples differ between two orders of one model: an id comes from a position" % n)]

    def sourced(self):
        """L2: every requirement has an evidence family beyond the model's own structure, and every cell outside the code is
        mapped or excluded with its reason - READ or ABSENT, never a third"""
        return self._rules("G-UNSOURCED") + [g for g in self._rules("S-CELL-MAPPED") if not self._code_cell(g.entity)]

    def corroborated(self):
        """L3: listed - a requirement only one family states is as right as that family"""
        return self._rules("G-ANCHOR")

    def carried(self):
        """L4: every requirement is carried on a pathway an epic reaches"""
        return self._rules("S-REQ-CARRIED", "G-REACH")

    def ended(self):
        """L5: every use case has its actor, trigger, goal, steps and two gulfs; every step and pathway carries a requirement and
        every pathway says where it ends"""
        return self._rules("S-UC-ACTOR", "S-UC-TRIGGER", "S-UC-GOAL", "S-UC-STEPS", "S-UC-GULF-EXEC", "S-UC-GULF-EVAL",
                           "S-KITE-STEP-CHILD", "S-STEP-CARRIES", "S-EXT-OUTCOME", "S-EXT-CARRIES", "S-PERSON-PRIMARY")

    def answered(self):
        """L6: every brainstormed question is inverted into the use case that answers it; every inferred requirement is a question
        for the owner, open and counted until the owner answers it"""
        return self._rules("S-Q-ANSWERED") + [
            Gap("", "req:%s" % r.id, "the owner's question: %s" % (r.why or r.text or "")[:170], owner=True)
            for r in self._reqs() if r.kind == "inferred"]

    def realized(self):
        """L7: every assertion holds against the code, so the code does what each requirement says. An inferred requirement's
        failing assertion is a defect the loop found, and a requirement that has no effect says why: the owner's, if inferred"""
        out, va = [], self.V.get("assertions") or {}
        for r in self._reqs():
            owner = r.kind == "inferred"
            if not r.effect:
                why = "no effect, and no reason it has none" if ucm.blank(r.absent) else r.absent
                out.append(Gap("", "req:%s" % r.id, "nothing checks the code against it: %s" % why,
                               owner=owner and not ucm.blank(r.absent)))
            for a in r.effect:
                v = va.get(a.id)
                if v is None:
                    out.append(Gap("", "assert:%s" % a.id, "not measured: no verdict (run the code adapter and pass --verdicts)"))
                elif v.get("holds") is None:
                    out.append(Gap("", "assert:%s" % a.id, "unread: %s" % v.get("why")))
                elif v.get("holds") is False:
                    out.append(Gap("", "assert:%s" % a.id, "%s: %s (%s)" % (
                        "the code lacks what the owner's question asks" if owner else "the code does not do what it says",
                        a.text(), v.get("why")), owner=owner))
        return out

    def falsifiable(self):
        """L8: every verdict was planted the other way and flipped - a check that cannot change its answer checks nothing"""
        va = self.V.get("assertions") or {}
        return [Gap("", "assert:%s" % a.id, "its plant %s" % va[a.id].get("plant")) for r in self._reqs() for a in r.effect
                if va.get(a.id, {}).get("holds") is not None and va[a.id].get("plant") != "flips"]

    def killed(self):
        """L14: every assertion that holds was broken in BEHAVIOUR - its matched statement mutated in the real source - and a test
        failed. A mutant every test passes is a hole of ours (a test is missing or blind); an adapter that mutates nothing, or a
        place no runner executes, is listed with its reason - checked by its shape only, never counted as killed."""
        out, va = [], self.V.get("assertions") or {}
        for r in self._reqs():
            for a in r.effect:
                v = va.get(a.id) or {}
                if v.get("holds") is not True:
                    continue
                k = v.get("kill")
                if not isinstance(k, dict):
                    out.append(Gap("", "assert:%s" % a.id, "shape only: the code adapter mutated no code, so no test was asked", failing=False))
                elif k.get("killed") is False:
                    out.append(Gap("", "assert:%s" % a.id, "a mutant survived every test: %s" % k.get("why")))
                elif k.get("killed") is None:
                    out.append(Gap("", "assert:%s" % a.id, "not killable: %s" % k.get("why"), failing=False))
        return out

    def pinned(self):
        """L15: every branch of the code the tests cover was broken every way the adapter knows - its condition negated, each
        boundary moved, each && made ||, its body emptied - and each break failed a test. A survivor is a branch whose behaviour
        no test pins: intention nobody checks. It is ours, and the requirement claiming the branch is named so the missing test
        lands where it belongs."""
        bs = self.V.get("branches")
        if bs is None:
            return [Gap("", "the branches", "not measured: the code adapter broke no branch (cs_effect_check does, for C#)", failing=False)]
        claims = collections.defaultdict(list)
        for c, r in self.kg.db.execute("SELECT s, o FROM edge WHERE p = 122 AND s LIKE 'cell:S-BRANCH:%'"):
            claims[c.split(":", 2)[2]].append(r[4:])
        out = []
        for key, b in sorted(bs.items()):
            who = ", ".join(sorted(claims.get(key, []))) or "no requirement"
            if b.get("killed") is False:
                out.append(Gap("", "cell:S-BRANCH:%s" % key, "%s (claimed by %s)" % (b.get("why"), who)))
            elif b.get("killed") is None:
                out.append(Gap("", "cell:S-BRANCH:%s" % key, "not pinned: %s" % b.get("why"), failing=False))
        return out

    def own_check(self):
        """L16: when a requirement's code is broken, ITS OWN check fails - a test it names (checks:) - not merely some test. A
        kill by another test proves the code is checked, not that the requirement's check checks it: a blind or renamed check
        passes while an unrelated test happens to notice. Per mutant: each break of its effects' statements must fail one of
        its checks. A check naming a test that never ran is a hole; a requirement whose code is killed and names no check is
        listed."""
        ms = self.V.get("mutants")
        if ms is None:
            return [Gap("", "the checks", "not measured: no mutants were run (cs_effect_check does, for C#)", failing=False)]
        ran = set(self.V.get("tests") or [])
        # the branches a requirement CLAIMS are its code as much as its effects' statements (REVIEW_1 #7, 10-29): its check must
        # notice them break too, or a claim is a word with no test behind it
        claimed = collections.defaultdict(set)
        for c, r in self.kg.db.execute("SELECT s, o FROM edge WHERE p = 122 AND s LIKE 'cell:S-BRANCH:%'"):
            claimed[r[4:]].add(c.split(":", 2)[2])
        out = []
        for r in self._reqs():
            aids = {a.id for a in r.effect} | claimed[r.id]
            mine_ms = [(k, m) for k, m in sorted(ms.items()) if aids & set(m.get("of") or []) and m.get("killed") is True]
            if not mine_ms:
                continue
            if not r.checks:
                by = sorted({t for _, m in mine_ms for t in m.get("by") or []})
                out.append(Gap("", "req:%s" % r.id, "names no test as its check (checks:); its code is caught by %s" % ", ".join(by[:3]), failing=False))
                continue
            mine = {c.rsplit(".", 1)[-1] for c in r.checks}
            gone = sorted(mine - ran) if ran else []
            if gone:
                out.append(Gap("", "req:%s" % r.id, "its check names a test that never ran: %s" % ", ".join(gone)))
            # REVIEW_2 #3 (10-29): naming more checks could only REMOVE holes - name all 41 tests and L16 reads 0. So each named
            # check must itself catch at least one break of this requirement's code; a check that catches none is decoration
            caught_by = {t for _, m in mine_ms for t in m.get("by") or []}
            idle = sorted((mine & ran) - caught_by) if ran else sorted(mine - caught_by)
            if idle:
                out.append(Gap("", "req:%s" % r.id, "names %d check(s) that catch none of its code's breaks: %s" % (len(idle), ", ".join(idle[:3]))))
            blind = [(k, m) for k, m in mine_ms if not mine & set(m.get("by") or [])]
            if blind:
                out.append(Gap("", "req:%s" % r.id, "%d break(s) of its code its own check misses (caught only by others): %s" % (
                    len(blind), "; ".join("%s <- %s" % (k.split(": ", 1)[-1], ", ".join((m.get("by") or [])[:2])) for k, m in blind[:2]))))
        return out

    def equivalent(self):
        """L19: a surviving mutant counts as no hole only when it is EQUIVALENT, and every equivalence has its proof: the same
        compiled IL as the original (Trivial Compiler Equivalence: a fact, counted), or the model's reason (meta.loop.equivalent:
        a person's judgement, listed). A declared equivalent a test KILLS is no equivalent - the claim is wrong; one whose mutant
        no longer exists names nothing (its statement's text changed: read it again). Both are ours. (Prior art: Papadakis et
        al. 2015; mutation can refute closure, never prove it - "closed" is relative to the operators.)"""
        ms = self.V.get("mutants")
        if ms is None:
            return [Gap("", "the mutants", "not measured: no mutants were run (cs_effect_check does, for C#)", failing=False)]
        out = []
        for key, why in sorted((self.V.get("equivalent_declared") or {}).items()):
            m = ms.get(key)
            if m is None:
                out.append(Gap("", "mut:%s" % key, "declared equivalent, but no such mutant: its statement's text changed - read it again"))
            elif m.get("killed") is True:
                out.append(Gap("", "mut:%s" % key, "declared equivalent, but a test kills it (%s): a real mutant" % ", ".join((m.get("by") or [])[:2])))
            else:
                out.append(Gap("", "mut:%s" % key, "declared equivalent: %s" % why, failing=False))
        for key, why in sorted((self.V.get("equivalent_lapsed") or {}).items()):
            out.append(Gap("", "mut:%s" % key, "a declared equivalent LAPSED: %s" % why))
        tce = sum(1 for m in ms.values() if m.get("equivalent") == "TCE")
        if tce:
            out.append(Gap("", "the mutants", "%d proven equivalent by their compiled IL (TCE), never run" % tce, failing=False))
        return out

    def justified(self):
        """L9: every code unit maps to a requirement or is excluded with its reason"""
        return [g for g in self._rules("S-CELL-MAPPED") if self._code_cell(g.entity)]

    def defined(self):
        """L10: every code name a requirement's text, a term or an assertion uses resolves in the code (or the pasted text) - a
        name only inferred requirements use is code the owner's question proposes. A requirement naming no code name and using
        no term is listed: which code does it mean?"""
        if self.V.get("names") is None:
            return [Gap("", "the names", "not measured: no verdicts (run the code adapter and pass --verdicts)")]
        says, users = self.V.get("says") or {}, collections.defaultdict(set)
        for rid, names in says.items():
            for n in names:
                users[n].add(rid)
        for r in self._reqs():
            for a in r.effect:
                for n in (a.target, a.by):
                    if n:
                        users[str(n)].add(r.id)
        for t in self.model.terms.values():
            for b in t.binds:
                users[b].add("term:%s" % t.id)
        out = [Gap("", "name:%s" % n, "names nothing in the code or the paste (used by %s)" % ", ".join(sorted(users[n])),
                   owner=self._owners_only(users[n])) for n, where in sorted(self.V["names"].items()) if where is None]
        out += [Gap("", "term:%s" % t.id, "binds no code name") for t in self.model.terms.values() if not t.binds]
        for r in self._reqs():
            if not says.get(r.id) and not any(re.search(r"(?i)(?<!\w)%s(?!\w)" % re.escape(t), r.text or "") for t in self.model.terms):
                out.append(Gap("", "req:%s" % r.id, "names no code name and uses no term: which code does it mean?", failing=False))
        return out

    def causal(self):
        """L11: every condition an assertion branches on is set by some requirement's effect, or is an entry condition set outside
        the model with its reason - a `branches` with no `sets` behind it is a requirement nobody wrote"""
        sets = {a.target for r in self._reqs() for a in r.effect if a.verb == "sets"}
        # a condition a CLAIMED method computes is made true by the requirements claiming it (REVIEW_1 #4b, 10-29): CanPay and
        # StockLeft were excused as "outside the model" while the covered code computed them; an entry the model computes itself
        # is listed, so the excuse is dropped instead of standing
        computed = {}
        for c in ucm.as_list((self.model.loop.get("branches") or {}).get("claims")):
            if isinstance(c, dict) and c.get("method"):
                computed[c["method"].rsplit(".", 1)[-1]] = (c["method"], ucm.as_list(c.get("reqs")))
        entry = {e.get("name") for e in ucm.as_list(self.model.loop.get("entry")) if isinstance(e, dict)}
        by = collections.defaultdict(list)
        for r in self._reqs():
            for a in r.effect:
                if a.verb == "branches" and a.target not in sets and a.target not in entry and a.target not in computed:
                    by[a.target].append(a)
        out = [Gap("", "name:%s" % n, "branched on (%s) but no requirement sets it and it is no entry condition: which requirement "
                   "makes it true?" % ", ".join(a.id for a in xs), owner=self._owners_only({a.req for a in xs}))
               for n, xs in sorted(by.items())]
        return out + [Gap("", "name:%s" % n, "an entry condition the model computes itself (%s, claimed by %s): drop the entry" % (
            computed[n][0], ", ".join(computed[n][1])), failing=False) for n in sorted(entry) if n in computed]

    def consistent(self):
        """L12: no two requirements set one name to different values at one statement or outcome; two requirements with one
        effect are listed (one requirement, or two that need telling apart?)"""
        by, eff = collections.defaultdict(set), collections.defaultdict(list)
        for r in self._reqs():
            for a in r.effect:
                if a.verb == "sets" and a.to is not None and "#" in str(a.at):
                    by[(a.target, str(a.at))].add((str(a.to).lower(), r.id))
            if r.effect:
                eff[tuple(sorted(a.text() for a in r.effect))].append(r.id)
        out = [Gap("", "name:%s" % t, "set to %s at %s: opposite effects" % (" and to ".join("%s by %s" % v for v in sorted(vals)), at))
               for (t, at), vals in sorted(by.items()) if len({v for v, _ in vals}) > 1]
        return out + [Gap("", " ".join(ids), "the same effect: one requirement, or two that need telling apart?", failing=False)
                      for ids in eff.values() if len(ids) > 1]

    def change_safe(self):
        """L13: every requirement's fingerprint is the one accepted in the lock; one that moved is STALE and everything carrying it
        is listed with it; one never accepted was never verified"""
        fps, lock, out, stale = self.kg.fingerprints(), self.read_lock(), [], set()
        for r in self._reqs():
            rid = "req:%s" % r.id
            if r.id not in lock:
                out.append(Gap("", rid, "never verified: no line in %s" % os.path.basename(self.lock_path)))
            elif lock[r.id][0] != fps.get(rid):
                stale.add(rid)
                out.append(Gap("", rid, "STALE: it, or the code it reads, changed after it was accepted (%s)" % lock[r.id][1]))
        carriers = sorted({s for s, o in self.kg.db.execute("SELECT s, o FROM edge WHERE p IN (107, 108)") if o in stale})
        return out + [Gap("", c, "carries a stale requirement: verify it with that one", failing=False) for c in carriers]

    # ------------------------------------------------------------------ the lock and the verdict

    def read_lock(self):
        out = {}
        if os.path.exists(self.lock_path):
            for line in io.open(self.lock_path, encoding="utf-8"):
                parts = line.rstrip("\n").split(" ", 2)
                if len(parts) >= 2 and not line.startswith("#"):
                    out[parts[0]] = (parts[1], parts[2] if len(parts) > 2 else "")
        return out

    def held(self):
        return [c for c in LOOP if not c.loop or self.model.closed_loop]

    def accept(self, why):
        """the lock written from the fingerprints now; refused (the blocking holes returned) while any hole of ours but the lock's
        own stands. A requirement whose fingerprint did not move keeps the reason it was accepted with"""
        blocking = [g for c in self.held() if c.id != "L13" for g in c.holes(self) if g.failing and not g.owner]
        if blocking:
            return blocking
        fps, old = self.kg.fingerprints(), self.read_lock()
        lines = ["# usecase_kg.py loop --accept: each requirement's fingerprint when it was last verified, and why. Never edit."]
        for r in self._reqs():
            fp = fps["req:%s" % r.id]
            lines.append("%s %s %s" % (r.id, fp, old[r.id][1] if r.id in old and old[r.id][0] == fp else why))
        io.open(self.lock_path, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
        return []

    def run(self, ours=False, limit=8):
        held, holes = self.held(), {}
        for c in held:
            holes[c.id] = c.holes(self)
        self.last_holes = holes          # what this run found, criterion by criterion (closed_loop keeps them beside its report)
        print("CLOSED LOOP  %s" % self.model.root)
        if not self.model.closed_loop:
            print("  NOT a closed-loop model (meta.loop.closed is not set): the criteria that need effects were not held")
        print("  %-4s %-13s %4s %5s %6s  the mistake it stops" % ("", "criterion", "ours", "owner", "listed"))
        for c in LOOP:
            h = holes.get(c.id)
            counts = ("-", "-", "-") if h is None else (sum(1 for g in h if g.failing and not g.owner),
                                                         sum(1 for g in h if g.owner), sum(1 for g in h if not g.failing))
            print("  %-4s %-13s %4s %5s %6s  %s%s" % ((c.id, c.name) + counts + (c.mistake, "" if h is not None else "  [not held]")))
        for c in held:
            for kind, pick in (("OURS", lambda g: g.failing and not g.owner), ("OWNER", lambda g: g.owner),
                               ("listed", lambda g: not g.failing)):
                gs = [g for g in holes[c.id] if pick(g)]
                if gs:
                    print("  %s %s, %s: %d" % (c.id, c.name, kind, len(gs)))
                    for g in gs[:limit]:
                        print("      %s: %s" % (g.entity, g.message))
                    if len(gs) > limit:
                        print("      ... %d more" % (len(gs) - limit))
        every = [g for h in holes.values() for g in h]
        n_ours, n_owner = sum(1 for g in every if g.failing and not g.owner), sum(1 for g in every if g.owner)
        # one owner question shows in several criteria (asked in L6, failing in L7, proposing a name in L10): count both
        asked = sum(1 for g in holes.get("L6", []) if g.owner)
        if n_ours:
            print("\n  LOOP BROKEN: %d hole(s) of ours - each one a way a mistake gets past the requirements" % n_ours)
        elif n_owner:
            print("\n  LOOP OPEN: every hole of ours is closed; %d owner hole(s), from %d question(s) only the owner can answer, "
                  "keep it open" % (n_owner, asked))
        else:
            print("\n  LOOP CLOSED: every criterion holds")
        return 1 if n_ours or (n_owner and not ours) else 0


def report(kg, limit, t_build=None):
    c, ch = kg.counts(), kg.chains()
    print("KNOWLEDGE GRAPH  %s" % kg.path)
    print("  %d entities, %d edges, %d literals; %d schema rules declared by the store%s" % (
        c["entities"], c["edges"], c["literals"], c["schema rules"], "" if t_build is None else "; built in %.0f ms" % t_build))
    t = time.perf_counter()
    gaps = kg.schema_gaps() + kg.graph_gaps()
    ks = kg.anchor_k()
    dist = collections.Counter(k for k, _, kind in ks.values() if kind != "inferred")
    print("  anchor count over %d requirements: %s   (k = distinct evidence families, the model's own structure excluded)" % (
        len(ks), ", ".join("k=%d: %d" % (k, dist[k]) for k in sorted(dist))))
    print("  chains: %d source sentences, %d reach an epic; %d use cases and pathways" % (
        ch["source sentences"], ch["reaching an epic"], ch["pathways"]))
    failing = [g for g in gaps if g.failing]
    listed = [g for g in gaps if not g.failing]
    print("  gaps found in %.1f ms: %d failing, %d listed (single-family requirements, possible duplicates)" % (
        (time.perf_counter() - t) * 1000, len(failing), len(listed)))
    by = collections.defaultdict(list)
    for g in gaps:
        by[g.rule].append(g)
    for rule in sorted(by):
        print("  %-18s %4d %s" % (rule, len(by[rule]), "" if by[rule][0].failing else "(listed, not failing)"))
        for g in by[rule][:limit]:
            print("      %s: %s" % (g.entity, g.message))
        if len(by[rule]) > limit:
            print("      ... %d more" % (len(by[rule]) - limit))
    return 1 if failing else 0


# ---------------------------------------------------------------------- selftest

AGREE = {  # usecase_model rule -> the graph rule(s) that must fire on the same plant
    "U2": ("G-REF",), "U3": ("S-KITE-STEP-CHILD",), "U4": ("S-EXT-OUTCOME",), "U5": ("S-REQ-CARRIED",),
    "U6": ("G-UNSOURCED",), "U7": ("S-CELL-MAPPED",), "U8": ("G-REF",), "U9": ("S-PERSON-PRIMARY",),
    "U10": ("S-REQ-MEASURE",), "U11": ("G-REF",), "U12": ("S-STEP-CARRIES",), "U13": ("S-UC-GULF-EVAL",),
}


def selftest():
    import tempfile
    tmp = tempfile.mkdtemp(prefix="ucg_selftest_")
    prose = os.path.join(tmp, "src.txt")
    io.open(prose, "w", encoding="utf-8").write("The reader wants to see the flow. The view must not lie.\n")
    meta, data = ucm._clean_data(prose)
    db = os.path.join(tmp, "g.sqlite")
    ok = True

    def gaps_of(m, d):
        kg = KnowledgeGraph.build(ucm.UseCaseModel.from_data(m, d), db)
        g = kg.schema_gaps() + kg.graph_gaps()
        kg.db.close()
        return g

    clean = [g for g in gaps_of(meta, data) if g.failing]
    print("selftest: clean model -> %d failing gap(s)%s" % (len(clean), "" if not clean else ": " + "; ".join(map(str, clean))))
    ok &= not clean
    for rule, what, plant in ucm.PLANTS:
        if rule not in AGREE:
            continue
        m, d = copy.deepcopy(meta), copy.deepcopy(data)
        plant(m, d)
        got = {g.rule for g in gaps_of(m, d) if g.failing}
        hit = any(r in got for r in AGREE[rule])
        ok &= hit
        print("selftest: %-4s %-40s graph says %-22s %s" % (rule, what, ",".join(sorted(got)) or "-", "AGREES" if hit else "DISAGREES"))
    # the graph's own gap classes, which the rules over YAML cannot see
    m, d = copy.deepcopy(meta), copy.deepcopy(data)
    d["spaces"][0]["cells"]["1.2"] = {"excluded": "a planted exclusion"}      # R-2 keeps only its framework cell: k 2 -> 1
    single = {g.entity for g in gaps_of(m, d) if g.rule == "G-ANCHOR"}
    hit = "req:R-2" in single and not [x for x in ucm.UseCaseModel.from_data(m, d).check()]
    ok &= hit
    print("selftest: G-ANCHOR R-2 loses its source sentence, rests on one family -> %s (the YAML rules see nothing: %s)" % (
        "LISTED" if "req:R-2" in single else "MISSED", "true" if not ucm.UseCaseModel.from_data(m, d).check() else "false"))
    m, d = copy.deepcopy(meta), copy.deepcopy(data)
    # R-4, not R-3: the clean model holds an R-3 since it gained a code space (10-28), and a second R-3 tested a duplicate id
    d["requirements"].append({"id": "R-4", "kind": "derived", "text": "t2", "measure": "m"})
    d["spaces"][1]["cells"]["D1.V1"] = ["R-2", "R-4"]
    d["spaces"][0]["cells"]["1.2"] = ["R-2", "R-4"]
    for u in d["usecases"]:
        for s in u.get("steps", []):
            if "R-2" in s.get("reqs", []):
                s["reqs"].append("R-4")
        for e in u.get("ext", []):
            if "R-2" in e.get("reqs", []):
                e["reqs"].append("R-4")
    dup = [g for g in gaps_of(m, d) if g.rule == "G-DUP"]
    hit = bool(dup) and not ucm.UseCaseModel.from_data(m, d).check()
    ok &= hit
    print("selftest: G-DUP   a second requirement with the same cells and carriers -> %s (YAML rules: %d findings)" % (
        "LISTED" if dup else "MISSED", len(ucm.UseCaseModel.from_data(m, d).check())))
    # stability: two builds of one model export byte-identical N-Triples
    kg = KnowledgeGraph.build(ucm.UseCaseModel.from_data(meta, data), db)
    a, b = os.path.join(tmp, "a.nt"), os.path.join(tmp, "b.nt")
    kg.export_ntriples(a)
    kg.db.close()
    kg = KnowledgeGraph.build(ucm.UseCaseModel.from_data(meta, data), db)
    kg.export_ntriples(b)
    kg.db.close()
    same = io.open(a, encoding="utf-8").read() == io.open(b, encoding="utf-8").read()
    ok &= same
    print("selftest: two builds export identical N-Triples -> %s" % same)
    ok &= loop_selftest(tmp, prose)
    print("selftest: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def _loop_data(prose):
    """the selftest model made a closed loop: each requirement with an effect, a term, an entry condition, and verdicts"""
    meta, data = ucm._clean_data(prose)
    meta["loop"] = {"closed": True, "entry": [{"name": "Input", "why": "the player's button, outside the model"}]}
    effects = {"R-1": [{"sets": "Flag", "to": True, "at": "G#s2"}], "R-2": [{"branches": "Flag", "at": "G#s1"}],
               "R-3": [{"calls": "Act", "args": ["Flag"], "at": "G#s1:then"}, {"branches": "Input", "at": "G#s3"}],
               "IR-1": [{"calls": "Fix", "at": "G"}]}
    for r in data["requirements"]:
        r["effect"] = effects[r["id"]]
    data["terms"] = [{"term": "flag", "binds": ["Flag"], "why": "the one condition"}]
    for r in data["requirements"]:
        if r["id"] != "IR-1":
            r["checks"] = ["T.One"]
    data["spaces"].append({"id": "S-BRANCH", "title": "every branch of the covered code (generated)", "basis": "code", "kind": "list",
                           "cells": {"G#b1": {"text": "if (x)", "maps": ["R-1"]}}})
    k = {"killed": True, "by": ["One"], "why": "the planted mutant failed a test"}
    mut = {"killed": True, "by": ["One"], "why": "failed: One", "place": "G", "what": "the condition negated", "mutant": "if (!(x))"}
    v = {"assertions": {"R-1#1": {"holds": True, "plant": "flips", "fp": "a1", "kill": dict(k)},
                        "R-2#1": {"holds": True, "plant": "flips", "fp": "b1", "kill": dict(k)},
                        "R-3#1": {"holds": True, "plant": "flips", "fp": "c1", "kill": dict(k)},
                        "R-3#2": {"holds": True, "plant": "flips", "fp": "c2", "kill": dict(k)},
                        "IR-1#1": {"holds": False, "plant": "flips", "fp": "d1", "why": "no statement calls Fix"}},
         "names": {"Flag": "code", "Act": "code", "Input": "code", "Fix": None}, "says": {},
         "tests": ["One"],
         "branches": {"G#b1": {"killed": True, "place": "G", "why": "2 of 2 mutant(s) killed", "text": "if (x)"}},
         "mutants": {"G#s1: the condition negated": dict(mut, of=["R-1#1", "R-2#1", "G#b1"]),
                     "G#s3: the call removed": dict(mut, of=["R-3#1", "R-3#2"], what="the call removed")}}
    return meta, data, v


LOOP_PLANTS = [  # (criterion, what, plant(meta, data, verdicts), the kind of hole it must raise: ours, owner or listed)
    ("L2", "a derived requirement loses its framework cell", lambda m, d, v: d["spaces"][1]["cells"].update({"D1.V1": ["R-1"]}), "ours"),
    ("L3", "a requirement left with one family", lambda m, d, v: d["spaces"][0]["cells"].update({"1.2": {"excluded": "x"}}), "listed"),
    ("L4", "a requirement no pathway carries", lambda m, d, v: (d["requirements"].append(
        {"id": "R-9", "kind": "authored", "text": "t", "measure": "m", "sources": ["OWNER"], "effect": [{"calls": "Act", "at": "G#s1:then"}]}),
        v["assertions"].update({"R-9#1": {"holds": True, "plant": "flips", "fp": "e"}})), "ours"),
    ("L5", "a pathway with no end", lambda m, d, v: ucm._uc(d, "UC-1")["ext"][0].pop("out"), "ours"),
    ("L6", "a question nobody inverts", lambda m, d, v: d["questions"][0].pop("inverse"), "ours"),
    ("L6", "an inferred requirement", lambda m, d, v: None, "owner"),
    ("L7", "code that does not do what R-3 says", lambda m, d, v: v["assertions"]["R-3#1"].update(holds=False), "ours"),
    ("L8", "a verdict its plant could not flip", lambda m, d, v: v["assertions"]["R-1#1"].update(plant="does not flip: x"), "ours"),
    ("L9", "a code node no requirement maps", lambda m, d, v: d["spaces"][3]["cells"].pop("K2Node_IfThenElse_0"), "ours"),
    ("L10", "a name the code does not have", lambda m, d, v: v["names"].update(Flag=None), "ours"),
    ("L10", "a name only the owner's question proposes", lambda m, d, v: None, "owner"),
    ("L11", "a branch on a condition nothing sets", lambda m, d, v: d["requirements"][1].update(effect=[{"branches": "Other", "at": "G#s1"}]), "ours"),
    ("L11", "an entry the model's own claimed method computes", lambda m, d, v: m["loop"].update(
        branches={"claims": [{"method": "G.Input", "reqs": ["R-3"], "why": "computed"}]}), "listed"),
    ("L12", "R-3 sets Flag false where R-1 sets it true", lambda m, d, v: d["requirements"][2]["effect"].append(
        {"sets": "Flag", "to": False, "at": "G#s2"}), "ours"),
    ("L13", "R-1's text edited after it was accepted", lambda m, d, v: d["requirements"][0].update(text="t edited"), "ours"),
    ("L13", "the code R-1 reads changed", lambda m, d, v: v["assertions"]["R-1#1"].update(fp="a2"), "ours"),
    ("L14", "a mutant of R-1's code every test passes", lambda m, d, v: v["assertions"]["R-1#1"].update(
        kill={"killed": False, "why": "the value replaced by default -> every test passed"}), "ours"),
    ("L14", "an adapter that mutated no code", lambda m, d, v: v["assertions"]["R-2#1"].pop("kill"), "listed"),
    ("L15", "a branch mutant every test passes", lambda m, d, v: v["branches"]["G#b1"].update(
        killed=False, why="1 mutant(s) survived every test: boundary <= to < (1)"), "ours"),
    ("L15", "a branch no test reaches", lambda m, d, v: v["branches"]["G#b1"].update(killed=None, why="no test covers G"), "listed"),
    ("L16", "R-3's code caught, but not by its own check", lambda m, d, v: v["mutants"]["G#s3: the call removed"].update(by=["Other"]), "ours"),
    ("L16", "R-1's check names a test that never ran", lambda m, d, v: d["requirements"][0].update(checks=["T.Gone"]), "ours"),
    ("L16", "R-3 names no check", lambda m, d, v: d["requirements"][2].pop("checks"), "listed"),
    ("L16", "a branch R-1 claims, caught only by another test", lambda m, d, v: v["mutants"].update({"G#s1: boundary < to <= (1)": dict(
        killed=True, by=["Other"], why="failed: Other", place="G", what="boundary < to <= (1)", mutant="if (x <= y)", of=["G#b1"])}), "ours"),
    ("L19", "a declared equivalent a test kills", lambda m, d, v: v.update(equivalent_declared={"G#s1: the condition negated": "same"}), "ours"),
    ("L19", "an equivalent whose rests_on method changed", lambda m, d, v: v.update(equivalent_lapsed={"G#s3: the call removed": "rests on G.Act, changed"}), "ours"),
    ("L16", "R-1 names a check that catches none of its breaks", lambda m, d, v: d["requirements"][0].update(checks=["T.One", "T.Idle"])
                                                                                  or v["tests"].append("Idle"), "ours"),
    ("L19", "a declared equivalent naming no mutant", lambda m, d, v: v.update(equivalent_declared={"G@gone: x": "same"}), "ours"),
    ("L19", "a survivor declared equivalent", lambda m, d, v: (v["mutants"]["G#s3: the call removed"].update(killed=False, by=[]),
                                                               v.update(equivalent_declared={"G#s3: the call removed": "Act is idempotent"})), "listed"),
]


def loop_selftest(tmp, prose):
    """the clean closed loop passes and takes its lock; each criterion raises its own hole on its own plant (L1 by comparison)"""
    root, ok = os.path.join(tmp, "loopmodel"), True
    os.makedirs(root, exist_ok=True)

    def gate(m, d, v, name):
        model = ucm.UseCaseModel.from_data(m, d, root=root)
        return LoopGate(KnowledgeGraph.build(model, os.path.join(tmp, name + ".sqlite"), v), model, v)
    meta, data, v = _loop_data(prose)
    g = gate(meta, data, v, "clean")
    blocking = g.accept("the selftest's clean model")
    ours = [h for c in LOOP for h in c.holes(g) if h.failing and not h.owner]
    good = not blocking and not ours and not ucm.UseCaseModel.from_data(meta, data, root=root).check()
    ok &= good
    print("selftest: LOOP clean closed loop -> accepted %s, %d hole(s) of ours%s %s" % (
        not blocking, len(ours), "" if not ours else ": " + "; ".join(map(str, ours[:3])), "PASS" if good else "FAIL"))
    model = ucm.UseCaseModel.from_data(meta, data, root=root)
    bigger = copy.deepcopy(data)
    bigger["requirements"][0]["text"] = "a different requirement"
    apart = g.triples_apart(model, ucm.UseCaseModel.from_data(meta, bigger, root=root))
    good = not g.identity() and apart > 0
    ok &= good
    print("selftest: LOOP L1  one model in two orders: identical; a changed model: %d triples apart -> %s" % (apart, "PASS" if good else "FAIL"))
    for i, (cid, what, plant, kind) in enumerate(LOOP_PLANTS):
        m, d, vv = copy.deepcopy(meta), copy.deepcopy(data), copy.deepcopy(v)
        plant(m, d, vv)
        crit = next(c for c in LOOP if c.id == cid)
        got = crit.holes(gate(m, d, vv, "plant%d" % i))     # a store per plant: Windows will not delete one still open
        hit = [h for h in got if (kind == "ours" and h.failing and not h.owner) or (kind == "owner" and h.owner) or
               (kind == "listed" and not h.failing)]
        ok &= bool(hit)
        print("selftest: LOOP %-4s %-44s -> %s %s" % (cid, what, "CAUGHT (%s)" % kind if hit else "MISSED", hit[0].entity if hit else ""))
    return ok


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["build", "gaps", "loop"])
    ap.add_argument("folder", nargs="?")
    ap.add_argument("--db")
    ap.add_argument("--nt", help="also write the graph as sorted N-Triples")
    ap.add_argument("--max", type=int, default=12)
    ap.add_argument("--verdicts", help="loop: the code adapter's verdicts.json (each assertion held to the code)")
    ap.add_argument("--ours", action="store_true", help="loop: fail on holes of ours only; the owner's questions are still printed")
    ap.add_argument("--accept", help="loop: write the lock now, with this reason - refused while a hole of ours stands")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "loop":
        if not a.folder:
            ap.error("loop needs the model folder")
        try:
            model = ucm.UseCaseModel.load(a.folder)
        except (RuntimeError, OSError) as e:
            print("usecase_kg: cannot read the model: %s" % e, file=sys.stderr)
            return 2
        v = json.load(io.open(a.verdicts, encoding="utf-8")) if a.verdicts else None
        kg = KnowledgeGraph.build(model, a.db or os.path.join(tempfile.mkdtemp(prefix="usecase_kg_loop_"), "g.sqlite"), v)
        gate = LoopGate(kg, model, v)
        if a.accept:
            blocking = gate.accept(a.accept)
            if blocking:
                print("usecase_kg loop: REFUSED - no lock is taken while %d hole(s) of ours stand:" % len(blocking))
                for g in blocking[:a.max]:
                    print("      %s" % g)
                return 1
            print("usecase_kg loop: the lock is written - %s" % gate.lock_path.replace("\\", "/"))
        return gate.run(a.ours, a.max)
    if not a.cmd or not a.db:
        ap.error("build <model folder> --db PATH, or gaps --db PATH, or --selftest")
    if a.cmd == "build":
        if not a.folder:
            ap.error("build needs the model folder")
        t = time.perf_counter()
        try:
            model = ucm.UseCaseModel.load(a.folder)
        except (RuntimeError, OSError) as e:
            print("usecase_kg: cannot read the model: %s" % e, file=sys.stderr)
            return 2
        kg = KnowledgeGraph.build(model, a.db)
        t_build = (time.perf_counter() - t) * 1000
        if a.nt:
            n = kg.export_ntriples(a.nt)
            print("  wrote %d triples to %s" % (n, a.nt))
        return report(kg, a.max, t_build)
    if not os.path.exists(a.db):
        print("usecase_kg: no store at %s - build it first" % a.db, file=sys.stderr)
        return 2
    return report(KnowledgeGraph(a.db), a.max)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
