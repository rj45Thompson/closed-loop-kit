#!/usr/bin/env python3
"""loop_stats.py - the closed loop's TELEMETRY: every pass measured, one row each, in a morph log beside the model.

WHY (2026-10-29): telemetry is a pillar of the loop. Any stat that measures growth or improvement is measured, and each loop
brainstorms what else to measure.
A loop that improves itself has to be able to SHOW it: whether its net got wider (more mutants, more operators, more decisions),
whether what it catches grew faster than what slips past, what the owner still has to answer, and what each pass cost. A number
written only in a report is gone at the next pass; a row in <model>/../stats.jsonl is a curve. STATS.md says what each stat is,
why it measures growth and how it is computed; its brainstorm list grows every pass.

THE PIECES:
  PassStats   one pass's numbers, every one READ from what the pass wrote (verdicts.json, the graph, the report) - never estimated
  MorphLog    the rows, in order: append one, read them back, and the deltas between passes
  Trend       one stat across the passes, read against the direction its kind should move (improving / worsening / stalled / done)
  Predictors  CANDIDATE predictive pairs: a change in X this pass against the change in Y the next, with n, anecdotes and
              arithmetic (mechanical) pairs labelled
  NextImprovement  the stats choose the next improvement: worsening beats stalled; the graph-for-GNN tune is due every 5 passes
  PeriodicReport   every 20 passes: trends, the graph's growth (this model and the loop's own), predictors, tunes, the pick

    python -X utf8 loop_stats.py record <model> [--report R.md] [--seconds S] [--note "..."]
    python -X utf8 loop_stats.py show <model> [--last 10]
    python -X utf8 loop_stats.py trends <model>
    python -X utf8 loop_stats.py report <model> [<model> ...]
    python -X utf8 loop_stats.py --selftest
"""
import argparse
import collections
import io
import json
import os
import re
import sqlite3
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# the project the model's paths are relative to: this repo, or a developer's own (CLOSED_LOOP_ROOT, set by the kit's installer)
ROOT = os.path.abspath(os.environ.get("CLOSED_LOOP_ROOT") or os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import usecase_model as ucm  # noqa: E402

TOOLS = ["closed_loop.py", "cs_effect_check.py", "usecase_kg.py", "usecase_model.py", "graph_gnn.py", "ab_loop.py", "loop_stats.py"]
MAPS = 122          # cell -> requirement in usecase_kg's graph (graph_gnn.MAPS; graph_gnn needs torch, and this file must run without it)


def owner_pillars(db_path):
    """how concentrated the owning of code is (SD-3: some requirements are pillars): the share of branch-to-requirement links the top 3
    requirements hold, and how many requirements hold 80% of them. {} on a graph with no such link: absent, never 0"""
    db = sqlite3.connect(db_path)
    owned = collections.Counter(o for _, o in db.execute("SELECT s, o FROM edge WHERE p = ? AND s LIKE 'cell:S-BRANCH:%'", (MAPS,)))
    db.close()
    total = sum(owned.values())
    if not total:
        return {}
    held, k80 = 0, None
    for i, (_, n) in enumerate(owned.most_common(), 1):
        held += n
        if k80 is None and held >= 0.8 * total:
            k80 = i
    return {"pillar_top3_share": round(sum(n for _, n in owned.most_common(3)) / float(total), 3), "pillars_80": k80}


def op_kind(what):
    """an operator's KIND: 'boundary <= to < (2)' -> boundary, "Pay's argument 1 made default" -> argument made default"""
    w = re.sub(r"\s*\(\d+\)$", "", what or "")
    w = re.sub(r"^(boundary|equality|logic|arithmetic) .*", r"\1", w)
    w = re.sub(r"^\w+'s argument \d+ made default$", "argument made default", w)
    w = re.sub(r"^the number \d+ made \d+$", "number made one more", w)
    w = re.sub(r"^the text .* emptied$", "text emptied", w)
    w = re.sub(r"^(true|false) made (true|false)$", "bool flipped", w)
    return w


class PassStats(object):
    """One pass's numbers, each READ from what the pass left: verdicts.json (mutants, effects, decisions, tests), the graph
    (nodes and edges by kind), the model (musts, mays, requirements, use cases), the report (holes per criterion, the GNN's
    scores, the chains). A number this cannot read is left out of the row, never filled in."""

    def __init__(self, model_dir, report=None, seconds=None, note=None):
        self.model_dir, self.report, self.seconds, self.note = model_dir, report, seconds, note

    def collect(self):
        kg = os.path.join(self.model_dir, "..", "kg")
        name = os.path.basename(os.path.dirname(os.path.abspath(self.model_dir)))
        v = json.load(io.open(os.path.join(kg, "verdicts.json"), encoding="utf-8"))
        ms, va, bs = v.get("mutants") or {}, v.get("assertions") or {}, v.get("branches") or {}
        row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": name}
        if self.note:
            row["note"] = self.note
        if self.seconds is not None:
            row["seconds"] = round(float(self.seconds), 1)
        # THE NET: how hard this pass tried to break the code
        row["mutants"] = len(ms)
        row["operator_kinds"] = len({op_kind(m.get("what")) for m in ms.values()})
        row["decisions"] = len(bs)
        # WHAT IT CAUGHT, WHAT SLIPPED
        killed = [m for m in ms.values() if m.get("killed") is True]
        surv = [m for m in ms.values() if m.get("killed") is False]
        declared = v.get("equivalent_declared") or {}
        row["killed"] = len(killed)
        row["survived"] = len(surv)
        row["survived_unexplained"] = sum(1 for k, m in ms.items() if m.get("killed") is False and k not in declared)
        row["equivalent_tce"] = sum(1 for m in ms.values() if m.get("equivalent") == "TCE")
        row["equivalent_declared"] = len(declared)
        row["equivalent_lapsed"] = len(v.get("equivalent_lapsed") or {})
        row["no_verdict"] = sum(1 for m in ms.values() if m.get("killed") is None and m.get("equivalent") != "TCE")
        row["kill_rate"] = round(len(killed) / float(len(killed) + row["survived_unexplained"]), 4) if killed else None
        row["decisions_pinned"] = sum(1 for b in bs.values() if b.get("killed") is True)
        row["effects"] = len(va)
        row["effects_killed"] = sum(1 for a in va.values() if (a.get("kill") or {}).get("killed") is True)
        row["effects_shape_only"] = sum(1 for a in va.values() if a.get("holds") and (a.get("kill") or {}).get("killed") is None)
        # WHERE THE GAPS ARE FOUND: survivors and kills per operator kind (which operators earn their run time)
        row["survivors_by_operator"] = dict(collections.Counter(op_kind(m.get("what")) for k, m in ms.items()
                                                                if m.get("killed") is False and k not in declared))
        row["kills_by_operator"] = dict(collections.Counter(op_kind(m.get("what")) for m in killed))
        # THE TESTS: how many, how many each kills (the strongest and the idle ones)
        tests = v.get("tests") or []
        by_test = collections.Counter(t for m in killed for t in m.get("by") or [])
        row["tests"] = len(tests)
        row["tests_idle"] = sum(1 for t in tests if by_test.get(t, 0) == 0)
        row["test_kills_top"] = by_test.most_common(5)
        # THE MODEL: what must hold, what only the owner can settle
        model = ucm.UseCaseModel.load(self.model_dir)
        row["requirements"] = len(model.requirements)
        row["musts"] = sum(1 for r in model.requirements.values() if r.kind != "inferred")
        row["mays"] = sum(1 for r in model.requirements.values() if r.kind == "inferred")
        row["use_cases"] = len(model.usecases)
        row["claims"] = len(ucm.as_list((model.loop.get("branches") or {}).get("claims")))
        # THE GRAPH
        dbp = os.path.join(kg, "%s.sqlite" % name)
        if os.path.exists(dbp):
            db = sqlite3.connect(dbp)
            row["graph_nodes"] = db.execute("SELECT COUNT(*) FROM entity").fetchone()[0]
            row["graph_edges"] = db.execute("SELECT COUNT(*) FROM edge").fetchone()[0]
            row["graph_by_kind"] = dict(db.execute("SELECT kind, COUNT(*) FROM entity GROUP BY kind").fetchall())
            db.close()
            row.update(owner_pillars(dbp))
        # WHERE THE QUESTIONS ARE ASKED (who/what/why/when/where/before/after per level of the model)
        row.update(FacetGrid([u.raw for u in model.usecases.values()]).stats())
        # THE BUZZSAW: the assumptions the requirements rest on, and the widest the owner has to answer
        if os.path.exists(dbp):
            import buzzsaw
            vp = os.path.join(kg, "verdicts.json")
            row.update(buzzsaw.Buzzsaw(dbp, vp if os.path.exists(vp) else None).stats())
        # THE REPORT: holes per criterion, the GNN, the chains
        if self.report and os.path.exists(self.report):
            text = io.open(self.report, encoding="utf-8").read()
            holes = {}
            for m in re.finditer(r"^\s+(L\d+)\s+[A-Z-]+\s+(\d+)\s+(\d+)\s+(\d+)\s", text, re.M):
                holes[m.group(1)] = [int(m.group(2)), int(m.group(3)), int(m.group(4))]
            row["holes"] = holes                                  # criterion -> [ours, owner, listed]
            row["holes_ours"] = sum(h[0] for h in holes.values())
            g = {n: float(s) for n, s in re.findall(r"^\s+(random|words|method|effects|gnn)\s+[\d.]+\s+[\d.]+\s+([\d.]+)", text, re.M)}
            if g:
                row["gnn_mrr"] = g
            c = re.search(r"Chain \S+ .*?: (\d+) steps, (\d+) ungrounded; (\d+) owner question", text)
            if c:
                row["chain"] = {"steps": int(c.group(1)), "ungrounded": int(c.group(2)), "blind": int(c.group(3))}
            row["loop"] = "HOLDS" if "closed_loop: HOLDS" in text else "BROKEN" if "BROKEN" in text else None
        # THE TOOL ITSELF: how big, and how much of it is proved by its own plants
        row["tool_lines"] = sum(sum(1 for _ in io.open(os.path.join(HERE, f), encoding="utf-8")) for f in TOOLS if os.path.exists(os.path.join(HERE, f)))
        try:
            row["tool_commit"] = subprocess.run(["git", "-C", os.path.join(ROOT, ".opus-tools"), "rev-parse", "--short", "HEAD"],
                                                capture_output=True, text=True).stdout.strip() or None
        except OSError:
            pass
        return {k: v for k, v in row.items() if v is not None}


class MorphLog(object):
    """The rows, one per pass, at <model>/../stats.jsonl: how the loop and the model morphed. Read back with deltas between passes."""

    def __init__(self, model_dir):
        self.path = os.path.join(model_dir, "..", "stats.jsonl")

    def rows(self):
        return [json.loads(l) for l in io.open(self.path, encoding="utf-8")] if os.path.exists(self.path) else []

    def add(self, row):
        rows = self.rows()
        row = dict(row, n=(rows[-1].get("n", len(rows)) + 1) if rows else 1)
        io.open(self.path, "a", encoding="utf-8").write(json.dumps(row, sort_keys=True) + "\n")
        return row

    def show(self, last=10):
        rows = self.rows()[-last:]
        keys = ["mutants", "operator_kinds", "decisions", "killed", "survived", "survived_unexplained", "equivalent_tce", "equivalent_declared",
                "kill_rate", "decisions_pinned", "effects_killed", "effects_shape_only", "tests", "mays", "assumptions", "holes_ours", "graph_nodes", "seconds"]
        print("MORPH LOG  %s  (%d pass(es))" % (os.path.normpath(self.path), len(self.rows())))
        for r in rows:
            print("  pass %-3s %s  %s" % (r.get("n"), r.get("time", "")[:16], r.get("note", "")[:90]))
            print("      " + "  ".join("%s=%s" % (k, r[k]) for k in keys if k in r))
            if r.get("gnn_mrr"):
                print("      GNN MRR " + ", ".join("%s %.2f" % kv for kv in sorted(r["gnn_mrr"].items())))
        return rows


# every stat a trend is read for: its KIND (STATS.md) and the direction growth means for it (+1 up is good, -1 down is good,
# 0 no direction of its own). graph_nodes / graph_edges are the record growing
KINDS = {
    "mutants": ("NET", +1), "operator_kinds": ("NET", +1), "decisions": ("NET", +1), "tests": ("NET", +1),
    "graph_nodes": ("NET", +1), "graph_edges": ("NET", +1), "requirements": ("NET", +1), "use_cases": ("NET", +1),
    "killed": ("CLOSURE", +1), "decisions_pinned": ("CLOSURE", +1), "effects_killed": ("CLOSURE", +1), "kill_rate": ("CLOSURE", +1),
    "pinned_share": ("CLOSURE", +1), "gnn": ("CLOSURE", +1), "gnn_margin": ("CLOSURE", +1),
    "survived_unexplained": ("OPEN", -1), "holes_ours": ("OPEN", -1), "effects_shape_only": ("OPEN", -1), "no_verdict": ("OPEN", -1),
    "mays": ("OPEN", -1), "tests_idle": ("OPEN", -1),
    "seconds": ("COST", 0), "seconds_per_mutant": ("COST", -1), "tool_lines": ("TOOL", 0),
    # where the model asks who/what/why/... (FacetGrid) and the assumptions a buzzsaw flips (buzzsaw.py)
    "facets_answered": ("CLOSURE", +1), "facets_open": ("OPEN", -1), "facets_never_asked": ("TOOL", 0),
    "may_radius_max": ("OPEN", -1), "entry_unread": ("OPEN", -1),
    # how few requirements own the code (owner_pillars, SD-3): no direction of its own, improvement's triggers read a run of it
    "pillar_top3_share": ("TOOL", 0), "pillars_80": ("TOOL", 0),
}


def value(row, stat):
    """a stat's value in one row, the DERIVED ones computed from what the row holds (never filled in when a part is missing)"""
    if stat in row and isinstance(row[stat], (int, float)) and not isinstance(row[stat], bool):
        return float(row[stat])
    g = row.get("gnn_mrr") or {}
    if stat == "gnn":
        return g.get("gnn")
    if stat == "gnn_margin":
        others = [v for k, v in g.items() if k != "gnn"]
        return round(g["gnn"] - max(others), 4) if "gnn" in g and others else None
    if stat == "pinned_share" and row.get("decisions"):
        return round(row.get("decisions_pinned", 0) / float(row["decisions"]), 4) if "decisions_pinned" in row else None
    if stat == "seconds_per_mutant" and row.get("seconds") and row.get("mutants"):
        return round(row["seconds"] / float(row["mutants"]), 3)
    return None


class FacetGrid(object):
    """WHERE the model asks who / what / why / when / where / before / after: those questions belong in
    all the relevant places, and whether they are asked is itself a stat. Each LEVEL of the model - a use case,
    a step, an extension - has a field for some of the seven and none for the rest. A field filled is ANSWERED, a field left empty
    is OPEN, and no field at all is NEVER ASKED at that level. Read from the model's own files, never inferred. Whether asking more
    helps or hurts accuracy is for the A/B pairs to measure; these counts go into the morph log so Predictors can pair them.
    MEASURED 10-29: every use case of all three models answers all seven (the 19 rules require them), so the gaps are one level
    down - a step records only what it does and why, never who, when, where, before or after."""

    FACETS = ("who", "what", "why", "when", "where", "before", "after")
    FIELDS = {"use case": {"who": "actor", "what": "name", "why": "why", "when": "trigger", "where": "where", "before": "pre", "after": "success"},
              "step": {"what": "do", "why": "reqs"},
              "extension": {"when": "when", "what": "do", "why": "reqs", "after": "out"}}

    def __init__(self, usecases):
        self.usecases = [u for u in usecases if isinstance(u, dict)]

    @staticmethod
    def of(model_dir):
        return FacetGrid([u.raw for u in ucm.UseCaseModel.load(model_dir).usecases.values()])

    def items(self, level):
        if level == "use case":
            return self.usecases
        key = "steps" if level == "step" else "ext"
        return [x for u in self.usecases for x in ucm.as_list(u.get(key)) if isinstance(x, dict)]

    def grid(self):
        """{(level, facet): None (never asked there) or (answered, open)}"""
        out = {}
        for level, fields in self.FIELDS.items():
            its = self.items(level)
            for f in self.FACETS:
                if f not in fields:
                    out[(level, f)] = None
                    continue
                filled = sum(1 for x in its if not (ucm.blank(x.get(fields[f])) if not isinstance(x.get(fields[f]), list) else not x.get(fields[f])))
                out[(level, f)] = (filled, len(its) - filled)
        return out

    def stats(self):
        g = self.grid()
        return {"facets_answered": sum(v[0] for v in g.values() if v), "facets_open": sum(v[1] for v in g.values() if v),
                "facets_never_asked": sum(1 for v in g.values() if v is None)}

    def lines(self):
        g = self.grid()
        out = []
        for level in self.FIELDS:
            n = len(self.items(level))
            cells = ["%s %s" % (f, "never asked" if g[(level, f)] is None else "%d/%d" % (g[(level, f)][0], n)) for f in self.FACETS]
            out.append("%s (%d): %s" % (level, n, ", ".join(cells)))
        return out


class Trend(object):
    """One stat across the passes of a morph log, read against the direction its kind should move: IMPROVING, WORSENING, STALLED
    (the last three values within 1% of each other), DONE (an OPEN stat stalled at 0, a share stalled at 1), MOVING (a stat with
    no direction of its own) or TOO FEW (under three passes hold it). The slope is least squares per pass."""

    FLAT = 0.01

    def __init__(self, stat, rows):
        self.stat = stat
        self.kind, self.good = KINDS.get(stat, ("?", 0))
        self.points = [(r.get("n", i + 1), v) for i, r in enumerate(rows) for v in [value(r, stat)] if v is not None]

    def slope(self):
        if len(self.points) < 2:
            return None
        xs, ys = [float(n) for n, _ in self.points], [v for _, v in self.points]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        den = sum((x - mx) ** 2 for x in xs)
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else None

    def verdict(self):
        vals = [v for _, v in self.points]
        if len(vals) < 3:
            return "too few"
        last = vals[-3:]
        if max(last) - min(last) <= self.FLAT * (max(abs(v) for v in vals) or 1.0):
            if (self.good < 0 and last[-1] == 0) or (self.stat in ("pinned_share", "kill_rate") and last[-1] >= 1.0):
                return "done"
            return "stalled"
        if self.good == 0:
            return "moving"
        return "improving" if (vals[-1] - vals[-3]) * self.good > 0 else "worsening"

    def line(self):
        if not self.points:
            return "%s: no pass holds it" % self.stat
        (n0, v0), (n1, v1) = self.points[0], self.points[-1]
        sl = self.slope()
        want = {1: "should rise", -1: "should fall", 0: "no direction of its own"}[self.good]
        # the verdict reads the LAST THREE passes, so the line shows them: "0 -> 42, improving" alone reads as a contradiction
        # when the last three were 92, 47, 42 (MEASURED 10-29 on the shops' log)
        return "%s (%s, %s): %s -> %s over passes %s-%s%s; last three %s: %s" % (
            self.stat, self.kind, want, fmt(v0), fmt(v1), n0, n1, ", %+.3g a pass" % sl if sl is not None else "",
            ", ".join(fmt(v) for _, v in self.points[-3:]), self.verdict().upper())


def fmt(v):
    return ("%d" % v) if float(v).is_integer() else ("%.3g" % v)


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1
        i = j + 1
    return r


def spearman(xs, ys):
    """the rank correlation, or None when either side never changes"""
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


class Predictors(object):
    """CANDIDATE predictive stats: for every pair of
    stats, the rank correlation of the change in X at one pass with the change in Y at the NEXT pass - does X moving now tell
    what Y does next? Each pair says its n, and under ANECDOTE points it is labelled an anecdote: a handful of passes can show
    a pattern worth watching, never prove one. A pair one stat makes by arithmetic (killed is part of mutants) is tagged
    MECHANICAL: it predicts nothing new."""

    ANECDOTE = 8
    MECHANICAL = {frozenset(p) for p in [
        ("mutants", "killed"), ("mutants", "survived_unexplained"), ("mutants", "no_verdict"), ("mutants", "graph_nodes"),
        ("mutants", "graph_edges"), ("graph_nodes", "graph_edges"), ("decisions", "decisions_pinned"), ("decisions", "pinned_share"),
        ("decisions_pinned", "pinned_share"), ("gnn", "gnn_margin"), ("killed", "kill_rate"), ("survived_unexplained", "kill_rate"),
        ("mutants", "seconds"), ("seconds", "seconds_per_mutant"), ("mutants", "seconds_per_mutant"), ("tests", "tests_idle"),
        ("decisions", "graph_nodes"), ("decisions", "graph_edges")]}

    def __init__(self, rows, min_n=4):
        self.rows, self.min_n = rows, min_n

    def deltas(self, stat):
        out = {}
        prev = None
        for i, r in enumerate(self.rows):
            n, v = r.get("n", i + 1), value(r, stat)
            if v is not None and prev is not None and prev[0] == n - 1:
                out[n] = v - prev[1]
            prev = (n, v) if v is not None else None
        return out

    def pairs(self, top=8):
        d = {s: self.deltas(s) for s in KINDS}
        found = []
        for x in KINDS:
            for y in KINDS:
                if x == y:
                    continue
                ns = sorted(n for n in d[x] if n + 1 in d[y])
                if len(ns) < self.min_n:
                    continue
                r = spearman([d[x][n] for n in ns], [d[y][n + 1] for n in ns])
                if r is not None:
                    found.append({"x": x, "y": y, "r": round(r, 2), "n": len(ns), "mechanical": frozenset((x, y)) in self.MECHANICAL,
                                  "anecdote": len(ns) < self.ANECDOTE})
        found.sort(key=lambda p: (p["mechanical"], -abs(p["r"])))
        return found[:top]

    @staticmethod
    def line(p):
        return "a change in %s %s the next pass's change in %s (rank r = %+.2f, n = %d)%s%s" % (
            p["x"], "goes WITH" if p["r"] > 0 else "goes AGAINST", p["y"], p["r"], p["n"],
            "; MECHANICAL: one is made from the other" if p["mechanical"] else "",
            "; an ANECDOTE until %d passes" % Predictors.ANECDOTE if p["anecdote"] else "")


class NextImprovement(object):
    """The stats themselves choose what to improve next. Each AREA names the stats that
    speak for it; a WORSENING one scores 3, a STALLED one 2 (DONE scores nothing: there is nothing left there to win). The
    graph-for-GNN tune (graph_gnn.py tune) is also due once TUNE_EVERY passes go by without one. The picks come ranked, each
    with the trend lines that chose it, so the choice can be argued with."""

    AREAS = [
        ("tests", "write tests for what got past every test", ("survived_unexplained", "holes_ours", "pinned_share", "effects_shape_only")),
        ("graph-for-gnn", "reshape the graph the GNN reads: graph_gnn.py tune <graph>", ("gnn_margin", "gnn")),
        ("net", "widen the net: a new kind of break", ("operator_kinds",)),
        ("speed", "make a pass cheaper per break", ("seconds_per_mutant",)),
        ("owner", "put the open mays to the owner, widest blast radius first (buzzsaw.py; theirs, never ours to settle)", ("mays", "may_radius_max")),
        ("assumptions", "remove or re-point the entry conditions nothing reads (buzzsaw.py)", ("entry_unread",)),
        ("questions", "answer the who / what / why ... fields the model leaves open (FacetGrid)", ("facets_open",)),
        ("coverage", "bring more code under the requirements", ("decisions", "no_verdict")),
    ]
    SCORE = {"worsening": 3, "stalled": 2}
    TUNE_EVERY = 5

    FIRED = 3            # a fired improvement trigger (improvement.py, RC-4) speaks as loudly as a worsening trend

    def __init__(self, rows, tunes=(), fired=()):
        """`fired`: the improvement triggers that fired on these rows ({id, area, detail}): each adds FIRED to the area it names"""
        self.rows, self.tunes, self.fired = rows, list(tunes), list(fired)

    def picks(self):
        out = []
        for area, what, stats in self.AREAS:
            trends = [Trend(s, self.rows) for s in stats]
            # an area scores its WORST stat, never the sum: two stats read from one source (gnn and gnn_margin) stalled together
            # are one reason, not two (MEASURED 10-29 in the selftest: the sum ranked a stalled GNN over a worsening survivor count)
            score = max([self.SCORE.get(t.verdict(), 0) for t in trends] or [0])
            why = [t.line() for t in trends if self.SCORE.get(t.verdict())]
            if area == "graph-for-gnn":
                last_n = self.rows[-1].get("n", len(self.rows)) if self.rows else 0
                tuned = max([t.get("pass", 0) for t in self.tunes] or [0])
                if last_n - tuned >= self.TUNE_EVERY or not self.tunes:
                    score += 1
                    why.append("due: %s" % ("never tuned" if not self.tunes else "last tuned at pass %d, %d passes ago" % (tuned, last_n - tuned)))
            for f in self.fired:
                if f.get("area") == area:
                    score = max(score, self.FIRED)
                    why.append("%s fired: %s" % (f["id"], f.get("detail", "")))
            if score:
                out.append({"area": area, "what": what, "score": score, "why": why})
        # a hole of ours outranks every other pick (KT-6k / P5-D, 10-29): after a planted break the pick said "tune the GNN" while the
        # missing test stood. The tests area comes first, its why names the count, every other area keeps its score
        holes = (self.rows[-1].get("holes_ours") or 0) if self.rows else 0
        if holes > 0:
            t = next((p for p in out if p["area"] == "tests"), None)
            if t is None:
                t = {"area": "tests", "what": self.AREAS[0][1], "score": 0, "why": []}
                out.append(t)
            # where they stand, read every pass: improvement's IMP-1 only speaks when this name changes
            by = self.rows[-1].get("holes") or {}
            most = max(by, key=lambda k: (by[k][0], k)) if by else None
            t["why"].insert(0, "%d hole%s of ours standing in the last pass%s" % (holes, "" if holes == 1 else "s",
                                                                                  ", most in %s (%d)" % (most, by[most][0]) if most else ""))
            return sorted(out, key=lambda p: (p is not t, -p["score"]))
        return sorted(out, key=lambda p: -p["score"])


class PeriodicReport(object):
    """Every EVERY passes, to see whether the graph of the requirements is growing: one report beside the morph log with each stat's trend, the
    GRAPH'S GROWTH (nodes, edges, and nodes by kind, first pass to last) for the model and for the closed loop's OWN requirements
    model (selfmodel/), the predictive candidates, the graph-for-GNN tunes so far, and what the stats pick next."""

    EVERY = 20
    SELF = os.path.join(HERE, "selfmodel", "model")

    def __init__(self, model_dirs):
        seen, self.models = set(), []
        for m in list(model_dirs) + [self.SELF]:
            k = os.path.normcase(os.path.abspath(m))
            if k not in seen and os.path.isdir(m):
                seen.add(k)
                self.models.append(m)

    @staticmethod
    def due(rows):
        return bool(rows) and rows[-1].get("n", len(rows)) % PeriodicReport.EVERY == 0

    @staticmethod
    def growth(rows):
        g = [r for r in rows if "graph_nodes" in r]
        if not g:
            return ["no pass recorded the graph's size"]
        a, b = g[0], g[-1]
        out = ["the graph: %d -> %d nodes (%+d), %d -> %d edges (%+d), passes %s-%s" % (
            a["graph_nodes"], b["graph_nodes"], b["graph_nodes"] - a["graph_nodes"], a.get("graph_edges", 0), b.get("graph_edges", 0),
            b.get("graph_edges", 0) - a.get("graph_edges", 0), a.get("n"), b.get("n"))]
        ka, kb = a.get("graph_by_kind") or {}, b.get("graph_by_kind") or {}
        moved = sorted(((k, ka.get(k, 0), kb.get(k, 0)) for k in set(ka) | set(kb) if ka.get(k, 0) != kb.get(k, 0)), key=lambda t: -abs(t[2] - t[1]))
        out += ["  %s: %d -> %d (%+d)" % (k, x, y, y - x) for k, x, y in moved]
        return out

    def text(self):
        lines = ["# The closed loop's periodic report", "", "Written %s. Every number is READ from a morph log (stats.jsonl beside each model); "
                 "a stat a pass did not write is left out, never filled in." % time.strftime("%Y-%m-%d %H:%M")]
        for m in self.models:
            rows = MorphLog(m).rows()
            name = os.path.basename(os.path.dirname(os.path.abspath(m)))
            lines += ["", "## %s (%d pass%s)" % (name, len(rows), "" if len(rows) == 1 else "es"), ""]
            if not rows:
                lines.append("- no pass recorded yet")
                continue
            lines += ["### Graph growth", ""] + ["- " + l.strip() for l in self.growth(rows)]
            lines += ["", "### Every stat's trend", ""] + ["- " + Trend(s, rows).line() for s in KINDS if Trend(s, rows).points]
            pc = Predictors(rows).pairs()
            lines += ["", "### Predictive candidates (lag one pass)", ""] + (["- " + Predictors.line(p) for p in pc] or ["- too few passes to pair any"])
            lines += ["", "### Where the questions are asked (who / what / why / when / where / before / after)", ""]
            lines += ["- " + l for l in FacetGrid.of(m).lines()]
            kgdb = os.path.join(m, "..", "kg", "%s.sqlite" % name)
            if os.path.exists(kgdb):
                import buzzsaw
                vp = os.path.join(m, "..", "kg", "verdicts.json")
                mays = [(x, r) for x, r in buzzsaw.Buzzsaw(kgdb, vp if os.path.exists(vp) else None).ranked() if x.kind == "inferred requirement"]
                lines += ["", "### Buzzsaw: the owner's questions, widest blast radius first", ""]
                lines += ["- %s (radius %d): %s" % (x.node, r.score, x.text[:110]) for x, r in mays[:5]] or ["- no inferred requirement"]
            tunes = GnnTunes(m).rows()
            lines += ["", "### Graph-for-GNN tunes", ""] + (["- pass %s: %s" % (t.get("pass"), t.get("summary")) for t in tunes] or ["- none yet"])
            lines += ["", "### What the stats pick next", ""]
            for p in NextImprovement(rows, tunes).picks() or [{"area": "none", "what": "no stat is stalled or worsening", "score": 0, "why": []}]:
                lines.append("- **%s** (score %d): %s" % (p["area"], p["score"], p["what"]))
                lines += ["  - " + w for w in p["why"]]
        return "\n".join(lines) + "\n"

    def write(self, out=None):
        rows = MorphLog(self.models[0]).rows()
        out = out or os.path.join(self.models[0], "..", "reports", "LOOP_REPORT_%03d.md" % (rows[-1].get("n", len(rows)) if rows else 0))
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        io.open(out, "w", encoding="utf-8").write(self.text())
        return out


class GnnTunes(object):
    """The graph-for-GNN tunes run on a model's graph, one row each, at <model>/../gnn_tune.jsonl (graph_gnn.py tune writes them)"""

    def __init__(self, model_dir):
        self.path = os.path.join(model_dir, "..", "gnn_tune.jsonl")

    def rows(self):
        return [json.loads(l) for l in io.open(self.path, encoding="utf-8")] if os.path.exists(self.path) else []


def summary(model_dir):
    """the lines a pass prints after its row: the picks, and whether the periodic report is due (closed_loop.py calls this)"""
    rows = MorphLog(model_dir).rows()
    out = []
    fired = []
    try:
        import improvement
        reqs, _ = improvement.load()
        for r in reqs:
            f, why = improvement.fires(r, rows)
            area = (r.get("action") or {}).get("area")
            if f and area:
                fired.append({"id": r["id"], "area": area, "detail": why})
    except FileNotFoundError:
        pass                         # a kit without the improvement model: the trends alone choose
    for p in NextImprovement(rows, GnnTunes(model_dir).rows(), fired).picks()[:3]:
        out.append("the stats pick %s (score %d): %s" % (p["area"], p["score"], p["what"]))
    if PeriodicReport.due(rows):
        out.append("the periodic report: %s" % PeriodicReport([model_dir]).write())
    return out


def selftest():
    """a planted verdicts.json and report give exactly the numbers in them; a number the pass did not write is absent, not 0"""
    import shutil
    import tempfile
    t = tempfile.mkdtemp(prefix="loop_stats_selftest_")
    meta, data = ucm._clean_data(os.path.join(t, "src.txt"))
    io.open(os.path.join(t, "src.txt"), "w", encoding="utf-8").write("The reader wants to see the flow.\n")
    md = os.path.join(t, "demo", "model")
    os.makedirs(md)
    import yaml
    io.open(os.path.join(md, "meta.yaml"), "w", encoding="utf-8").write(yaml.safe_dump({"meta": meta}))
    for k, val in data.items():
        io.open(os.path.join(md, "%s.yaml" % k), "w", encoding="utf-8").write(yaml.safe_dump({k: val}))
    os.makedirs(os.path.join(t, "demo", "kg"))
    v = {"mutants": {"A@1: the condition negated": {"killed": True, "by": ["T1"], "what": "the condition negated"},
                     "A@1: boundary < to <= (1)": {"killed": False, "what": "boundary < to <= (1)"},
                     "A@2: boundary < to <= (2)": {"killed": False, "what": "boundary < to <= (2)"},
                     "A@3: the number 0 made 1": {"killed": None, "equivalent": "TCE", "what": "the number 0 made 1"}},
         "assertions": {"R-1#1": {"holds": True, "kill": {"killed": True}}}, "branches": {"A#b0": {"killed": True}},
         "tests": ["T1", "T2"], "equivalent_declared": {"A@2: boundary < to <= (2)": "same"}}
    io.open(os.path.join(t, "demo", "kg", "verdicts.json"), "w", encoding="utf-8").write(json.dumps(v))
    rep = os.path.join(t, "R.md")
    io.open(rep, "w", encoding="utf-8").write("  L14  KILLED           1     0      2  x\n  L15  PINNED           0     0      0  y\n  gnn     0.62  0.74  0.71\n")
    ok = True
    try:
        g = os.path.join(t, "pillars.sqlite")
        db = sqlite3.connect(g)
        db.execute("CREATE TABLE edge(s TEXT, p INTEGER, o TEXT, fam INTEGER)")
        owners = {"R-1": 6, "R-2": 2, "R-3": 1, "R-4": 1}          # 10 links: the top 3 hold 9, two requirements reach 80%
        db.executemany("INSERT INTO edge VALUES (?, 122, ?, 0)", [("cell:S-BRANCH:T.M#b%s%d" % (q, i), "req:" + q, ) for q, n in owners.items() for i in range(n)])
        db.execute("INSERT INTO edge VALUES ('cell:S-EFFECT:x', 122, 'req:R-9', 0)")          # not a branch cell: owns no code
        db.execute("INSERT INTO edge VALUES ('cell:S-BRANCH:T.M#z', 134, 'req:R-9', 0)")      # a branch cell, another relation
        db.commit()
        p1 = owner_pillars(g)
        db.execute("DELETE FROM edge")
        db.commit()
        db.close()
        good = p1 == {"pillar_top3_share": 0.9, "pillars_80": 2} and owner_pillars(g) == {}
        ok &= good
        print("selftest: PILLARS the top 3's share of the owned code and how many hold 80%%, branch cells only; no link is absent, not 0 %s%s" % (
            "PASS" if good else "FAIL", "" if good else " %s" % p1))
        r = PassStats(md, rep, seconds=12.34).collect()
        want = {"mutants": 4, "killed": 1, "survived": 2, "survived_unexplained": 1, "equivalent_tce": 1, "equivalent_declared": 1,
                "kill_rate": 0.5, "tests": 2, "tests_idle": 1, "decisions_pinned": 1, "holes_ours": 1, "seconds": 12.3}
        bad = {k: (r.get(k), w) for k, w in want.items() if r.get(k) != w}
        good = not bad and r["survivors_by_operator"] == {"boundary": 1} and r["gnn_mrr"] == {"gnn": 0.71} and "chain" not in r
        good = good and "pillars_80" not in r            # no graph beside this pass: absent, never 0
        ok &= good
        print("selftest: every number READ from what the pass wrote; an unwritten one absent %s%s" % ("PASS" if good else "FAIL", "" if good else " %s" % bad))
        dbp = os.path.join(t, "demo", "kg", "demo.sqlite")
        db = sqlite3.connect(dbp)
        db.executescript("CREATE TABLE entity(id TEXT PRIMARY KEY, kind TEXT, label TEXT, sub TEXT);CREATE TABLE edge(s TEXT, p INTEGER, o TEXT, fam INTEGER);"
                         "CREATE TABLE literal(s TEXT, p INTEGER, v TEXT);INSERT INTO edge VALUES ('cell:S-BRANCH:T.M#b0', 122, 'req:R-1', 0);")
        db.commit()
        db.close()
        r2 = PassStats(md, rep, seconds=12.34).collect()
        os.remove(dbp)
        good = r2.get("pillar_top3_share") == 1.0 and r2.get("pillars_80") == 1
        ok &= good
        print("selftest: a pass beside a graph records its pillars %s%s" % ("PASS" if good else "FAIL", "" if good else " %s" % r2.get("pillars_80")))
        log = MorphLog(md)
        a, b = log.add(r), log.add(r)
        good = (a["n"], b["n"]) == (1, 2) and len(log.rows()) == 2
        ok &= good
        print("selftest: the morph log numbers its passes in order %s" % ("PASS" if good else "FAIL"))
        rows = [{"n": i + 1, "survived_unexplained": v, "mutants": 100 + 10 * i, "gnn_mrr": {"gnn": 0.7, "words": 0.5},
                 "operator_kinds": 9 + i, "holes_ours": 0 if i > 1 else 5 - i} for i, v in enumerate([9, 5, 4, 6, 8])]
        t1, t2, t3 = Trend("survived_unexplained", rows), Trend("gnn_margin", rows), Trend("holes_ours", rows)
        good = (t1.verdict(), t2.verdict(), t3.verdict(), Trend("tests", rows).verdict(), Trend("mutants", rows[:2]).verdict()) == \
            ("worsening", "stalled", "done", "too few", "too few")
        ok &= good
        print("selftest: TREND an OPEN stat rising is worsening, a flat margin stalled, an OPEN stat flat at 0 done, an absent or "
              "two-pass stat too few %s" % ("PASS" if good else "FAIL"))
        hp = NextImprovement([{"n": 1, "holes_ours": 3}]).picks()
        fp = NextImprovement([{"n": 1}], fired=[{"id": "IMP-8", "area": "speed", "detail": "idle 5"}]).picks()
        good_h = hp and hp[0]["area"] == "tests" and "3 holes of ours" in hp[0]["why"][0] and any(p["area"] == "graph-for-gnn" for p in hp) \
            and [p["area"] for p in NextImprovement([{"n": 1, "holes_ours": 0}]).picks()] == ["graph-for-gnn"] \
            and fp[0]["area"] == "speed" and fp[0]["score"] == NextImprovement.FIRED and "IMP-8 fired" in fp[0]["why"][-1] \
            and "most in" not in hp[0]["why"][0] \
            and "48 holes of ours standing in the last pass, most in L13 (27)" == NextImprovement(
                [{"n": 1, "holes_ours": 48, "holes": {"L1": [0, 9, 9], "L13": [27, 0, 43], "L15": [21, 0, 2]}}]).picks()[0]["why"][0]
        ok &= bool(good_h)
        print("selftest: HOLES FIRST a hole of ours puts the tests first with its count, the others keep theirs; none changes nothing; a "
              "fired improvement trigger lifts the area it names %s" % ("PASS" if good_h else "FAIL"))
        picks = NextImprovement(rows, [{"pass": 5}]).picks()
        areas = [p["area"] for p in picks]
        never = NextImprovement(rows).picks()
        good = areas[:2] == ["tests", "graph-for-gnn"] and "net" not in areas and \
            any("never tuned" in w for p in never if p["area"] == "graph-for-gnn" for w in p["why"])
        ok &= good
        print("selftest: NEXT the stats pick tests (worsening) before the GNN tune (stalled, never tuned), and never a rising "
              "net %s %s" % ("PASS" if good else "FAIL", "" if good else areas))
        prow = [{"n": i + 1, "tests": x, "survived_unexplained": y, "mutants": 10, "graph_nodes": 50 + i}
                for i, (x, y) in enumerate([(1, 9), (3, 9), (4, 7), (9, 6), (10, 1), (12, 0)])]
        best = Predictors(prow).pairs(top=40)
        hit = [p for p in best if (p["x"], p["y"]) == ("tests", "survived_unexplained")]
        mech = Predictors([{"n": i + 1, "mutants": m, "killed": k} for i, (m, k) in enumerate([(10, 5), (20, 9), (25, 15), (40, 20), (41, 30), (60, 31)])]).pairs(40)
        good = bool(hit) and hit[0]["n"] == 4 and hit[0]["anecdote"] and all(p["mechanical"] for p in mech if {p["x"], p["y"]} == {"mutants", "killed"})
        ok &= good
        print("selftest: PREDICT a lagged pair is found with its n and called an anecdote; killed against mutants is tagged "
              "mechanical %s" % ("PASS" if good else "FAIL"))
        rep_rows = [dict(r, n=i + 1, graph_nodes=100 + 7 * i, graph_edges=300 + 20 * i, graph_by_kind={"mutant": 40 + 5 * i, "test": 3})
                    for i, r in enumerate([{}] * 20)]
        g = PeriodicReport.growth(rep_rows)
        good = PeriodicReport.due(rep_rows) and not PeriodicReport.due(rep_rows[:19]) and g[0].startswith("the graph: 100 -> 233 nodes (+133)") \
            and any(l.strip() == "mutant: 40 -> 135 (+95)" for l in g) and not any(l.strip().startswith("test:") for l in g)
        ok &= good
        print("selftest: REPORT due at pass 20 and not 19; the graph's growth read first pass to last, by kind, the unchanged "
              "kinds left out %s %s" % ("PASS" if good else "FAIL", "" if good else g))
        fg = FacetGrid([{"actor": "A1", "name": "Buy", "why": "", "trigger": "t", "where": "Shop", "pre": "p", "success": "s",
                         "steps": [{"do": "pay", "reqs": ["R-1"]}, {"do": "", "reqs": []}],
                         "ext": [{"at": 1, "when": "short of money", "do": ["refuse"], "out": "", "reqs": ["R-2"]}]}])
        g, st = fg.grid(), fg.stats()
        good = g[("use case", "why")] == (0, 1) and g[("use case", "who")] == (1, 0) and g[("step", "what")] == (1, 1) \
            and g[("step", "who")] is None and g[("extension", "after")] == (0, 1) and st["facets_never_asked"] == 8 \
            and st["facets_open"] == 4
        ok &= good
        print("selftest: FACETS an empty field is open, a missing field is never asked, per level of the model %s %s" % (
            "PASS" if good else "FAIL", "" if good else st))
    finally:
        shutil.rmtree(t, ignore_errors=True)
    print("selftest: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["record", "show", "trends", "report"])
    ap.add_argument("model", nargs="?")
    ap.add_argument("more", nargs="*", help="report: further models (the closed loop's own selfmodel is always added)")
    ap.add_argument("--report")
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--note")
    ap.add_argument("--last", type=int, default=10)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.cmd or not a.model:
        ap.error("record <model> | show <model> | --selftest")
    if a.cmd == "record":
        row = MorphLog(a.model).add(PassStats(a.model, a.report, a.seconds, a.note).collect())
        print("loop_stats: pass %d recorded -> %s" % (row["n"], os.path.normpath(MorphLog(a.model).path)))
        return 0
    if a.cmd == "trends":
        rows = MorphLog(a.model).rows()
        print("TRENDS  %s  (%d pass(es))" % (os.path.normpath(MorphLog(a.model).path), len(rows)))
        for st in KINDS:
            t = Trend(st, rows)
            if t.points:
                print("  " + t.line())
        print("GRAPH GROWTH")
        for l in PeriodicReport.growth(rows):
            print("  " + l)
        print("WHERE THE QUESTIONS ARE ASKED")
        for l in FacetGrid.of(a.model).lines():
            print("  " + l)
        print("PREDICTIVE CANDIDATES (lag one pass)")
        for p in Predictors(rows).pairs():
            print("  " + Predictors.line(p))
        print("WHAT THE STATS PICK NEXT")
        for p in NextImprovement(rows, GnnTunes(a.model).rows()).picks():
            print("  %s (score %d): %s" % (p["area"], p["score"], p["what"]))
            for w in p["why"]:
                print("      " + w)
        return 0
    if a.cmd == "report":
        print("loop_stats: the periodic report -> %s" % PeriodicReport([a.model] + a.more).write())
        return 0
    MorphLog(a.model).show(a.last)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
