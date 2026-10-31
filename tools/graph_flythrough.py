#!/usr/bin/env python3
"""graph_flythrough.py - any closed loop graph as a page you can fly through, laid out so its layers read: why, what, proof.

WHY (10-30): to show the node graph, how it is laid out and how it maps code to requirements, as a fly-through. The graph
is the record, and it had no picture: its only readers were queries. This lays out what the store holds and invents nothing: every node
and edge on the page is a row in the sqlite the loop wrote, and the tour follows one thread that really runs through it, from a source
to a requirement, the step that needs it, Claude's written reasoning about it, the code, a branch, a break of that branch and the test
that caught it. A break no test caught glows red.

  GraphLayout  the store's nodes and edges, placed: a layer by kind, an angle by the code each node belongs to
  Tour         the one real thread the page flies along; refuses to invent one when the graph holds none
  Hero         the fly-through as the first screen of a whole document that scrolls on past it
  Readers      the learned parts as readers standing outside the rings, one per goal, with the goals and stats lists that work on each other
  page         the single HTML file (three.js from cdnjs, the data inline; WASD to fly, drag to orbit, N and P for the tour)

`--hide` keeps private names off the page: a label matching it is dropped, and the node stays. Nothing else in the file names a project.
`--anonymous` is for a graph whose names may not be published at all: every label goes whatever it says, the tour speaks in roles ("a
requirement", "the test that caught it"), positions, kinds and counts stay, and a page that still holds a name from the store is refused, not written.
`--hero` writes a whole document instead of a fragment for a host: the graph fills the first screen and the page scrolls on to what `--below`
holds. A wheel or a swipe scrolls the page (ctrl and the wheel zooms), and the keys fly only while the graph is in view and nobody is typing.
`--goals` draws what gnn_goals.py --write makes: a reader per goal, a Goals panel and a Stats panel, and a Show view of the readers. Picking a goal flies to its
reader, lights what it reads and predicts, shows its score and filters the stats to the ones it uses. Picking a stat lights the readers that use it and, where
the stat plainly counts one kind of node, those nodes. Trying is not reaching: a goal that has not reached its goal looks different in the list and in the graph.

    py -X utf8 graph_flythrough.py <store.sqlite> --out page.html [--title "..."] [--hide "NameA|NameB"]
    py -X utf8 graph_flythrough.py <store.sqlite> --out page.html --anonymous --hero --title "..." --headline "..." --tagline "..." --below fragment.html --cue "Read on" "#anchor" [--goals gnn_goals.json]
    py -X utf8 graph_flythrough.py --selftest
"""
import argparse
import collections
import hashlib
import html as markup_text
import io
import json
import math
import os
import re
import sqlite3
import sys
import tempfile

# (kind, height, radius, spread in radius, colour, point size): top to bottom, why it exists, what the code does, what proves it
LAYERS = [("source", 96, 46, 6, "#F4F2EC", 5), ("actor", 96, 30, 4, "#F4F2EC", 4), ("question", 78, 62, 6, "#F7C873", 5),
          ("requirement", 62, 76, 0, "#F08A3C", 10), ("use case", 44, 44, 0, "#C9B6F2", 8), ("step", 36, 92, 3, "#C9B6F2", 6),
          ("extension", 32, 104, 3, "#C9B6F2", 4), ("chain", 20, 20, 0, "#7FE0C3", 9), ("chain step", 20, 60, 0, "#7FE0C3", 7),
          ("space", 12, 28, 0, "#5B6875", 3), ("name", 8, 40, 6, "#5B6875", 3), ("code", 0, 118, 0, "#7FB2E5", 9),
          ("assertion", -18, 128, 5, "#9CD0FF", 3), ("cell", -24, 130, 8, "#4FA3D9", 3), ("mutant", -54, 146, 22, "#5C7C99", 2),
          ("test", -92, 84, 0, "#8BE28B", 7)]
SURVIVED = "#FF4D4D"
INFERRED = "#FFF0A0"   # a requirement the walk inferred, against the orange of the ones people wrote
REL_COLOUR = {"maps to": "#F08A3C", "cites": "#7FE0C3", "binds requirement": "#C9B6F2", "infers requirement": "#C9B6F2", "stated by": "#F4F2EC",
              "is at": "#7FB2E5", "mutated as": "#4FA3D9", "killed by": "#3E6B4E", "checked by": "#8BE28B", "has step": "#C9B6F2"}
OTHER_EDGE = "#34495C"
KILLED_BY = "killed by"
BRIGHT_EDGES = 150     # a relation with more edges than this dims in proportion: lines add up, and 5,000 of them burn to plain white
GNN_LEFT_OUT = {"mutant", "test", "chain", "chain step"}    # graph_gnn.LEFT_OUT (that file needs torch, this one must run without it)
NOT_DRAWN = ("doc", "stat", "goal")     # doc_nodes.KINDS (its selftest holds the two alike): no layer yet, so they would land on the page as an unlabelled grey ring
# (hash token, what the view shows, the relations and kinds it keeps): a named question the graph can answer by itself
VIEWS = [("all", "Everything", None, None),
         ("maps", "How code maps to requirements", {"maps to", "is at"}, {"requirement", "cell", "code"}),
         ("proof", "Every break, and what caught it", {"mutated as", "killed by"}, {"cell", "mutant", "test"}),
         ("reasoning", "Claude's reasoning, and what it stands on", {"cites", "reasons"}, {"chain", "chain step", "requirement", "code"}),
         ("why", "Requirements and the use cases that need them", {"stated by", "binds requirement", "infers requirement", "has step", "next step"},
          {"source", "requirement", "use case", "step", "extension"}),
         ("heard", "What the graph neural network reads", None, {row[0] for row in LAYERS} - GNN_LEFT_OUT)]
OUR_RELATIONS = set(REL_COLOUR) | {r for v in VIEWS if v[2] for r in v[2]}       # what an anonymous page may still call a relation
OUR_WORDS = OUR_RELATIONS | {row[0] for row in LAYERS}                           # the tool's own vocabulary, which a store may also use
ROLES = {"requirement": "a requirement", "code": "a method", "test": "a test", "use case": "a use case", "step": "a use case step",
         "chain step": "a step of reasoning", "cell": "a branch", "mutant": "a break"}       # what a node is called when it may not be named
# What each kind of learned part reads, which the goals file does not say. A goal may say for itself with `reads`, and then this is not asked.
READS_OF_MODEL = {"graph network": {row[0] for row in LAYERS} - GNN_LEFT_OUT, "small model on the break's features": {"mutant"}}
READS_IN_WORDS = {"graph network": "every kind of node but the breaks, the tests and the reasoning", "small model on the break's features": "the break nodes"}
MEASURE_OF_MODEL = {"graph network": "Scored by mean reciprocal rank, where 1.0 always ranks the right answer first.",     # graph_gnn.evaluate
                    "small model on the break's features": "Scored by the share of what it looks for found in the first fifth of its ranking."}     # survivor_rank.by_method
# The kind of node a stat plainly counts, by record. Checked against the last pass of a real store: mutants, requirements, tests, use_cases and effects equal
# that kind's count, decisions is the branches among the cells, and every field of a break's verdict sits on a break. Anything else stays unlinked rather than
# guessed at: graph_by_kind and graph_nodes count every kind, and musts, mays and claims do not name a kind.
STAT_SITS_ON = {"mutant": {"morph": ("mutants", "killed", "survived", "no_verdict", "survived_unexplained", "equivalent_tce", "equivalent_declared", "equivalent_lapsed", "kill_rate",
                                     "kills_by_operator", "survivors_by_operator", "operator_kinds"),
                           "verdicts": ("equivalent", "file", "killed", "place", "statement", "what", "why"),
                           "survivor_rank": ("breaks", "survived"), "e1_3_cheap": ("breaks", "survived", "equiv_n", "equiv_pos")},
                "requirement": {"morph": ("requirements",), "gnn_tasks": ("shape.requirements",), "gnn_curve": ("requirements",)},
                "test": {"morph": ("tests", "tests_idle", "test_kills_top")},
                "use case": {"morph": ("use_cases",)},
                "cell": {"morph": ("decisions", "decisions_claimed", "decisions_pinned"), "gnn_curve": ("linked_cells",)},
                "code": {"gnn_tasks": ("shape.methods",), "gnn_curve": ("methods",)},
                "assertion": {"morph": ("effects", "effects_killed", "effects_shape_only", "effects_with_survivor")}}
VERDICT_WORDS = {"DECIDES": "Code reads it, and it changes what runs next.", "TRIGGERED": "A rule fires on it, and what fires becomes a work item.", "WATCHED": "It is trended or printed, and nothing acts on it.",
                 "BANKED": "Kept for a learner that does not exist yet.", "IDENTITY": "It says what a row is; it is not a measurement.", "UNREAD": "Nothing reads it."}     # telemetry_audit's verdicts
FONTS = "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&display=swap"


def hash01(text):
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16) / float(0xFFFFFFFF)


def circular_mean(angles, default):
    if not angles:
        return default
    return math.atan2(sum(math.sin(a) for a in angles), sum(math.cos(a) for a in angles))


class GraphLayout(object):
    """The store's nodes and edges with a place for each: a layer by kind, an angle by the code it belongs to"""

    def __init__(self, store, hide=None, anonymous=False):
        db = sqlite3.connect(store)
        self.hide = re.compile(hide) if hide else None
        self.anonymous = anonymous
        self.prop = dict(db.execute("select id,label from prop"))
        self.nodes = {i: {"id": i, "kind": k, "label": l or "", "sub": s} for i, k, l, s in db.execute("select id,kind,label,sub from entity") if k not in NOT_DRAWN}
        self.edges = sorted({(s, self.prop.get(p, str(p)), o) for s, p, o in db.execute("select s,p,o from edge") if s in self.nodes and o in self.nodes})
        db.close()
        self.out = collections.defaultdict(list)
        self.into = collections.defaultdict(list)
        for s, p, o in self.edges:
            self.out[s].append((p, o))
            self.into[o].append((p, s))
        self.killed = {s for s, p, _ in self.edges if p == KILLED_BY}
        self.place()

    def of_kind(self, kind):
        return sorted(i for i, n in self.nodes.items() if n["kind"] == kind)

    def survived(self, node_id):
        return self.nodes[node_id]["kind"] == "mutant" and node_id not in self.killed

    def neighbour_angles(self, node_id, angle):
        return [angle[o] for _, o in self.out[node_id] if o in angle] + [angle[s] for _, s in self.into[node_id] if s in angle]

    def place(self):
        angle = {}
        code = self.of_kind("code")
        for n, c in enumerate(code):                    # sorted by name, so a class's methods sit side by side
            angle[c] = 2 * math.pi * n / max(1, len(code))
        sector = 2 * math.pi / max(1, len(code))
        for kind in ("cell", "assertion", "mutant"):    # each decision and each break under the method it belongs to
            for i in self.of_kind(kind):
                at = [o for p, o in self.out[i] if p == "is at" and o in angle]
                base = angle[at[0]] if at else 2 * math.pi * hash01(i)
                angle[i] = base + (hash01(i + "a") - 0.5) * sector * 0.9
        # the rest settle where what they touch is, the even kinds spaced on their ring so a popular angle does not pile them up
        for kind in ("requirement", "test", "step", "extension", "use case", "chain step", "chain", "question", "source", "actor", "name", "space"):
            ids = self.of_kind(kind)
            means = sorted(ids, key=lambda i: circular_mean(self.neighbour_angles(i, angle), 2 * math.pi * hash01(i)))
            for rank, i in enumerate(means):
                mean = circular_mean(self.neighbour_angles(i, angle), 2 * math.pi * hash01(i))
                even = -math.pi + 2 * math.pi * rank / max(1, len(ids))
                angle[i] = circular_mean([mean, even], even)
        layer = {row[0]: row for row in LAYERS}
        for i, n in self.nodes.items():
            kind, height, radius, spread, _, _ = layer.get(n["kind"], ("other", -110, 60, 10, "#5B6875", 2))
            r = radius + (hash01(i + "r") - 0.5) * 2 * spread
            jitter = (hash01(i + "y") - 0.5) * (14 if n["kind"] == "mutant" else 4)
            a = angle.get(i, 2 * math.pi * hash01(i))
            n["x"], n["y"], n["z"] = r * math.cos(a), height + jitter, r * math.sin(a)

    def clean(self, text):
        """the text, or nothing if it holds a name to keep off the page; an anonymous page keeps no text from the store at all"""
        return "" if self.anonymous or (self.hide and self.hide.search(text)) else text

    def short(self, node_id):
        """what a reader is shown for a node: its id without the kind's prefix, unless that names something hidden"""
        text = node_id.split(":", 1)[-1]
        return self.clean(text)[:90]

    def names_on(self, page_text):
        """Which names of the store are on a page. A whole id or label, or the identifier-shaped words inside one (A.B, R-14, SomeMethod): a
        plain English word in a label says nothing private, and would match any sentence on the page. The relations the tool itself names are its own words."""
        found = set()
        for n in self.nodes.values():
            for text in (n["id"], n["id"].split(":", 1)[-1], n["label"]):
                words = [w.rstrip(".-") for w in re.findall(r"[A-Za-z][\w.#-]+", text) if re.search(r"[A-Z0-9_#.-]", w[1:].rstrip(".-"))]
                found.update(x for x in [text] + words if len(x) >= 3 and x in page_text and x not in OUR_WORDS)
        # whole words only: Add is not a name inside AdditiveBlending, nor ec4 inside vec4
        return sorted(x for x in found if re.search(r"(?<![\w.#-])%s(?![\w#-])" % re.escape(x), page_text))


class Tour(object):
    """One thread that really runs through the graph: source, requirement, step, reasoning, code, branch, break, test.
    A node with no name to show is spoken of by its role, so a hidden name leaves a sentence, never a hole in one."""

    def __init__(self, layout, hero=False):
        self.g = layout
        self.zoom = "ctrl and scroll to zoom" if hero else "scroll to zoom"
        self.writer = "the harness" if hero else "the loop"     # a hero's reader has not been told what the loop is yet

    def thread(self):
        g = self.g
        best = None
        for cs in g.of_kind("chain step"):
            cited = [o for p, o in g.out[cs] if p == "cites"]
            for req in [c for c in cited if g.nodes[c]["kind"] == "requirement"]:
                for code in [c for c in cited if g.nodes[c]["kind"] == "code"]:
                    for cell in [s for p, s in g.into[req] if p == "maps to" and g.nodes[s]["kind"] == "cell"]:
                        if ("is at", code) not in g.out[cell]:
                            continue
                        for _, mut in [e for e in g.out[cell] if e[0] == "mutated as"]:
                            tests = [o for p, o in g.out[mut] if p == KILLED_BY]
                            if not tests:
                                continue
                            found = {"chain step": cs, "requirement": req, "code": code, "cell": cell, "mutant": mut, "test": sorted(tests)[0],
                                     "source": next((o for p, o in sorted(g.out[req]) if p == "stated by"), None),
                                     "step": next((s for p, s in sorted(g.into[req]) if p == "binds requirement"), None)}
                            score = (found["source"] is not None) + (found["step"] is not None)
                            if best is None or score > best[0]:
                                best = (score, found)
        return best[1] if best else None

    def survivor_near(self, code):
        g = self.g
        near = [s for p, s in g.into[code] if p == "is at" and g.survived(s)]
        anywhere = [m for m in g.of_kind("mutant") if g.survived(m)]
        pick = sorted(near) or sorted(anywhere)
        return pick[0] if pick else None

    def stops(self):
        g = self.g
        t = self.thread()
        total = "%s nodes and %s edges." % (format(len(g.nodes), ","), format(len(g.edges), ","))
        out = [{"title": "The whole graph", "text": total + " Top to bottom: why it exists, what the code does, what proves it. Every point is a row %s wrote." % self.writer, "node": None, "path": []}]
        if not t:
            return out                      # no thread in this graph: the page says only what it holds
        def add(title, text, node, *pair):
            out.append({"title": title, "text": text, "node": node, "path": list(pair)})
        req_id = g.short(t["requirement"]) or "A requirement"
        req_text = g.clean(g.nodes[t["requirement"]]["label"])[:120]
        says = g.clean(g.nodes[t["chain step"]]["label"])[:160]
        step_name = g.short(t["step"]) if t["step"] else ""
        if t["source"]:
            add("Where it comes from", "A requirement has to come from somewhere it can be checked against: here, %s." % (g.short(t["source"]) or "a named source"), t["source"])
        add("The requirement", (req_id + ": " + req_text) if req_text and req_text != req_id else "%s, a node of its own. Everything below hangs off it." % req_id, t["requirement"], *([t["requirement"], t["source"]] if t["source"] else []))
        if t["step"]:
            add("The step that needs it", "A use case step binds it into a chain of cause and effect" + (": %s." % step_name if step_name else "."), t["step"], t["step"], t["requirement"])
        add("Claude's reasoning", ("\"%s\" " % says if says else "One step of a written argument. ") + "It cites the requirement and the code, and the graph checks those two really are joined.", t["chain step"], t["chain step"], t["requirement"])
        add("The code", "%s, a node of its own. The reasoning cites it by name, and a name the code does not have fails the run." % (g.short(t["code"]) or "A method"), t["code"], t["chain step"], t["code"])
        add("One decision", "A branch of that method, and the requirement that owns it. This link is how code maps to requirements.", t["cell"], t["cell"], t["requirement"], t["cell"], t["code"])
        add("Broken on purpose", g.short(t["mutant"]) or "One break of that decision. It runs alone against the tests, and one of them has to fail.", t["mutant"], t["cell"], t["mutant"])
        add("Caught", "%s failed when this break ran. That is the requirement being held." % (g.short(t["test"]) or "The test that caught it"), t["test"], t["mutant"], t["test"])
        survivor = self.survivor_near(t["code"])
        if survivor:
            add("And one that got past", (g.short(survivor) + ". " if g.short(survivor) else "") + "No test caught this break. Either it changes nothing, and that has to be proven, or a test is missing.", survivor)
        pointing = "Point at a node to see what kind it is." if g.anonymous else "Point at a node to name it."
        out.append({"title": "The whole graph", "text": "Fly anywhere: drag to orbit, %s, W A S D to move. %s" % (self.zoom, pointing), "node": None, "path": []})
        return out


def data(layout, title, hero=False, readers=None):
    g = layout
    ids = sorted(g.nodes)
    index = {i: n for n, i in enumerate(ids)}
    kinds = [row[0] for row in LAYERS] + ["other"]
    colours = {row[0]: row[4] for row in LAYERS}
    sizes = {row[0]: row[5] for row in LAYERS}
    pos, kind, flag = [], [], []
    for i in ids:
        n = g.nodes[i]
        pos += [round(n["x"], 1), round(n["y"], 1), round(n["z"], 1)]
        kind.append(kinds.index(n["kind"]) if n["kind"] in kinds else len(kinds) - 1)
        flag.append(1 if g.survived(i) else 2 if n["kind"] == "requirement" and n["sub"] == "inferred" else 0)
    rels = sorted({p for _, p, _ in g.edges})
    edges = []
    for s, p, o in g.edges:
        edges += [index[s], index[o], rels.index(p)]
    named = set(ROLES)
    # with no names to show, a pointed-at node still says what kind it is: an empty label keeps it pointable
    labels = {str(index[i]): g.short(i) for i in ids if g.nodes[i]["kind"] in named and (g.short(i) or g.anonymous)}
    stops = Tour(g, hero).stops()
    if readers:
        stops[len(stops) - 1 if len(stops) > 1 else 1:len(stops) - 1 if len(stops) > 1 else 1] = readers.stops()      # before the closing overview
    for st in stops:
        st["node"] = index.get(st["node"])
        st["path"] = [index[x] for x in st["path"] if x in index]
    counts = collections.Counter(g.nodes[i]["kind"] for i in ids)
    per_rel = collections.Counter(p for _, p, _ in g.edges)
    views = [{"id": v, "title": t, "rels": None if r is None else [rels.index(x) for x in sorted(r) if x in rels], "kinds": None if k is None else [kinds.index(x) for x in sorted(k)]}
             for v, t, r, k in VIEWS]
    # a store may name its relations after what it holds: an anonymous page keeps only the words this tool itself uses
    rel_names = [r if not g.anonymous or r in OUR_RELATIONS else "relation %d" % n for n, r in enumerate(rels)]
    if readers:
        views.insert(1, readers.view(rels))
    extra = readers.data(kinds, rels) if readers else {}
    return {**extra, "title": title, "hero": hero, "roles": ROLES if g.anonymous else None, "pos": pos, "kind": kind, "flag": flag, "edges": edges, "labels": labels, "stops": stops, "kinds": kinds,
            "colour": [colours.get(k, "#5B6875") for k in kinds], "size": [sizes.get(k, 2) for k in kinds], "rels": rel_names,
            "relColour": [REL_COLOUR.get(r, OTHER_EDGE) for r in rels], "relWeight": [round(min(1.0, math.sqrt(BRIGHT_EDGES / float(per_rel[r]))), 3) for r in rels],
            "survived": SURVIVED, "inferred": INFERRED, "counts": counts, "nodes": len(ids), "edgeCount": len(g.edges), "survivors": flag.count(1), "views": views}


STYLE = r"""<style>
:root { --bg:#0B1219; --panel:rgba(16,26,36,.86); --fg:#F4F2EC; --muted:#A9B7C4; --line:#2E4052; --accent:#F08A3C; color-scheme:dark }
html,body { height:100%; margin:0; background:var(--bg); color:var(--fg); font:15px/1.45 'IBM Plex Sans',system-ui,sans-serif; overflow:hidden }
canvas { position:fixed; inset:0; width:100%; height:100%; display:block; cursor:grab }
.hud { position:fixed; pointer-events:none; }
#head { top:calc(16px + env(safe-area-inset-top,0px)); left:max(16px,env(safe-area-inset-left,0px)); right:16px }
#head h1 { margin:0; font:600 22px/1.2 'Space Grotesk',system-ui,sans-serif; text-wrap:balance }
#head p { margin:4px 0 0; color:var(--muted); font-size:13px; font-variant-numeric:tabular-nums }
#layers { top:calc(84px + env(safe-area-inset-top,0px)); left:max(16px,env(safe-area-inset-left,0px)); display:flex; flex-direction:column; gap:6px; font-size:12px; color:var(--muted); max-width:240px }
#layers b { color:var(--fg); font-weight:600 }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px }
#card { left:50%; transform:translateX(-50%); bottom:calc(16px + env(safe-area-inset-bottom,0px)); width:min(680px,calc(100vw - 32px)); background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px 16px; pointer-events:auto; box-sizing:border-box }
#card h2 { margin:0 0 4px; font:600 18px/1.25 'Space Grotesk',system-ui,sans-serif; color:var(--accent) }
#card p { margin:0; min-height:3em; overflow-wrap:anywhere }
#ctl { display:flex; gap:8px; align-items:center; margin-top:10px; font-size:13px; color:var(--muted); flex-wrap:wrap }
button { background:#1D2C3A; color:var(--fg); border:1px solid var(--line); border-radius:8px; padding:6px 12px; font:inherit; cursor:pointer }
button:hover,button:focus-visible { border-color:var(--accent); outline:none }
select { background:#1D2C3A; color:var(--fg); border:1px solid var(--line); border-radius:8px; padding:6px 8px; font:inherit; max-width:100% }
#head,#layers { text-shadow:0 1px 4px #0B1219,0 0 12px #0B1219 }
#tip { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:4px 8px; font-size:12px; max-width:320px; display:none; overflow-wrap:anywhere }
@media (max-width:640px) { #layers { display:none } #head h1 { font-size:18px } }
body.shot .hud { display:none !important }
</style>
"""

MARKUP = r"""<canvas id="c" aria-label="A three dimensional graph of requirements, code, breaks and tests"></canvas>
/*HEAD*/
<div class="hud" id="tip"></div>
<div class="hud" id="card"><h2 id="ct"></h2><p id="cx"></p>
<div id="ctl"><button id="prev" type="button">Previous</button><button id="play" type="button">Pause</button><button id="next" type="button">Next</button><span id="pos"></span><label for="view">Show</label><select id="view"></select><span id="hint">/*HINT*/</span>/*CUE*/</div></div>
"""

SCRIPT = r"""<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const D = /*DATA*/;
const $ = (id) => document.getElementById(id);
const HERO = D.hero, stage = $('stage');
if(!HERO) $('t').textContent = D.title;
$('s').textContent = (HERO?D.title+': ':'') + D.nodes.toLocaleString('en') + ' nodes, ' + D.edgeCount.toLocaleString('en') + ' edges, ' + D.survivors.toLocaleString('en') + ' breaks no test caught (red)';
const legend = [['#F08A3C','Why','sources, requirements, use cases'],['#7FE0C3','Reasoning',"Claude's chain, each step citing what it stands on"],['#7FB2E5','What','the code, method by method'],['#8BE28B','Proof','branches, every break, the tests']];
for (const [c,a,b] of legend) { const d = document.createElement('div'); const s = document.createElement('span'); s.className='dot'; s.style.background=c; const i = document.createElement('b'); i.textContent=a; d.append(s,i,document.createTextNode(': '+b)); $('layers').append(d); }
let renderer=null; try { renderer=new THREE.WebGLRenderer({canvas:$('c'),antialias:true,preserveDrawingBuffer:true}); } catch(e) { console.error(e); }
if(!renderer){ $('ct').textContent='The graph could not be drawn'; $('cx').textContent='This browser did not start WebGL, so there is no graph to show.'; $('ctl').querySelectorAll(':scope > :not(a)').forEach((x)=>{ x.style.display='none'; }); throw new Error('no WebGL: the graph is not drawn'); }
renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));
const scene = new THREE.Scene(); scene.background = new THREE.Color('#0B1219');
const camera = new THREE.PerspectiveCamera(50,1,1,4000);
const N = D.kind.length;
const P = (i) => new THREE.Vector3(D.pos[3*i],D.pos[3*i+1],D.pos[3*i+2]);
// edges
const edgeCount = D.edges.length/3, edgeCol = new Float32Array(edgeCount*6), relCol = D.relColour.map((c)=>new THREE.Color(c));
{ const pos = new Float32Array(edgeCount*6);
  for (let e=0;e<edgeCount;e++){ const a=D.edges[3*e],b=D.edges[3*e+1]; for (let k=0;k<3;k++){ pos[6*e+k]=D.pos[3*a+k]; pos[6*e+3+k]=D.pos[3*b+k]; } }
  var edgeGeo = new THREE.BufferGeometry(); edgeGeo.setAttribute('position',new THREE.BufferAttribute(pos,3)); edgeGeo.setAttribute('color',new THREE.BufferAttribute(edgeCol,3));
  scene.add(new THREE.LineSegments(edgeGeo,new THREE.LineBasicMaterial({vertexColors:true,transparent:true,opacity:.28,blending:THREE.AdditiveBlending,depthWrite:false}))); }
// nodes
const pointShader = { uniforms:{scale:{value:600},time:{value:0}}, transparent:true, depthWrite:false, blending:THREE.AdditiveBlending,
  vertexShader:'attribute float size; attribute vec3 color; varying vec3 vC; uniform float scale; void main(){ vC=color; vec4 mv=modelViewMatrix*vec4(position,1.0); gl_PointSize=size*scale/-mv.z; gl_Position=projectionMatrix*mv; }',
  fragmentShader:'varying vec3 vC; void main(){ float d=length(gl_PointCoord-vec2(.5))*2.0; float a=smoothstep(1.0,.15,d); gl_FragColor=vec4(vC*(0.55+0.9*a*a),a); }' };
const nodeMat = new THREE.ShaderMaterial(pointShader);
const nodeSize=new Float32Array(N);
{ const pos=new Float32Array(D.pos), col=new Float32Array(N*3); const cs=D.colour.map((c)=>new THREE.Color(c)), red=new THREE.Color(D.survived), inf=new THREE.Color(D.inferred);
  for (let i=0;i<N;i++){ const c=D.flag[i]===1?red:D.flag[i]===2?inf:cs[D.kind[i]]; col[3*i]=c.r; col[3*i+1]=c.g; col[3*i+2]=c.b; }
  var nodeGeo=new THREE.BufferGeometry(); nodeGeo.setAttribute('position',new THREE.BufferAttribute(pos,3)); nodeGeo.setAttribute('color',new THREE.BufferAttribute(col,3)); nodeGeo.setAttribute('size',new THREE.BufferAttribute(nodeSize,1));
  scene.add(new THREE.Points(nodeGeo,nodeMat)); }
// a view keeps the relations and kinds it names: a line drawn black adds nothing, a point of size 0 draws nothing. What `lit` names stands out from all of it, dimmed so it can still be seen
let viewNow=D.views[0], lit=null, wasLit=false, onPaint=()=>{}, onStop=()=>{}, stopView=()=>null, onFrame=()=>{}, onResize=()=>{}, overReader=()=>false, clickReader=()=>{};
const nodeBase=nodeGeo.attributes.color.array.slice();
function setView(id){ viewNow=D.views.find((x)=>x.id===id)||D.views[0]; $('view').value=viewNow.id; paint(); }
function paint(){ const v=viewNow;
  const keepRel=!lit&&v.rels&&new Set(v.rels), keepKind=!lit&&v.kinds&&new Set(v.kinds);
  const gain=(v.gain||(keepRel?1.8:1))*Number(new URLSearchParams(location.search).get('dim')||1);    // fewer lines on screen, so each can be brighter; a still can dim the rest so the thread reads
  for (let e=0;e<edgeCount;e++){ const r=D.edges[3*e+2], on=(!keepRel||keepRel.has(r))&&(!keepKind||(keepKind.has(D.kind[D.edges[3*e]])&&keepKind.has(D.kind[D.edges[3*e+1]]))), w=lit?(lit.rels.has(r)?Math.min(1,D.relWeight[r]*4):.07*D.relWeight[r]):on?Math.min(1,D.relWeight[r]*gain):0, c=relCol[r]; for (let k=0;k<2;k++){ edgeCol[6*e+3*k]=c.r*w; edgeCol[6*e+3*k+1]=c.g*w; edgeCol[6*e+3*k+2]=c.b*w; } }
  edgeGeo.attributes.color.needsUpdate=true;
  for (let i=0;i<N;i++) nodeSize[i]=((!keepKind||keepKind.has(D.kind[i]))?(D.flag[i]===1?7:D.size[D.kind[i]]):0)*(lit&&!lit.kinds.has(D.kind[i])?.5:1);
  nodeGeo.attributes.size.needsUpdate=true;
  if(lit||wasLit){ const a=nodeGeo.attributes.color; for(let i=0;i<N;i++){ const f=lit&&!lit.kinds.has(D.kind[i])?.2:1; for(let k=0;k<3;k++) a.array[3*i+k]=nodeBase[3*i+k]*f; } a.needsUpdate=true; wasLit=!!lit; }
  onPaint(); }
for (const v of D.views){ const o=document.createElement('option'); o.value=v.id; o.textContent=v.title; $('view').append(o); }
$('view').onchange=(e)=>setView(e.target.value);
// the thread: the edges flown so far, as rods the eye can follow, and pulses running along them
const threadMax = 40; const rods = new THREE.Group(); scene.add(rods); const rodMat = new THREE.MeshBasicMaterial({color:'#FFD9A8'}), up=new THREE.Vector3(0,1,0);
function drawThread(){ while(rods.children.length){ const m=rods.children[0]; m.geometry.dispose(); rods.remove(m); }
  for (const [p,q] of thread.slice(0,threadMax)){ const a=P(p),b=P(q),d=b.clone().sub(a),m=new THREE.Mesh(new THREE.CylinderGeometry(.45,.45,d.length(),8),rodMat); m.position.copy(a).addScaledVector(d,.5); m.quaternion.setFromUnitVectors(up,d.clone().normalize()); rods.add(m);
    for (const end of [a,b]){ const s=new THREE.Mesh(new THREE.SphereGeometry(1.5,12,12),rodMat); s.position.copy(end); rods.add(s); } } }
const pulseGeo = new THREE.BufferGeometry(); const pulsePos=new Float32Array(threadMax*3), pulseCol=new Float32Array(threadMax*3), pulseSize=new Float32Array(threadMax).fill(16); pulseCol.fill(1);
pulseGeo.setAttribute('position',new THREE.BufferAttribute(pulsePos,3)); pulseGeo.setAttribute('color',new THREE.BufferAttribute(pulseCol,3)); pulseGeo.setAttribute('size',new THREE.BufferAttribute(pulseSize,1)); pulseGeo.setDrawRange(0,0);
scene.add(new THREE.Points(pulseGeo,nodeMat));
// camera: a target, an orbit around it, and a flight between views
const reduced=matchMedia&&matchMedia('(prefers-reduced-motion: reduce)').matches;
const HOME = {x:0,y:-8,z:0,dist:430,theta:.7,phi:1.28};
const view = Object.assign({},HOME); let from=null,to=null,tStart=0; const TRAVEL=2200, DWELL=4200;
let stop=0, playing=true, lastAdvance=0; const thread=[];
function viewFor(s){ const own=stopView(s); if(own) return own; if(s.node===null||s.node===undefined) return Object.assign({},HOME); const p=P(s.node); const a=Math.atan2(p.z,p.x); return {x:p.x,y:p.y,z:p.z,dist:s.title==='The requirement'||s.title==="Claude's reasoning"?120:95,theta:a+.35,phi:1.22}; }
function short(a){ return a-Math.round(a/(2*Math.PI))*2*Math.PI; }
function goTo(i,instant){ stop=(i+D.stops.length)%D.stops.length; const s=D.stops[stop];
  $('ct').textContent=s.title; $('cx').textContent=s.text; $('pos').textContent=(stop+1)+' of '+D.stops.length;
  thread.length=0; for(let k=0;k<=stop;k++){ const p=D.stops[k].path; for(let j=0;j+1<p.length;j+=2) thread.push([p[j],p[j+1]]); }
  drawThread(); pulseGeo.setDrawRange(0,Math.min(thread.length,threadMax));
  onStop(s); flyTo(viewFor(s),instant); }
function flyTo(t,instant){ to=t; if(instant||reduced){ Object.assign(view,to); from=null; } else { from=Object.assign({},view); to.theta=from.theta+short(to.theta-from.theta); tStart=performance.now(); } lastAdvance=performance.now(); }
const ease=(t)=>t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2;
function setPlaying(v){ playing=v; $('play').textContent=v?'Pause':'Play'; lastAdvance=performance.now(); }
$('next').onclick=()=>{ setPlaying(false); goTo(stop+1); }; $('prev').onclick=()=>{ setPlaying(false); goTo(stop-1); }; $('play').onclick=()=>setPlaying(!playing);
// orbit, zoom, fly
const keys={}; let drag=null; const cv=$('c');
cv.addEventListener('pointerdown',(e)=>{ drag={x:e.clientX,y:e.clientY,moved:false,touch:e.pointerType==='touch'}; cv.setPointerCapture(e.pointerId); if(!HERO){ setPlaying(false); from=null; } cv.style.cursor='grabbing'; });
cv.addEventListener('pointermove',(e)=>{ if(drag){ if(!drag.moved){ const dx=Math.abs(e.clientX-drag.x),dy=Math.abs(e.clientY-drag.y); if(HERO&&(dx+dy<4||(drag.touch&&dx<=dy))) return; drag.moved=true; if(HERO){ setPlaying(false); from=null; } } view.theta+=(e.clientX-drag.x)*.005; view.phi=Math.max(.15,Math.min(3,view.phi-(e.clientY-drag.y)*.005)); drag={x:e.clientX,y:e.clientY}; } else hover(e); });
const release=()=>{ drag=null; cv.style.cursor='grab'; }; cv.addEventListener('pointerup',(e)=>{ const tapped=drag&&!drag.moved; release(); if(tapped) clickReader(e); }); cv.addEventListener('pointercancel',release);
cv.addEventListener('wheel',(e)=>{ if(HERO&&!e.ctrlKey&&!e.metaKey) return; e.preventDefault(); from=null; setPlaying(false); view.dist=Math.max(20,Math.min(1400,view.dist*Math.exp(e.deltaY*.001))); },{passive:false});
const flying=()=>!HERO||stage.getBoundingClientRect().bottom>innerHeight*.5, typing=(e)=>e.target.closest&&e.target.closest('input,textarea,select,[contenteditable]');
addEventListener('keydown',(e)=>{ if(typing(e)||(HERO&&(e.ctrlKey||e.metaKey||e.altKey||!flying()))) return; const k=e.key.toLowerCase(); keys[k]=true; if(!HERO&&k===' '&&e.target===document.body){ e.preventDefault(); setPlaying(!playing);} if(k==='n'||k==='arrowright'){ setPlaying(false); goTo(stop+1);} if(k==='p'||k==='arrowleft'){ setPlaying(false); goTo(stop-1);} if('wasdqe'.includes(k)){ setPlaying(false); from=null; } });
addEventListener('keyup',(e)=>{ keys[e.key.toLowerCase()]=false; });
// hover: the nearest named node on screen
const named=Object.keys(D.labels).map(Number); const v3=new THREE.Vector3();
function hover(e){ if(overReader(e)) return; let best=-1,bd=18*18; const box=cv.getBoundingClientRect(),w=cv.clientWidth,h=cv.clientHeight,ex=e.clientX-box.left,ey=e.clientY-box.top; for(const i of named){ v3.set(D.pos[3*i],D.pos[3*i+1],D.pos[3*i+2]).project(camera); if(v3.z>1) continue; const x=(v3.x+1)/2*w-ex,y=(1-v3.y)/2*h-ey,d=x*x+y*y; if(d<bd){bd=d;best=i;} }
  const t=$('tip'); if(best<0){ t.style.display='none'; return; } const kind=D.kinds[D.kind[best]]; t.textContent=D.labels[best]?kind+': '+D.labels[best]:D.roles?D.roles[kind]:kind; t.style.display='block'; t.style.left=Math.min(ex+14,w-330)+'px'; t.style.top=(ey+14)+'px'; }
// a hero sizes to its stage, stands back far enough on a tall thin screen that the rings fit its width, and centres the view in the room the card and the headline leave
function frameCard(){ const w=cv.clientWidth,h=cv.clientHeight,card=$('card').getBoundingClientRect().height,top=$('head').getBoundingClientRect().height; camera.setViewOffset(w,h,0,Math.round(w<=640?(card-top)/2:card*.38),w,h); }
function resize(){ const box=HERO?stage.getBoundingClientRect():null,w=box?box.width:innerWidth,h=box?box.height:innerHeight; renderer.setSize(w,h,false); camera.aspect=w/h; camera.updateProjectionMatrix(); nodeMat.uniforms.scale.value=h*.9;
  if(HERO){ HOME.dist=D.readers?600*Math.max(1,.88/camera.aspect):470*Math.max(1,.95/camera.aspect); frameCard(); }
  onResize(); }
addEventListener('resize',resize); resize();
let last=performance.now();
function frame(now){ const gap=now-last,dt=Math.min(.05,gap/1000); last=now;
  if(HERO){ if(!flying()) for(const k in keys) keys[k]=false; if(stage.getBoundingClientRect().bottom<=0){ lastAdvance+=gap; tStart+=gap; requestAnimationFrame(frame); return; } }
  if(from&&to){ const u=Math.min(1,(now-tStart)/TRAVEL),e=ease(u); for(const k of ['x','y','z','dist','theta','phi']) view[k]=from[k]+(to[k]-from[k])*e; if(u>=1) from=null; }
  else if(playing){ if(!reduced) view.theta+=dt*.05; if(now-lastAdvance>TRAVEL+DWELL) goTo(stop+1); }
  const sp=view.dist*dt*.7, fx=Math.cos(view.theta),fz=Math.sin(view.theta);
  if(keys.w){ view.x-=fx*sp; view.z-=fz*sp; } if(keys.s){ view.x+=fx*sp; view.z+=fz*sp; } if(keys.a){ view.x-=fz*sp; view.z+=fx*sp; } if(keys.d){ view.x+=fz*sp; view.z-=fx*sp; } if(keys.q) view.y-=sp; if(keys.e) view.y+=sp;
  const sr=Math.sin(view.phi); camera.position.set(view.x+view.dist*sr*Math.cos(view.theta),view.y+view.dist*Math.cos(view.phi),view.z+view.dist*sr*Math.sin(view.theta)); camera.lookAt(view.x,view.y,view.z);
  const pp=pulseGeo.attributes.position; const phase=reduced?.5:(now/1100)%1; thread.slice(0,threadMax).forEach((e,n)=>{ const a=P(e[0]),b=P(e[1]); pp.setXYZ(n,a.x+(b.x-a.x)*phase,a.y+(b.y-a.y)*phase,a.z+(b.z-a.z)*phase); }); pp.needsUpdate=true;
  renderer.render(scene,camera); onFrame(); requestAnimationFrame(frame); }
/*READERS*/
// #maps opens that view, #s3 that tour stop. ?shot=1&view=maps&stop=4&theta=.7&phi=1.2&dist=300&x=0&y=0&z=0 frames one still for a slide: the same data, no panels
const hash=location.hash.slice(1), q=new URLSearchParams(location.search), shot=q.has('shot');
setView(q.get('view')||hash);
if(/^s\d+$/.test(hash)){ goTo(Number(hash.slice(1)),true); setPlaying(false); } else { goTo(Number(q.get('stop')||0),true); if(shot||reduced) setPlaying(false); }
if(HERO||D.readers) resize();      // again, now the card holds its text
if(shot){ document.body.classList.add('shot'); for(const k of ['x','y','z','dist','theta','phi']) if(q.has(k)) view[k]=Number(q.get(k)); }
window.__ready=true; requestAnimationFrame(frame);
</script>
"""

FIXED_HEAD = """<div class="hud" id="head"><h1 id="t"></h1><p id="s"></p></div>
<div class="hud" id="layers"></div>"""
TEMPLATE = "<title>Requirements Graph Fly-Through</title>\n" + STYLE + MARKUP.replace("/*HEAD*/", FIXED_HEAD).replace("/*HINT*/", "Drag to orbit, W A S D to fly").replace("/*CUE*/", "") + SCRIPT

HERO_STYLE = r"""<style>
html,body { height:auto; overflow:visible }
#stage { position:relative; height:100vh; height:100svh; min-height:540px; overflow:hidden; background:var(--bg) }
#stage canvas { position:absolute; inset:0; touch-action:pan-y pinch-zoom }
#stage .hud { position:absolute; z-index:2 }
#stage::before { content:''; position:absolute; inset:0 0 auto 0; height:min(62%,480px); z-index:1; pointer-events:none; background:linear-gradient(180deg,rgba(11,18,25,.92) 0%,rgba(11,18,25,.66) 48%,rgba(11,18,25,0) 100%) }
#stage #head { top:calc(clamp(20px,3.2vw,44px) + env(safe-area-inset-top,0px)); left:max(clamp(16px,3.2vw,48px),env(safe-area-inset-left,0px)); right:clamp(16px,3.2vw,48px) }
#stage #head h1 { font:700 clamp(30px,4.5vw,62px)/1.06 'Space Grotesk',system-ui,sans-serif; letter-spacing:-.015em; max-width:21ch }
#stage #head #tag { margin:14px 0 0; font-size:clamp(16px,1.5vw,21px); line-height:1.35; color:var(--fg); max-width:44ch }
#stage #head #s { margin-top:10px; max-width:60ch }
#stage #layers { position:static; margin-top:16px }
#stage #card { width:min(760px,calc(100vw - 32px)) }
#hint { flex:1 1 auto }
#cue { margin-left:auto; padding:7px 14px; border-radius:8px; background:var(--accent); color:#0B1219; font-weight:600; text-decoration:none; white-space:nowrap }
#cue:hover,#cue:focus-visible { background:#F7A566; outline:2px solid var(--fg); outline-offset:2px }
@media (prefers-reduced-motion:no-preference) { html { scroll-behavior:smooth } }
@media (hover:none) { #hint { display:none } }
@media (max-height:600px) { #stage #layers { display:none } }
#nojs { position:absolute; z-index:2; left:clamp(16px,3.2vw,48px); right:16px; bottom:24px; margin:0; color:var(--muted) }
@media (max-width:640px) {
  #stage #head h1 { font-size:clamp(25px,7.6vw,30px) }
  #stage #head #tag { margin-top:8px; font-size:15px }
  #stage #head #s { margin-top:6px; font-size:12px }
  #hint { display:none }
  #ctl label { position:absolute; width:1px; height:1px; overflow:hidden; clip-path:inset(50%) }
  #stage #card { padding:10px 12px }
  #card h2 { font-size:16px }
  #card p { font-size:14px; line-height:1.35; min-height:5.4em }
  #ctl { gap:6px; margin-top:8px }
  #ctl button { padding:6px 10px }
  #ctl select { flex:1 1 40%; min-width:0; font-size:16px }
  #cue { flex:1 1 100%; margin-left:0; text-align:center }
}
</style>
"""


class Hero(object):
    """The fly-through as the first screen of a page that goes on below it. A page somebody reads has to let them leave the graph, so: a whole
    document (doctype, charset, viewport) rather than a fragment, a stage one screen high rather than a fixed canvas, and the host's own text under it"""

    def __init__(self, headline, tagline="", below="", cue=None, description=None):
        self.headline, self.tagline, self.below, self.cue, self.description = headline, tagline, below, cue, description

    def head(self):
        tag = '<p id="tag">%s</p>' % markup_text.escape(self.tagline, quote=False) if self.tagline else ""
        return '<div class="hud" id="head"><h1 id="t">%s</h1>%s<p id="s"></p><div id="layers"></div></div>' % (markup_text.escape(self.headline, quote=False), tag)

    def document(self):
        styles = re.findall(r"<style>.*?</style>", self.below, re.S)       # the host's styles belong in the head
        body = re.sub(r"<style>.*?</style>\s*", "", self.below, flags=re.S)
        cue = '<a id="cue" href="%s">%s <span aria-hidden="true">&darr;</span></a>' % (markup_text.escape(self.cue[1], quote=True), markup_text.escape(self.cue[0], quote=False)) if self.cue else ""
        said = markup_text.escape(self.description or self.tagline, quote=True)
        name = markup_text.escape(self.headline, quote=True)
        stage = MARKUP.replace("/*HEAD*/", self.head()).replace("/*HINT*/", "Drag to orbit, ctrl and scroll to zoom, W A S D to fly").replace("/*CUE*/", cue)
        return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                + '<title>%s</title>\n<meta name="description" content="%s">\n<meta property="og:title" content="%s">\n<meta property="og:description" content="%s">\n<meta property="og:type" content="website">\n' % (name, said, name, said)
                + '<link rel="icon" href="data:,">\n<link rel="stylesheet" href="%s">\n' % markup_text.escape(FONTS, quote=True)
                + STYLE + HERO_STYLE + "".join(styles) + '\n</head>\n<body>\n<section id="stage" aria-label="The graph">\n' + stage
                + '<noscript><style>#card { display:none }</style><p id="nojs">The graph needs JavaScript. The text below does not.</p></noscript>\n</section>\n' + body + "\n" + SCRIPT + "</body>\n</html>\n")


READER_STYLE = r"""<style>
#stage #labels, #labels { inset:0; overflow:hidden }
.fly-rl { position:absolute; left:0; top:0; padding:2px 7px; font-size:12px; line-height:1.3; white-space:nowrap; background:rgba(11,18,25,.74); border:1px solid var(--line); border-radius:6px; pointer-events:auto; cursor:pointer }
.fly-rl.planned, .fly-rl.borrowed { color:var(--muted) }
.fly-rl.on { border-color:var(--accent); background:rgba(16,26,36,.96) }
.fly-rl[hidden] { display:none }
.fly-panel { box-sizing:border-box; display:flex; flex-direction:column; width:290px; background:var(--panel); border:1px solid var(--line); border-radius:12px; pointer-events:auto; overflow:hidden }
#stage .fly-panel { z-index:3 }
#stage #card { z-index:3 }
.fly-panel:empty { display:none }
.fly-panel header { display:flex; align-items:baseline; gap:8px; padding:10px 12px 6px; flex:none }
.fly-panel h2 { margin:0; font:600 16px/1.2 'Space Grotesk',system-ui,sans-serif }
.fly-panel .fly-n { color:var(--muted); font-size:12px; font-variant-numeric:tabular-nums }
.fly-panel .fly-x { display:none; margin-left:auto; padding:2px 10px }
.fly-panel .fly-list { flex:1 1 auto; min-height:0; overflow:auto; padding:0 6px 8px; scrollbar-width:thin; scrollbar-color:var(--line) transparent }
.fly-panel .fly-row { display:flex; align-items:flex-start; gap:9px; width:100%; box-sizing:border-box; padding:6px 7px; text-align:left; background:none; border:1px solid transparent; border-radius:8px; font-size:13px; line-height:1.3 }
.fly-panel .fly-row:hover, .fly-panel .fly-row:focus-visible { background:#16222E; border-color:var(--line); outline:none }
.fly-panel .fly-row[aria-pressed="true"] { background:#1D2C3A; border-color:var(--accent) }
.fly-panel .fly-row.marked { border-color:#4A6275; background:#141F2A }
.fly-panel .fly-row.faded { opacity:.45 }
.fly-panel .fly-row .fly-t { display:block; color:var(--fg) }
.fly-panel .fly-row .fly-w { display:block; color:var(--muted); font-size:12px }
.fly-glyph { flex:none; box-sizing:border-box; width:13px; height:13px; margin-top:2px; border-radius:50%; border:2px solid #FF8F6B }
.fly-glyph.reached { background:#FFD27A; border-color:#FFD27A }
.fly-glyph.doubted { border-color:#FFD27A; background:radial-gradient(circle,#CDBB8E 38%,transparent 42%) }
.fly-glyph.partly { border-color:#FFD27A; background:linear-gradient(90deg,#FFD27A 50%,transparent 50%) }
.fly-glyph.borrowed, .fly-glyph.planned { border-color:#7E8E9E; border-style:dashed }
#goals { left:max(clamp(16px,3.2vw,48px),env(safe-area-inset-left,0px)); bottom:16px }
#stats { right:16px; top:16px; bottom:16px }
.fly-notes { margin:8px 6px 0; padding-top:8px; border-top:1px solid var(--line); color:var(--muted); font-size:12px; line-height:1.35 }
.fly-notes p { margin:0 0 6px }
.fly-find { display:flex; gap:6px; padding:0 8px 6px; flex:none }
.fly-find input { flex:1 1 auto; min-width:0; box-sizing:border-box; padding:5px 8px; background:#16222E; color:var(--fg); border:1px solid var(--line); border-radius:7px; font:inherit; font-size:13px }
.fly-find input:focus-visible { outline:none; border-color:var(--accent) }
.fly-find button { flex:none; padding:4px 8px; font-size:12px }
.fly-find button[aria-pressed="true"] { border-color:var(--accent) }
.fly-because { padding:0 12px 6px; color:var(--muted); font-size:12px; flex:none }
.fly-because[hidden] { display:none }
.fly-group { margin:8px 7px 2px; color:var(--muted); font-size:11.5px; font-weight:600; letter-spacing:.02em }
.fly-group[hidden], .fly-panel .fly-row[hidden] { display:none }
.fly-stat { align-items:baseline; justify-content:space-between; font-variant-ligatures:none }
.fly-stat .fly-nm { overflow-wrap:anywhere; font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:12px; color:var(--fg) }
.fly-stat.fly-none .fly-nm { color:var(--muted) }
.fly-stat .fly-tag { flex:none; color:var(--muted); font-size:11px; font-variant-numeric:tabular-nums }
#card #cs { margin-top:6px }
#card #cs:empty { display:none }
#card #cs p { margin:4px 0 0; min-height:0; font-size:13.5px; line-height:1.35; color:var(--fg) }
#card #cs p.quiet { color:var(--muted); font-size:12.5px }
#card #cs p.fly-doubt { color:#E8D9A8; font-size:13px }
#card #cs b { font-weight:600 }
#card #cs .miss b { color:#FF9E7E }
#card #cs .hit b { color:#FFD27A }
#clear { margin-top:6px; padding:3px 9px; font-size:12px }
.fly-toggle { display:none }
@media (min-width:981px) { #card, #stage #card { width:min(760px,calc(100% - 660px)); min-width:320px } }
@media (min-width:981px) and (max-height:760px) { #stage #layers { display:none } }
@media (max-width:980px) {
  .fly-toggle { display:inline-block }
  .fly-panel { display:none; left:8px; right:8px; top:auto; bottom:8px; width:auto; max-height:74%; overscroll-behavior:contain }
  #goals, #stats { left:8px; right:8px; top:auto; bottom:8px }
  .fly-panel.open { display:flex; background:#101A24 }
  #stage .fly-panel.open { z-index:4 }
  .fly-panel .fly-x { display:block }
  .fly-panel .fly-row { padding:8px 7px }
  .fly-find input { font-size:16px }
}
@media (max-width:640px) {
  #ctl select { flex:1 1 30%; font-size:16px }
  #card #cs p { font-size:13px }
  #card #cs { max-height:32vh; overflow:auto }
  #card #cs p.quiet, #card #cs .fly-rule { display:none }
}
</style>
"""

PANELS = """<div class="hud" id="labels"></div>
<aside class="hud fly-panel" id="goals" aria-label="Goals"></aside>
<aside class="hud fly-panel" id="stats" aria-label="Stats"></aside>
"""

TOGGLES = """<button class="fly-toggle" id="goalsBtn" type="button" aria-expanded="false" aria-controls="goals">Goals</button><button class="fly-toggle" id="statsBtn" type="button" aria-expanded="false" aria-controls="stats">Stats</button>"""

READER_SCRIPT = r"""// ---- the readers: a large node per goal outside the rings, rays into what it reads. Filled has reached its goal, hollow has not, because trying is not reaching
const READER_COLOUR={reached:'#FFD27A',doubted:'#FFD27A',partly:'#FFD27A',trying:'#FF8F6B',borrowed:'#7E8E9E',planned:'#7E8E9E'}, RAY_SAMPLE=16, RAY_VIEW=140;     // all the rays burn to white where they meet: a selected reader gets them all
const kindCount=D.kinds.map(()=>0); for(let i=0;i<N;i++) kindCount[D.kind[i]]++;
const relKinds=D.rels.map(()=>new Set()); for(let e=0;e<edgeCount;e++){ const r=relKinds[D.edges[3*e+2]]; r.add(D.kind[D.edges[3*e]]); r.add(D.kind[D.edges[3*e+1]]); }
const readerNodes=D.readers.map((r)=>{ const colour=new THREE.Color(READER_COLOUR[r.state]), small=r.state==='planned'||r.state==='borrowed', R=small?6:11, at=new THREE.Vector3(r.pos[0],r.pos[1],r.pos[2]), group=new THREE.Group(); group.position.copy(at);
  const shellMat=new THREE.LineBasicMaterial({color:colour,transparent:true,depthWrite:false}); group.add(new THREE.LineSegments(new THREE.WireframeGeometry(new THREE.IcosahedronGeometry(R,1)),shellMat));
  let coreMat=null; if(r.state==='reached'||r.state==='partly'||r.state==='doubted'){ coreMat=new THREE.MeshBasicMaterial({color:r.state==='doubted'?'#CDBB8E':colour,transparent:true}); group.add(new THREE.Mesh(new THREE.IcosahedronGeometry(r.state==='reached'?R*.94:r.state==='doubted'?R*.68:R*.42,1),coreMat)); }
  scene.add(group);
  const ids=[]; for(let n=0;n<N;n++) if(r.reads.includes(D.kind[n])) ids.push(n);
  ids.sort((a,b)=>((a*2654435761)>>>0)-((b*2654435761)>>>0));      // a fixed shuffle, so the first few rays are a fair sample of them all
  const ends=ids.map(P); if(r.uses!==null) ends.push(new THREE.Vector3(D.readers[r.uses].pos[0],D.readers[r.uses].pos[1],D.readers[r.uses].pos[2]));
  const p=new Float32Array(ends.length*6); ends.forEach((e,k)=>{ p[6*k]=at.x; p[6*k+1]=at.y; p[6*k+2]=at.z; p[6*k+3]=e.x; p[6*k+4]=e.y; p[6*k+5]=e.z; });
  const geo=new THREE.BufferGeometry(); geo.setAttribute('position',new THREE.BufferAttribute(p,3));
  const rayMat=new THREE.LineBasicMaterial({color:colour,transparent:true,blending:THREE.AdditiveBlending,depthWrite:false}), rays=new THREE.LineSegments(geo,rayMat); scene.add(rays);
  return {r,at,group,shellMat,coreMat,rays,rayMat,rayCount:ends.length,nodeCount:ids.length}; });
const wide=()=>innerWidth>980, fix=(x)=>Number(x).toFixed(3);
let focus=null, savedView=null;
const wordsOf=(r)=>{ const won=r.scores.filter((s)=>s.achieved).length; if(r.state==='reached') return 'Reached its goal'; if(r.state==='doubted') return 'Reached, but doubted'; if(r.state==='partly') return 'Reached on '+won+' of '+r.scores.length; if(r.state==='trying') return r.scores.length?'Has not reached it':'Trained, no score on file'; if(r.state==='borrowed') return "Uses another goal's model"; return r.waits?'Planned, waits for '+r.waits:'Planned'; };
const seesOf=(x)=>{ const r=x.r; if(r.state==='planned') return 'Nothing to read yet.'; if(r.state==='borrowed') return 'It uses the model of "'+D.readers[r.uses].title+'" and has no score of its own.'; return 'Reads '+x.nodeCount.toLocaleString('en')+' nodes: '+r.sees+'. '+(r.predicts!==null?'Predicts the "'+r.learns+'" links.':'Learns from '+r.learns+'.'); };
onPaint=()=>{ const all=viewNow.id==='readers', drawn=all||viewNow.id==='all'||viewNow.id==='heard';
  readerNodes.forEach((x,i)=>{ const on=!!focus&&focus.readers.has(i), whole=on&&focus.goal!==undefined, n=whole?x.rayCount:!focus&&all?RAY_VIEW:RAY_SAMPLE;
    x.rays.visible=focus?on:drawn; x.rays.geometry.setDrawRange(0,2*Math.min(n,x.rayCount)); x.rayMat.opacity=whole?.2*Math.sqrt(562/Math.max(562,x.rayCount)):on?.34:all?.16:.2;
    const o=focus&&!on?.28:1; x.shellMat.opacity=o; if(x.coreMat) x.coreMat.opacity=o; x.group.scale.setScalar(on?1.15:1); }); };
function litOfGoal(i){ const r=D.readers[i], source=r.uses!==null?D.readers[r.uses]:r, kinds=new Set(source.reads), rels=new Set(), readers=new Set([i]); if(r.predicts!==null){ rels.add(r.predicts); relKinds[r.predicts].forEach((k)=>kinds.add(k)); } if(r.uses!==null) readers.add(r.uses); return {goal:i,readers,lit:{kinds,rels}}; }
function setFocus(f){ focus=f; lit=f?f.lit:null; paint(); refreshLists(); }
function readerView(i){ const p=D.readers[i].pos, a=Math.atan2(p[2],p[0]), scale=HOME.dist/600; return {x:p[0]*.6,y:p[1]*.8,z:p[2]*.6,dist:430*scale,theta:a+.5,phi:1.24}; }
stopView=(s)=>s.goal!==undefined?readerView(s.goal):null;
function goalCard(i){ const r=D.readers[i], box=$('cs'); box.textContent='';
  for(const s of r.scores){ const d=document.createElement('p'); d.className=s.achieved?'hit':'miss'; const b=document.createElement('b'), rule=document.createElement('span'); b.textContent=s.project+': '; rule.className='fly-rule'; rule.textContent=' ('+s.rule+')'; d.append(b,document.createTextNode('learned '+fix(s.learned)+' against plain '+fix(s.plain)),rule,document.createTextNode(', '+s.wins+' of '+s.seeds+' seeds won. '+(s.achieved?(s.doubted?'Reached, but doubted.':'Reached.'):'Not reached.')+(s.flips?' It changes with the seed.':''))); box.append(d); }
  for(const why of r.doubts){ const q=document.createElement('p'); q.className='fly-doubt'; q.textContent='Doubt: '+why; box.append(q); }
  if(r.note){ const q=document.createElement('p'); q.className='fly-doubt'; q.textContent=r.note; box.append(q); }
  const quiet=[r.scores.length?r.measure:'',r.asks.length?'It asks for what beat it: '+r.asks.join(', ')+'.':''].filter(Boolean).join(' ');
  if(quiet){ const q=document.createElement('p'); q.className='quiet'; q.textContent=quiet; box.append(q); } }
function clearButton(){ const b=document.createElement('button'); b.id='clear'; b.type='button'; b.textContent='Clear selection'; b.onclick=()=>clearFocus(true); $('cs').append(b); }
function selectGoal(i){ closePanels(); setPlaying(false); if(focus&&focus.goal===i){ clearFocus(true); return; } setFocus(litOfGoal(i)); const r=D.readers[i]; $('ct').textContent=r.title; goalCard(i); $('cx').textContent=wordsOf(r)+'. '+seesOf(readerNodes[i]); clearButton(); if(HERO) frameCard(); flyTo(readerView(i)); }
function selectStat(i){ closePanels(); setPlaying(false); if(focus&&focus.stat===i){ clearFocus(true); return; } const s=D.statRows[i]; setFocus({stat:i,readers:new Set(s.goals),lit:{kinds:new Set(s.on===null?[]:[s.on]),rels:new Set()}});
  $('ct').textContent=s.name; const titles=s.goals.map((g)=>D.readers[g].title), asked=s.asked.map((g)=>D.readers[g].title);
  $('cx').textContent=[s.record+'.',s.verdict,titles.length?'Used by: '+titles.join('; ')+'.':'No goal uses it yet.',asked.length?'Asked for by: '+asked.join('; ')+'.':'',s.on===null?'It is not linked to one kind of node.':'It counts the '+(D.roles?(D.roles[D.kinds[s.on]]||D.kinds[s.on]).replace(/^an? /,''):D.kinds[s.on])+' nodes, lit here.',s.read?'A requirement reads it.':''].filter(Boolean).join(' ');
  $('cs').textContent=''; clearButton(); if(HERO) frameCard(); flyTo(Object.assign({},HOME)); }
function clearFocus(fly){ const s=D.stops[stop]; $('ct').textContent=s.title; $('cx').textContent=s.text; $('cs').textContent=''; setFocus(null); if(HERO) frameCard(); if(fly) flyTo(viewFor(s)); }
onStop=(s)=>{ $('cs').textContent=''; if(s.view){ if(savedView===null) savedView=viewNow.id; viewNow=D.views.find((x)=>x.id===s.view)||viewNow; $('view').value=viewNow.id; } else if(savedView!==null){ viewNow=D.views.find((x)=>x.id===savedView)||D.views[0]; $('view').value=viewNow.id; savedView=null; }
  if(s.goal!==undefined){ setFocus(litOfGoal(s.goal)); goalCard(s.goal); } else setFocus(null); };
$('view').addEventListener('change',()=>{ savedView=null; if(focus) clearFocus(false); });
// the two lists, which work on each other: a goal filters the stats to the ones it uses, a stat marks the goals that use it
function head(id,title,count){ const h=document.createElement('header'), t=document.createElement('h2'), n=document.createElement('span'), x=document.createElement('button'); t.textContent=title; n.className='fly-n'; n.textContent=count; x.className='fly-x'; x.type='button'; x.textContent='Close'; x.onclick=closePanels; h.append(t,n,x); $(id).append(h); }
head('goals','Goals',String(D.readers.length));
const goalList=document.createElement('div'); goalList.className='fly-list';
const goalRows=D.readers.map((r,i)=>{ const b=document.createElement('button'); b.type='button'; b.className='fly-row'; b.setAttribute('aria-pressed','false'); const g=document.createElement('span'); g.className='fly-glyph '+r.state; g.setAttribute('aria-hidden','true'); const c=document.createElement('span'), t=document.createElement('span'), w=document.createElement('span'); t.className='fly-t'; t.textContent=r.title; w.className='fly-w'; w.textContent=wordsOf(r); c.append(t,w); b.append(g,c); b.onclick=()=>selectGoal(i); goalList.append(b); return b; });
if(D.readerNotes.length){ const n=document.createElement('div'); n.className='fly-notes'; for(const line of D.readerNotes){ const p=document.createElement('p'); p.textContent=line; n.append(p); } goalList.append(n); }
$('goals').append(goalList);
head('stats','Stats',String(D.statRows.length));
const because=document.createElement('div'); because.className='fly-because'; because.hidden=true; $('stats').append(because);
const finder=document.createElement('div'); finder.className='fly-find'; const query=document.createElement('input'); query.type='search'; query.placeholder='Find a stat'; query.setAttribute('aria-label','Find a stat'); const usedOnly=document.createElement('button'); usedOnly.type='button'; usedOnly.textContent='Used'; usedOnly.title='Only the stats a goal uses'; usedOnly.setAttribute('aria-pressed','false'); finder.append(query,usedOnly); $('stats').append(finder);
const statList=document.createElement('div'); statList.className='fly-list'; const statGroups=new Map(), statTags=[];
const statRows=D.statRows.map((s,i)=>{ if(!statGroups.has(s.record)){ const h=document.createElement('div'); h.className='fly-group'; h.textContent=s.record; statList.append(h); statGroups.set(s.record,{h,rows:[]}); }
  const b=document.createElement('button'); b.type='button'; b.className='fly-row fly-stat'+(s.goals.length?'':' fly-none'); b.setAttribute('aria-pressed','false'); const n=document.createElement('span'), t=document.createElement('span'); n.className='fly-nm'; n.textContent=s.name; t.className='fly-tag'; statTags.push(t); b.append(n,t); b.onclick=()=>selectStat(i); statList.append(b); statGroups.get(s.record).rows.push(i); return b; });
$('stats').append(statList);
function refreshLists(){ const g=focus&&focus.goal!==undefined?focus.goal:null, st=focus&&focus.stat!==undefined?focus.stat:null, words=query.value.trim().toLowerCase(), used=usedOnly.getAttribute('aria-pressed')==='true';
  goalRows.forEach((b,i)=>{ b.setAttribute('aria-pressed',String(g===i)); b.classList.toggle('marked',st!==null&&D.statRows[st].goals.includes(i)); b.classList.toggle('faded',st!==null&&!D.statRows[st].goals.includes(i)); });
  statRows.forEach((b,i)=>{ const s=D.statRows[i]; b.setAttribute('aria-pressed',String(st===i)); b.hidden=(g!==null&&!s.goals.includes(g)&&!s.asked.includes(g))||(g===null&&((used&&!s.goals.length)||(words&&!s.name.toLowerCase().includes(words)))); b.classList.toggle('marked',g!==null&&s.asked.includes(g)); statTags[i].textContent=g!==null&&s.asked.includes(g)?'asked':s.goals.length?s.goals.length+(s.goals.length===1?' goal':' goals'):''; });
  statGroups.forEach((v)=>{ v.h.hidden=v.rows.every((i)=>statRows[i].hidden); });
  finder.hidden=g!==null; finder.style.display=g!==null?'none':''; because.hidden=g===null; if(g!==null){ const n=statRows.filter((b)=>!b.hidden).length; because.textContent=n?n+(n===1?' stat':' stats')+' that "'+D.readers[g].title+'" uses or asks for':'No stat is linked to "'+D.readers[g].title+'" yet'; }
  labels.forEach((b,i)=>b.classList.toggle('on',!!focus&&focus.readers.has(i))); }
query.oninput=refreshLists; usedOnly.onclick=()=>{ usedOnly.setAttribute('aria-pressed',String(usedOnly.getAttribute('aria-pressed')!=='true')); refreshLists(); };
const labels=D.readers.map((r,i)=>{ const b=document.createElement('button'); b.type='button'; b.className='fly-rl '+r.state; b.textContent=r.title+(r.state==='doubted'?' (doubted)':''); b.tabIndex=-1; b.hidden=true; b.onclick=()=>selectGoal(i); $('labels').append(b); return b; });
function closePanels(){ for(const [p,t] of [['goals','goalsBtn'],['stats','statsBtn']]){ $(p).classList.remove('open'); $(t).setAttribute('aria-expanded','false'); } }
for(const [p,t,other] of [['goals','goalsBtn','stats'],['stats','statsBtn','goals']]) $(t).onclick=()=>{ const open=!$(p).classList.contains('open'); closePanels(); if(open){ $(p).classList.add('open'); $(t).setAttribute('aria-expanded','true'); } };
addEventListener('keydown',(e)=>{ if(e.key==='Escape'){ closePanels(); if(focus) clearFocus(true); } });
function readerAt(e){ const box=cv.getBoundingClientRect(), w=cv.clientWidth, h=cv.clientHeight, ex=e.clientX-box.left, ey=e.clientY-box.top; let best=-1, bd=30*30; readerNodes.forEach((x,i)=>{ v3.copy(x.at).project(camera); if(v3.z>1) return; const dx=(v3.x+1)/2*w-ex, dy=(1-v3.y)/2*h-ey, d=dx*dx+dy*dy; if(d<bd){ bd=d; best=i; } }); return {best,ex,ey,w}; }
overReader=(e)=>{ const a=readerAt(e), t=$('tip'); if(a.best<0){ if(!drag) cv.style.cursor='grab'; return false; } cv.style.cursor='pointer'; const r=D.readers[a.best]; t.textContent=r.title+': '+wordsOf(r); t.style.display='block'; t.style.left=Math.min(a.ex+14,a.w-330)+'px'; t.style.top=(a.ey+14)+'px'; return true; };
clickReader=(e)=>{ const a=readerAt(e); if(a.best>=0) selectGoal(a.best); };
const labelWidth=D.readers.map(()=>0); let keepOff=[], rightEnd=0, tick=0;
function measureLabels(){ const box=cv.getBoundingClientRect(), near=(id)=>{ const e=$(id); if(!e||!e.offsetParent) return null; const r=e.getBoundingClientRect(); return {l:r.left-box.left,r:r.right-box.left,t:r.top-box.top,b:r.bottom-box.top}; };
  keepOff=['t','tag','s','layers','goals','stats','card'].map(near).filter(Boolean); const st=near('stats'); rightEnd=st?st.l-8:cv.clientWidth-6; }
onFrame=()=>{ if(tick++%20===0) measureLabels(); const w=cv.clientWidth, h=cv.clientHeight, allowed=wide();
  readerNodes.forEach((x,i)=>{ v3.copy(x.at).project(camera); const b=labels[i], chosen=!!focus&&focus.readers.has(i), trained=x.r.state!=='planned'&&x.r.state!=='borrowed'; let show=v3.z<1&&(chosen||(allowed&&trained));
    if(show){ const px=(v3.x+1)/2*w, py=(1-v3.y)/2*h; if(b.hidden) b.hidden=false; const lw=labelWidth[i]||(labelWidth[i]=b.offsetWidth); let lx=px+18; if(lx+lw>rightEnd) lx=px-18-lw; lx=Math.max(6,lx); const ly=py-11;
      if(!chosen&&keepOff.some((o)=>lx<o.r&&lx+lw>o.l&&ly<o.b&&ly+22>o.t)) show=false; else b.style.transform='translate('+Math.round(lx)+'px,'+Math.round(ly)+'px)'; }
    if(b.hidden===show) b.hidden=!show; }); };
onResize=()=>{ const g=$('goals'); if(wide()){ const top=(stage||document.body).getBoundingClientRect().top, a=$('head').getBoundingClientRect().bottom, l=$('layers').getBoundingClientRect().bottom; g.style.top=(Math.max(a,l)-top+12)+'px'; } else g.style.top=''; tick=0; };
"""


class Readers(object):
    """The learned parts drawn as what they are: readers standing outside the graph, one per goal, each with rays into what it reads and lit lines along what it
    predicts. What a goal learns and how it scored come from the file gnn_goals.py writes, never by hand here; what a model reads comes from READS_OF_MODEL unless the
    goal says. Trying is not reaching, so `state` is the first thing drawn: reached on every project, reached but doubted, on some, on none, using another goal's model, or only planned.
    Beside them, the two lists that work on each other: the goals we try for and the stats we could try with."""

    RADIUS = 214
    GOLDEN = math.pi * (3 - math.sqrt(5))       # any number of goals spread evenly round the ring, with no gap to redo when a goal is added
    HEIGHT = {"graph network": 44, "small model on the break's features": -72}
    IDLE_HEIGHT = 80                             # a reader with no model of its own stands apart from those that read: it reads nothing yet

    def __init__(self, book, layout):
        self.book, self.g = book, layout
        self.goals = book["goals"]
        self.index = {g["id"]: n for n, g in enumerate(self.goals)}
        self.projects, self.rules, self.records = book.get("projects") or {}, book.get("rules") or {}, book.get("records") or {}
        self.asks = collections.defaultdict(list)
        for a in (book.get("asks") or {}).get("goals_ask_stats") or []:
            if a["goal"] in self.index and a["stat"] not in self.asks[a["goal"]]:
                self.asks[a["goal"]].append(a["stat"])

    @staticmethod
    def state(goal):
        scores = list((goal.get("scores") or {}).values())
        if scores:
            won = sum(1 for s in scores if s["achieved"])
            if won == len(scores):
                return "doubted" if any(s.get("doubted") for s in scores) else "reached"
            return "partly" if won else "trying"
        if goal.get("uses"):
            return "borrowed"
        return "trying" if goal.get("state") == "trained" else "planned"

    def reads(self, goal):
        """the kinds it reads: its own word if it gives one, else its model's. No model reads nothing yet, and a model this tool has not been told about stops the build"""
        if goal.get("reads"):
            return set(goal["reads"])
        model = goal.get("model")
        if model is None:
            return set()
        if model not in READS_OF_MODEL:
            raise ValueError("goal %s has a model this tool cannot draw: %s. Add it to READS_OF_MODEL, or give the goal `reads`" % (goal.get("id"), model))
        return READS_OF_MODEL[model]

    def place(self, n, goal):
        a = n * self.GOLDEN
        y = self.HEIGHT.get(goal.get("model"), self.IDLE_HEIGHT)
        return [round(self.RADIUS * math.cos(a), 1), y, round(self.RADIUS * math.sin(a), 1)]

    def scores(self, goal):
        out = []
        for key, s in (goal.get("scores") or {}).items():
            if key not in self.projects:
                raise ValueError("goal %s has a score on a project the file does not name: the page shows names, never keys" % goal.get("id"))
            out.append({"project": self.projects[key], "learned": s["learned"], "plain": s["plain"], "rule": self.rules.get(s["rule"], s["rule"]), "wins": s["wins"], "seeds": s["seeds"],
                        "achieved": s["achieved"], "flips": s["flips"], "doubted": bool(s.get("doubted"))})
        return sorted(out, key=lambda x: x["project"])

    def rows(self, kinds):
        """one row per stat a reader can tell apart: rows of the same name under the same record's name are merged. A stat naming a goal the file does not hold stops the build"""
        merged = collections.OrderedDict()
        for s in self.book["stats"]:
            label = self.records.get(s["record"], "Other records")
            row = merged.setdefault((label, s["name"]), {"record": label, "name": s["name"], "goals": set(), "asked": set(), "on": None, "verdict": "", "read": False})
            for goal in s.get("goals") or []:
                if goal not in self.index:
                    raise ValueError("stat %s names a goal the file does not hold: %s" % (s["name"], goal))
                row["goals"].add(self.index[goal])
            row["read"] = row["read"] or bool(s.get("read"))
            row["verdict"] = row["verdict"] or VERDICT_WORDS.get(s.get("verdict"), "")
            for kind, records in STAT_SITS_ON.items():
                if s["name"] in records.get(s["record"], ()):
                    row["on"] = kinds.index(kind)
            if s["record"] == "morph":         # the plain rules stand on the stats each pass records
                row["asked"].update(self.index[g] for g, stats in self.asks.items() if s["name"] in stats)
        return [dict(r, goals=sorted(r["goals"]), asked=sorted(r["asked"])) for r in merged.values()]

    def view(self, rels):
        learned = sorted({g.get("learns") for g in self.goals if g.get("learns") in rels})
        return {"id": "readers", "title": "The readers", "rels": [rels.index(r) for r in learned], "kinds": None, "gain": 0.7}

    def stops(self):
        """the tour's two stops: all the readers, then one that is trying"""
        states = [self.state(g) for g in self.goals]
        trained = sum(1 for g in self.goals if g.get("scores"))
        out = [{"title": "The readers", "text": "Each large node outside the rings is one thing the harness tries to learn: %d goals, %d of them trained. Its rays go to what it reads. Solid has reached its goal, hollow has not, and a core in a shell is part way there or in doubt." % (len(self.goals), trained),
                "node": None, "path": [], "view": "readers"}]
        trying = next((n for n, s in enumerate(states) if s in ("trying", "partly") and self.goals[n].get("scores")), None)
        if trying is not None:
            g = self.goals[trying]
            out.append({"title": "Trying is not reaching", "text": "Goal: %s. It is trying, and it has not got there. The score below is against the plain rule it has to beat." % g["trained_to"].lower(),
                        "node": None, "path": [], "goal": trying})
        return out

    def data(self, kinds, rels):
        readers = []
        for n, g in enumerate(self.goals):
            reads = self.reads(g)
            model = g.get("model")
            readers.append({"title": g["trained_to"], "state": self.state(g), "pos": self.place(n, g), "reads": [kinds.index(k) for k in sorted(reads) if k in kinds],
                            "predicts": rels.index(g["learns"]) if g.get("learns") in rels else None, "learns": g.get("learns") or "", "uses": self.index.get(g.get("uses")),
                            "waits": g.get("waits_for"), "sees": READS_IN_WORDS.get(model, ", ".join(sorted(reads))), "measure": MEASURE_OF_MODEL.get(model, ""),
                            "scores": self.scores(g), "asks": self.asks.get(g["id"], []), "doubts": [d["why"] for d in g.get("doubts") or []], "note": g.get("note") or ""})
        notes = ["%s (%s)" % ((p["text"][0].upper() + p["text"][1:]).rstrip("."), p["rests_on"]) + "." for p in self.book.get("patterns") or []]
        offers = ", ".join("%s (%d links)" % (c["relation"], c["links"]) for c in (self.book.get("candidates") or [])[:5])
        if offers:
            notes.append("The graph offers these next, relations nothing predicts yet: %s." % offers)
        return {"readers": readers, "statRows": self.rows(kinds), "readerNotes": notes}


def page(store, out, title="The requirements graph", hide=None, anonymous=False, hero=None, goals=None):
    g = GraphLayout(store, hide, anonymous)
    readers = Readers(json.load(io.open(goals, encoding="utf-8")), g) if goals else None
    d = data(g, title, hero is not None, readers)
    text = hero.document() if hero else TEMPLATE
    if readers:
        for mark, add in (('<div class="hud" id="tip"></div>', PANELS), ('<span id="hint">', TOGGLES), ('<div id="ctl">', '<div id="cs"></div>')):
            assert text.count(mark) == 1, mark
            text = text.replace(mark, add + mark)
        text = text.replace("</head>", READER_STYLE + "</head>", 1) if "</head>" in text else READER_STYLE + text
    text = text.replace("/*READERS*/", READER_SCRIPT if readers else "").replace("/*DATA*/", json.dumps(d, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/"))
    if anonymous:
        names = g.names_on(text)
        if names:
            raise ValueError("anonymous page refused, it still holds %d name(s) from the store: %s" % (len(names), ", ".join(names[:12])))
    io.open(out, "w", encoding="utf-8", newline="\n").write(text)
    return d


def selftest():
    t = tempfile.mkdtemp(prefix="flythrough_selftest_")
    store = os.path.join(t, "g.sqlite")
    db = sqlite3.connect(store)
    db.executescript("create table prop(id integer primary key,label text,aliases text); create table entity(id text primary key,kind text,label text,sub text);"
                     "create table edge(s text,p integer,o text,fam integer);")
    rels = ["maps to", "is at", "mutated as", "killed by", "cites", "stated by", "binds requirement"]
    db.executemany("insert into prop values(?,?,'')", list(enumerate(rels)))
    nodes = [("source:SECRETNAME-1", "source", ""), ("req:R-1", "requirement", "the rule, by SECRETNAME"), ("step:U-1.1", "step", ""), ("chain step:C#1", "chain step", ""),
             ("code:A.B", "code", ""), ("cell:S#b0", "cell", ""), ("mut:A.B: kept", "mutant", ""), ("mut:A.B: past", "mutant", ""), ("test:T1", "test", "")]
    db.executemany("insert into entity values(?,?,?,'')", nodes)
    edges = [("cell:S#b0", 0, "req:R-1"), ("cell:S#b0", 1, "code:A.B"), ("cell:S#b0", 2, "mut:A.B: kept"), ("cell:S#b0", 2, "mut:A.B: past"), ("mut:A.B: kept", 1, "code:A.B"), ("mut:A.B: past", 1, "code:A.B"),
             ("mut:A.B: kept", 3, "test:T1"), ("chain step:C#1", 4, "req:R-1"), ("chain step:C#1", 4, "code:A.B"), ("req:R-1", 5, "source:SECRETNAME-1"), ("step:U-1.1", 6, "req:R-1")]
    db.executemany("insert into edge values(?,?,?,1)", edges)
    db.commit()
    db.close()
    out = os.path.join(t, "p.html")
    d = page(store, out, "T", hide="SECRETNAME")
    text = io.open(out, encoding="utf-8").read()
    good = d["nodes"] == 9 and d["edgeCount"] == 11 and d["survivors"] == 1 and len(d["pos"]) == 27 and "SECRETNAME" not in text
    print("selftest: PAGE every node and edge of the store is on the page, the break no test caught is flagged, a hidden name reaches no label %s" % ("PASS" if good else "FAIL"))
    ok = good
    titles = [s["title"] for s in d["stops"]]
    want = ["The whole graph", "Where it comes from", "The requirement", "The step that needs it", "Claude's reasoning", "The code", "One decision", "Broken on purpose", "Caught", "And one that got past", "The whole graph"]
    good = titles == want
    print("selftest: TOUR the thread runs source, requirement, step, reasoning, code, branch, break, test, and on to a break that got past %s" % ("PASS" if good else "FAIL"))
    ok = ok and good
    by = {v["id"]: v for v in d["views"]}
    good = [d["rels"][r] for r in by["maps"]["rels"]] == ["is at", "maps to"] and by["all"]["rels"] is None and d["kinds"][by["maps"]["kinds"][0]] in ("cell", "code", "requirement")
    print("selftest: VIEWS a view names the relations the store really holds, and everything is everything %s" % ("PASS" if good else "FAIL"))
    ok = ok and good
    # ANONYMOUS: a store naming things everywhere it can, a relation included. The named page first: a check that cannot fail proves nothing
    db = sqlite3.connect(store)
    db.execute("insert into prop values(7,'SECRETNAME relation','')")
    db.execute("insert into edge values('test:T1',7,'code:A.B',1)")
    db.execute("update entity set sub='SECRETNAME sub' where id='code:A.B'")
    db.commit()
    db.close()
    names = ["SECRETNAME", "R-1", "U-1.1", "A.B", "C#1", "S#b0", "the rule"]
    page(store, out, "T")
    named = io.open(out, encoding="utf-8").read()
    d = page(store, out, "T", anonymous=True)
    text = io.open(out, encoding="utf-8").read()
    broken = [s["text"] for s in d["stops"] if not re.match(r'^[A-Z0-9"].*\.$', s["text"]) or re.search(r"  |: \.| ,|None", s["text"])]
    good = all(n in named for n in names) and not any(n in text for n in names) and d["nodes"] == 9 and d["edgeCount"] == 12 and set(d["labels"].values()) == {""} \
        and [s["title"] for s in d["stops"]] == want and not broken and any(r.startswith("relation ") for r in d["rels"])
    print("selftest: ANONYMOUS no name from the store on the page, in a label, the tour or a relation, every node kept, the tour still whole sentences %s" % ("PASS" if good else "FAIL"))
    ok = ok and good
    # --hide on every name has the same holes to fall into
    d = page(store, out, "T", hide=".")
    broken = [s["text"] for s in d["stops"] if not re.match(r'^[A-Z0-9"].*\.$', s["text"]) or re.search(r"  |: \.| ,|None", s["text"])]
    good = not broken
    print("selftest: HIDE a hidden name leaves a sentence, never a hole in one %s" % ("PASS" if good else "FAIL"))
    ok = ok and good
    # the refusal: text from outside the store is the one way back in, so a name in the title or the host's page stops the build and leaves no file
    refused = 0
    for label, kwargs in (("title", {"title": "about U-1.1"}), ("below", {"title": "T", "hero": Hero("H", below="<p>see SECRETNAME-1</p>")})):
        spare = os.path.join(t, "refused_%s.html" % label)
        try:
            page(store, spare, anonymous=True, **kwargs)
        except ValueError as refusal:
            refused += 1 if not os.path.exists(spare) and "name(s) from the store" in str(refusal) else 0
    good = refused == 2
    print("selftest: REFUSED an anonymous page still holding a name from the store is not written %s" % ("PASS" if good else "FAIL"))
    ok = ok and good
    # HERO: a whole document, the host's page after the stage and before the script, its style in the head, the plain fragment left as it was
    d = page(store, out, "T", anonymous=True, hero=Hero("H & L", tagline="the <tag>", below="<style>.x{}</style><main id=\"m\">below</main>", cue=("Read on", "#m")))
    text = io.open(out, encoding="utf-8").read()
    head, _, body = text.partition("</head>")
    good = text.startswith("<!doctype html>") and '<meta charset="utf-8">' in head and 'name="viewport"' in head and text.count("<body>") == 1 and text.rstrip().endswith("</html>") \
        and "<title>H &amp; L</title>" in head and "<h1 id=\"t\">H &amp; L</h1>" in body and "the &lt;tag&gt;" in body and ".x{}" in head and ".x{}" not in body \
        and body.index("</section>") < body.index("<main") < body.index("three.min.js") and 'href="#m"' in body and d["hero"] is True \
        and "html,body { height:auto; overflow:visible }" in head and head.index("overflow:hidden") < head.index("overflow:visible") and "/*" not in re.sub(r"<script.*", "", body, flags=re.S)
    print("selftest: HERO a whole document, the host's page between the stage and the script, its style in the head %s" % ("PASS" if good else "FAIL"))
    ok = ok and good
    d = page(store, out, "T")
    text = io.open(out, encoding="utf-8").read()
    good = text.startswith("<title>") and "<!doctype" not in text and d["hero"] is False and 'id="stage"' not in text and 'id="cue"' not in text and "/*HEAD*/" not in text and "/*HINT*/" not in text
    print("selftest: FRAGMENT with no hero the page is still the fragment a host wraps %s" % ("PASS" if good else "FAIL"))
    ok = ok and good
    # GOALS: every state a goal can be in, a stat with goals, one with none, one sitting on a kind, and project keys that must never reach the page
    def scored(*won):
        return {"KEY%d" % n: {"learned": 0.5, "plain": 0.4, "rule": "kills", "wins": 3 if w else 0, "seeds": 3, "achieved": w, "flips": False} for n, w in enumerate(won)}
    book = {"projects": {"KEY0": "Project zero", "KEY1": "Project one"}, "rules": {"kills": "a plain count"}, "records": {"morph": "Each pass"},
            "patterns": [{"text": "a pattern", "rests_on": "an anecdote"}], "candidates": [{"relation": "cites", "links": 3}], "asks": {"goals_ask_stats": [{"goal": "g", "stat": "mutants"}]},
            "goals": [{"id": "g", "trained_to": "Reads the graph", "model": "graph network", "learns": "maps to", "state": "trained", "scores": scored(False, False)},
                      {"id": "b", "trained_to": "Half way", "model": "graph network", "learns": "killed by", "state": "trained", "scores": scored(True, False)},
                      {"id": "s", "trained_to": "Reads breaks", "model": "small model on the break's features", "learns": "earlier passes", "state": "trained", "scores": scored(True, True)},
                      {"id": "h", "trained_to": "Borrows", "learns": "the same model", "state": "uses another", "uses": "s"},
                      {"id": "p", "trained_to": "Not yet", "learns": "history", "state": "planned", "waits_for": "20 outcomes"},
                      {"id": "d", "trained_to": "In doubt", "model": "small model on the break's features", "learns": "earlier passes", "state": "trained", "doubts": [{"why": "a stated doubt"}], "note": "a stated note",
                       "scores": dict(scored(True, True), KEY1=dict(scored(True)["KEY0"], doubted=True))}],
            "stats": [{"name": "mutants", "record": "morph", "verdict": "DECIDES", "goals": ["s"], "read": True}, {"name": "alone", "record": "morph", "verdict": None, "goals": [], "read": False},
                      {"name": "same", "record": "morph", "verdict": "WATCHED", "goals": ["g"], "read": False}, {"name": "same", "record": "morph", "verdict": "WATCHED", "goals": ["b"], "read": False}]}
    goals_file = os.path.join(t, "goals.json")
    json.dump(book, io.open(goals_file, "w", encoding="utf-8"))
    d = page(store, out, "T", anonymous=True, goals=goals_file)
    text = io.open(out, encoding="utf-8").read()
    r = d["readers"]
    left = {d["kinds"][k] for k in r[0]["reads"]}
    stat = {x["name"]: x for x in d["statRows"]}
    good = [x["state"] for x in r] == ["trying", "partly", "reached", "borrowed", "planned", "doubted"] and r[5]["doubts"] == ["a stated doubt"] and r[5]["note"] == "a stated note" and any(x["doubted"] for x in r[5]["scores"]) and not any(x["doubted"] for x in r[2]["scores"]) and not left & GNN_LEFT_OUT and "requirement" in left \
        and [d["kinds"][k] for k in r[2]["reads"]] == ["mutant"] and r[4]["reads"] == [] and r[3]["uses"] == 2 and d["rels"][r[0]["predicts"]] == "maps to" and r[2]["predicts"] is None \
        and stat["alone"]["goals"] == [] and d["kinds"][stat["mutants"]["on"]] == "mutant" and stat["alone"]["on"] is None and stat["same"]["goals"] == [0, 1] and len(d["statRows"]) == 3 \
        and stat["mutants"]["asked"] == [0] and "KEY0" not in text and "Project zero" in text and "SECRETNAME" not in text and 'id="goals"' in text and 'id="stats"' in text and "/*READERS*/" not in text
    print("selftest: GOALS a reader per goal, reads from its model, trying, reaching and doubting told apart, a stat with no goal and one on a kind, no project key on the page %s" % ("PASS" if good else "FAIL"))
    ok = ok and good
    views = [v["id"] for v in d["views"]]
    titles = [s["title"] for s in d["stops"]]
    good = "readers" in views and d["views"][views.index("readers")]["kinds"] is None and titles[-3:] == ["The readers", "Trying is not reaching", "The whole graph"] and d["stops"][-2]["goal"] == 0 and d["stops"][-3]["view"] == "readers"
    print("selftest: READERS TOUR a Show view of the readers, and two stops before the close, the second on a goal that is trying %s" % ("PASS" if good else "FAIL"))
    ok = ok and good
    refused = 0
    for change in ({"model": "a model nobody told us about"}, {"scores": {"NOKEY": book["goals"][0]["scores"]["KEY0"]}}):
        bad = json.loads(json.dumps(book))
        bad["goals"][0].update(change)
        json.dump(bad, io.open(goals_file, "w", encoding="utf-8"))
        try:
            page(store, out, "T", goals=goals_file)
        except ValueError:
            refused += 1
    bad = json.loads(json.dumps(book))
    bad["stats"][0]["goals"] = ["no such goal"]
    json.dump(bad, io.open(goals_file, "w", encoding="utf-8"))
    try:
        page(store, out, "T", goals=goals_file)
    except ValueError:
        refused += 1
    good = refused == 3
    print("selftest: READERS REFUSE a model it cannot draw, a project with no name and a stat naming no goal stop the build %s" % ("PASS" if good else "FAIL"))
    ok = ok and good
    bad = json.loads(json.dumps(book))
    bad["goals"][4].update({"reads": ["test"], "model": "something new"})
    json.dump(bad, io.open(goals_file, "w", encoding="utf-8"))
    d = page(store, out, "T", goals=goals_file)
    good = [d["kinds"][k] for k in d["readers"][4]["reads"]] == ["test"]
    print("selftest: READERS OWN WORD a goal that says what it reads is taken at its word %s" % ("PASS" if good else "FAIL"))
    ok = ok and good
    d = page(store, out, "T")
    text = io.open(out, encoding="utf-8").read()
    good = "readers" not in d and 'id="goals"' not in text and "/*READERS*/" not in text and "Trying is not reaching" not in text and [v["id"] for v in d["views"]][0] == "all"
    print("selftest: NO GOALS with no goals file the page is as it was, none of this on it %s" % ("PASS" if good else "FAIL"))
    ok = ok and good
    # the script asks for its elements by id, and one it cannot find kills the whole page at load: a browser shows that and a build does not
    missing = []
    json.dump(book, io.open(goals_file, "w", encoding="utf-8"))
    for kwargs in ({}, {"goals": goals_file}, {"hero": Hero("H", below="<main id=\"m\"></main>")}, {"hero": Hero("H"), "goals": goals_file, "anonymous": True}):
        page(store, out, "T", **kwargs)
        text = io.open(out, encoding="utf-8").read()
        have = set(re.findall(r"""\bid=["']([^"']+)["']""", text)) | ({"stage"} if "hero" not in kwargs else set())    # the plain page has no stage, and asks for it knowing that
        missing += [x for x in re.findall(r"\$\('(\w+)'\)", text) if x not in have]
    good = not missing
    print("selftest: ELEMENTS every element the script asks for is in the markup, in all four shapes of page %s" % ("PASS" if good else "FAIL: %s" % sorted(set(missing))))
    ok = ok and good
    db = sqlite3.connect(store)
    db.execute("delete from edge where p=4")
    db.commit()
    db.close()
    d = page(store, out, "T", hide="SECRETNAME")
    good = [s["title"] for s in d["stops"]] == ["The whole graph"]
    print("selftest: NO THREAD a graph with no reasoning gets the overview alone, never an invented thread %s" % ("PASS" if good else "FAIL"))
    ok = ok and good
    print("selftest: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv):
    if "--selftest" in argv:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("store")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="The requirements graph")
    ap.add_argument("--hide")
    ap.add_argument("--anonymous", action="store_true", help="drop every name the store holds; the tour speaks in roles; a page still holding a name is refused")
    ap.add_argument("--hero", action="store_true", help="a whole document: the graph as its first screen, the page scrolling on below")
    ap.add_argument("--headline", help="--hero: the headline over the graph")
    ap.add_argument("--tagline", default="", help="--hero: the line under the headline")
    ap.add_argument("--below", help="--hero: a file of HTML for what follows the graph (its <style> blocks move to the head)")
    ap.add_argument("--cue", nargs=2, metavar=("LABEL", "HREF"), help="--hero: the link in the tour card that takes the reader down the page")
    ap.add_argument("--description", help="--hero: the page description, the tagline if not given")
    ap.add_argument("--goals", help="the file gnn_goals.py --write makes: a reader per goal, the Goals and Stats panels, and a Show view of the readers")
    a = ap.parse_args(argv)
    if a.hero and not a.headline:
        ap.error("--hero needs --headline")
    hero = Hero(a.headline, a.tagline, io.open(a.below, encoding="utf-8").read() if a.below else "", a.cue, a.description) if a.hero else None
    d = page(a.store, a.out, a.title, a.hide, a.anonymous, hero, a.goals)
    print("graph_flythrough: %d nodes, %d edges, %d breaks no test caught, %d tour stops -> %s (%d KB)" % (d["nodes"], d["edgeCount"], d["survivors"], len(d["stops"]), a.out, os.path.getsize(a.out) // 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
