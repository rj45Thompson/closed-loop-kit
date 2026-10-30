#!/usr/bin/env python3
"""graph_gnn.py - a graph neural network over the closed loop's knowledge graph that PROPOSES the links a model is missing, and is
measured against plain baselines before anything it says is shown.

WHY (2026-10-29): a graph neural network could help link code to requirements. The loop's criteria are exact queries: they
say a branch is claimed by no requirement, they cannot say WHICH requirement it belongs to. Every claim so far was typed by hand
(meta.loop.branches.claims). A GNN learns from the links the graph already holds - which requirement owns the branches around a
branch, which requirements have effects in its method, what its statement says - and ranks the requirements for a branch that has
none. It is a PROPOSER, never a verdict (iron law 1: a guess is not READ): its proposals are printed as questions, and a person or
the exact checks accept one. The rule, learned on an earlier project: it earns that place only by beating the hand baselines on held-out
links, so `eval` hides real links, ranks the requirements for each, and prints the GNN beside random, word overlap and "the same
method's requirements". If it does not beat them, it says so and `propose` refuses to run.

THE INPUT IS WHAT A NEW BRANCH WOULD HAVE. Two leaks are closed (each MEASURED on the shops graph, 10-29): a branch cell's text
carries "[claimed with its method: ...]", the very answer, so it is cut; and the mutant and test nodes join a branch to the
assertions of the requirement whose effect matched it, so measurement nodes (mutant, test, chain step) are left out.

    python -X utf8 graph_gnn.py eval <store.sqlite> [--folds 5] [--seed 0]
    python -X utf8 graph_gnn.py propose <store.sqlite> [--top 3]
    python -X utf8 graph_gnn.py --selftest
"""
import argparse
import collections
import io
import json
import math
import os
import random
import re
import sqlite3
import sys

import torch

MAPS = 122                     # cell -> requirement (the relation proposed)
LEFT_OUT = ("mutant", "test", "chain", "chain step")    # measurements and argument chains: they hold the answer, a new branch has none
DIM = 512
STOP = set("a an the is it its of to in on at and or not what when who how why does do did which where this that with be are was "
           "for from by as if then else so no one every each any can must here only than more".split())


def words(s):
    out = []
    for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s or ""):
        for part in [w] + re.split(r"_|(?<=[a-z0-9])(?=[A-Z])", w):
            p = part.lower()
            if len(p) > 2 and p not in STOP:
                out.append(p)
    return out


class GraphView(object):
    """The knowledge graph as a GNN reads it: the nodes a new branch could see (measurement nodes left out), their kinds, their
    words, the undirected edges between them, and the `maps to` links from branch cells to requirements - the ones to learn and
    to hide. Its breadcrumb: usecase_kg.py KnowledgeGraph.build writes every node and edge it reads."""

    def __init__(self, db_path, space="S-BRANCH", saved_view=True):
        # the relations the GNN does NOT hear, chosen by GraphTuner and kept beside the graph (gnn_view.json) with the numbers
        # that justified them; a tune itself starts from the full graph (saved_view=False)
        self.view_path = os.path.join(os.path.dirname(os.path.abspath(db_path)), "gnn_view.json")
        saved = json.load(io.open(self.view_path, encoding="utf-8")) if saved_view and os.path.exists(self.view_path) else {}
        self.drop, self.add = set(saved.get("drop") or []), set(saved.get("add") or [])
        db = sqlite3.connect(db_path)
        ents = [(i, k, l or "", s or "") for i, k, l, s in db.execute("SELECT id, kind, label, sub FROM entity ORDER BY id") if k not in LEFT_OUT]
        self.ids = [e[0] for e in ents]
        self.index = {e: n for n, e in enumerate(self.ids)}
        self.kind = {e[0]: e[1] for e in ents}
        text = collections.defaultdict(list)
        for e, k, l, s in ents:
            text[e].append(l)
        for s, v in db.execute("SELECT s, v FROM literal WHERE p = 133"):
            if s in self.index:
                text[s].append(re.sub(r"\s*\[claimed with its method:.*$", "", v))       # the leak: the claim's own reason
        for e in self.ids:
            if e.startswith("code:"):
                text[e].append(e[5:].replace(".", " "))
        self.text = {e: " ".join(text[e]) for e in self.ids}
        self.edges = [(s, o, p) for s, p, o in db.execute("SELECT s, p, o FROM edge") if s in self.index and o in self.index]
        self.cells = sorted(e for e in self.ids if e.startswith("cell:%s:" % space))
        self.reqs = sorted(e for e in self.ids if self.kind[e] == "requirement")
        self.links = sorted((s, o) for s, o, p in self.edges if p == MAPS and s in set(self.cells))
        # the code place each branch sits at (the cell's `is at` edge, else its key's Type.Method)
        self.place = {}
        for s, o, p in self.edges:
            if p == 152 and s in self.index and self.kind.get(s) == "cell":
                self.place[s] = o
        for c in self.cells:
            self.place.setdefault(c, "code:" + c.split(":", 2)[2].split("#")[0])
        # the requirements whose EFFECTS sit at each code place (requirement -asserts-> assertion -is at-> code): what a new branch
        # in that method can already see before anyone claims it (REVIEW_2 #8 measured this path alone at MRR 0.62)
        owner_of = {o: s for s, o, p in self.edges if p == 134}
        self.effects_at = collections.defaultdict(collections.Counter)
        for s, o, p in self.edges:
            if p == 152 and s in owner_of:
                self.effects_at[o][owner_of[s]] += 1
        db.close()

    def features(self):
        """kind one-hot + hashed words of each node's label and text"""
        kinds = sorted(set(self.kind.values()))
        x = torch.zeros(len(self.ids), len(kinds) + DIM)
        for n, e in enumerate(self.ids):
            x[n, kinds.index(self.kind[e])] = 1.0
            ws = words(self.text[e])
            for w in ws:
                x[n, len(kinds) + (hash_word(w) % DIM)] += 1.0 / math.sqrt(len(ws))
        return x

    def adjacency(self, hidden=()):
        """one row-normalised adjacency per relation GROUP, the hidden links taken out (both directions): what a node hears from
        its claims, from the code, and from the model's structure is learned apart (an R-GCN, Schlichtkrull et al. 2018, in small)"""
        hide = set(hidden)
        n = len(self.ids)
        out = []
        edges = self.edges + [e for name in sorted(self.add) for e in self.derived(name)]
        for group in RELATION_GROUPS:
            a = torch.zeros(n, n)
            for s, o, p in edges:
                if (p == MAPS and (s, o) in hide) or group_of(p) != group or p in self.drop:
                    continue
                i, j = self.index[s], self.index[o]
                a[i, j] = a[j, i] = 1.0
            out.append(a / a.sum(1, keepdim=True).clamp(min=1.0))
        return out


    def derived(self, name):
        """the links a tune may ADD (GraphTuner: the graph shaped for the GNN), each read off the graph, never invented:
        `same class` joins the methods of one type (code:Type.Method); `carried together` joins the requirements one step carries"""
        if name == "same class":
            by = collections.defaultdict(list)
            for e in self.ids:
                if self.kind[e] == "code" and "." in e[5:]:
                    by[e[5:].rsplit(".", 1)[0]].append(e)
            return [(a, b, SAME_CLASS) for ms in by.values() for i, a in enumerate(ms) for b in ms[i + 1:]]
        if name == "carried together":
            by = collections.defaultdict(set)
            for s, o, p in self.edges:
                if p in (107, 108):
                    by[s].add(o)
            pairs = {tuple(sorted((a, b))) for rs in by.values() for a in rs for b in rs if a < b}
            return [(a, b, CARRIED_TOGETHER) for a, b in sorted(pairs)]
        raise KeyError("no derived link called %r (GraphView.derived knows: same class, carried together)" % name)


RELATION_GROUPS = ("claims", "code", "model")
SAME_CLASS, CARRIED_TOGETHER = 900, 901      # derived links (GraphView.derived): never written to the graph, only heard by the GNN
DERIVED = ("same class", "carried together")


def group_of(p):
    """a relation's group: a cell's claim (122), the code (asserts, is at, sets, branches on, names, binds), or the model's structure"""
    return "claims" if p == MAPS else "code" if p in (134, 135, 137, 138, 139, 140, 141, 152, SAME_CLASS) else "model"


def hash_word(w):
    h = 2166136261
    for ch in w.encode("utf-8"):
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h


class SageEncoder(torch.nn.Module):
    """two GraphSAGE-mean layers over TYPED relations: a node is its own words plus, per relation group, the mean of its
    neighbours' - so a branch hears its method through the code relations, the method hears the requirements whose effects sit in
    it, and the branch's siblings' owners come through the claims, each with weights of its own"""

    def __init__(self, d_in, d_hid=128, d_out=64, groups=len(RELATION_GROUPS)):
        super().__init__()
        self.l1 = torch.nn.Linear((1 + groups) * d_in, d_hid)
        self.l2 = torch.nn.Linear((1 + groups) * d_hid, d_out)

    def forward(self, x, adj):
        h = torch.relu(self.l1(torch.cat([x] + [a @ x for a in adj], 1)))
        return self.l2(torch.cat([h] + [a @ h for a in adj], 1))


class LinkProposer(object):
    """Trains the encoder on the graph's visible `maps to` links (each against requirements it does not map to) and ranks every
    requirement for a cell by the dot product of their embeddings."""

    def __init__(self, view, seed=0, epochs=300):
        self.v, self.seed, self.epochs = view, seed, epochs

    def fit(self, train_links, hidden):
        torch.manual_seed(self.seed)
        rnd = random.Random(self.seed)
        v = self.v
        self.x, self.a = v.features(), v.adjacency(hidden)
        self.enc = SageEncoder(self.x.shape[1])
        self.base = Baselines(v, self.seed)
        self.visible = list(train_links)
        # the PAIR SCORER: the embeddings' dot product plus three exact signals - the words shared, the effects in the cell's
        # method, the claims of its siblings - with weights learned; the GNN adds what the signals miss, never hides them
        # the signals start trusted (weight 1) and the embedding term bounded (cosine times a learned scale): an untrained or
        # overfitted embedding can then only ADD to what the signals say, not drown it (MEASURED 10-29 on the plant: 0.78 -> see selftest)
        self.w = torch.nn.Parameter(torch.ones(3))
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        opt = torch.optim.Adam(list(self.enc.parameters()) + [self.w, self.scale], lr=0.01, weight_decay=5e-4)
        owned = collections.defaultdict(set)
        for c, r in train_links:
            owned[c].add(r)
        if not train_links:
            raise RuntimeError("no link to learn from")
        reqs = [v.index[r] for r in v.reqs]
        # MESSAGE vs SUPERVISION links (MEASURED 10-29: training on links the cell could still see through its own claim edge taught
        # the embeddings "my claim neighbour is my owner" - MRR 0.58 on the plant, below the plain baselines). Each epoch hides
        # half the training METHODS' claims from the message passing and scores the loss on those alone: the situation of new code.
        places = sorted({v.place.get(c) for c, _ in train_links})
        for _ in range(self.epochs):
            hide = set(rnd.sample(places, max(1, len(places) // 2)))
            sup = [(c, r) for c, r in train_links if v.place.get(c) in hide]
            self.visible = [(c, r) for c, r in train_links if v.place.get(c) not in hide]
            adj = v.adjacency(list(hidden) + sup)
            pos = [(v.index[c], v.index[r]) for c, r in sup]
            neg = []
            for c, r in sup:
                choices = [q for q in reqs if v.ids[q] not in owned[c]]
                neg.append((v.index[c], rnd.choice(choices) if choices else v.index[r]))
            h = self.enc(self.x, adj)
            loss = torch.nn.functional.softplus(-self._score(h, pos)).mean() + torch.nn.functional.softplus(self._score(h, neg)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.visible = list(train_links)
        with torch.no_grad():
            self.h = self.enc(self.x, self.a)
        return self

    def signals(self, c, r):
        """the three exact signals of one (cell, requirement) pair: words shared, effects in the cell's method, siblings' claims"""
        v = self.v
        place = v.place.get(c)
        sib = sum(1 for cc, rr in self.visible if rr == r and cc != c and v.place.get(cc) == place)
        return [self.base.cos(c, r), float(v.effects_at.get(place, {}).get(r, 0)), float(sib)]

    def _score(self, h, pairs):
        idx_c, idx_r = [c for c, _ in pairs], [r for _, r in pairs]
        sig = torch.tensor([self.signals(self.v.ids[c], self.v.ids[r]) for c, r in pairs])
        hn = torch.nn.functional.normalize(h, dim=1)
        return self.scale * (hn[idx_c] * hn[idx_r]).sum(1) + sig @ self.w

    def scores(self, cell):
        v = self.v
        ci = v.index[cell]
        with torch.no_grad():
            return self._score(self.h, [(ci, v.index[r]) for r in v.reqs])

    def rank(self, cell):
        s = self.scores(cell)
        return [r for _, r in sorted(zip(s.tolist(), self.v.reqs), key=lambda t: (-t[0], t[1]))]

    def confidence(self, cell):
        """the softmax weight of the top requirement among all: a proposal below a threshold is withheld (abstained)"""
        return float(torch.softmax(self.scores(cell), 0).max())


class Baselines(object):
    """What the GNN must beat: RANDOM; WORDS - the requirements whose text shares most words with the branch's statement and its
    method's name (tf-idf cosine); METHOD - the requirements that own the other branches of the same method, by how many."""

    def __init__(self, view, seed=0):
        self.v, self.rnd = view, random.Random(seed)
        docs = {e: collections.Counter(words(view.text[e])) for e in view.cells + view.reqs}
        df = collections.Counter(w for d in docs.values() for w in d)
        n = len(docs)
        self.vec = {e: {w: c * math.log((1 + n) / (1 + df[w])) for w, c in d.items()} for e, d in docs.items()}

    def random(self, cell, visible):
        rs = list(self.v.reqs)
        self.rnd.shuffle(rs)
        return rs

    def cos(self, c, r):
        a, b = self.vec[c], self.vec[r]
        na = math.sqrt(sum(x * x for x in a.values())) or 1.0
        nb = math.sqrt(sum(x * x for x in b.values())) or 1.0
        return sum(a[w] * b.get(w, 0.0) for w in a) / (na * nb)

    def words(self, cell, visible):
        return sorted(self.v.reqs, key=lambda r: (-self.cos(cell, r), r))

    def effects(self, cell, visible):
        """the requirements whose effects sit in the cell's method, by how many, then by words - no learning (REVIEW_2 #8)"""
        at = self.v.effects_at.get(self.v.place.get(cell), {})
        return sorted(self.v.reqs, key=lambda r: (-at.get(r, 0), -self.cos(cell, r), r))

    def combined(self, cell, visible):
        """effects and words and siblings, summed by hand - the strongest no-learning ranker this graph allows"""
        place = self.v.place.get(cell)
        at = self.v.effects_at.get(place, {})
        sib = collections.Counter(r for c, r in visible if c != cell and self.v.place.get(c) == place)
        return sorted(self.v.reqs, key=lambda r: (-(at.get(r, 0) + sib[r] + self.cos(cell, r)), r))

    def method(self, cell, visible):
        place = self.v.place.get(cell)
        votes = collections.Counter(r for c, r in visible if c != cell and self.v.place.get(c) == place)
        return sorted(self.v.reqs, key=lambda r: (-votes[r], r))


def metrics(ranks):
    """Hits@1, Hits@3 and MRR over (ranked requirements, the true ones) - the trace-link literature's measures"""
    h1 = h3 = rr = 0.0
    for ranked, truth in ranks:
        first = min(ranked.index(t) for t in truth) + 1
        h1 += first <= 1
        h3 += first <= 3
        rr += 1.0 / first
    n = max(1, len(ranks))
    return {"hits@1": h1 / n, "hits@3": h3 / n, "mrr": rr / n, "n": len(ranks)}


def evaluate(view, folds=5, seed=0, epochs=300, group="cell"):
    """k-fold over the branch cells: each fold's cells lose every `maps to` link (from the graph AND the training), the GNN and the
    baselines rank the requirements for each, and the true owners' best rank is scored. Folds by CELL, so no cell is half-seen;
    or by METHOD (group="method"): every branch of a method hidden at once - NEW code, whose siblings own nothing yet, the case a
    proposer exists for (a by-cell fold lets the same-method baseline read the answer off the siblings a method claim made)."""
    cells = sorted({c for c, _ in view.links})
    units = sorted({view.place.get(c) for c in cells}) if group == "method" else cells
    rnd = random.Random(seed)
    rnd.shuffle(units)
    truth = collections.defaultdict(set)
    for c, r in view.links:
        truth[c].add(r)
    out = collections.defaultdict(list)
    calib = []
    for k in range(folds):
        pick = set(units[k::folds])
        test = {c for c in cells if (view.place.get(c) if group == "method" else c) in pick}
        hidden = [(c, r) for c, r in view.links if c in test]
        visible = [(c, r) for c, r in view.links if c not in test]
        gnn = LinkProposer(view, seed + k, epochs).fit(visible, hidden)
        base = Baselines(view, seed + k)
        for c in sorted(test):
            ranked = gnn.rank(c)
            out["gnn"].append((ranked, truth[c]))
            calib.append((gnn.confidence(c), ranked[0] in truth[c]))
            for name in BASELINES:
                out[name].append((getattr(base, name)(c, visible), truth[c]))
    res = {name: metrics(r) for name, r in out.items()}
    # CALIBRATION: how often the top proposal is right when the GNN is at least this sure, and how many cells reach it
    res["calibration"] = {t: {"precision": (sum(1 for p, ok in calib if p >= t and ok) / float(max(1, sum(1 for p, _ in calib if p >= t)))),
                              "coverage": sum(1 for p, _ in calib if p >= t) / float(max(1, len(calib)))} for t in (0.3, 0.5, 0.7, 0.9)}
    return res


BASELINES = ("random", "words", "method", "effects", "combined")
# the bar a proposal must clear to be shown at all: on the shops (10-29, folded by method) the GNN was right 94% of the time when at
# least 0.7 sure, on 51% of new code - below the bar it abstains rather than guess (iron law 1)
SURE = 0.7


def report(res, group="cell"):
    lines = ["  %-8s %7s %7s %7s  (n=%d held-out branch cells, folded by %s)" % ("ranker", "hits@1", "hits@3", "MRR", res["gnn"]["n"], group)]
    for name in BASELINES + ("gnn",):
        m = res[name]
        lines.append("  %-8s %7.2f %7.2f %7.2f" % (name, m["hits@1"], m["hits@3"], m["mrr"]))
    if res.get("calibration"):
        lines.append("  calibration: " + "; ".join("sure >= %.1f: right %.0f%% of %.0f%% of cells" % (t, 100 * c["precision"], 100 * c["coverage"])
                                                 for t, c in sorted(res["calibration"].items())))
    best = max(BASELINES, key=lambda n: res[n]["mrr"])
    beats = res["gnn"]["mrr"] > res[best]["mrr"]
    lines.append("  the GNN %s the best baseline (%s, MRR %.2f vs %.2f)%s" % (
        "BEATS" if beats else "does NOT beat", best, res["gnn"]["mrr"], res[best]["mrr"],
        "" if beats else " - its proposals are not shown (the rule: it earns the place by beating the hand baseline)"))
    return "\n".join(lines), beats


class GraphTuner(object):
    """The graph shaped for the GNN: the node graph tuned to work better with it, chosen
    every so often by the stats (loop_stats.NextImprovement). Each VARIANT leaves one relation group, or one of the
    busiest relations, out of what the GNN hears; the pair scorer's exact signals and the `maps to` labels are never touched. Each
    is measured on held-out METHODS exactly as evaluate() measures. A variant is KEPT only when it beats the full graph by MARGIN
    on the tuning seeds AND again on a confirming seed it was not chosen on - a choice that fits only the seeds it was picked on is
    noise. The kept drop goes to gnn_view.json beside the graph (GraphView reads it); a tune that keeps nothing removes it. Every
    tune appends one row to gnn_tune.jsonl beside the model, which the morph log's report reads."""

    MARGIN = 0.02

    def __init__(self, db_path, folds=5, epochs=300, tune_seeds=(0, 1, 2), confirm_seed=3, busiest=4):
        self.db, self.folds, self.epochs = db_path, folds, epochs
        self.tune_seeds, self.confirm_seed, self.busiest = tuple(tune_seeds), confirm_seed, busiest
        self.view = GraphView(db_path, saved_view=False)
        self.log = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(db_path))), "gnn_tune.jsonl")

    def mrr(self, variant, seed):
        drop, add = variant
        self.view.drop, self.view.add = set(drop), set(add)
        return evaluate(self.view, self.folds, seed, self.epochs, "method")["gnn"]["mrr"]

    def variants(self):
        """[(name, (drop, add))]: each relation group left out, each of the busiest relations left out, each derived link added
        (tune 1 on the shops, 10-29, only left things out and kept nothing: removing can only find noise, adding can find signal)"""
        heard = collections.Counter(p for _, _, p in self.view.edges if p != MAPS)
        out = []
        for g in RELATION_GROUPS:
            ps = frozenset(p for p in heard if group_of(p) == g)
            if ps:
                out.append(("without the %s relations" % g, (ps, frozenset())))
        for p, n in heard.most_common(self.busiest):
            if frozenset([p]) not in {v[0] for _, v in out}:
                out.append(("without '%s' (%d edges)" % (relation_name(p), n), (frozenset([p]), frozenset())))
        for name in DERIVED:
            k = len(self.view.derived(name))
            if k:
                out.append(("with '%s' links (%d)" % (name, k), (frozenset(), frozenset([name]))))
        return out

    @staticmethod
    def keep(full_tune, var_tune, full_confirm, var_confirm, margin=MARGIN):
        mean = lambda xs: sum(xs) / float(len(xs))
        return mean(var_tune) - mean(full_tune) >= margin and var_confirm - full_confirm >= margin

    def run(self, pass_n=None, say=print):
        mean = lambda xs: sum(xs) / float(len(xs))
        full = [self.mrr((frozenset(), frozenset()), sd) for sd in self.tune_seeds]
        say("  the full graph: MRR %s" % ", ".join("%.3f" % x for x in full))
        scores = []
        for name, variant in self.variants():
            sc = [self.mrr(variant, sd) for sd in self.tune_seeds]
            scores.append((name, variant, sc))
            say("  %s: MRR %s (%+.3f)" % (name, ", ".join("%.3f" % x for x in sc), mean(sc) - mean(full)))
        row = {"time": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"), "pass": pass_n, "graph": os.path.basename(self.db),
               "folds": self.folds, "epochs": self.epochs, "seeds": list(self.tune_seeds), "full": full,
               "variants": {name: sc for name, _, sc in scores}}
        best = max(scores, key=lambda t: mean(t[2])) if scores else None
        kept = False
        if best and mean(best[2]) - mean(full) >= self.MARGIN:
            fc, vc = self.mrr((frozenset(), frozenset()), self.confirm_seed), self.mrr(best[1], self.confirm_seed)
            row["confirm"] = {"seed": self.confirm_seed, "full": fc, "variant": vc, "name": best[0]}
            kept = self.keep(full, best[2], fc, vc, self.MARGIN)
            say("  confirming %s on seed %d: %.3f against the full graph's %.3f" % (best[0], self.confirm_seed, vc, fc))
        row["kept"] = best[0] if kept else None
        if kept:
            row["summary"] = "KEPT %s: MRR %.3f -> %.3f on the tuning seeds, %.3f -> %.3f on the confirming seed" % (
                best[0], mean(full), mean(best[2]), row["confirm"]["full"], row["confirm"]["variant"])
            io.open(self.view.view_path, "w", encoding="utf-8").write(json.dumps(
                {"drop": sorted(best[1][0]), "add": sorted(best[1][1]), "why": row["summary"], "at": row["time"]}, indent=1) + "\n")
        else:
            row["summary"] = "kept nothing: the full graph's MRR %.3f%s" % (mean(full), (
                "; the best variant (%s, %.3f) %s" % (best[0], mean(best[2]), "did not hold on the confirming seed" if "confirm" in row
                                                      else "beat it by less than %.2f" % self.MARGIN)) if best else "")
            if os.path.exists(self.view.view_path):
                os.remove(self.view.view_path)
                row["summary"] += "; the earlier kept view removed"
        io.open(self.log, "a", encoding="utf-8").write(json.dumps(row, sort_keys=True) + "\n")
        say("graph_gnn: tune %s (-> %s)" % (row["summary"], self.log))
        return row


def relation_name(p):
    """the graph's own name for a relation (usecase_kg.PROPS)"""
    import usecase_kg
    return {pid: label for pid, label, _ in usecase_kg.PROPS}.get(p, "relation %d" % p)


def selftest():
    """a planted graph where the owner of a branch is learnable only from structure two hops away (the requirement whose effect
    sits in the branch's method) and the words carry nothing: the GNN must beat random and words; and the leak cut must hold"""
    import os
    import tempfile
    d = tempfile.mkdtemp(prefix="graph_gnn_selftest_")
    p = os.path.join(d, "g.sqlite")
    db = sqlite3.connect(p)
    db.executescript("CREATE TABLE entity(id TEXT PRIMARY KEY, kind TEXT, label TEXT, sub TEXT);"
                     "CREATE TABLE edge(s TEXT, p INTEGER, o TEXT, fam INTEGER);"
                     "CREATE TABLE literal(s TEXT, p INTEGER, v TEXT);")
    rnd = random.Random(1)
    reqs = ["req:R-%d" % i for i in range(6)]
    for r in reqs:
        db.execute("INSERT INTO entity VALUES (?, 'requirement', ?, 'authored')", (r, r[4:]))
        db.execute("INSERT INTO literal VALUES (?, 133, ?)", (r, "zz%d qq" % rnd.randrange(1000)))
    for m in range(12):
        place, owner = "code:T.M%d" % m, reqs[m % 6]
        db.execute("INSERT INTO entity VALUES (?, 'code', ?, NULL)", (place, place[5:]))
        a = "assert:%s#m%d" % (owner[4:], m)
        db.execute("INSERT INTO entity VALUES (?, 'assertion', 'x', 'sets')", (a,))
        db.execute("INSERT INTO edge VALUES (?, 134, ?, 16)", (owner, a))
        db.execute("INSERT INTO edge VALUES (?, 152, ?, 32)", (a, place))
        for b in range(3):
            c = "cell:S-BRANCH:T.M%d#b%d" % (m, b)
            db.execute("INSERT INTO entity VALUES (?, 'cell', ?, 'S-BRANCH')", (c, c[14:]))
            db.execute("INSERT INTO literal VALUES (?, 133, ?)", (c, "if (x%d)  [claimed with its method: %s]" % (rnd.randrange(1000), owner)))
            db.execute("INSERT INTO edge VALUES (?, 152, ?, 32)", (c, place))
            db.execute("INSERT INTO edge VALUES (?, 122, ?, 32)", (c, owner))
    db.commit()
    db.close()
    v = GraphView(p)
    leak = any("claimed with its method" in t for t in v.text.values())
    res = evaluate(v, folds=3, seed=0, epochs=200)
    ok = not leak and res["gnn"]["mrr"] > res["random"]["mrr"] + 0.2 and res["gnn"]["mrr"] > res["words"]["mrr"]
    print("selftest: the claim's own words are cut from the input: %s" % ("PASS" if not leak else "FAIL"))
    print(report(res)[0])
    print("selftest: the GNN learns an owner two hops away that the words do not carry: %s" % ("PASS" if ok else "FAIL"))
    # the tuner keeps a variant only when the confirming seed agrees; a saved view is heard by GraphView, and a tune ignores it
    keep_ok = GraphTuner.keep([0.70, 0.72], [0.76, 0.78], 0.71, 0.77) and not GraphTuner.keep([0.70, 0.72], [0.76, 0.78], 0.71, 0.70) \
        and not GraphTuner.keep([0.70, 0.72], [0.715, 0.725], 0.71, 0.75)
    io.open(os.path.join(d, "gnn_view.json"), "w", encoding="utf-8").write(json.dumps({"drop": [152]}))
    heard = GraphView(p)
    fresh = GraphView(p, saved_view=False)
    # LINKS are counted (non-zero entries): a row-normalised adjacency sums to the number of linked NODES, which adding a link
    # between two already-linked methods leaves unchanged (the first plant compared sums and could not see an add)
    links = lambda v: int((v.adjacency()[RELATION_GROUPS.index("code")] > 0).sum())
    view_ok = heard.drop == {152} and fresh.drop == set() and links(heard) < links(fresh)
    io.open(os.path.join(d, "gnn_view.json"), "w", encoding="utf-8").write(json.dumps({"add": ["same class"]}))
    more = GraphView(p)
    add_ok = len(more.derived("same class")) == 66 and links(more) == links(fresh) + 2 * 66
    good = keep_ok and view_ok and add_ok
    ok = ok and good
    print("selftest: TUNE a variant is kept only when the confirming seed agrees too; a kept drop or a kept ADD is heard, a tune "
          "starts from the full graph %s" % ("PASS" if good else "FAIL"))
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["eval", "propose", "tune"])
    ap.add_argument("db", nargs="?")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--group", choices=["cell", "method"], default="method",
                    help="hide links a cell at a time, or a whole method at a time (new code: the default, and the honest test)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.cmd or not a.db:
        ap.error("eval <store.sqlite> | propose <store.sqlite> | --selftest")
    if a.cmd == "tune":
        stats = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(a.db))), "stats.jsonl")
        rows = [json.loads(l) for l in io.open(stats, encoding="utf-8")] if os.path.exists(stats) else []
        t = GraphTuner(a.db, a.folds, a.epochs, tune_seeds=(a.seed, a.seed + 1, a.seed + 2), confirm_seed=a.seed + 3)
        print("graph_gnn: tuning the graph %s for the GNN: %d variant(s), folded by method, seeds %s, confirm on %d" % (
            a.db, len(t.variants()), list(t.tune_seeds), t.confirm_seed))
        t.run(rows[-1].get("n") if rows else None)
        return 0
    v = GraphView(a.db)
    if v.drop or v.add:
        print("graph_gnn: the GNN's view (%s): without %s; with %s" % (v.view_path, ", ".join(relation_name(p) for p in sorted(v.drop)) or "nothing",
                                                                      ", ".join(sorted(v.add)) or "nothing"))
    print("graph_gnn: %d nodes (measurements left out), %d edges, %d branch cells, %d requirements, %d maps-to links" % (
        len(v.ids), len(v.edges), len(v.cells), len(v.reqs), len(v.links)))
    res = evaluate(v, a.folds, a.seed, a.epochs, a.group)
    text, beats = report(res, a.group)
    print(text)
    if a.cmd == "eval":
        return 0
    if not beats:
        print("graph_gnn: propose REFUSED - the GNN did not beat the hand baseline on held-out links")
        return 1
    gnn = LinkProposer(v, a.seed, a.epochs).fit(v.links, [])
    owned = collections.defaultdict(set)
    for c, r in v.links:
        owned[c].add(r)
    for c in v.cells:
        conf = gnn.confidence(c)
        top = [r for r in gnn.rank(c) if r not in owned[c]][:a.top]
        print("  %s  owned by %s; %s" % (c[14:], ", ".join(sorted(x[4:] for x in owned[c])) or "NOBODY",
                                         ("the GNN also proposes %s (%.0f%% sure)" % (", ".join(r[4:] for r in top), 100 * conf)) if conf >= SURE
                                         else "the GNN abstains (%.0f%% sure, below %.0f%%)" % (100 * conf, 100 * SURE)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
