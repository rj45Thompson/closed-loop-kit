#!/usr/bin/env python3
"""ask.py - the questions a requirements model cannot answer for itself, one at a time, from the holes in its chain.

WHY (2026-10-31). A producer who writes a rough epic cannot say what they forgot: what was forgotten leaves no trace in their text. But the chain
of use cases that grow.py builds from it does leave one. A step needs something nobody provides; a sentence of the text is cited by nothing; a
goal has no alternate path; two statements cannot both be true. Each is a typed hole, and a hole with its data in it is a question. This is the
queue of those questions and the record of their answers. It is the tool's main claim made runnable: the holes of the chain generate the
questions nobody thought to ask.

THE CLASS MODEL (each class is one concept; its docstring says what it is and where its evidence lives):

  Hole          a gap in the model, typed by SHAPE: disagree | need | uncited | noalt | altreq | facet. Its key is stable (it names the place), so the
                same hole is the same question across sessions and across rebuilds
  Question      a Hole said in plain words a producer can answer (no internal id in the sentence), with what the agent in hand needs to record the
                answer (the command, the hole's place) kept on separate lines
  Disagreement  the detector for two statements that cannot both hold: the same noun with two different numbers, or one statement the negation of
                another. A candidate, never a fact: it raises a question and a person rules
  ModelFiles    the YAML files of a model, read and written back without losing the leading comment block
  Ledger        answers.yaml: every reply, in order, with its source. The reason a question never returns, and how a session resumes: all state is on
                disk, none in a process
  Queue         the open Holes of a model, minus the Ledger's, ranked
  Asker         applies one reply to the model in one transaction (rolled back if it adds an error the drafter would have to fix), then asks again

A reply is exactly one of: an authored answer (the producer's words become a requirement, or fill a field, with the producer as its source), ABSENT
with the reason (nothing there, and why: the cell is set aside, the requirement dropped, the field says none), CONFIRMED (the producer agrees with
what the text or the draft says), or LATER (parked, still counted as open). "Probably fine" is none of these and records nothing.

RANK. Bands first, then how much of the chain rests on the answer. Two statements that disagree go first, because any answer built on either is
suspect. A need nobody authored (INFERRED) is next: its answer can add steps and requirements. Then a sentence of the text that nothing cites. Then
a goal with no alternate path. Then the fields of a goal, last: an answer there fills one field and changes nothing else. Inside a band, the hole
whose answer the most requirements share a step with goes first (its dependents). This is a design choice and says so: ranking all holes by dependents
alone would open with the epic's own "why", which depends on everything and teaches the least.

  py -X utf8 closed_loop.py ask <model>                                  the next question (exit 1), or "nothing open" (exit 0)
  py -X utf8 closed_loop.py ask <model> --answer "<their words>" [--check "<how it would be seen to hold>"] [--place UC:STEP]
                                         [--when "<cause>" --at N --out "rejoin N|end success|end minimal"] [--who "<name>"]
  py -X utf8 closed_loop.py ask <model> --confirm | --absent "<reason>" | --later "<reason>"
  py -X utf8 closed_loop.py ask <model> --status | --list                 how many are open and of what kind; every open question, ranked
  py -X utf8 ask.py --selftest

Exit 0: nothing is open. 1: a question is open (printed), or a reply was refused. 2: the model cannot be read.
"""
import argparse
import collections
import copy
import glob
import io
import json
import os
import re
import shutil
import sys
import tempfile

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import usecase_model as ucm  # noqa: E402
import grow  # noqa: E402
import source_rules  # noqa: E402

BANDS = {"disagree": 0, "need": 1, "uncited": 2, "vague": 2, "noalt": 3, "altreq": 3, "facet": 4}
LABELS = {"disagree": ("statement that disagrees with another", "statements that disagree with another"),
          "need": ("need nobody authored", "needs nobody authored"), "uncited": ("sentence nothing captures", "sentences nothing captures"),
          "noalt": ("goal with no alternate path", "goals with no alternate path"), "altreq": ("alternate path with no rule", "alternate paths with no rule"),
          "facet": ("field of a goal", "fields of goals"), "vague": ("vague word in your text", "vague words in your text")}
FACET_PHRASE = {
    "trigger": "what starts it", "pre": "what has to be true before it can start", "minimal": "what must still be true afterwards if it fails partway",
    "success": "how you will know it worked", "why": "why {a} does it",
    "exec": "how {a} carries it out (what they press, say or choose)", "eval": "how {a} sees that it worked",
}
TEXT_FIELDS = ("trigger", "pre", "minimal", "success", "why")
GULF_FIELDS = ("exec", "eval")
NUMBER_WORDS = dict(zip("zero one two three four five six seven eight nine ten eleven twelve".split(), range(13)))
QUALIFIERS = {("up", "to"): "up to", ("at", "most"): "at most", ("at", "least"): "at least", ("no", "more"): "at most", ("no", "fewer"): "at least",
              ("exactly",): "exactly", ("maximum", "of"): "at most", ("minimum", "of"): "at least", ("only",): "at most"}
NEGATION = re.compile(r"\b(not|never|no|cannot|can't|won't|without|neither|nor|n't)\b", re.I)
SKIP_NOUNS = {"of", "the", "a", "an", "in", "on", "times", "or", "and", "to", "per", "each", "every", "for", "from", "at", "by", "with", "into", "more", "than"}


# words that leave a quantity, a limit or a degree to the reader. The weak-word list of requirements standards (IEEE 830, ISO/IEC/IEEE 29148) cut to what a game, a
# product or a ticket really says: a conservative list, because each hit costs the producer a question. A heuristic that picks whom to ask, never a refusal
VAGUE = re.compile(r"\b(small|large|big|tiny|huge|short|(?:a )?few|several|many|some|various|multiple|numerous|enough|sufficient|plenty|a lot|a bit|a little|"
                   r"quick|quickly|fast|slow|slows|slowed|slower|slowly|soon|often|usually|sometimes|occasionally|normal|normally|appropriate|appropriately|adequate|"
                   r"reasonable|reasonably|suitable|as needed|as required|if possible|where possible|if necessary|etc|and so on|and/or|similar|easy|simple|intuitive|"
                   r"efficient|efficiently|flexible|robust|minimal|optimal)\b", re.I)


class AnswerRefused(Exception):
    """A reply the tool will not record: the words say what is missing. Nothing is written."""


def stem(word):
    w = word.lower()
    for suffix in ("ies", "es", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[:-len(suffix)] + ("y" if suffix == "ies" else "")
    return w


def quoted(text, limit=240):
    t = re.sub(r"\s+", " ", text or "").strip()
    return t if len(t) <= limit else t[:limit - 1].rstrip() + "…"


class Disagreement(object):
    """Two statements that cannot both hold. (1) NUMBER: the same noun with two different counts under the same qualifier, in statements that
    share at least two other content words ("the board has 9 cells" against "the board has 16 cells"). (2) POLARITY: statements that say nearly
    the same words (a word overlap of three quarters) and exactly one of them carries a negation. Heuristics: they pick whom to ask and never
    rule; a refinement ("a player may pass" and "a player may not pass on the first turn") is flagged too and the producer says so."""

    def __init__(self, req_id, text):
        self.id, self.text = req_id, text or ""
        self.words = source_rules.words(self.text) - {"cannot", "never", "without", "neither", "nor"}
        self.negated = bool(NEGATION.search(self.text))
        self.counts = self._counts()

    def _counts(self):
        """[(qualifier, noun, {values}, context)] for every count in the statement, the context taken from its own SENTENCE only (a long answer holds several rules, and
        two sentences of it share words by accident): "6 pits" -> {6}; "1 or 2 cells" -> {1, 2}; "1 to 3 cells" -> {1, 2, 3}"""
        def number(t):
            return int(t) if t.isdigit() else NUMBER_WORDS.get(t)
        out = []
        for sentence in re.split(r"(?<=[.!?])\s+", self.text):
            toks = re.findall(r"[a-z0-9']+", sentence.lower())
            ctx_words = source_rules.words(sentence) - {"cannot", "never", "without", "neither", "nor"}
            i = 0
            while i < len(toks):
                value = number(toks[i])
                if value is None or (toks[i] == "one" and toks[i + 1:i + 2] == ["of"]):
                    i += 1
                    continue
                start, values = i, {value}
                while i + 2 < len(toks) and toks[i + 1] in ("or", "to") and number(toks[i + 2]) is not None:
                    nxt = number(toks[i + 2])
                    values |= set(range(value, nxt + 1)) if toks[i + 1] == "to" and nxt >= value else {nxt}
                    value = nxt
                    i += 2
                j = i + 1
                while j < len(toks) and toks[j] in SKIP_NOUNS:
                    j += 1
                i += 1
                if j >= len(toks) or number(toks[j]) is not None:
                    continue
                qual = ""
                for n in (2, 1):
                    key = tuple(toks[max(0, start - n):start])
                    if key in QUALIFIERS:
                        qual = QUALIFIERS[key]
                        break
                noun = stem(toks[j])
                out.append((qual, noun, values, {stem(w) for w in ctx_words} - {noun}))
        return out

    def against(self, other):
        """-> a reason, or None"""
        for q1, n1, v1, c1 in self.counts:
            for q2, n2, v2, c2 in other.counts:
                if n1 == n2 and q1 == q2 and not (v1 & v2) and len(c1 & c2) >= 2:
                    return "number"
        if self.negated != other.negated and len(self.words) >= 3 and len(other.words) >= 3:
            union = self.words | other.words
            if len(self.words & other.words) / float(len(union)) >= 0.75:
                return "polarity"
        return None


class ModelFiles(object):
    """The YAML files of a model folder, as the loader reads them (every *.yaml, recursively), kept as plain data so one reply can change
    several and write them back together. The leading comment block of a file is kept; a comment inside a file is lost on rewrite, and the
    rewrite says so."""

    def __init__(self, folder):
        self.folder = folder
        self.files = collections.OrderedDict()
        for p in sorted(glob.glob(os.path.join(folder, "**", "*.yaml"), recursive=True)):
            text = io.open(p, encoding="utf-8").read()
            head, inner = [], False
            lines = text.splitlines()
            for i, l in enumerate(lines):
                if l.startswith("#") or not l.strip():
                    head.append(l)
                else:
                    inner = any(x.lstrip().startswith("#") for x in lines[i:])
                    break
            self.files[p] = {"data": yaml.safe_load(text) or {}, "head": head, "text": text, "dirty": False, "lost_comments": inner}

    def entries(self, key):
        for p, f in self.files.items():
            for item in f["data"].get(key) or []:
                yield p, item

    def find(self, key, ident):
        for p, item in self.entries(key):
            if str(item.get("id")) == str(ident):
                return p, item
        return None, None

    def touch(self, path):
        self.files[path]["dirty"] = True

    def answers_path(self):
        p = os.path.join(self.folder, "answers.yaml")
        if p not in self.files:
            self.files[p] = {"data": {"answers": []}, "head": [], "text": "", "dirty": True, "lost_comments": False}
        return p

    def append(self, key, item):
        p = self.answers_path()
        self.files[p]["data"].setdefault(key, []).append(item)
        self.touch(p)

    def remove(self, key, ident):
        p, item = self.find(key, ident)
        if item is not None:
            self.files[p]["data"][key].remove(item)
            self.touch(p)
        return item

    def save(self):
        """-> {path: original text}, so a caller can put everything back"""
        before = {}
        for p, f in self.files.items():
            if not f["dirty"]:
                continue
            before[p] = f["text"]
            body = grow.dump(f["data"])
            head = "\n".join(f["head"]).rstrip("\n")
            text = (head + "\n" if head.strip() else "") + body
            tmp = p + ".tmp"
            io.open(tmp, "w", encoding="utf-8", newline="\n").write(text)
            os.replace(tmp, p)
            f["text"], f["dirty"] = text, False
        return before


class Ledger(object):
    """answers.yaml: every reply in order. An entry: n, key (the hole), shape, kind (authored | absent | confirmed | deferred), question (the
    sentence asked), text / check / reason, source (the id of the quoted answer), made (requirement ids), changed (what the reply did to the model).
    The latest entry for a key decides: answered, or deferred."""

    def __init__(self, entries):
        self.entries = list(entries)

    def latest(self):
        out = {}
        for e in self.entries:
            out[e.get("key")] = e
        return out

    def answered(self):
        """keys with an answer on record. A reply to a goal's fields names the fields it settled, each of which counts as its own key"""
        done = {k for k, e in self.latest().items() if e.get("kind") in ("authored", "absent", "confirmed")}
        for e in self.entries:
            if e.get("kind") in ("authored", "absent"):
                for f in e.get("fields") or {}:
                    done.add("facet:%s:%s" % (e.get("key", "").split(":")[1], f))
        return done

    def deferred(self):
        return {k for k, e in self.latest().items() if e.get("kind") == "deferred"}

    def next_n(self):
        return max([int(e.get("n") or 0) for e in self.entries] + [0]) + 1


class Hole(object):
    """A gap in the model. `key` names the place and is stable across sessions: need:<req>, uncited:<space>:<cell>, facet:<use case>:text|gulf (a goal's fields, asked together),
    noalt:<use case>, altreq:<use case>:<ext>, disagree:<req>|<req>. `data` is what the question and the reply need."""

    def __init__(self, shape, key, dependents, data):
        self.shape, self.key, self.dependents, self.data = shape, key, dependents, data
        self.band = BANDS[shape]


class Question(object):
    """A Hole said in plain words. `sentence` is what the producer is asked and holds no internal id; `how` lines are for the agent in hand."""

    def __init__(self, hole, sentence, why, modes):
        self.hole, self.sentence, self.why, self.modes = hole, sentence, why, modes
        self.key = hole.key


class Queue(object):
    """The open Holes of a model, ranked, with their questions. Built fresh from the files every time: nothing about a session is remembered
    anywhere but answers.yaml. `mine` are the findings only the drafter can fix; a queue over a model that has them is refused, because a
    question asked over a broken chain would be asked of the wrong thing."""

    def __init__(self, folder):
        self.folder = folder
        self.model = ucm.UseCaseModel.load(folder)
        self.files = ModelFiles(folder)
        self.ledger = Ledger((self.files.files.get(os.path.join(folder, "answers.yaml"), {"data": {}})["data"]).get("answers") or [])
        self.where = {}
        mp = os.path.join(folder, "source", "map.json")
        if os.path.exists(mp):
            self.where = (json.load(io.open(mp, encoding="utf-8")).get("where")) or {}
        self.findings = self.model.check()
        self.mine, self.theirs = grow.split_findings(self.findings)
        self.holes = self._holes()
        done, parked = self.ledger.answered(), self.ledger.deferred()
        live = []
        for h in self.holes:
            if h.shape == "facet":
                h.data["fields"] = [f for f in h.data["fields"] if "facet:%s:%s" % (h.data["uc"], f) not in done]
                if h.data["fields"]:
                    live.append(h)
            elif h.key not in done:
                live.append(h)
        self.open = sorted([h for h in live if h.key not in parked], key=lambda h: (h.band, -h.dependents)) + \
            sorted([h for h in live if h.key in parked], key=lambda h: (h.band, -h.dependents))
        self.parked = parked

    # ------------------------------------------------------------ the chain
    def pathways(self):
        """[(use case id, 'step N' | 'ext X', step text or when, [req ids])]"""
        out = []
        for uc in self.model.usecases.values():
            for s in uc.steps:
                out.append((uc.id, "step %d" % s.n, s.text, list(s.reqs)))
            for e in uc.ext:
                out.append((uc.id, "ext %s" % e.id, e.when, list(e.reqs)))
        return out

    def dependents(self, req_ids):
        ids = set(req_ids)
        reach = set(ids)
        for _, _, _, rs in self.pathways():
            if ids & set(rs):
                reach |= set(rs)
        return len(reach)

    def sentence_text(self, space_id, key):
        sp = self.model.spaces.get(space_id)
        if sp is None:
            return key
        if sp.kind == "sentences":
            src = self.model.sources.get(sp.source)
            for k, t in ucm.sentences(self.model.resolve(src.path)):
                if k == key:
                    return t
        c = sp.cells.get(key)
        return (c.text if c is not None and c.text else key)

    def where_of(self, key):
        return self.where.get(key) or ("line %s" % key.split(".")[0])

    def uc_name(self, uc_id):
        uc = self.model.usecases.get(uc_id)
        return uc.name if uc else uc_id

    def uc_title(self, uc_id):
        """a goal by name; the whole (a kite) as 'the whole of "X"', so a producer sees that the question is about everything"""
        uc = self.model.usecases.get(uc_id)
        return 'the whole of "%s"' % uc.name if uc is not None and uc.level == "kite" else '"%s"' % self.uc_name(uc_id)

    def actor_name(self, uc_id):
        uc = self.model.usecases.get(uc_id)
        a = self.model.actors.get(uc.actor) if uc else None
        return (a.name or a.id) if a else "the player"

    # ------------------------------------------------------------ holes
    def _holes(self):
        m, out = self.model, []
        raw = lambda key, ident: self.files.find(key, ident)[1] or {}
        fields = collections.OrderedDict()
        for r in m.requirements.values():
            if r.kind == "inferred":
                carriers = [(u, w, t) for u, w, t, rs in self.pathways() if r.id in rs]
                out.append(Hole("need", "need:%s" % r.id, self.dependents([r.id]), {"req": r.id, "carriers": carriers, "needs": raw("requirements", r.id).get("needs")}))
        claims = [(r.id, Disagreement(r.id, r.text)) for r in m.requirements.values()]
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                why = claims[i][1].against(claims[j][1])
                if why:
                    a, b = sorted((claims[i][0], claims[j][0]))
                    out.append(Hole("disagree", "disagree:%s|%s" % (a, b), self.dependents([a, b]), {"a": a, "b": b, "why": why}))
        for f in self.theirs:
            if f.rule == "U7":
                sp, _, cell = f.where.partition(":")
                out.append(Hole("uncited", "uncited:%s:%s" % (sp, cell), 1, {"space": sp, "cell": cell, "set_aside": None}))
            elif f.rule == "U17":
                uc = m.usecases[f.where]
                out.append(Hole("noalt", "noalt:%s" % f.where, self.dependents([r for s in uc.steps for r in s.reqs]), {"uc": f.where, "kite": uc.level == "kite"}))
            elif f.rule == "U4":
                uid, _, eid = f.where.partition(" ")
                uc = m.usecases[uid]
                out.append(Hole("altreq", "altreq:%s:%s" % (uid, eid), self.dependents([r for s in uc.steps for r in s.reqs]), {"uc": uid, "ext": eid}))
            elif f.rule == "U1":
                fields.setdefault((f.where, "text"), []).append(f.message.split()[1])
            elif f.rule == "U13":
                fields.setdefault((f.where, "gulf"), []).append("exec" if "execution" in f.message else "eval")
        for (uid, part), fs in fields.items():
            order = TEXT_FIELDS if part == "text" else GULF_FIELDS
            out.append(Hole("facet", "facet:%s:%s" % (uid, part), self.dependents(m.subtree_reqs(uid)), {"uc": uid, "part": part, "fields": [x for x in order if x in fs]}))
        # a vague word inside a sentence a requirement stands on: the sentence leaves a quantity, a limit or a degree to the reader
        for sp in m.spaces.values():
            if sp.kind != "sentences":
                continue
            for k, c in sp.cells.items():
                if c.maps:
                    terms = []
                    for hit in VAGUE.finditer(self.sentence_text(sp.id, k)):
                        if hit.group(0).lower() not in [t.lower() for t in terms]:
                            terms.append(hit.group(0))
                    if terms:
                        out.append(Hole("vague", "vague:%s:%s" % (sp.id, k), self.dependents(c.maps), {"space": sp.id, "cell": k, "terms": terms, "reqs": list(c.maps)}))
        # a sentence the drafting agent set aside that reads like a rule: the agent's word is not the producer's
        for sp in m.spaces.values():
            if sp.kind != "sentences":
                continue
            raw_sp = raw("spaces", sp.id)
            for k, c in sp.cells.items():
                rc = (raw_sp.get("cells") or {}).get(k)
                if not c.maps and not ucm.blank(c.excluded) and isinstance(rc, dict) and rc.get("by") == "drafting agent" and not rc.get("confirmed"):
                    text = self.sentence_text(sp.id, k)
                    if grow.STRICT.search(text):
                        out.append(Hole("uncited", "uncited:%s:%s" % (sp.id, k), 1, {"space": sp.id, "cell": k, "set_aside": c.excluded}))
        return out

    # ------------------------------------------------------------ words
    def question(self, hole):
        d, m = hole.data, self.model
        if hole.shape == "need":
            r = m.requirements[d["req"]]
            needs = d.get("needs")
            if d["carriers"]:
                uid, place, said = d["carriers"][0]
                if place.startswith("step"):
                    lead = 'When "%s"' % quoted(said, 120)
                else:
                    lead = 'When %s' % quoted(said, 120)
                if needs:
                    s = '%s, we need to know %s, and nothing you wrote says where that comes from. We read it as: "%s" Is that right, or what should happen?' % (
                        lead, quoted(needs, 120), quoted(r.text))
                else:
                    s = '%s, we need something that nothing you wrote provides. We read it as: "%s" Is that right, or what should happen?' % (lead, quoted(r.text))
            else:
                s = 'A step of the plan seems to need this and nobody wrote it: "%s" Is that right, or what should happen?' % quoted(r.text)
            why = "need: %s is INFERRED (%s)" % (r.id, quoted(r.why, 200))
            return Question(hole, s, why, ("confirm", "answer", "absent", "later"))
        if hole.shape == "disagree":
            a, b = m.requirements[d["a"]], m.requirements[d["b"]]
            s = 'These two statements cannot both hold: "%s" (%s) and "%s" (%s). Which is right, or what is the rule?' % (
                quoted(a.text, 160), self.provenance(a), quoted(b.text, 160), self.provenance(b))
            return Question(hole, s, "disagree: %s and %s (%s)" % (a.id, b.id, d["why"]), ("answer", "absent", "later"))
        if hole.shape == "vague":
            text = self.sentence_text(d["space"], d["cell"])
            terms = " and ".join('"%s"' % t for t in d["terms"])
            s = 'Your text says: "%s" (%s). How exactly much, how many or how far is %s? Give a number, a rule or an example.' % (quoted(text), self.where_of(d["cell"]), terms)
            return Question(hole, s, "vague: %s in sentence %s, carried by %s" % (", ".join(d["terms"]), d["cell"], ", ".join(d["reqs"])), ("answer", "absent", "later"))
        if hole.shape == "uncited":
            text = self.sentence_text(d["space"], d["cell"])
            sp = m.spaces[d["space"]]
            if sp.kind == "sentences":
                place = self.where_of(d["cell"])
                if d["set_aside"]:
                    s = 'Your text says: "%s" (%s). It was set aside as: %s. Is that right, or is it a rule the finished work must follow?' % (
                        quoted(text), place, quoted(d["set_aside"], 120))
                    return Question(hole, s, "uncited: sentence %s, set aside by the drafting agent" % d["cell"], ("confirm", "answer", "later"))
                s = 'Your text says: "%s" (%s). Nothing captures it yet. Is it a rule the finished work must follow, and if so what exactly must hold?' % (quoted(text), place)
            else:
                s = 'In "%s" there is "%s", and no rule covers it. Does the finished work do this, and if so what must hold?' % (sp.title or sp.id, quoted(text))
            return Question(hole, s, "uncited: %s cell %s" % (d["space"], d["cell"]), ("answer", "absent", "later"))
        g, a = self.uc_name(d["uc"]), self.actor_name(d["uc"])
        if hole.shape == "noalt":
            s = 'While %s is going on, what can go wrong or stop it partway, and what should happen then?' % self.uc_title(d["uc"])
            return Question(hole, s, "noalt: %s has %d steps and no alternate path" % (d["uc"], len(m.usecases[d["uc"]].steps)), ("answer", "absent", "later"))
        if hole.shape == "altreq":
            e = next(x for x in m.usecases[d["uc"]].ext if x.id == d["ext"])
            s = 'During %s, when %s, the plan says: %s. What rule is that? State it as something that must always be true.' % (
                self.uc_title(d["uc"]), quoted(e.when, 120), quoted("; ".join(e.do), 160))
            return Question(hole, s, "altreq: %s %s carries no requirement" % (d["uc"], d["ext"]), ("answer", "absent", "later"))
        parts = [FACET_PHRASE[f].format(a=a) for f in d["fields"]]
        if len(parts) == 1:
            s = 'About %s: %s?' % (self.uc_title(d["uc"]), parts[0])
        else:
            s = 'About %s, tell us whichever of these you can, and leave the rest: %s.' % (self.uc_title(d["uc"]), "; ".join(parts))
        return Question(hole, s, "facet: %s of %s" % (", ".join(d["fields"]), d["uc"]), ("fields", "later"))

    def provenance(self, r):
        raw = self.files.find("requirements", r.id)[1] or {}
        if r.kind == "inferred":
            return "our reading of the plan"
        if any(str(s).startswith("ANS-") for s in r.sources):
            return "your answer to an earlier question"
        cites = raw.get("cite") or []
        return "your text, %s" % self.where_of(cites[0]) if cites else "your text"

    def questions(self):
        return [self.question(h) for h in self.open]


def counts_by_shape(holes):
    c = collections.Counter(h.shape for h in holes)
    return ", ".join("%d %s" % (c[s], LABELS[s][0 if c[s] == 1 else 1]) for s in BANDS if c[s])


def status_lines(q, display):
    lines = []
    open_n = len(q.open)
    parked = sum(1 for h in q.open if h.key in q.parked)
    lines.append("%d open question(s)%s%s" % (open_n, ": " + counts_by_shape(q.open) if open_n else "",
                                              " (%d parked for later)" % parked if parked else ""))
    latest = q.ledger.latest()
    kinds = collections.Counter(e.get("kind") for e in latest.values())
    lines.append("%d answered so far (%d authored, %d absent with a reason, %d confirmed)" % (
        sum(kinds[k] for k in ("authored", "absent", "confirmed")), kinds["authored"], kinds["absent"], kinds["confirmed"]))
    return lines


def how_lines(q, display):
    modes, key = q.modes, q.key
    c = "closed_loop.py ask %s" % display
    out = []
    if q.hole.shape == "need":
        out += ['%s --confirm                                   (they agree with the requirement as written)' % c,
                '%s --answer "<their words>" [--check "<how it would be seen to hold>"]   (they say it differently)' % c,
                '%s --absent "<why it does not apply>"          (the plan does not need it)' % c]
    elif q.hole.shape == "uncited" and "confirm" in modes:
        out += ['%s --confirm                                   (it is background, not a rule)' % c,
                '%s --answer "<the rule>" --check "<how it would be seen to hold>" --place <goal>:<step>' % c]
    elif q.hole.shape == "uncited":
        out += ['%s --answer "<the rule>" --check "<how it would be seen to hold>" --place <goal>:<step>' % c,
                '%s --absent "<why it states no rule>"' % c]
    elif q.hole.shape == "vague":
        out += ['%s --answer "<the exact amount, rule or example>" --check "<how it would be seen to hold>"' % c,
                '%s --absent "<why it is left open>"   (the builder may choose)' % c]
    elif q.hole.shape == "disagree":
        out += ['%s --answer "<the right statement>" --check "<how it would be seen to hold>"   (replaces both)' % c,
                '%s --absent "<why they do not conflict>"' % c]
    elif q.hole.shape == "noalt":
        kite = q.hole.data.get("kite")
        out += ['%s --answer "<what happens>" --when "<the cause>" --at <step number> --out "rejoin N|end success|end minimal" --check "<how it would be seen>"%s' % (
                    c, ' --place <goal>:<step>   (a goal step must carry it too)' if kite else ""),
                '%s --absent "<why nothing can stop it>"' % c]
    elif q.hole.shape == "altreq":
        out += ['%s --answer "<the rule>" --check "<how it would be seen to hold>"' % c,
                '%s --absent "<why this alternate path does not happen>"' % c]
    else:
        fs = q.hole.data["fields"]
        out += ['%s %s' % (c, " ".join('--set %s="<their words>"' % f for f in fs)) + ('   (add --check "<how it would be seen to hold>")' if q.hole.data["part"] == "gulf" else ""),
                '%s %s   (a field with nothing to say)' % (c, " ".join('--none %s="<why there is none>"' % f for f in fs))]
    out.append('%s --later "<why not now>"' % c)
    return out


def show_next(folder, out=print, header="NEXT QUESTION", display=None):
    display = display or folder
    try:
        q = Queue(folder)
    except (RuntimeError, OSError, yaml.YAMLError) as e:
        out("ask: cannot read the model %s: %s" % (folder, e))
        return 2
    if q.mine:
        out("ask: the model has %d error(s) only the drafter can fix, so no question is asked yet (a question over a broken chain asks about the wrong thing):" % len(q.mine))
        for f in q.mine[:12]:
            out("  %s" % f)
        return 1
    if not q.open:
        out("%s: nothing is open. Every hole of the chain has an answer on record." % header)
        for l in status_lines(q, display):
            out("  " + l)
        rest = [f for f in q.findings]
        if rest:
            out("  The 19 rules still list %d hole(s), each answered ABSENT or confirmed: %s. The rules have no place to record an absence in a gulf, an alternate path or a "
                "step, so these stay open by rule." % (len(rest), ", ".join("%s x%d" % (k, v) for k, v in sorted(collections.Counter(f.rule for f in rest).items()))))
        else:
            out("  The 19 rules: PASS.")
        return 0
    top = q.questions()[0]
    out("%s  (%d open)" % (header, len(q.open)))
    out("")
    out("  " + top.sentence)
    out("")
    out("  --- for the agent: ask the producer the sentence above, in those words. %s" % top.why)
    for l in how_lines(top, display):
        out("      " + l)
    if len(q.open) > 1:
        out("  Then ask again; or `closed_loop.py ask %s --status` says where this stands, `--list` shows every open question." % display)
    return 1


# ---------------------------------------------------------------------- answering

class Asker(object):
    """Applies ONE reply to the top open question (or the one named by --key), as one transaction over the model's files."""

    def __init__(self, folder, display=None):
        self.folder, self.display = folder, display or folder

    def reply(self, mode, text=None, check=None, reason=None, place=None, when=None, at=None, out=None, who=None, key=None, sets=None, nones=None):
        q = Queue(self.folder)
        if q.mine:
            raise AnswerRefused("the model has %d error(s) only the drafter can fix; nothing is asked or recorded until they are" % len(q.mine))
        if not q.open:
            raise AnswerRefused("nothing is open: there is no question to answer")
        qs = q.questions()
        top = next((x for x in qs if x.key == key), None) if key else qs[0]
        if top is None:
            raise AnswerRefused("no open question has the key %r" % key)
        if mode not in top.modes:
            raise AnswerRefused("this question takes %s, not --%s" % (" or ".join("--" + m for m in top.modes), mode))
        who = who or "the producer"
        n = q.ledger.next_n()
        files = q.files
        entry = {"n": n, "key": top.key, "shape": top.hole.shape, "kind": {"answer": "authored", "absent": "absent", "confirm": "confirmed", "later": "deferred", "fields": "authored"}[mode],
                 "question": top.sentence, "who": who}
        changed, made = [], []
        if mode == "later":
            if grow.blank(reason):
                raise AnswerRefused("--later needs a reason: why not now, or who to ask")
            entry["reason"] = reason
        elif mode == "absent":
            if grow.blank(reason):
                raise AnswerRefused("--absent needs the reason there is nothing: ABSENT without a reason is just silence")
            entry["reason"] = reason
            self.apply_absent(q, top, files, reason, entry, changed)
        elif mode == "confirm":
            self.apply_confirm(q, top, files, n, entry, changed)
        elif mode == "fields":
            entry["text"] = "; ".join("%s: %s" % kv for kv in list((sets or {}).items()) + [(k, "none - " + v) for k, v in (nones or {}).items()])
            entry["check"] = check
            self.apply_fields(q, top, files, n, sets or {}, nones or {}, check, entry, changed, made)
            text = "; ".join("%s: %s" % kv for kv in (sets or {}).items()) or None
        else:
            if grow.blank(text):
                raise AnswerRefused("--answer needs the producer's words")
            entry["text"] = text
            self.apply_answer(q, top, files, n, text, check, place, when, at, out, who, entry, changed, made)
        if mode in ("answer", "confirm") or (mode == "fields" and sets):
            files.append("sources", {"id": "ANS-%d" % n, "kind": "quote", "cite": "answer %d, from %s" % (n, who),
                                     "text": text or "confirmed as written: %s" % top.sentence})
        entry["source"] = "ANS-%d" % n if mode in ("answer", "confirm") or (mode == "fields" and sets) else None
        entry["made"], entry["changed"] = made, changed
        files.append("answers", entry)
        return self.commit(q, files, top, entry)

    def commit(self, q, files, top, entry):
        before = files.save()
        try:
            after = Queue(self.folder)
        except (RuntimeError, OSError, yaml.YAMLError) as e:
            self.restore(before)
            raise AnswerRefused("the reply would leave the model unreadable (%s); nothing was recorded" % e)
        added = {str(f) for f in after.mine} - {str(f) for f in q.mine}
        if added:
            self.restore(before)
            raise AnswerRefused("the reply would add %d error(s) to the chain, so nothing was recorded: %s" % (len(added), "; ".join(sorted(added)[:3])))
        return entry

    def restore(self, before):
        for p, text in before.items():
            if text == "" and os.path.basename(p) == "answers.yaml":
                os.remove(p)
            else:
                io.open(p, "w", encoding="utf-8", newline="\n").write(text)

    # ------------------------------------------------------------ shapes
    def new_requirement(self, q, files, n, text, check, cites, closed, src=None, extra=None):
        if grow.blank(check):
            raise AnswerRefused("--check is needed: how would anyone see this requirement hold or fail? (a requirement with no measure is an error of its own)")
        taken = {str(i.get("id")) for _, i in files.entries("requirements")}
        rid = next(c for c in ["RA%d" % n] + ["RA%d%s" % (n, x) for x in "bcdefgh"] if c not in taken)      # one reply can make two (a goal's two gulfs)
        r = {"id": rid, "kind": "authored", "sources": ["ANS-%d" % n] + ([src] if cites and src else []), "text": text, "measure": check}
        if cites:
            r["cite"] = list(cites)
        if closed:
            r["effect"] = {"absent": "stated by the producer; not yet in code"}
        r.update(extra or {})
        files.append("requirements", r)
        return rid

    def resolve_place(self, q, place):
        if grow.blank(place):
            names = []
            for uc in q.model.usecases.values():
                if uc.level != "kite":
                    names.append("%s: %s" % (uc.name, "; ".join("%d %s" % (s.n, quoted(s.text, 40)) for s in uc.steps)))
            raise AnswerRefused("--place <goal>:<step> is needed, so the new requirement is carried by a step. Goals and steps: " + " | ".join(names))
        g, _, s = str(place).rpartition(":")
        uc = q.model.usecases.get(g) or next((u for u in q.model.usecases.values() if u.name.lower() == g.lower()), None)
        if uc is None or not s.isdigit() or not (1 <= int(s) <= len(uc.steps)):
            raise AnswerRefused("--place %r names no step: use <goal id or name>:<step number>" % place)
        if uc.level == "kite":
            raise AnswerRefused("place it on a step of a goal, not on a step of the whole epic: a requirement only the epic carries is rule U16")
        return uc, int(s)

    def carry(self, files, uc_id, step, rid):
        _, raw = files.find("usecases", uc_id)
        steps = raw.setdefault("steps", [])
        steps[step - 1].setdefault("reqs", []).append(rid)
        files.touch(files.find("usecases", uc_id)[0])

    def apply_answer(self, q, top, files, n, text, check, place, when, at, out, who, entry, changed, made):
        shape, d, closed = top.hole.shape, top.hole.data, q.model.closed_loop
        entry["check"] = check
        if shape == "need":
            p, raw = files.find("requirements", d["req"])
            raw["was"] = raw.get("text")
            raw.update({"kind": "authored", "text": text})
            raw.pop("why", None)
            raw["inferred_from"] = raw.pop("at", None)
            raw["sources"] = list(raw.get("sources") or []) + ["ANS-%d" % n]
            if not grow.blank(check):
                raw["measure"] = check
            files.touch(p)
            changed.append("%s: inferred, now authored by the producer's words" % d["req"])
        elif shape == "vague":
            sp = q.model.spaces[d["space"]]
            rid = self.new_requirement(q, files, n, text, check, [d["cell"]], closed, src=sp.source)
            placed = []
            for uid, where, _, rs in q.pathways():
                if set(rs) & set(d["reqs"]):
                    p, raw = files.find("usecases", uid)
                    kind, _, ident = where.partition(" ")
                    holder = raw["steps"][int(ident) - 1] if kind == "step" else next(e for e, x in zip(raw["ext"], q.model.usecases[uid].ext) if x.id == ident)
                    if rid not in holder.setdefault("reqs", []):
                        holder["reqs"].append(rid)
                        placed.append("%s %s" % (uid, where))
                        files.touch(p)
            if not placed:
                raise AnswerRefused("no step or alternate path carries the requirement this sentence became, so the amount has nowhere to go")
            p, rawsp = files.find("spaces", d["space"])
            cell = rawsp["cells"][d["cell"]]
            if isinstance(cell, dict):
                cell["maps"] = list(cell.get("maps") or []) + [rid]
            else:
                rawsp["cells"][d["cell"]] = list(cell) + [rid]
            files.touch(p)
            made.append(rid)
            changed.append("%s adds the amount, carried where the sentence's requirement is (%s)" % (rid, ", ".join(placed[:3])))
        elif shape == "uncited":
            sp = q.model.spaces[d["space"]]
            cites = [d["cell"]] if sp.kind == "sentences" else []
            uc, step = self.resolve_place(q, place)
            rid = self.new_requirement(q, files, n, text, check, cites, closed, src=sp.source)
            self.carry(files, uc.id, step, rid)
            p, rawsp = files.find("spaces", d["space"])
            cell = rawsp["cells"].get(d["cell"])
            if isinstance(cell, dict):
                cell.pop("excluded", None)
                cell.pop("by", None)
                cell["maps"] = list(cell.get("maps") or []) + [rid]
            else:
                rawsp["cells"][d["cell"]] = list(cell or []) + [rid]
            files.touch(p)
            made.append(rid)
            changed.append("%s cell %s now maps to %s, carried by step %d of %s" % (d["space"], d["cell"], rid, step, uc.id))
        elif shape == "disagree":
            rid = self.new_requirement(q, files, n, text, check, [], closed)
            old = [d["a"], d["b"]]
            cites, srcs = [], ["ANS-%d" % n]
            for o in old:
                raw = files.find("requirements", o)[1] or {}
                cites += list(raw.get("cite") or [])
            _, rawnew = files.find("requirements", rid)
            if cites and "S-SRC" in q.model.sources:
                rawnew["cite"] = cites
            rawnew["replaces"] = old
            for o in old:
                files.remove("requirements", o)
            self.replace_everywhere(files, old, rid)
            made.append(rid)
            changed.append("%s and %s replaced by %s" % (old[0], old[1], rid))
        elif shape == "noalt":
            uc = q.model.usecases[d["uc"]]
            if grow.blank(when) or grow.blank(out) or at is None:
                raise AnswerRefused('a noalt answer needs --when "<the cause>", --at <step number> and --out "rejoin N|end success|end minimal"')
            if not ucm.OUT_RE.match(str(out)):
                raise AnswerRefused('--out must be "rejoin N", "end success" or "end minimal"')
            at = int(at)
            if not (1 <= at <= len(uc.steps)):
                raise AnswerRefused("--at %d is not a step of this goal (1 to %d)" % (at, len(uc.steps)))
            host = self.resolve_place(q, place) if uc.level == "kite" else None     # a kite's own alternate path is a summary: a goal's step must carry the rule too (U16)
            rid = self.new_requirement(q, files, n, text, check, [], closed)
            _, raw = files.find("usecases", uc.id)
            raw.setdefault("ext", []).append({"at": at, "when": when, "do": [text], "out": out, "reqs": [rid]})
            files.touch(files.find("usecases", uc.id)[0])
            if host:
                self.carry(files, host[0].id, host[1], rid)
            made.append(rid)
            changed.append("%s gains an alternate path at step %d, carried by %s" % (uc.id, at, rid))
        elif shape == "altreq":
            uc = q.model.usecases[d["uc"]]
            host = self.resolve_place(q, place) if uc.level == "kite" else None
            rid = self.new_requirement(q, files, n, text, check, [], closed)
            idx = next(i for i, e in enumerate(uc.ext) if e.id == d["ext"])
            p, raw = files.find("usecases", d["uc"])
            raw["ext"][idx].setdefault("reqs", []).append(rid)
            files.touch(p)
            if host:
                self.carry(files, host[0].id, host[1], rid)
            made.append(rid)
            changed.append("%s %s now carries %s" % (d["uc"], d["ext"], rid))

    def apply_fields(self, q, top, files, n, sets, nones, check, entry, changed, made):
        """a reply to a goal's fields: --set field=text for what the producer said, --none field=reason for what does not exist"""
        d, closed = top.hole.data, q.model.closed_loop
        uid = d["uc"]
        p, raw = files.find("usecases", uid)
        bad = [f for f in list(sets) + list(nones) if f not in d["fields"]]
        if bad:
            raise AnswerRefused("this question is about %s, not %s" % (", ".join(d["fields"]), ", ".join(bad)))
        both = [f for f in sets if f in nones]
        if both:
            raise AnswerRefused("%s is both set and none" % ", ".join(both))
        if not sets and not nones:
            raise AnswerRefused("name at least one field: --set <field>=\"<their words>\" or --none <field>=\"<why there is none>\"")
        for f, v in list(sets.items()) + list(nones.items()):
            if grow.blank(v):
                raise AnswerRefused("%s has no words: --none needs the reason there is nothing" % f if f in nones else "%s has no words" % f)
        settled = {}
        for f, v in sets.items():
            if f in GULF_FIELDS:
                rid = self.new_requirement(q, files, n, v, check, [], closed)
                host_uc, step = self.host_step(q, uid, f)
                self.carry(files, host_uc, step, rid)
                raw.setdefault("gulf", {})[f] = {"text": v, "reqs": [rid]}
                made.append(rid)
                changed.append("%s gulf of %s: %s, carried by step %d of %s" % (uid, f, rid, step, host_uc))
            else:
                raw[f] = v
                raw.setdefault("answered", {})[f] = "ANS-%d" % n
                changed.append("%s %s set" % (uid, f))
            settled[f] = "ANS-%d" % n
        for f, why in nones.items():
            if f in GULF_FIELDS:
                raw.setdefault("gulf", {})[f] = {"text": "none - %s" % why, "absent": why, "reqs": []}
                changed.append("%s gulf of %s: none (rule U13 stays open by rule)" % (uid, f))
            else:
                raw[f] = "none - %s" % why
                raw.setdefault("answered", {})[f] = "ABSENT"
                changed.append("%s %s: none, with the reason" % (uid, f))
            settled[f] = "ABSENT"
        files.touch(p)
        entry["fields"] = settled
        entry["kind"] = "authored" if sets else "absent"

    def host_step(self, q, uid, which):
        uc = q.model.usecases[uid]
        if uc.level != "kite":
            return uid, (1 if which == "exec" else len(uc.steps))
        for s in uc.steps:
            for child in s.includes:
                c = q.model.usecases.get(child)
                if c is not None:
                    return child, (1 if which == "exec" else len(c.steps))
        for c in q.model.usecases.values():
            if c.parent.get("uc") == uid:
                return c.id, (1 if which == "exec" else len(c.steps))
        raise AnswerRefused("the epic has no goal below it to carry this")

    def replace_everywhere(self, files, old, new):
        def swap(lst):
            out = []
            for r in lst or []:
                r = new if r in old else r
                if r not in out:
                    out.append(r)
            return out
        for p, item in files.entries("usecases"):
            for s in item.get("steps") or []:
                if "reqs" in s:
                    s["reqs"] = swap(s["reqs"])
            for e in item.get("ext") or []:
                if "reqs" in e:
                    e["reqs"] = swap(e["reqs"])
            for g in (item.get("gulf") or {}).values():
                if isinstance(g, dict) and "reqs" in g:
                    g["reqs"] = swap(g["reqs"])
            files.touch(p)
        for p, sp in files.entries("spaces"):
            cells = sp.get("cells") or {}
            for k, c in list(cells.items()):
                if isinstance(c, dict) and "maps" in c:
                    c["maps"] = swap(c["maps"])
                elif isinstance(c, list):
                    cells[k] = swap(c)
            files.touch(p)

    def apply_confirm(self, q, top, files, n, entry, changed):
        shape, d = top.hole.shape, top.hole.data
        if shape == "need":
            p, raw = files.find("requirements", d["req"])
            raw["kind"] = "authored"
            raw["inferred_from"] = raw.pop("why", None)
            raw.pop("at", None)
            raw["sources"] = list(raw.get("sources") or []) + ["ANS-%d" % n]
            files.touch(p)
            changed.append("%s: inferred, now authored: the producer confirmed it as written" % d["req"])
        else:
            p, rawsp = files.find("spaces", d["space"])
            cell = rawsp["cells"][d["cell"]]
            cell["confirmed"] = "ANS-%d" % n
            files.touch(p)
            changed.append("%s cell %s: set aside, and the producer confirmed it" % (d["space"], d["cell"]))

    def apply_absent(self, q, top, files, reason, entry, changed):
        shape, d = top.hole.shape, top.hole.data
        if shape == "need":
            alone = [(u, w, t) for u, w, t, rs in q.pathways() if rs == [d["req"]] and not any(
                x.includes for x in q.model.usecases[u].steps if w == "step %d" % x.n)]
            if alone:
                raise AnswerRefused('dropping it would leave %s with nothing it must do. If the plan does not need it, say what that step must do instead (--answer), '
                                    'or confirm it' % "; ".join('%s of "%s" ("%s")' % (w, q.uc_name(u), quoted(t, 60)) for u, w, t in alone[:3]))
            r = files.remove("requirements", d["req"])
            self.drop_everywhere(files, d["req"])
            entry["removed"] = {"id": d["req"], "text": (r or {}).get("text")}
            changed.append("%s dropped: the plan does not need it" % d["req"])
        elif shape == "disagree":
            changed.append("%s and %s kept: the producer says they do not conflict" % (d["a"], d["b"]))
        elif shape == "vague":
            changed.append("sentence %s left as written: %s" % (d["cell"], reason))
        elif shape == "uncited":
            p, rawsp = files.find("spaces", d["space"])
            cell = rawsp["cells"].get(d["cell"])
            note = "%s (the producer)" % reason
            rawsp["cells"][d["cell"]] = {"excluded": note, "by": "the producer"} if not isinstance(cell, dict) else dict(cell, excluded=note, by="the producer")
            files.touch(p)
            changed.append("%s cell %s set aside with the producer's reason" % (d["space"], d["cell"]))
        elif shape == "noalt":
            p, raw = files.find("usecases", d["uc"])
            raw["no_alt"] = reason
            files.touch(p)
            changed.append("%s: the producer says nothing can stop it (rule U17 stays open by rule)" % d["uc"])
        elif shape == "altreq":
            p, raw = files.find("usecases", d["uc"])
            uc = q.model.usecases[d["uc"]]
            idx = next(i for i, e in enumerate(uc.ext) if e.id == d["ext"])
            if any(c.parent.get("uc") == uc.id and str(c.parent.get("ext")) == d["ext"] for c in q.model.usecases.values()):
                raise AnswerRefused("a goal hangs below this alternate path; it cannot be dropped")
            raw["ext"].pop(idx)
            if not raw["ext"]:
                raw.pop("ext")
            files.touch(p)
            changed.append("%s %s dropped: that alternate path does not happen" % (d["uc"], d["ext"]))

    def drop_everywhere(self, files, rid):
        for p, item in files.entries("usecases"):
            for s in item.get("steps") or []:
                if rid in (s.get("reqs") or []):
                    s["reqs"].remove(rid)
            for e in item.get("ext") or []:
                if rid in (e.get("reqs") or []):
                    e["reqs"].remove(rid)
            for g in (item.get("gulf") or {}).values():
                if isinstance(g, dict) and rid in (g.get("reqs") or []):
                    g["reqs"].remove(rid)
            files.touch(p)
        for p, sp in files.entries("spaces"):
            for k, c in (sp.get("cells") or {}).items():
                lst = c.get("maps") if isinstance(c, dict) else c
                if isinstance(lst, list) and rid in lst:
                    lst.remove(rid)
            files.touch(p)


# ---------------------------------------------------------------------- command line

# ---------------------------------------------------------------------- the selftest: the queue, the ranking, the words, the record, the plants

def scripted_reply(folder, q, text="Yes, as stated."):
    """answer the top question the way a cooperative producer would, one reply of the kind the question takes -> the ledger entry"""
    top = q.questions()[0]
    a = Asker(folder)
    shape = top.hole.shape
    if shape == "need":
        return a.reply("confirm")
    if shape == "facet":
        sets = {f: ("%s: %s" % (f, text)) for f in top.hole.data["fields"]}
        return a.reply("fields", sets=sets, check="a person tries it and sees that it holds")
    if shape == "vague":
        return a.reply("answer", text="Exactly two cells, never more.", check="move three cells and see it refused")
    if shape == "uncited":
        if "confirm" in top.modes:
            return a.reply("confirm")
        return a.reply("absent", reason="background, not a rule")
    if shape == "noalt":
        return a.reply("answer", text="the goal stops where it is", when="something stops it", at=1, out="end minimal", check="stop it and look at what is left",
                       place="UC-1:1" if top.hole.data.get("kite") else None)
    if shape == "altreq":
        return a.reply("answer", text="the game says why", check="trigger it and read the message")
    return a.reply("absent", reason="they do not conflict")


def selftest():
    import grow
    tmp = tempfile.mkdtemp(prefix="ask_selftest_")
    ok = True

    def check(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print("selftest: %-82s %s%s" % (name, "PASS" if good else "FAIL", "" if good else "  " + str(detail)[:400]))

    def fresh(name, draft=None):
        src, model = grow.write_fixture(os.path.join(tmp, name), draft if draft is not None else grow.fixture_draft())
        rc, said = grow.run_grow(src, model)
        assert rc in (0, 1), said
        return model

    def keys(folder):
        return [x.key for x in Queue(folder).questions()]

    def refused(fn):
        try:
            fn()
        except AnswerRefused as e:
            return str(e)
        return None

    # --- the detector, by table: each pair is a fact about statements, with the reason it should or should not fire
    table = [
        ("The board has 12 pits, six for each player.", "The board has 16 pits, eight for each player.", "number"),
        ("The board has 12 pits, six for each player.", "The board has 12 pits, six for each player.", None),
        ("A player cannot pick up from an empty pit.", "A player can pick up from an empty pit.", "polarity"),
        ("A player cannot pick up from an empty pit.", "A player cannot pick up from an empty pit.", None),
        ("A game has up to 4 players.", "A game has exactly 2 players.", None),
        ("Each pit starts with 4 seeds.", "A turn takes 4 seeds.", None),
        ("Two players take part.", "The game has a winner.", None),
        ("A runner moves 1 or 2 cells in a turn.", "A runner standing in a lit cell may move only 1 cell.", None),
        ("A runner moves 1 or 2 cells in a turn.", "A runner moves 3 cells in a turn.", "number"),
        ("A runner moves 1 to 3 cells in a turn.", "A runner moves 2 cells in a turn.", None),
        ("A runner moves 1 to 3 cells in a turn.", "A runner moves 5 cells in a turn.", "number"),
        ("A caravan carries at most 6 units.", "A caravan carries at most 8 units.", "number"),
        ("The road has 8 towns in a line. Both caravans start in town 1 with 20 coins.", "Town n buys and sells salt at n+1 coins per unit, from 2 coins in town 1 to 9 coins in town 8.", None),
    ]
    for a, b, want in table:
        got = Disagreement("a", a).against(Disagreement("b", b))
        check("DETECTOR %-60s -> %s" % ("%r vs %r" % (a[:22], b[:22]), want or "no question"), got == want, got)

    # --- the queue of a fresh model: shapes, ranking, plain words
    model = fresh("rank")
    q = Queue(model)
    shapes = collections.Counter(h.shape for h in q.open)
    check("QUEUE a fresh model: 2 needs, 2 goals with no alternate path, 8 groups of goal fields", shapes == {"need": 2, "noalt": 2, "facet": 8}, dict(shapes))
    bands = [h.band for h in q.open]
    check("QUEUE ranked by band: disagree, need, uncited, alternate path, goal fields", bands == sorted(bands), bands)
    within = all(a.dependents >= b.dependents for a, b in zip(q.open, q.open[1:]) if a.band == b.band)
    check("QUEUE inside a band, the answer more of the chain rests on comes first", within, [(h.key, h.dependents) for h in q.open])
    sentences = [x.sentence for x in q.questions()]
    check("WORDS no question sentence holds an internal id (%d sentences)" % len(sentences), not any(grow.INTERNAL_ID.search(s) for s in sentences),
          [s for s in sentences if grow.INTERNAL_ID.search(s)])
    check("WORDS every question sentence is a question, in plain words, ending in a mark", all(s.rstrip().endswith(("?", ".")) and len(s) > 25 for s in sentences))
    check("WORDS the first question names the step it comes from and what is needed", "Seeds are placed" in sentences[0] and "how many seeds each pit starts with" in sentences[0], sentences[0])
    check("QUEUE every question carries a key that names its place, stable across two reads", keys(model) == [x.key for x in Queue(model).questions()])

    # --- replies are refused unless they are whole
    r = refused(lambda: Asker(model).reply("absent"))
    check("REFUSED --absent with no reason (ABSENT without a reason is silence)", r and "reason" in r, r)
    r = refused(lambda: Asker(model).reply("later"))
    check("REFUSED --later with no reason", r and "reason" in r, r)
    r = refused(lambda: Asker(model).reply("fields", sets={"exec": "x"}))
    check("REFUSED a field the top question is not about", r and "not" in r, r)
    gulf_key = next(k for k in keys(model) if k.endswith(":gulf") and "UC-1" in k)
    r = refused(lambda: Asker(model).reply("fields", sets={"exec": "they press a pit"}, key=gulf_key))
    check("REFUSED a gulf answer with no check (a requirement needs a way to be seen to hold)", r and "--check" in r, r)
    before = {p: open(p, encoding="utf-8").read() for p in glob.glob(os.path.join(model, "*.yaml"))}
    check("REFUSED replies wrote nothing", before == {p: open(p, encoding="utf-8").read() for p in glob.glob(os.path.join(model, "*.yaml"))})

    # --- confirm a need: the requirement becomes authored with the producer as source; the question is gone and never returns
    k0 = keys(model)[0]
    e = Asker(model).reply("confirm", who="Sam")
    q2 = Queue(model)
    r6 = q2.model.requirements["R6"]
    check("CONFIRM a need: inferred -> authored, sourced to the producer's answer, recorded in the ledger",
          r6.kind == "authored" and any(s.startswith("ANS-") for s in r6.sources) and e["who"] == "Sam" and e["key"] == k0 and e["kind"] == "confirmed", (r6.kind, r6.sources, e))
    check("CONFIRM the answered question is not in the queue, and the model has no error for the drafter", k0 not in keys(model) and not q2.mine, (k0, q2.mine))

    # --- a contradiction an answer makes raises a question; a consistent answer raises none (control)
    m2 = fresh("contradict")
    Asker(m2).reply("fields", sets={"exec": "The board is laid out as 16 pits, eight for each player."}, check="count the pits", key="facet:UC-1:gulf")
    q = Queue(m2)
    top = q.questions()[0]
    check("CONTRADICT an answer saying 16 pits against the text's 12 -> a disagreement is the first question",
          top.hole.shape == "disagree" and "12 pits" in top.sentence and "16 pits" in top.sentence and "your answer to an earlier question" in top.sentence, top.sentence)
    m3 = fresh("consistent")
    Asker(m3).reply("fields", sets={"exec": "The board is laid out as 12 pits, six for each player."}, check="count the pits", key="facet:UC-1:gulf")
    check("CONTROL a consistent answer raises no disagreement", not any(h.shape == "disagree" for h in Queue(m3).open))
    m4 = fresh("polarity")
    Asker(m4).reply("fields", sets={"exec": "A player can pick up from an empty pit."}, check="pick from an empty pit", key="facet:UC-2:gulf")
    top = Queue(m4).questions()[0]
    check("CONTRADICT an answer permitting what the text forbids -> a disagreement", top.hole.shape == "disagree" and "cannot pick up" in top.sentence, top.sentence)
    # the ruling replaces both and the pair never returns
    pair = top.key
    refused_no_check = refused(lambda: Asker(m4).reply("answer", text="A player cannot pick up from an empty pit."))
    check("REFUSED a ruling with no check", refused_no_check and "--check" in refused_no_check, refused_no_check)
    Asker(m4).reply("answer", text="A player cannot pick up from an empty pit, whoever asks.", check="try it as each player")
    q4 = Queue(m4)
    carried = {r for u in q4.model.usecases.values() for s in u.steps for r in s.reqs}
    check("RULING both statements are replaced by the ruling, carried where they were, and the pair is closed",
          pair not in [h.key for h in q4.holes] and not q4.mine and "R3" not in q4.model.requirements and any(r.startswith("RA") and "whoever" in q4.model.requirements[r].text for r in carried),
          (pair, sorted(carried)))

    # --- uncited: a sentence nothing cites is asked, and answered into a requirement a step carries
    d = grow.fixture_draft()
    d["requirements"][4].update(inferred="step 1 of 'Start a game' seats two players and no sentence says how many", needs="how many players")
    d["requirements"][4].pop("cite")
    d["requirements"][4].pop("quote")
    m5 = fresh("uncited", d)
    q5 = Queue(m5)
    unc = [x for x in q5.questions() if x.hole.shape == "uncited"]
    check("UNCITED a sentence no requirement cites is a question quoting it and saying where it stands", len(unc) == 1 and "a game for two players" in unc[0].sentence and "line 1" in unc[0].sentence, [x.sentence for x in unc])
    r = refused(lambda: Asker(m5).reply("answer", text="Exactly two players take part.", check="count them", key=unc[0].key))
    check("REFUSED an answer with no --place lists the goals and steps to choose from", r and "Take a turn" in r and "--place" in r, r)
    Asker(m5).reply("answer", text="Exactly two players take part.", check="count them", place="UC-1:1", key=unc[0].key)
    q5 = Queue(m5)
    cell = q5.model.spaces["S1"].cells["1.1"]
    check("UNCITED the answer becomes a requirement that cites that sentence, is sourced to the answer, is carried by the step, and the cell is closed",
          cell.maps and cell.maps[0].startswith("RA") and "uncited:S1:1.1" not in keys(m5) and not q5.mine
          and "ANS-" in " ".join(q5.model.requirements[cell.maps[0]].sources), (cell.maps, q5.mine))
    r = refused(lambda: Asker(fresh("kite_place", d)).reply("answer", text="Exactly two players take part.", check="count them", place="EPIC-1:1", key="uncited:S1:1.1"))
    check("ROLLBACK/REFUSED a requirement placed on a step of the whole epic is refused", r and "epic" in r, r)

    # --- no alternate path: an answer adds one, with its cause, its step and where it ends
    m6 = fresh("noalt")
    Asker(m6).reply("answer", text="Play stops and the stores are compared.", when="a player leaves", at=2, out="end minimal", check="leave mid-game", key="noalt:UC-3")
    q6 = Queue(m6)
    uc3 = q6.model.usecases["UC-3"]
    check("NOALT an answer gives the goal an alternate path at the step named, ending as stated, carrying a new requirement",
          len(uc3.ext) == 1 and uc3.ext[0].at == 2 and uc3.ext[0].out == "end minimal" and uc3.ext[0].reqs and "noalt:UC-3" not in keys(m6) and not q6.mine, (uc3.ext, q6.mine))
    r = refused(lambda: Asker(m6).reply("answer", text="x", when="y", at=2, out="sideways", check="z", key="noalt:EPIC-1"))
    check("REFUSED an outcome that is not rejoin N / end success / end minimal", r and "--out" in r, r)

    # --- facets: asked together, a partial reply leaves the rest, the rest never repeat
    m7 = fresh("facets")
    k = "facet:UC-2:text"
    Asker(m7).reply("fields", sets={"trigger": "the previous turn ends", "pre": "both players are seated"}, key=k)
    rest = next(x for x in Queue(m7).questions() if x.key == k)
    check("FACETS a partial reply settles those fields; the question is asked again about the rest only",
          rest.hole.data["fields"] == ["minimal", "success", "why"] and "what starts it" not in rest.sentence and "how you will know" in rest.sentence, rest.sentence)
    Asker(m7).reply("fields", sets={"success": "the seeds are sown"}, nones={"minimal": "a refused choice changes nothing", "why": "none"}, key=k)
    check("FACETS --none records a field with the producer's reason; the question closes", k not in keys(m7))
    uc2 = Queue(m7).model.usecases["UC-2"]
    check("FACETS the fields hold the producer's words, and the none holds its reason", uc2.raw["trigger"] == "the previous turn ends" and uc2.raw["minimal"].startswith("none - "), uc2.raw)

    # --- an answered question never returns, even if the hole is put back
    m8 = fresh("never")
    Asker(m8).reply("confirm")
    Asker(m8).reply("confirm")
    snapshot = os.path.join(tmp, "never_snapshot")
    shutil.copytree(m8, snapshot)
    # put the holes back: restore the pre-answer model files, keep only the ledger
    src8, model8 = grow.write_fixture(os.path.join(tmp, "never_original"), grow.fixture_draft())
    grow.run_grow(src8, model8)
    for f in ("requirements.yaml", "usecases.yaml", "spaces.yaml", "people.yaml", "meta.yaml"):
        shutil.copy2(os.path.join(model8, f), os.path.join(m8, f))
    back = Queue(m8)
    check("NEVER RETURNS a question answered earlier stays out of the queue though the model is put back to the state that raised it",
          any(h.key == "need:R6" for h in back.holes) and "need:R6" not in [h.key for h in back.open] and "need:R7" not in [h.key for h in back.open],
          ([h.key for h in back.holes][:4], [h.key for h in back.open][:4]))

    # --- resume: a session that stops and starts again asks what an uninterrupted one would have asked next, and repeats nothing
    straight = fresh("straight")
    seq_a, sent_a = [], []
    for _ in range(5):
        qa = Queue(straight)
        seq_a.append(qa.questions()[0].key)
        sent_a.append(qa.questions()[0].sentence)
        scripted_reply(straight, qa)
    split = fresh("split")
    seq_b = []
    for _ in range(2):
        qb = Queue(split)
        seq_b.append(qb.questions()[0].key)
        scripted_reply(split, qb)
    import subprocess
    out1 = subprocess.run([sys.executable, "-X", "utf8", os.path.join(HERE, "ask.py"), "ask", split], capture_output=True, text=True, encoding="utf-8")
    for _ in range(3):
        qb = Queue(split)
        seq_b.append(qb.questions()[0].key)
        scripted_reply(split, qb)
    check("RESUME a session stopped after two answers and restarted as a separate process prints the third question, exit 1 (one is open)",
          out1.returncode == 1 and sent_a[2] in out1.stdout and seq_a[0] not in out1.stdout.split("for the agent")[0], out1.stdout[:300] + out1.stderr[:200])
    check("RESUME the five questions asked across the stop are the five an uninterrupted run asks, in order, none twice", seq_a == seq_b and len(set(seq_b)) == 5, (seq_a, seq_b))

    # --- a need the plan does not have: dropped when something else still carries its steps, refused when a step would be left with nothing
    d8 = grow.fixture_draft()
    d8["requirements"].append({"id": "R8", "inferred": "step 3 of 'Take a turn' hands the move on, and no sentence says whose turn comes next", "needs": "whose turn comes next",
                               "says": "After the seeds are sown the turn passes to the other player.", "check": "play two turns and see who moves"})
    d8["goals"][1]["steps"][2]["reqs"].append("R8")
    m8b = fresh("drop", d8)
    r = refused(lambda: Asker(m8b).reply("absent", reason="not needed", key="need:R6"))
    check("REFUSED dropping a need that is the only thing a step carries names the step and offers an answer", r and "Start a game" in r and "--answer" in r, r)
    Asker(m8b).reply("absent", reason="the game has no such rule", key="need:R8")
    q8 = Queue(m8b)
    check("ABSENT a need the plan does not have is dropped from the model and from every step; the steps keep their other requirements",
          "R8" not in q8.model.requirements and q8.model.usecases["UC-2"].steps[2].reqs == ["R2"] and "need:R8" not in keys(m8b) and not q8.mine, q8.mine)
    last = [e for e in q8.ledger.entries if e["key"] == "need:R8"][-1]
    check("ABSENT the dropped requirement's words and the reason stay on record in the ledger", last["removed"]["text"].startswith("After the seeds") and last["reason"] == "the game has no such rule", last)

    # --- a vague word in a sentence a requirement stands on is asked about, and the amount is carried where the sentence's requirement is
    text_v = grow.FIXTURE_TEXT.replace("picks up every seed in one of their own pits", "picks up a few seeds from one of their own pits")
    dv = grow.fixture_draft()
    dv["requirements"][1].update(quote="picks up a few seeds from one of their own pits", says="A turn picks up a few seeds from one of the player's own pits.")
    srcv, mv = grow.write_fixture(os.path.join(tmp, "vague"), dv, text_v)
    grow.run_grow(srcv, mv)
    qv = Queue(mv)
    vq = [x for x in qv.questions() if x.hole.shape == "vague"]
    check("VAGUE a cited sentence with a weak word is a question quoting the sentence, naming the word, with no id in it",
          len(vq) == 1 and '"a few"' in vq[0].sentence and "picks up a few seeds" in vq[0].sentence and not grow.INTERNAL_ID.search(vq[0].sentence), [x.sentence for x in vq])
    order = [x.hole.shape for x in qv.questions()]
    check("VAGUE ranks after the needs and before the goals without an alternate path", order.index("vague") > order.index("need") and order.index("vague") < order.index("noalt"), order[:6])
    check("VAGUE the control text (no weak word) raises none", not any(h.shape == "vague" for h in Queue(fresh("novague")).holes))
    e = Asker(mv).reply("answer", text="Exactly three seeds, never more.", check="pick up four seeds: refused", key=vq[0].key)
    qv = Queue(mv)
    newr = e["made"][0]
    carriers = [(u, w) for u, w, _, rs in qv.pathways() if newr in rs]
    base = [(u, w) for u, w, _, rs in qv.pathways() if "R2" in rs]
    check("VAGUE the amount becomes a requirement sourced to the answer, cites that sentence, and is carried by every step that carries the sentence's own requirement",
          carriers == base and carriers and vq[0].key not in keys(mv) and not qv.mine and "3.1" in qv.model.spaces["S1"].cells and newr in qv.model.spaces["S1"].cells["3.1"].maps, (carriers, base))
    mv2 = fresh("vague_absent", dv) if False else None
    srcv2, mv2 = grow.write_fixture(os.path.join(tmp, "vague2"), dv, text_v)
    grow.run_grow(srcv2, mv2)
    before = {p: open(p, encoding="utf-8").read() for p in glob.glob(os.path.join(mv2, "*.yaml")) if not p.endswith("answers.yaml")}
    Asker(mv2).reply("absent", reason="the builder may choose how many", key="vague:S1:3.1")
    after = {p: open(p, encoding="utf-8").read() for p in glob.glob(os.path.join(mv2, "*.yaml")) if not p.endswith("answers.yaml")}
    check("VAGUE ABSENT with a reason closes the question and changes nothing in the model", before == after and "vague:S1:3.1" not in keys(mv2))

    # --- the transaction: a reply whose changes would break the chain is rolled back whole, whatever wrote it
    m14 = fresh("rollback")
    qr = Queue(m14)
    asker = Asker(m14)
    files = qr.files
    _, uc_raw = files.find("usecases", "UC-2")
    uc_raw["steps"][0]["reqs"].append("R404")
    files.touch(files.find("usecases", "UC-2")[0])
    on_disk = {p: open(p, encoding="utf-8").read() for p in glob.glob(os.path.join(m14, "*.yaml"))}
    r = refused(lambda: asker.commit(qr, files, qr.questions()[0], {}))
    check("ROLLBACK a change that would leave the chain with an error is refused and every file is put back exactly",
          r and "nothing was recorded" in r and on_disk == {p: open(p, encoding="utf-8").read() for p in glob.glob(os.path.join(m14, "*.yaml"))}, r)

    # --- parked questions stay open and come back last
    m9 = fresh("later")
    first = keys(m9)[0]
    Asker(m9).reply("later", reason="ask the designer on Monday")
    after = keys(m9)
    check("LATER a parked question stays counted open, but is asked after every question that is not parked", after[-1] == first and after[0] != first and len(after) == len(keys(fresh("later_ref"))), (first, after[:3]))

    # --- it stops when the queue is empty; stopped early it says how many are open; a model broken by the drafter is not questioned
    m10 = fresh("empty")
    n0 = len(Queue(m10).open)
    asked = []
    for _ in range(200):
        qq = Queue(m10)
        if not qq.open:
            break
        asked.append(qq.questions()[0].key)
        scripted_reply(m10, qq)
    lines = []
    rc = show_next(m10, out=lines.append)
    final = Queue(m10)
    check("EMPTY answering every question empties the queue: exit 0, nothing repeated, %d asked of %d first open (answers add questions too)" % (len(asked), n0),
          rc == 0 and not final.open and len(asked) == len(set(asked)) and "nothing is open" in "\n".join(lines), (rc, asked[-3:], lines[:3]))
    check("EMPTY the 19 rules then PASS, every hole answered with words (nothing was absent)", not final.findings and any("PASS" in l for l in lines), [str(f) for f in final.findings[:3]])
    m11 = fresh("early")
    scripted_reply(m11, Queue(m11))
    scripted_reply(m11, Queue(m11))
    qe = Queue(m11)
    st = status_lines(qe, m11)
    check("STATUS stopped early says how many are open and of what kind", st[0].startswith("%d open question(s)" % len(qe.open)) and "goals with no alternate path" in st[0] or "fields" in st[0], st)
    m12 = fresh("allabsent")
    for _ in range(200):
        qq = Queue(m12)
        if not qq.open:
            break
        top = qq.questions()[0]
        if top.hole.shape == "facet":
            Asker(m12).reply("fields", nones={f: "nothing to say" for f in top.hole.data["fields"]})
        elif top.hole.shape == "need":
            Asker(m12).reply("confirm")
        else:
            Asker(m12).reply("absent", reason="nothing there")
    lines = []
    rc = show_next(m12, out=lines.append)
    rules_left = sorted({f.rule for f in Queue(m12).findings})
    check("ABSENT every question answered ABSENT empties the queue, and says which rules stay open by rule (U13, U17 have no place for an absence)",
          rc == 0 and set(rules_left) <= {"U13", "U17"} and "stay open by rule" in "\n".join(lines), (rc, rules_left))
    m13 = fresh("broken")
    p = os.path.join(m13, "usecases.yaml")
    t = io.open(p, encoding="utf-8").read().replace("R5", "R404", 1)
    io.open(p, "w", encoding="utf-8", newline="\n").write(t)
    lines = []
    rc = show_next(m13, out=lines.append)
    check("BROKEN a chain with an error only the drafter can fix is not questioned", rc == 1 and "only the drafter can fix" in "\n".join(lines) and "QUESTION" not in "\n".join(lines), lines[:3])

    # --- a hand-written model: the kit's own example has one inferred requirement, and the same queue asks it
    ex = next((os.path.join(b, "example_py", "model") for b in (HERE, os.path.join(HERE, "..", "closed_loop_kit")) if os.path.isdir(os.path.join(b, "example_py", "model"))), None)
    if ex is None:
        print("selftest: EXAMPLE skipped: example_py/model is not beside this tool (not counted as a pass)")
    else:
        copy_ex = os.path.join(tmp, "example_py_model")
        shutil.copytree(ex, copy_ex, ignore=shutil.ignore_patterns("*.lock", "generated_*"))
        before = {str(f) for f in ucm.UseCaseModel.load(copy_ex).check()}
        qx = Queue(copy_ex)
        check("EXAMPLE a hand-written model: its one inferred requirement is the first question, in plain words",
              qx.open and qx.open[0].key == "need:W-4" and "refused" in qx.questions()[0].sentence and not grow.INTERNAL_ID.search(qx.questions()[0].sentence), [x.sentence for x in qx.questions()[:2]])
        Asker(copy_ex).reply("confirm")
        after = {str(f) for f in ucm.UseCaseModel.load(copy_ex).check()}
        check("EXAMPLE confirming it adds no rule hole to the hand-written model", after <= before, after - before)
    shutil.rmtree(tmp, ignore_errors=True)
    print("selftest: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["ask"])
    ap.add_argument("model", nargs="?")
    ap.add_argument("--answer")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--absent")
    ap.add_argument("--later")
    ap.add_argument("--check")
    ap.add_argument("--place")
    ap.add_argument("--when")
    ap.add_argument("--at", type=int)
    ap.add_argument("--out")
    ap.add_argument("--who")
    ap.add_argument("--set", action="append", default=[], metavar='FIELD="TEXT"', help="a goal's field, in the producer's words (repeatable)")
    ap.add_argument("--none", action="append", default=[], metavar='FIELD="WHY"', help="a goal's field that does not exist, with the reason (repeatable)")
    ap.add_argument("--key", help="answer this open question rather than the top one")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd != "ask" or not a.model:
        ap.error("ask <model folder>, or --selftest")
    folder = os.path.abspath(a.model)
    if not os.path.isdir(folder):
        print("ask: %s is not a folder" % a.model)
        return 2
    def pairs(items):
        out = collections.OrderedDict()
        for it in items:
            k, eq, v = it.partition("=")
            if not eq:
                raise SystemExit("ask: %r should be FIELD=\"TEXT\"" % it)
            out[k.strip()] = v.strip()
        return out
    sets, nones = pairs(a.set), pairs(a.none)
    modes = [m for m, v in (("answer", a.answer), ("confirm", a.confirm or None), ("absent", a.absent), ("later", a.later), ("fields", (sets or nones) or None)) if v is not None]
    if len(modes) > 1:
        print("ask: one reply at a time: %s" % ", ".join("--" + m for m in modes))
        return 1
    try:
        if a.status or a.list:
            q = Queue(folder)
            for l in status_lines(q, a.model):
                print(l)
            if a.list:
                for i, x in enumerate(q.questions(), 1):
                    print("%3d. [%s%s] %s" % (i, LABELS[x.hole.shape][0], ", parked" if x.key in q.parked else "", x.sentence))
            if q.mine:
                print("the model has %d error(s) only the drafter can fix; no question is asked until they are" % len(q.mine))
            return 1 if q.open or q.mine else 0
        if modes:
            mode = modes[0]
            try:
                e = Asker(folder, a.model).reply(mode, text=a.answer, check=a.check, reason=a.absent if mode == "absent" else a.later, place=a.place,
                                                 when=a.when, at=a.at, out=a.out, who=a.who, key=a.key, sets=sets, nones=nones)
            except AnswerRefused as err:
                print("ask: NOT RECORDED: %s" % err)
                return 1
            print("recorded answer %d (%s): %s" % (e["n"], e["kind"], "; ".join(e["changed"]) or "parked"))
            print("")
        return show_next(folder, display=a.model)
    except (RuntimeError, OSError, yaml.YAMLError) as err:
        print("ask: cannot read the model %s: %s" % (a.model, err))
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
