#!/usr/bin/env python3
"""usecase_model.py - a use case model kept as DATA, and the gap checker that reads it.

WHY THIS EXISTS (2026-10-27): the markdown gap checker was suspected of missing holes, perhaps hundreds. Use case descriptions
also needed a hierarchy (child use cases under their parent's steps), and every pathway of a use case description, not only the
main scenario, has to be accounted for, because between them the pathways should carry every possible requirement.

MEASURED BEFORE IT WAS WRITTEN: a model the markdown checker
passed (interrogate.py 25 questions, coverage.py 20/20, source_rules.py 0 holes) had 2 of 20 requirements that no step
carried, 14 of 14 extension pathways that named no requirement, 10 of 14 with no stated outcome, and 22 of 44 source
sentences never checked, because the rule-word filter does not fire on an essay. The markdown checker reads table rows;
extensions, the hierarchy and who-carries-which-requirement are prose to it. Here they are fields, so each one is counted.

THE MODEL: a folder of YAML files. Any file may hold any of these top-level keys; lists are concatenated across files.
  meta:          {subject, mode, story: {text, clauses: [{text, source}]}}
  actors:        [{id, name, kind: person | ai | system, profile}]
  sources:       [{id, kind: prose | quote | standing | framework, path (prose), text, cite}]
  spaces:        [{id, title, basis: prose | framework | language | brainstorm | code, kind: sentences | grid | extract | list,
                   source (sentences), dims + surfaces (grid), file + pattern (extract),
                   cells: {key: [req ids] | {maps: [req ids], text} | {excluded: reason}}}]
  requirements:  [{id, kind: authored | derived | inferred, text, measure, sources: [source ids], why (inferred),
                   effect: [{sets | calls | branches | reads: <name>, to, args, by (sets: through a call),
                             at: <graph>[#sN[:<outcome>]]}] | {absent: why}}]
  terms:         [{term, binds: [code names], why}]   a word of the requirements bound to what it means in the code
  meta.loop:     {closed: true, entry: [{name, why}]}  held to the closed-loop criteria (usecase_kg.py loop); entry names are
                   conditions set outside the model (an engine call, another Blueprint), each with the reason
  usecases:      [{id, name, level: kite | sea | fish, parent: {uc, step | ext}, actor, supporting, trigger, pre, minimal,
                   success, why, where, gulf: {exec: {text, reqs}, eval: {text, reqs}} (kite and sea),
                   steps: [{do, reqs, includes}], ext: [{at, when, do: [...], includes, out: "rejoin N" | "end success" |
                   "end minimal", reqs}]}]
  questions:     [{id, persona, text, inverse}]   the brainstorm: each question is answered by (inverted into) a use case

THE RULES. Every one is a code, and --selftest plants one fault per code and requires that code to fire:
  U1  FIELDS     a use case lacks a field, has a bad level, or a main scenario outside 3-11 steps (2-11 for fish)
  U2  PARENT     a sea or fish use case has no parent, the parent or its step is missing, a child ranks above its parent,
                 or the parents form a cycle
  U3  KITE       a kite step decomposes into nothing: no child use case and no include
  U4  EXTENSION  an alternate pathway lacks its step, its condition, its handling, a valid outcome or a requirement
  U5  CARRIER    a requirement that no step and no extension carries - a requirement no pathway realises
  U6  SOURCE     a requirement with no source: authored without an authored source, derived without a space cell,
                 inferred without a reason
  U7  CELL       a closed space's cell is missing, unmapped, mapped to an unknown requirement, or excluded with no reason
  U8  REF        an id that resolves to nothing (requirement, include, actor, persona, source)
  U9  ACTOR      a person who is primary actor of no use case
  U10 MEASURE    a requirement with no way to check it
  U11 QUESTION   a brainstormed question that inverts into no use case
  U12 STEP       a step that carries no requirement and includes nothing that does
  U13 GULF       a kite or sea use case that does not say how the person acts (execution) and how they see the result
                 (evaluation), each with requirements carried in its own subtree (Norman's two gulfs)
  U14 STORY      a problem story clause with no source
  U15 DUP        two things with one id
  U16 KITE-ONLY  a requirement carried only by kite (summary) steps, realised by no user goal or subfunction
  U17 NO-ALT     a use case of 3+ steps with no alternate pathway: what happens when a step cannot complete was never written
  U18 EFFECT     in a closed-loop model, a requirement with no effect and no reason it has none, or an assertion with no
                 single verb, no place, or a `to`/`args` its verb does not take
  U19 TERM       a term that binds no code name, or an entry condition with no reason
INFERRED requirements are not holes: they are the questions only the owner can answer, and are listed as such.

    python -X utf8 usecase_model.py <model folder> [--max 20] [--json out.json]
    python -X utf8 usecase_model.py --selftest

Exit 0: no holes. Exit 1: holes (each printed with its rule and where). Exit 2: the model cannot be read.
"""
import argparse
import collections
import copy
import glob
import io
import json
import os
import re
import sys
import tempfile

try:
    import yaml
except ImportError:  # the model is YAML: without a reader nothing can be checked, and a silent pass would be a lie
    yaml = None

LEVELS = {"kite": 0, "sea": 1, "fish": 2}
REQ_KINDS = ("authored", "derived", "inferred")
AUTHORED_SOURCES = ("prose", "quote", "standing")
OUT_RE = re.compile(r"^\s*(?:rejoin\s+(\d+)|end\s+(success|minimal))\s*$")
PLACEHOLDER = re.compile(r"^\s*(?:todo|tbd|\?+|-|n/?a)?\s*$", re.I)
ABBREV = re.compile(r"\b(?:vs|e\.g|i\.e|etc)\.$", re.I)
RULES = {
    "U1": "FIELDS", "U2": "PARENT", "U3": "KITE", "U4": "EXTENSION", "U5": "CARRIER", "U6": "SOURCE", "U7": "CELL",
    "U8": "REF", "U9": "ACTOR", "U10": "MEASURE", "U11": "QUESTION", "U12": "STEP", "U13": "GULF", "U14": "STORY",
    "U15": "DUP", "U16": "KITE-ONLY", "U17": "NO-ALT", "U18": "EFFECT", "U19": "TERM",
}


def blank(value):
    """a field that says nothing: missing, empty, or a placeholder such as TODO"""
    return value is None or (isinstance(value, str) and PLACEHOLDER.match(value) is not None)


def as_list(value):
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def sentences(path):
    """Every sentence of a prose source, keyed "line.k": headings and rule lines skipped, a bullet's lead label kept with
    its sentence, split at a full stop, question or exclamation mark followed by a space and a capital, never after an
    abbreviation. The checker splits the source itself so a sentences space cannot skip one quietly."""
    out = []
    for n, raw in enumerate(io.open(path, encoding="utf-8").read().splitlines(), 1):
        t = raw.strip()
        if not t or re.fullmatch(r"[=\-]{3,}", t) or (t.upper() == t and re.search(r"[A-Z]{3}", t)):
            continue
        t = re.sub(r"^[*\-]\s*", "", t)
        start, k = 0, 0
        for m in re.finditer(r"[.!?](?=\s+[A-Z(\"])", t):
            piece = t[start:m.end()].strip()
            if ABBREV.search(piece):
                continue
            k += 1
            out.append(("%d.%d" % (n, k), piece))
            start = m.end()
        if t[start:].strip():
            k += 1
            out.append(("%d.%d" % (n, k), t[start:].strip()))
    return out


class Finding(object):
    """One hole: the rule it breaks, where in the model, and what is wrong."""

    def __init__(self, rule, where, message):
        self.rule, self.where, self.message = rule, where, message

    def __str__(self):
        return "[%s %s] %s: %s" % (self.rule, RULES[self.rule], self.where, self.message)


class Actor(object):
    """Someone or something that takes part: a person (with ergonomic needs), an AI, or a system."""

    def __init__(self, d):
        self.id, self.name, self.kind, self.profile = d.get("id"), d.get("name"), d.get("kind"), d.get("profile")


class Source(object):
    """Where an authored requirement comes from: a prose file (every sentence counts), a quote, a standing rule, a framework."""

    def __init__(self, d):
        self.id, self.kind, self.path = d.get("id"), d.get("kind"), d.get("path")
        self.text, self.cite = d.get("text"), d.get("cite")


class Cell(object):
    """One point of a closed space, accounted for by the requirements it maps to, or excluded with a reason."""

    def __init__(self, space, key, raw):
        self.space, self.key = space, str(key)
        self.maps, self.excluded, self.text = [], None, None
        if isinstance(raw, dict):
            self.maps, self.excluded, self.text = as_list(raw.get("maps")), raw.get("excluded"), raw.get("text")
        else:
            self.maps = as_list(raw)

    @property
    def ref(self):
        return "%s:%s" % (self.space, self.key)


class Space(object):
    """A closed set the requirements are measured against. Its keys come from somewhere other than the requirements:
    the sentences of a prose source, a cross product of a framework's dimensions and the view's surfaces, or names
    extracted from a file. A list space is closed only by its own provenance, and says so."""

    def __init__(self, d):
        self.id, self.title, self.basis, self.kind = d.get("id"), d.get("title"), d.get("basis"), d.get("kind")
        self.source, self.file, self.pattern, self.dir = d.get("source"), d.get("file"), d.get("pattern"), d.get("dir")
        self.dims = [str(x["id"] if isinstance(x, dict) else x) for x in as_list(d.get("dims"))]
        self.surfaces = [str(x["id"] if isinstance(x, dict) else x) for x in as_list(d.get("surfaces"))]
        self.cells = collections.OrderedDict((str(k), Cell(self.id, k, v)) for k, v in (d.get("cells") or {}).items())
        self.base = d.get("_base")

    def expected_keys(self, model):
        """the keys this space must hold, or None for a list space; raises ValueError when they cannot be read"""
        if self.kind == "grid":
            return ["%s.%s" % (a, b) for a in self.dims for b in self.surfaces]
        if self.kind == "sentences":
            src = model.sources.get(self.source)
            if src is None or not src.path:
                raise ValueError("sentences space %s names source %r, which has no path" % (self.id, self.source))
            return [k for k, _ in sentences(model.resolve(src.path))]
        if self.kind == "extract":
            text = io.open(model.resolve(self.file), encoding="utf-8").read()
            seen = []
            for m in re.finditer(self.pattern, text):
                if m.group(1) not in seen:
                    seen.append(m.group(1))
            if not seen:
                raise ValueError("extract space %s: pattern matched nothing in %s" % (self.id, self.file))
            return seen
        if self.kind == "files":
            # the keys are FILE NAMES in a folder (group 1 of the pattern, matched whole): a vocabulary that lives one class per
            # file, e.g. the Blueprint language's K2Node_*.h in the engine - a class added there is a missing cell here
            root = model.resolve(self.dir)
            if not os.path.isdir(root):
                raise ValueError("files space %s: %s is not a folder" % (self.id, root))
            keys = []
            for f in sorted(os.listdir(root)):
                m = re.fullmatch(self.pattern, f)
                if m:
                    keys.append(m.group(1))
            if not keys:
                raise ValueError("files space %s: no file in %s matches %s" % (self.id, root, self.pattern))
            return keys
        return None


class Requirement(object):
    """Something the product must do, with how to check it and where it comes from. Authored: the owner's words. Derived:
    a framework's cell applied to this product. Inferred: the chain needs it and nobody stated it - a question."""

    def __init__(self, d):
        self.id, self.kind, self.text, self.measure = d.get("id"), d.get("kind"), d.get("text"), d.get("measure")
        self.sources, self.why = as_list(d.get("sources")), d.get("why")
        # the tests that check it, by name (Fixture.Test or Test): the closed loop's L16 holds them to its code - when its code is
        # broken, one of THESE must fail, not merely some test (10-29)
        self.checks = [str(c) for c in as_list(d.get("checks"))]
        e = d.get("effect")
        self.absent = e.get("absent") if isinstance(e, dict) else None
        self.effect = [] if isinstance(e, dict) else [Assertion(self.id, i + 1, a) for i, a in enumerate(as_list(e))]


class Assertion(object):
    """One claim of a requirement's EFFECT, in the code's own terms: at a place - a graph `G`, a statement `G#sN`, or one outcome
    of a decision `G#sN:then` - the code SETS a name (to a value), CALLS a function (with these arguments), BRANCHES on a
    condition, or READS a name. It is what an AI must make true when it writes the code, and the oracle that checks the code it
    wrote: the closed loop's unit (in a closed loop the requirements, not the code, are what an AI acts on). A `sets` is a
    postcondition and a `branches` a precondition, so causal closure and contradiction are queries over these, not prose."""

    VERBS = ("sets", "calls", "branches", "reads")

    def __init__(self, req, n, d):
        self.req, self.n, self.raw = req, n, d if isinstance(d, dict) else {}
        verbs = [v for v in self.VERBS if v in self.raw]
        self.verbs = verbs
        self.verb = verbs[0] if len(verbs) == 1 else None
        self.target = str(self.raw[self.verb]) if self.verb else None
        self.to, self.at = self.raw.get("to"), self.raw.get("at")
        self.args = [str(a) for a in as_list(self.raw.get("args"))]
        # `sets X by F`: the statement calls F with X among its arguments - a write through a call, the only way a Blueprint
        # changes an array (Array_Add, Array_RemoveItem). Without it no requirement could be seen producing a list, and the
        # loop's causal criterion said so on its first run over the online sample (10-28: Out_Actors, ConeActors)
        self.by = self.raw.get("by")

    @property
    def id(self):
        return "%s#%d" % (self.req, self.n)

    def text(self):
        """the canonical form, one line: what the graph stores and the fingerprint reads"""
        return "%s %s%s%s%s at %s" % (self.verb, self.target, "" if self.by is None else " by %s" % self.by,
                                       "" if self.to is None else " to %s" % self.to,
                                       "" if not self.args else " with %s" % ", ".join(self.args), self.at)


class Term(object):
    """One word of the requirements bound to the code names it means (`lock range` -> LockOnSphere): a requirement an AI acts on
    must mean one thing, and a word two readers bind differently is a mistake the requirements carry, not the code."""

    def __init__(self, d):
        self.id, self.binds, self.why = d.get("term"), [str(b) for b in as_list(d.get("binds"))], d.get("why")


class Step(object):
    """One step of a main scenario: what happens, the requirements it carries, and a use case it includes."""

    def __init__(self, n, d):
        self.n, self.text = n, d.get("do")
        self.reqs, self.includes = as_list(d.get("reqs")), as_list(d.get("includes"))


class Extension(object):
    """An alternate pathway: at which step, on what condition, what happens, and where it ends - back into the main
    scenario at a step, or out of the use case with its success or minimal guarantee."""

    def __init__(self, uc, letter, d):
        self.uc, self.at, self.letter = uc, d.get("at"), letter
        self.when, self.do, self.includes = d.get("when"), as_list(d.get("do")), as_list(d.get("includes"))
        self.out, self.reqs = d.get("out"), as_list(d.get("reqs"))

    @property
    def id(self):
        return "%s%s" % (self.at, self.letter)


class Pathway(object):
    """One way through a use case - its main scenario or one extension - and the requirements it carries."""

    def __init__(self, uc, name, reqs):
        self.uc, self.name, self.reqs = uc, name, reqs


class UseCase(object):
    """A goal of an actor at one level (kite: summary, sea: user goal, fish: subfunction), linked to its parent step so
    the hierarchy is a tree of goals, each with its main scenario and every alternate pathway."""

    FIELDS = ("id", "name", "level", "actor", "trigger", "pre", "minimal", "success", "why")

    def __init__(self, d):
        self.raw = d
        self.id, self.name, self.level, self.actor = d.get("id"), d.get("name"), d.get("level"), d.get("actor")
        self.parent, self.supporting = d.get("parent") or {}, as_list(d.get("supporting"))
        self.gulf = d.get("gulf") or {}
        self.steps = [Step(i + 1, s) for i, s in enumerate(as_list(d.get("steps")))]
        letters, self.ext = collections.Counter(), []
        for e in as_list(d.get("ext")):
            at = e.get("at")
            self.ext.append(Extension(self.id, "abcdefghijklmnopqrstuvwxyz"[letters[at]], e))
            letters[at] += 1

    def pathways(self):
        main = sorted({r for s in self.steps for r in s.reqs})
        return [Pathway(self.id, "main", main)] + [Pathway(self.id, e.id, sorted(set(e.reqs))) for e in self.ext]


class Question(object):
    """A question a persona brings to the view, inverted into the use case that answers it."""

    def __init__(self, d):
        self.id, self.persona, self.text, self.inverse = d.get("id"), d.get("persona"), d.get("text"), d.get("inverse")


class UseCaseModel(object):
    """The whole model: loads a folder of YAML, checks it against the rules, and counts what it holds."""

    def __init__(self, root="."):
        self.root = root
        self.meta, self.actors, self.sources, self.spaces = {}, {}, {}, collections.OrderedDict()
        self.requirements, self.usecases, self.questions = collections.OrderedDict(), collections.OrderedDict(), []
        self.terms = collections.OrderedDict()
        self.dups = []

    @property
    def loop(self):
        """meta.loop: {closed, entry} - a model held to the closed-loop criteria says so; entry names are set outside it"""
        return self.meta.get("loop") or {}

    @property
    def closed_loop(self):
        return bool(self.loop.get("closed"))

    def resolve(self, path):
        return path if os.path.isabs(path) else os.path.normpath(os.path.join(self.root, path))

    @classmethod
    def load(cls, folder):
        if yaml is None:
            raise RuntimeError("PyYAML is not installed - the model cannot be read, so nothing can be checked")
        files = sorted(glob.glob(os.path.join(folder, "**", "*.yaml"), recursive=True))
        if not files:
            raise RuntimeError("no .yaml files under %s" % folder)
        data = collections.defaultdict(list)
        meta = {}
        for f in files:
            d = yaml.safe_load(io.open(f, encoding="utf-8")) or {}
            meta.update(d.get("meta") or {})
            for key in ("actors", "sources", "spaces", "requirements", "usecases", "questions", "terms"):
                for item in as_list(d.get(key)):
                    item = dict(item)
                    item["_file"] = os.path.relpath(f, folder)
                    data[key].append(item)
        m = cls.from_data(meta, data, root=folder)
        m.merge_recorded_checks()
        return m

    def merge_recorded_checks(self):
        """the tests a pass's verdicts show catching each requirement's breaks (record_checks.py writes <model>/record_checks.json from them): added to
        what a requirement names itself, never replacing it. An id the model no longer holds is ignored here and listed by record_checks --check"""
        path = os.path.join(self.root, "record_checks.json")
        if not os.path.isfile(path):
            return
        for rid, names in (json.load(io.open(path, encoding="utf-8")).get("checks") or {}).items():
            if rid in self.requirements:
                r = self.requirements[rid]
                r.checks = sorted(set(r.checks) | set(str(n) for n in names))

    @classmethod
    def from_data(cls, meta, data, root="."):
        m = cls(root)
        m.meta = meta or {}

        def put(table, obj, kind):
            if obj.id in table:
                m.dups.append((kind, obj.id))
            table[obj.id] = obj
        for d in data.get("actors", []):
            put(m.actors, Actor(d), "actor")
        for d in data.get("sources", []):
            put(m.sources, Source(d), "source")
        for d in data.get("spaces", []):
            put(m.spaces, Space(d), "space")
        for d in data.get("requirements", []):
            put(m.requirements, Requirement(d), "requirement")
        for d in data.get("usecases", []):
            put(m.usecases, UseCase(d), "use case")
        for d in data.get("terms", []):
            put(m.terms, Term(d), "term")
        seen = set()
        for d in data.get("questions", []):
            q = Question(d)
            if q.id in seen:
                m.dups.append(("question", q.id))
            seen.add(q.id)
            m.questions.append(q)
        return m

    # ------------------------------------------------------------------ derived tables

    def children(self):
        """(parent id, 'step'|'ext', step number or extension id) -> [child ids]"""
        out = collections.defaultdict(list)
        for uc in self.usecases.values():
            p = uc.parent
            if p.get("uc"):
                where = ("step", p.get("step")) if p.get("step") is not None else ("ext", str(p.get("ext")))
                out[(p["uc"],) + where].append(uc.id)
        return out

    def carriers(self):
        """requirement id -> [use case:pathway] that carry it"""
        out = collections.defaultdict(list)
        for uc in self.usecases.values():
            for s in uc.steps:
                for r in s.reqs:
                    out[r].append("%s step %d" % (uc.id, s.n))
            for e in uc.ext:
                for r in e.reqs:
                    out[r].append("%s %s" % (uc.id, e.id))
        return out

    def mapped_from(self):
        """requirement id -> [space cell refs that map to it]"""
        out = collections.defaultdict(list)
        for sp in self.spaces.values():
            for c in sp.cells.values():
                for r in c.maps:
                    out[r].append(c.ref)
        return out

    def subtree_reqs(self, uc_id):
        kids = collections.defaultdict(list)
        for uc in self.usecases.values():
            if uc.parent.get("uc"):
                kids[uc.parent["uc"]].append(uc.id)
        reqs, stack, seen = set(), [uc_id], set()
        while stack:
            u = stack.pop()
            if u in seen or u not in self.usecases:
                continue
            seen.add(u)
            for p in self.usecases[u].pathways():
                reqs |= set(p.reqs)
            stack.extend(kids[u])
            for s in self.usecases[u].steps:
                stack.extend(s.includes)
        return reqs

    # ------------------------------------------------------------------ the rules

    def check(self):
        f = []
        add = lambda rule, where, msg: f.append(Finding(rule, where, msg))
        for kind, i in self.dups:
            add("U15", i, "two %ss share this id" % kind)
        for c in as_list((self.meta.get("story") or {}).get("clauses")):
            if blank(c.get("source")):
                add("U14", "story", "clause %r has no source" % (c.get("text") or "")[:60])
            elif c.get("source") not in self.sources:
                add("U8", "story", "clause source %r is not a known source" % c.get("source"))
        kids = self.children()
        carried = self.carriers()
        mapped = self.mapped_from()
        primary = collections.Counter(uc.actor for uc in self.usecases.values())

        for uc in self.usecases.values():
            w = uc.id or "(no id)"
            for fld in UseCase.FIELDS:
                if blank(uc.raw.get(fld)):
                    add("U1", w, "no %s" % fld)
            if uc.level not in LEVELS:
                add("U1", w, "level %r is not kite, sea or fish" % uc.level)
            lo = 2 if uc.level == "fish" else 3
            if not (lo <= len(uc.steps) <= 11):
                add("U1", w, "%d main steps; %s use cases take %d to 11" % (len(uc.steps), uc.level, lo))
            if uc.actor not in self.actors:
                add("U8", w, "actor %r is not a known actor" % uc.actor)
            for a in uc.supporting:
                if a not in self.actors:
                    add("U8", w, "supporting actor %r is not known" % a)
            # the hierarchy
            if uc.level == "kite":
                if uc.parent.get("uc"):
                    add("U2", w, "a kite use case has no parent")
            elif uc.level in LEVELS:
                p = self.usecases.get(uc.parent.get("uc"))
                if p is None:
                    add("U2", w, "parent %r is not a known use case" % uc.parent.get("uc"))
                else:
                    if uc.parent.get("step") is not None:
                        if not (1 <= int(uc.parent["step"]) <= len(p.steps)):
                            add("U2", w, "parent step %s is not a step of %s" % (uc.parent["step"], p.id))
                    elif uc.parent.get("ext") is not None:
                        if str(uc.parent["ext"]) not in {e.id for e in p.ext}:
                            add("U2", w, "parent extension %s is not an extension of %s" % (uc.parent["ext"], p.id))
                    else:
                        add("U2", w, "parent names no step and no extension")
                    if p.level in LEVELS and LEVELS[uc.level] < LEVELS[p.level]:
                        add("U2", w, "a %s use case cannot be the child of a %s" % (uc.level, p.level))
            seen, cur = set(), uc
            while cur is not None and cur.parent.get("uc"):
                if cur.id in seen:
                    add("U2", w, "the parents form a cycle")
                    break
                seen.add(cur.id)
                cur = self.usecases.get(cur.parent["uc"])
            # steps
            for s in uc.steps:
                if blank(s.text):
                    add("U1", w, "step %d says nothing" % s.n)
                for r in s.reqs:
                    if r not in self.requirements:
                        add("U8", w, "step %d carries %r, which is not a requirement" % (s.n, r))
                for i in s.includes:
                    if i not in self.usecases:
                        add("U8", w, "step %d includes %r, which is not a use case" % (s.n, i))
                included = any(self.usecases[i].pathways()[0].reqs for i in s.includes if i in self.usecases)
                if not s.reqs and not included:
                    add("U12", w, "step %d carries no requirement" % s.n)
                if uc.level == "kite" and not kids.get((uc.id, "step", s.n)) and not s.includes:
                    add("U3", w, "kite step %d decomposes into no use case" % s.n)
            # alternate pathways. A use case of 3+ steps with none: measured 2026-10-27, interrogate.py asked it of EPIC-4, -5
            # and -6 on the render while this checker passed them - the two checkers must ask the same questions
            if len(uc.steps) >= 3 and not uc.ext:
                add("U17", w, "%d steps and no alternate pathway: what happens when a step cannot complete?" % len(uc.steps))
            for e in uc.ext:
                ew = "%s %s" % (w, e.id)
                if not isinstance(e.at, int) or not (1 <= e.at <= len(uc.steps)):
                    add("U4", ew, "at %r is not a step of the main scenario" % e.at)
                if blank(e.when):
                    add("U4", ew, "no condition")
                if not e.do and not e.includes and not kids.get((uc.id, "ext", e.id)):
                    add("U4", ew, "no handling: no steps, no include, no child use case")
                m = OUT_RE.match(str(e.out or ""))
                if not m:
                    add("U4", ew, "outcome %r is not 'rejoin N', 'end success' or 'end minimal'" % e.out)
                elif m.group(1) and not (1 <= int(m.group(1)) <= len(uc.steps)):
                    add("U4", ew, "rejoins step %s, which the main scenario does not have" % m.group(1))
                if not e.reqs:
                    add("U4", ew, "carries no requirement")
                for r in e.reqs:
                    if r not in self.requirements:
                        add("U8", ew, "carries %r, which is not a requirement" % r)
                for i in e.includes:
                    if i not in self.usecases:
                        add("U8", ew, "includes %r, which is not a use case" % i)
            # Norman's gulfs, for the goals a person holds
            if uc.level in ("kite", "sea"):
                sub = self.subtree_reqs(uc.id)
                for g, name in (("exec", "execution"), ("eval", "evaluation")):
                    side = uc.gulf.get(g) or {}
                    if blank(side.get("text")) or not as_list(side.get("reqs")):
                        add("U13", w, "no gulf of %s: how the person %s" % (
                            name, "acts" if g == "exec" else "sees the result"))
                        continue
                    for r in as_list(side.get("reqs")):
                        if r not in self.requirements:
                            add("U8", w, "gulf of %s names %r, which is not a requirement" % (name, r))
                        elif r not in sub:
                            add("U13", w, "gulf of %s names %s, which nothing in this use case's subtree carries" % (name, r))

        for a in self.actors.values():
            if a.kind == "person" and not primary.get(a.id):
                add("U9", a.id, "%s is primary actor of no use case" % (a.name or a.id))

        for r in self.requirements.values():
            if r.kind not in REQ_KINDS:
                add("U6", r.id, "kind %r is not authored, derived or inferred" % r.kind)
            if blank(r.text):
                add("U1", r.id, "a requirement with no text")
            if blank(r.measure):
                add("U10", r.id, "no measure: nothing says how to check it")
            if not carried.get(r.id):
                add("U5", r.id, "no step and no extension carries it")
            elif all(self.usecases[c.split(" ")[0]].level == "kite" for c in carried[r.id]
                     if c.split(" ")[0] in self.usecases):
                # measured 2026-10-27: the render of the Blueprint-view model found REQ-BPV-13, -15, -22 and ED-7 carried only
                # by epic steps, which summarise; a requirement no user goal realises has no pathway a person actually walks
                add("U16", r.id, "carried only by kite (summary) steps: no user goal or subfunction realises it")
            for s in r.sources:
                if s not in self.sources:
                    add("U8", r.id, "source %r is not a known source" % s)
            cells = mapped.get(r.id, [])
            if r.kind == "authored":
                ok = any(self.sources[s].kind in AUTHORED_SOURCES for s in r.sources if s in self.sources) or any(
                    self.spaces[c.split(":")[0]].basis == "prose" for c in cells if c.split(":")[0] in self.spaces)
                if not ok:
                    add("U6", r.id, "authored, but no authored source or source sentence maps to it")
            elif r.kind == "derived":
                # a CODE basis (2026-10-28, the Blueprint view's online sample): a requirement READ from existing code - a
                # Blueprint node, a decision's outcome, a path - is derived from the code cell that maps it, as a framework's is
                if not any(self.spaces[c.split(":")[0]].basis in ("framework", "language", "brainstorm", "code")
                           for c in cells if c.split(":")[0] in self.spaces):
                    add("U6", r.id, "derived, but no framework, language, brainstorm or code cell maps to it")
            elif r.kind == "inferred" and blank(r.why):
                add("U6", r.id, "inferred, with no reason the chain needs it")
            if self.closed_loop and not r.effect and blank(r.absent):
                add("U18", r.id, "no effect and no reason it has none: what must the code do when this holds?")
            for a in r.effect:
                if a.verb is None:
                    add("U18", a.id, "an assertion takes exactly one of %s; it has %s" % (", ".join(Assertion.VERBS),
                                                                                       ", ".join(a.verbs) or "none"))
                if blank(a.at):
                    add("U18", a.id, "no place: at <graph>, <graph>#sN or <graph>#sN:<outcome>")
                if a.to is not None and a.verb != "sets":
                    add("U18", a.id, "`to` belongs to sets, not to %s" % a.verb)
                if a.args and a.verb != "calls":
                    add("U18", a.id, "`args` belong to calls, not to %s" % a.verb)
                if a.by is not None and a.verb != "sets":
                    add("U18", a.id, "`by` belongs to sets (a write through a call), not to %s" % a.verb)

        for sp in self.spaces.values():
            try:
                expected = sp.expected_keys(self)
            except (ValueError, OSError, re.error) as e:
                add("U7", sp.id, "its keys cannot be read: %s" % e)
                continue
            if expected is not None:
                for k in expected:
                    if k not in sp.cells:
                        add("U7", "%s:%s" % (sp.id, k), "missing from the space")
                for k in sp.cells:
                    if k not in expected:
                        add("U7", "%s:%s" % (sp.id, k), "not a key of this space (stale or misspelt)")
            for c in sp.cells.values():
                if not c.maps and blank(c.excluded):
                    add("U7", c.ref, "unmapped: no requirement, and no reason it is excluded")
                for r in c.maps:
                    if r not in self.requirements:
                        add("U7", c.ref, "maps to %r, which is not a requirement" % r)

        for q in self.questions:
            if q.persona not in self.actors:
                add("U8", q.id, "persona %r is not a known actor" % q.persona)
            if q.inverse not in self.usecases:
                add("U11", q.id, "%r inverts into no use case (%r)" % ((q.text or "")[:50], q.inverse))

        for t in self.terms.values():
            if not t.binds:
                add("U19", t.id, "binds no code name: what does the word mean in the code?")
        for e in as_list(self.loop.get("entry")):
            if not isinstance(e, dict) or blank(e.get("why")):
                add("U19", "entry %s" % (e.get("name") if isinstance(e, dict) else e), "set outside the model, with no reason why")
        return f

    def counts(self):
        levels = collections.Counter(uc.level for uc in self.usecases.values())
        paths = [p for uc in self.usecases.values() for p in uc.pathways()]
        kinds = collections.Counter(r.kind for r in self.requirements.values())
        cells = {sp.id: (len(sp.cells), sum(1 for c in sp.cells.values() if c.maps),
                         sum(1 for c in sp.cells.values() if not c.maps and not blank(c.excluded)))
                 for sp in self.spaces.values()}
        return {"usecases": dict(levels), "pathways": len(paths), "extensions": len(paths) - len(self.usecases),
                "steps": sum(len(uc.steps) for uc in self.usecases.values()), "requirements": dict(kinds),
                "cells": cells, "questions": len(self.questions), "actors": len(self.actors)}


def report(model, findings, limit):
    c = model.counts()
    print("USE CASE MODEL  %s" % model.root)
    print("  use cases  %s   steps %d   pathways %d (main %d + alternate %d)" % (
        ", ".join("%s %d" % (k, c["usecases"].get(k, 0)) for k in LEVELS), c["steps"], c["pathways"],
        sum(c["usecases"].values()), c["extensions"]))
    print("  requirements  %s   questions %d   actors %d" % (
        ", ".join("%s %d" % (k, c["requirements"].get(k, 0)) for k in REQ_KINDS), c["questions"], c["actors"]))
    for sid, (n, m, x) in c["cells"].items():
        print("  space %-4s %4d cells: %4d mapped, %3d excluded with a reason" % (sid, n, m, x))
    inferred = [r for r in model.requirements.values() if r.kind == "inferred"]
    if inferred:
        print("  QUESTIONS FOR THE OWNER (inferred requirements): %d" % len(inferred))
        for r in inferred:
            print("    %-8s %s" % (r.id, (r.text or "")[:120]))
    by = collections.defaultdict(list)
    for x in findings:
        by[x.rule].append(x)
    if not findings:
        print("\n  PASS - every use case, pathway, requirement, cell and question accounted for, by the %d rules" % len(RULES))
        return 0
    print("\n  %d HOLE(S)" % len(findings))
    for rule in sorted(by, key=lambda r: int(r[1:])):
        print("  %s %s: %d" % (rule, RULES[rule], len(by[rule])))
        for x in by[rule][:limit]:
            print("      %s: %s" % (x.where, x.message))
        if len(by[rule]) > limit:
            print("      ... %d more" % (len(by[rule]) - limit))
    return 1


# ---------------------------------------------------------------------- selftest: one planted fault per rule

def _clean_data(prose_path):
    meta = {"story": {"text": "s", "clauses": [{"text": "a reader wants to understand", "source": "OWNER"}]}}
    data = {
        "actors": [{"id": "A1", "name": "Reader", "kind": "person"}, {"id": "SYS", "name": "View", "kind": "system"}],
        "sources": [{"id": "SRC", "kind": "prose", "path": prose_path}, {"id": "OWNER", "kind": "quote", "text": "q"}],
        "spaces": [
            {"id": "S1", "basis": "prose", "kind": "sentences", "source": "SRC", "cells": {"1.1": ["R-1"], "1.2": ["R-2"]}},
            {"id": "S2", "basis": "framework", "kind": "grid", "dims": ["D1"], "surfaces": ["V1"],
             "cells": {"D1.V1": ["R-2"]}},
            {"id": "S3", "basis": "language", "kind": "files", "dir": _files_dir(prose_path, "nodes", ("N_A.h", "N_B.h", "README.md")),
             "pattern": r"(N_\w+)\.h", "cells": {"N_A": ["R-1"], "N_B": {"excluded": "abstract"}}},
            # a CODE basis: the nodes of a Blueprint text, each mapped to the requirement it was read as
            {"id": "S4", "basis": "code", "kind": "extract", "file": _code_file(prose_path),
             # (?m): the checker's re.finditer takes no flags, so a ^ without it would read only the FIRST node of the text
             "pattern": r'(?m)^Begin Object Class=\S+ Name="([^"]+)"',
             "cells": {"K2Node_IfThenElse_0": ["R-3"], "K2Node_Knot_1": {"excluded": "a reroute knot carries no logic of its own"}}},
        ],
        "requirements": [
            {"id": "R-1", "kind": "authored", "text": "t", "measure": "m", "sources": ["OWNER"]},
            {"id": "R-2", "kind": "derived", "text": "t", "measure": "m"},
            {"id": "R-3", "kind": "derived", "text": "t", "measure": "m"},
            {"id": "IR-1", "kind": "inferred", "text": "t", "measure": "m", "why": "the chain needs it"},
        ],
        "usecases": [
            {"id": "EPIC-1", "name": "k", "level": "kite", "actor": "A1", "trigger": "t", "pre": "p", "minimal": "m",
             "success": "s", "why": "w", "gulf": {"exec": {"text": "x", "reqs": ["R-1"]}, "eval": {"text": "v", "reqs": ["R-2"]}},
             "steps": [{"do": "a", "reqs": ["R-1"]}, {"do": "b", "reqs": ["R-1"], "includes": ["UC-1"]},
                       {"do": "c", "reqs": ["R-1"], "includes": ["UC-1"]}],
             "ext": [{"at": 1, "when": "w", "do": ["h"], "out": "end minimal", "reqs": ["R-1"]}]},
            {"id": "UC-1", "name": "s", "level": "sea", "parent": {"uc": "EPIC-1", "step": 1}, "actor": "A1", "trigger": "t",
             "pre": "p", "minimal": "m", "success": "s", "why": "w",
             "gulf": {"exec": {"text": "x", "reqs": ["R-1"]}, "eval": {"text": "v", "reqs": ["R-2"]}},
             "steps": [{"do": "a", "reqs": ["R-1"]}, {"do": "b", "reqs": ["R-2"]}, {"do": "c", "reqs": ["IR-1"]}],
             "ext": [{"at": 2, "when": "w", "do": ["h"], "out": "rejoin 3", "reqs": ["R-2"]}]},
            {"id": "UC-1.1", "name": "f", "level": "fish", "parent": {"uc": "UC-1", "step": 2}, "actor": "SYS",
             "trigger": "t", "pre": "p", "minimal": "m", "success": "s", "why": "w",
             "steps": [{"do": "a", "reqs": ["R-2"]}, {"do": "b", "reqs": ["R-2", "R-3"]}]},
        ],
        "questions": [{"id": "Q1", "persona": "A1", "text": "what does it do?", "inverse": "UC-1"}],
    }
    return meta, data


def _files_dir(prose_path, name, files):
    """a folder of files beside the selftest's prose, for a `files` space (made once, never changed by a plant)"""
    d = os.path.join(os.path.dirname(prose_path), name)
    os.makedirs(d, exist_ok=True)
    for f in files:
        io.open(os.path.join(d, f), "w", encoding="utf-8").close()
    return d


def _code_file(prose_path):
    """a Blueprint text of one node beside the selftest's prose, for a `code` space (made once, never changed by a plant)"""
    f = os.path.join(os.path.dirname(prose_path), "one_node.t3d")
    io.open(f, "w", encoding="utf-8").write('Begin Object Class=/Script/BlueprintGraph.K2Node_IfThenElse Name="K2Node_IfThenElse_0"\nEnd Object\n'
                                            'Begin Object Class=/Script/BlueprintGraph.K2Node_Knot Name="K2Node_Knot_1"\nEnd Object\n')
    return f


def _uc(data, uid):
    return next(u for u in data["usecases"] if u["id"] == uid)


PLANTS = [
    ("U1", "a sea use case with two steps", lambda m, d: _uc(d, "UC-1")["steps"].pop()),
    ("U2", "a fish whose parent is missing", lambda m, d: _uc(d, "UC-1.1")["parent"].update(uc="UC-9")),
    ("U3", "a kite step with no child and no include", lambda m, d: _uc(d, "EPIC-1")["steps"][2].pop("includes")),
    ("U4", "an extension with no outcome", lambda m, d: _uc(d, "UC-1")["ext"][0].pop("out")),
    ("U5", "a requirement no pathway carries", lambda m, d: d["requirements"].append(
        {"id": "R-9", "kind": "authored", "text": "t", "measure": "m", "sources": ["OWNER"]})),
    ("U6", "a derived requirement no cell maps to", lambda m, d: d["spaces"][1]["cells"].update({"D1.V1": ["R-1"]})),
    ("U7", "a grid cell missing", lambda m, d: d["spaces"][1].update(dims=["D1", "D2"])),
    ("U7", "a class added to a files space's folder", lambda m, d: d["spaces"][2].update(
        dir=_files_dir(os.path.dirname(d["spaces"][2]["dir"]) + os.sep + "x", "nodes_plus", ("N_A.h", "N_B.h", "N_C.h")))),
    ("U8", "a step carrying an unknown requirement", lambda m, d: _uc(d, "UC-1")["steps"][0]["reqs"].append("R-404")),
    ("U9", "a person who is primary nowhere", lambda m, d: d["actors"].append({"id": "A2", "name": "x", "kind": "person"})),
    ("U10", "a requirement with no measure", lambda m, d: d["requirements"][0].pop("measure")),
    ("U11", "a question that inverts into nothing", lambda m, d: d["questions"][0].update(inverse="UC-404")),
    ("U12", "a step that carries nothing", lambda m, d: _uc(d, "UC-1.1")["steps"][0].update(reqs=[])),
    ("U13", "a sea use case with no gulf of evaluation", lambda m, d: _uc(d, "UC-1")["gulf"].pop("eval")),
    ("U14", "a story clause with no source", lambda m, d: m["story"]["clauses"].append({"text": "x"})),
    ("U15", "two requirements with one id", lambda m, d: d["requirements"].append(dict(d["requirements"][0]))),
    ("U16", "a requirement only an epic step carries", lambda m, d: (d["requirements"].append(
        {"id": "R-8", "kind": "authored", "text": "t", "measure": "m", "sources": ["OWNER"]}),
        _uc(d, "EPIC-1")["steps"][0]["reqs"].append("R-8"))),
    ("U17", "a use case of 3 steps with no alternate pathway", lambda m, d: _uc(d, "UC-1").pop("ext")),
    ("U18", "a closed loop, one requirement with no effect", lambda m, d: (m.update(loop={"closed": True}), [
        r.update(effect=[{"sets": "X", "at": "G#s1"}]) for r in d["requirements"][1:]])),
    ("U18", "an assertion with two verbs", lambda m, d: d["requirements"][0].update(
        effect=[{"sets": "X", "calls": "F", "at": "G#s1"}])),
    ("U18", "an assertion with no place", lambda m, d: d["requirements"][0].update(effect=[{"calls": "F"}])),
    ("U18", "a write through a call on a branch", lambda m, d: d["requirements"][0].update(
        effect=[{"branches": "X", "by": "F", "at": "G#s1"}])),
    ("U19", "a term that binds nothing", lambda m, d: d.setdefault("terms", []).append({"term": "lock range", "binds": []})),
    ("U19", "an entry condition with no reason", lambda m, d: m.update(loop={"entry": [{"name": "IsDead"}]})),
]


def selftest():
    tmp = tempfile.mkdtemp(prefix="ucm_selftest_")
    prose = os.path.join(tmp, "src.txt")
    io.open(prose, "w", encoding="utf-8").write("The reader wants to see the flow. The view must not lie.\n")
    meta, data = _clean_data(prose)
    clean = UseCaseModel.from_data(meta, data).check()
    ok = True
    # every check line ends in its verdict, the form every tool's selftest uses and prove_all.py counts checks by
    print("selftest: clean model -> %d finding(s)%s %s" % (len(clean), "" if not clean else ": " + "; ".join(map(str, clean)), "PASS" if not clean else "FAIL"))
    ok &= not clean
    sentence_keys = [k for k, _ in sentences(prose)]
    print("selftest: sentence split of the clean source -> %s %s" % (sentence_keys, "PASS" if sentence_keys == ["1.1", "1.2"] else "FAIL"))
    ok &= sentence_keys == ["1.1", "1.2"]
    for rule, what, plant in PLANTS:
        m, d = copy.deepcopy(meta), copy.deepcopy(data)
        plant(m, d)
        got = {x.rule for x in UseCaseModel.from_data(m, d).check()}
        hit = rule in got
        ok &= hit
        print("selftest: %-4s %-40s -> %s %s %s" % (rule, what, "CAUGHT" if hit else "MISSED", sorted(got), "PASS" if hit else "FAIL"))
    # the prose split must also refuse to skip a sentence: a third sentence in the source is a missing cell
    io.open(prose, "a", encoding="utf-8").write("A third sentence nobody mapped.\n")
    got = {x.rule for x in UseCaseModel.from_data(meta, data).check()}
    print("selftest: U7   a source sentence added, no cell            -> %s %s %s" % (
        "CAUGHT" if "U7" in got else "MISSED", sorted(got), "PASS" if "U7" in got else "FAIL"))
    ok &= "U7" in got
    print("selftest: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?")
    ap.add_argument("--max", type=int, default=20, help="examples printed per rule")
    ap.add_argument("--json", help="write counts and findings here")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.folder:
        ap.error("name a model folder, or --selftest")
    try:
        model = UseCaseModel.load(a.folder)
    except (RuntimeError, OSError, yaml.YAMLError if yaml else RuntimeError) as e:
        print("usecase_model: cannot read the model: %s" % e, file=sys.stderr)
        return 2
    findings = model.check()
    rc = report(model, findings, a.max)
    if a.json:
        io.open(a.json, "w", encoding="utf-8").write(json.dumps(
            {"counts": model.counts(), "findings": [vars(x) for x in findings]}, indent=1))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
