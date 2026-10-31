#!/usr/bin/env python3
"""grow.py - a first requirements model from text someone already wrote (an epic, a PRD, a ticket export, a transcript), without inventing.

WHY (2026-10-31). The tool's main job is growing a rough epic into a larger set of requirements, so code is written to what the owner meant.
Until now it had no front door: an agent wrote a model by hand against a YAML format, and nothing checked that a requirement came from
anything. This is the door. What a program can do about prose is small and exact; what it cannot do is read. So the work is split, and the
split is the design:

  THE TOOL (deterministic)                          THE AGENT IN HAND (reads, judges)             THE TOOL AGAIN (refuses)
  numbers every sentence of the text, keeps the     reads the numbered sentences and writes       a requirement that cites nothing, cites a
  line (or ticket, or speaker) it stands on,        the draft: each requirement quotes the        sentence that is not there, quotes words
  prints a brief and writes a skeleton draft        words it stands on, or is INFERRED with the   the sentence lacks, or is inferred with no
                                                    reason the chain needs it                     reason: refused, nothing is written

No model is called and no key is needed: the agent that runs this command is the reader. The questions that come after (ask.py) are the part
the producer answers.

THE CLASS MODEL (each class is one concept; its docstring says what it is and where its evidence lives):

  SourceError      the text cannot be read at all (empty, no sentences, unknown kind): a loud stop, never an empty model
  Sentence         one sentence of the text: its key "line.k", its line, the place a person finds it (a ticket, a speaker), whether it carries a rule word
  SourceText       the text made readable: kind (document | tickets | transcript), normalised lines that keep the original line numbers, the
                   sentences, and what it could not read (ignored ticket fields); written beside the model so citations stay checkable
  DraftRequirement what the agent proposes: says + check + (cite + quote | inferred reason + needs)
  Draft            the agent's file (draft.yml): subject, actors, requirements, goals (use cases), exclusions with reasons
  Refusal          one defect in a draft, with a code (D1..D12); a refusal is never repaired silently
  Brief            the text the agent is given (BRIEF.md) and the skeleton draft with one stub per sentence that states a rule
  Grower           Draft + SourceText -> refusals, or a model folder; then the 19 rules; sorts what is left into the agent's errors and
                   the producer's questions
  Report           what `grow` prints: counts, the second reader's count, the rules' holes, the first question

The model it writes is the kit's own (usecase_model.py): people.yaml, requirements.yaml, usecases.yaml, spaces.yaml (one `sentences` space, so
every sentence of the text is a cell that is cited, set aside with a reason, or open), plus source/ (the numbered text and its map) and an empty
answers.yaml for ask.py. The working draft is draft.yml, not .yaml, because the model loader reads every *.yaml in the folder.

WHAT IS CITED, EXACTLY. A requirement cites sentence keys ("4.1": line 4, first sentence) and quotes at least two words that must occur in a
cited sentence. A key that is not a sentence of the text, a quote the sentence does not contain, or no cite and no inferred reason: refused (D1-D4).
An INFERRED requirement carries the reason the chain needs it and, computed here, the goal and step that carries it. A sentence nothing cites
and nothing sets aside stays open: it is a hole in the 19 rules (U7) and a question for the producer; a rule-bearing sentence the agent set
aside is asked about too, because "background" is the agent's word, not the producer's.

  py -X utf8 closed_loop.py grow <text file> --out <model folder> [--kind document|tickets|transcript]
      first run: writes <out>/source/, <out>/draft.yml, <out>/BRIEF.md and prints the brief; exit 3 = a draft is needed
      with a filled draft: refuses or writes the model, runs the 19 rules, prints the first question; exit 1 = refused or an agent error
  py -X utf8 grow.py --selftest

Exit 0: model written and structurally sound (questions may remain; `ask` takes them). 1: the draft is refused, or the model has an error only
the drafter can fix. 2: the text cannot be read. 3: waiting for the draft.

Ideas, with the licences they come from (ideas only, no code copied): Spec Kit's clarify step asks structured questions to fill what a spec leaves
open (MIT); BMAD-METHOD's analyst brainstorming and elicitation (MIT); Superpowers' brainstorming skill, asking one question at a time (MIT);
OpenSpec's explore step (MIT); Agent OS's shaping questions (MIT); Task Master's parse of a requirements document into tasks (MIT with the Commons
Clause, so the idea only); SpecTrace's import of requirements from markdown specs (MIT). Kiro and Claude Code's question tool are proprietary:
idea only. What is ours: every requirement carries its sentence or its chain reason, checked by a program, and the questions come from holes in
the use case chain rather than from a script.
"""
import argparse
import collections
import copy
import csv
import hashlib
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
import source_rules  # noqa: E402

FACETS = ("trigger", "pre", "minimal", "success", "why")
KINDS = ("document", "tickets", "transcript")
# a sentence that is probably a rule: the strict subset a PRODUCER is asked about when the agent set it aside. A heuristic: it picks who to
# ask and never refuses a draft ("a guess that can fail a build is worse than no check")
STRICT = re.compile(r"\b(must|must not|cannot|can't|can not|never|always|only|at most|at least|no more than|exactly|loses?|wins?|unless|may not|"
                    r"is not allowed|are not allowed|not allowed|has to|have to|required|forbidden)\b", re.I)
# an id of this tool's own making, in a sentence meant for a person
INTERNAL_ID = re.compile(r"\b(?:R\d+|RA\d+|UC-[\d.]+|EPIC-\d+|ANS-\d+|S-SRC)\b")
TURN = re.compile(r"^\s*(?:[\[(]?(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)[\])]?\s+)?([A-Za-z][\w.'\u2019 -]{0,40}?):\s+(\S.*)$")
CUE = re.compile(r"^\s*(\d{1,2}:\d{2}:\d{2}[.,]\d+|\d{1,2}:\d{2}[.,]\d+)\s*-->\s*\S+")
TICKET_KEY = ("key", "id", "issue key", "issue id", "ticket", "number", "ref")
TICKET_TITLE = ("summary", "title", "name", "subject")
TICKET_BODY = ("description", "body", "details", "story", "user story", "notes", "text")
MIN_REASON_WORDS = 5
MIN_QUOTE_WORDS = 2


class SourceError(Exception):
    """The text cannot be read at all: nothing there, nothing sentence-shaped, or a kind this tool does not know. A model grown from nothing
    would be an empty model that passes everything, so this stops the run instead."""


class Sentence(object):
    """One sentence of the text. `key` is "line.k" (the key usecase_model.sentences() gives it, so the space and the citation agree), `where` is
    where a person finds it: a line, a ticket and field, a speaker and time. `rule` is the broad rule-word flag (source_rules.RULE_WORDS);
    `strict` the narrower one a producer is asked about."""

    def __init__(self, key, text, where):
        self.key, self.text, self.where = key, text, where
        self.line = int(key.split(".")[0])
        self.strict = bool(STRICT.search(text))
        self.rule = (bool(source_rules.RULE_WORDS.search(text)) or self.strict) and len(source_rules.words(text)) >= 3


def squash(text):
    """a text made comparable: lower case, words only. A quote is checked against its sentence in this form, so a changed quotation mark or a
    bold marker does not refuse a true quote, and a changed word still does."""
    return " ".join(re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower().replace("\u2019", "'")))


def plain_cell(value):
    return "" if value is None else re.sub(r"\s+", " ", str(value)).strip()


class SourceText(object):
    """The text made readable. `lines` are the normalised lines (a heading, a code fence, a table rule are blanked; markers are stripped) and keep
    the ORIGINAL line numbers for a document, so "line 14" means line 14 of the file. For tickets and transcripts the normalised file is the
    reading order and `labels` maps each line back to its ticket and field or its speaker. `ignored` names what was seen and not read.
    Evidence: written to <model>/source/<stem>.txt and source/map.json, which the citations and the questions read back."""

    def __init__(self, origin, kind, lines, labels, headings, ignored):
        self.origin, self.kind, self.lines = origin, kind, lines
        self.labels, self.headings, self.ignored = labels, headings, ignored
        self.stem = re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.splitext(os.path.basename(origin))[0]) or "source"
        self.sentences = []
        self.sha = ""

    # ---------------------------------------------------------------- reading
    @classmethod
    def read(cls, path, kind=None):
        if not os.path.isfile(path):
            raise SourceError("%s is not a file" % path)
        raw = io.open(path, encoding="utf-8-sig", errors="replace").read()
        if not raw.strip():
            raise SourceError("%s is empty: there is nothing to grow a model from" % path)
        ext = os.path.splitext(path)[1].lower()
        kind = kind or cls.detect(raw, ext)
        if kind not in KINDS:
            raise SourceError("kind %r is not one of %s" % (kind, ", ".join(KINDS)))
        if kind == "tickets":
            lines, labels, ignored = cls._tickets(raw, ext)
            headings = {}
        elif kind == "transcript":
            lines, labels, ignored = cls._transcript(raw, ext)
            headings = {}
        else:
            lines, labels, headings = cls._document(raw)
            ignored = []
        src = cls(path, kind, lines, labels, headings, ignored)
        src.sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        src.index()
        return src

    @staticmethod
    def detect(raw, ext):
        if ext in (".csv", ".tsv"):
            return "tickets"
        if ext == ".json":
            return "tickets"
        if ext in (".vtt", ".srt"):
            return "transcript"
        body = [l for l in raw.splitlines() if l.strip()]
        turns = [TURN.match(l) for l in body]
        speakers = {m.group(2).strip() for m in turns if m}
        if len(body) >= 3 and sum(1 for m in turns if m) >= 0.5 * len(body) and len(speakers) >= 2:
            return "transcript"
        return "document"

    @staticmethod
    def _document(raw):
        """A markdown or plain text document -> (normalised lines, labels, headings). Line N of the result is line N of the file: a paragraph is joined onto the line it
        STARTS on (a text hard-wrapped at 80 columns is one sentence stream, not a sentence per line), and what is not prose (headings, fences, HTML comments, link
        definitions, a table of contents, table rules) is blanked. Measured on a public proposal of 400 lines: before this, 283 'sentences', most of them half a line, a
        heading or a link."""
        out, labels, heads = [], {}, {}
        fence, comment, section, open_at = False, False, None, None
        rows = raw.splitlines()
        for n, line in enumerate(rows, 1):
            s = line.strip()
            if comment:
                out.append("")
                comment = "-->" not in s
                continue
            if s.startswith("<!--"):
                comment = "-->" not in s
                out.append("")
                open_at = None
                continue
            if s.startswith("```") or s.startswith("~~~"):
                fence = not fence
                out.append("")
                open_at = None
                continue
            if fence:
                out.append("")
                continue
            m = re.match(r"^#{1,6}\s+(.*?)\s*#*$", s)
            if m:
                section = m.group(1)
                heads[n] = section
                out.append("")
                open_at = None
                continue
            if not s or re.match(r"^\[[^\]]+\]:\s*\S+", s) or re.match(r"^(?:[-*+]\s+)?\[[^\]]+\]\(#[^)]*\)\s*$", s):
                out.append("")           # a blank line ends a paragraph; a link definition and a table-of-contents entry are not prose
                if not s or s.startswith("["):
                    open_at = None
                continue
            starts = False
            if s.startswith("|"):
                nxt = rows[n].strip() if n < len(rows) else ""
                if re.fullmatch(r"[|:\s-]+", s) or re.fullmatch(r"\|[|:\s-]+", nxt):     # a rule line, or the header row above one
                    out.append("")
                    open_at = None
                    continue
                s = " - ".join(c.strip() for c in s.strip("|").split("|") if c.strip())
                starts = True
            if re.match(r"^(?:[-*+]|\d+[.)])\s+", s) or s.startswith(">"):
                starts = True
            s = re.sub(r"^>\s?", "", s)
            s = re.sub(r"^(?:[-*+]|\d+[.)])\s+(?:\[[ xX]\]\s+)?", "", s)
            s = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", s)
            s = s.replace("**", "").replace("__", "").replace("`", "")
            if open_at is not None and not starts and not re.search(r"[.!?][\"')\]]*$", out[open_at]):      # the line above is unfinished: this one continues it
                out[open_at] = out[open_at] + " " + s
                out.append("")
                continue
            out.append(s)
            open_at = len(out) - 1
            labels[n] = "line %d%s" % (n, "" if not section else ", under \"%s\"" % section)
        return out, labels, heads

    @staticmethod
    def _transcript(raw, ext):
        out, labels = [], {}
        if ext in (".vtt", ".srt"):
            when = ""
            for line in raw.splitlines():
                s = line.strip()
                if not s or s.upper().startswith("WEBVTT") or re.fullmatch(r"\d+", s):
                    continue
                m = CUE.match(s)
                if m:
                    when = m.group(1)[:8]
                    continue
                s = re.sub(r"<[^>]+>", "", s)
                out.append(s)
                labels[len(out)] = "at %s" % when if when else "line %d" % len(out)
            return out, labels, []
        for n, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                out.append("")
                continue
            m = TURN.match(line)
            if m:
                when, who, said = m.group(1), m.group(2).strip(), m.group(3).strip()
                out.append(said)
                labels[n] = "%s%s" % (who, ", at %s" % when if when else ", line %d" % n)
            else:
                out.append(line.strip())
                labels[n] = "line %d" % n
        return out, labels, []

    @staticmethod
    def _flatten(node):
        """the words of a value that may be a nested document (a tracker's rich text): the text nodes in order, never the structure's own labels"""
        if node is None:
            return []
        if isinstance(node, str):
            return [node]
        if isinstance(node, dict):
            if isinstance(node.get("text"), str):
                return [node["text"]]
            return [t for v in node.values() if isinstance(v, (dict, list)) for t in SourceText._flatten(v)]
        if isinstance(node, (list, tuple)):
            return [t for v in node for t in SourceText._flatten(v)]
        return [str(node)]

    @classmethod
    def _tickets(cls, raw, ext):
        if ext == ".json":
            data = json.loads(raw)
            if isinstance(data, dict):
                for k in ("issues", "tickets", "items", "results", "data"):
                    if isinstance(data.get(k), list):
                        data = data[k]
                        break
                else:
                    data = [data]
            rows = []
            for item in data:
                flat = dict(item)
                if isinstance(item.get("fields"), dict):
                    flat.update(item["fields"])
                rows.append({str(k).lower(): v for k, v in flat.items()})
        else:
            rd = csv.DictReader(io.StringIO(raw), delimiter="\t" if ext == ".tsv" else ",")
            rows = [{str(k).strip().lower(): v for k, v in r.items() if k} for r in rd]
        if not rows:
            raise SourceError("the ticket export holds no tickets")
        out, labels, seen, used = [], {}, set(), set()
        for i, r in enumerate(rows, 1):
            key = next((plain_cell(r[k]) for k in TICKET_KEY if plain_cell(r.get(k))), "ticket %d" % i)
            parts = []
            for name in TICKET_TITLE:
                if plain_cell(r.get(name)):
                    parts.append(("title", [plain_cell(r[name])]))
                    used.add(name)
                    break
            for name in TICKET_BODY:
                if r.get(name):
                    parts.append(("description", [t for v in cls._flatten(r[name]) for t in v.splitlines() if t.strip()]))
                    used.add(name)
                    break
            for name in sorted(r):
                if name.startswith("acceptance") or name.startswith("criteria"):
                    if r.get(name):
                        parts.append(("acceptance", [t for v in cls._flatten(r[name]) for t in v.splitlines() if t.strip()]))
                        used.add(name)
            for field, texts in parts:
                for t in texts:
                    t = re.sub(r"^\s*(?:[-*+]|\d+[.)]|\[[ xX]\])\s+", "", t).strip()
                    if not t:
                        continue
                    if field == "title" and not re.search(r"[.!?]$", t):
                        t += "."
                    out.append(t)
                    labels[len(out)] = "%s %s" % (key, field)
            out.append("")
            seen.add(key)
        ignored = sorted({k for r in rows for k in r} - used - set(TICKET_KEY))
        return out, labels, ignored

    # ---------------------------------------------------------------- numbering
    def text(self):
        return "\n".join(self.lines) + "\n"

    def index(self):
        d = tempfile.mkdtemp(prefix="grow_src_")
        try:
            p = os.path.join(d, "s.txt")
            io.open(p, "w", encoding="utf-8", newline="\n").write(self.text())
            found = ucm.sentences(p)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        if not found:
            raise SourceError("%s holds no sentences the tool can number (only headings, rules or blank lines)" % self.origin)
        self.sentences = [Sentence(k, t, self.labels.get(int(k.split(".")[0]), "line %s" % k.split(".")[0])) for k, t in found]
        self.by_key = collections.OrderedDict((s.key, s) for s in self.sentences)

    def write(self, folder):
        d = os.path.join(folder, "source")
        os.makedirs(d, exist_ok=True)
        rel = "source/%s.txt" % self.stem
        io.open(os.path.join(folder, rel), "w", encoding="utf-8", newline="\n").write(self.text())
        io.open(os.path.join(d, "map.json"), "w", encoding="utf-8", newline="\n").write(json.dumps(
            {"origin": os.path.basename(self.origin), "kind": self.kind, "sha256_16": self.sha, "ignored": self.ignored,
             "where": {s.key: s.where for s in self.sentences}}, indent=1, sort_keys=False) + "\n")
        return rel

    def describe(self):
        rules = sum(1 for s in self.sentences if s.rule)
        n = len(self.sentences)
        return "%s, %d sentence%s (%d carry%s a rule word)" % (self.kind, n, "" if n == 1 else "s", rules, "" if rules == 1 else "")


class Refusal(object):
    """One defect in a draft. Codes: D1 NO_GROUND (cites nothing, no inferred reason), D2 BAD_CITE (a key that is not a sentence), D3 NO_QUOTE,
    D4 QUOTE_ABSENT (the words are not in the cited sentence), D5 BOTH_GROUNDS (cites and is inferred), D6 THIN_REASON, D7 MISSING (says or
    check), D8 DUP (one id twice), D9 EXCLUDE (bad key, no reason, or also cited), D10 ACTOR (none, or an unknown one), D11 EMPTY, D12 ID_IN_TEXT
    (an internal id in words a person reads)."""

    NAMES = {"D1": "NO_GROUND", "D2": "BAD_CITE", "D3": "NO_QUOTE", "D4": "QUOTE_ABSENT", "D5": "BOTH_GROUNDS", "D6": "THIN_REASON",
             "D7": "MISSING", "D8": "DUP", "D9": "EXCLUDE", "D10": "ACTOR", "D11": "EMPTY", "D12": "ID_IN_TEXT"}

    def __init__(self, code, where, message):
        self.code, self.where, self.message = code, where, message

    def __str__(self):
        return "[%s %s] %s: %s" % (self.code, self.NAMES[self.code], self.where, self.message)


def as_keys(value):
    if value is None:
        return []
    items = value if isinstance(value, (list, tuple)) else re.split(r"[,\s]+", str(value))
    return [str(i).strip() for i in items if str(i).strip()]


class DraftRequirement(object):
    """What the agent proposes. Grounded ONE of two ways: `cite` (sentence keys) + `quote` (words that occur in them), or `inferred` (the reason
    the chain needs it, in a sentence) with an optional `needs` (the thing needed, in a producer's words, which the question uses)."""

    def __init__(self, d):
        self.raw = d
        self.id, self.says, self.check = d.get("id"), plain_cell(d.get("says")), plain_cell(d.get("check"))
        self.cite, self.quote = as_keys(d.get("cite")), plain_cell(d.get("quote"))
        self.inferred, self.needs = plain_cell(d.get("inferred")), plain_cell(d.get("needs"))


class Draft(object):
    """The agent's file, draft.yml. Keys: subject; actors [{name, kind, profile}]; requirements [DraftRequirement]; goals [use cases of level sea:
    name, actor, steps [{do, reqs}], ext [{at, when, do, out, reqs}], and any of trigger / pre / minimal / success / why / gulf that the TEXT
    states]; epic {actor, and any of the same facets} for the whole; exclude [{cites: [...], why}]. A facet the text does not state is left out:
    it becomes a question."""

    def __init__(self, data, path=""):
        self.data, self.path = data or {}, path
        self.subject = plain_cell(self.data.get("subject"))
        self.actors = list(self.data.get("actors") or [])
        self.requirements = [DraftRequirement(r) for r in (self.data.get("requirements") or []) if isinstance(r, dict)]
        self.goals = [g for g in (self.data.get("goals") or []) if isinstance(g, dict)]
        self.epic = self.data.get("epic") or {}
        self.exclude = [e for e in (self.data.get("exclude") or []) if isinstance(e, dict)]

    @classmethod
    def load(cls, path):
        try:
            data = yaml.safe_load(io.open(path, encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            raise SourceError("draft.yml is not valid YAML: %s" % e)
        if not isinstance(data, dict):
            raise SourceError("draft.yml must be a mapping at the top")
        return cls(data, path)

    def filled(self):
        return bool(self.requirements or self.goals)


class Grower(object):
    """Draft + SourceText -> refusals, or the model. `refuse()` is the whole grounding contract; `compile()` writes the model's data (nothing is
    invented: every field is the agent's or a sentence's, and a fixed structural id); `findings()` runs the 19 rules and sorts what is left into
    what only the drafter can fix and what only the producer can answer."""

    def __init__(self, source, draft, folder):
        self.source, self.draft, self.folder = source, draft, folder
        self.rel_source = "source/%s.txt" % source.stem

    # ---------------------------------------------------------------- the grounding contract
    def refuse(self):
        out, d, src = [], self.draft, self.source
        if not d.filled():
            return [Refusal("D11", "draft.yml", "no requirements and no goals: the draft says nothing")]
        keys = list(src.by_key)
        shown = "%s to %s" % (keys[0], keys[-1]) if keys else "none"
        seen = set()
        for r in d.requirements:
            w = "requirement %s" % (r.id or "(no id)")
            if blank(r.id):
                out.append(Refusal("D7", w, "no id"))
            elif r.id in seen:
                out.append(Refusal("D8", w, "two requirements share this id"))
            seen.add(r.id)
            for fld, val in (("says", r.says), ("check", r.check), ("needs", r.needs)):
                if blank(val):
                    if fld != "needs":
                        out.append(Refusal("D7", w, "no %s: %s" % (fld, "the requirement in words a test could hold it to" if fld == "says"
                                                                   else "how anyone would see it hold or fail")))
                elif INTERNAL_ID.search(val):
                    out.append(Refusal("D12", w, "%s holds an internal id (%s); a producer reads these words" % (fld, INTERNAL_ID.search(val).group(0))))
            if r.cite and r.inferred:
                out.append(Refusal("D5", w, "cites a sentence AND says it is inferred; it is one or the other"))
            elif not r.cite and not r.inferred:
                out.append(Refusal("D1", w, "cites no sentence and gives no reason it is inferred: where does it come from?"))
            if r.cite:
                bad = [k for k in r.cite if k not in src.by_key]
                for k in bad:
                    out.append(Refusal("D2", w, "cites %s, which is not a sentence of the text (the sentences are %s)" % (k, shown)))
                if blank(r.quote):
                    out.append(Refusal("D3", w, "cites %s but quotes none of its words" % ", ".join(r.cite)))
                elif not bad:
                    q = squash(r.quote)
                    if len(q.split()) < MIN_QUOTE_WORDS:
                        out.append(Refusal("D3", w, "the quote %r is under %d words; it proves nothing" % (r.quote, MIN_QUOTE_WORDS)))
                    elif not any(q in squash(src.by_key[k].text) for k in r.cite):
                        out.append(Refusal("D4", w, "the quote %r is not in %s" % (r.quote[:60], ", ".join(r.cite))))
            if r.inferred and len(r.inferred.split()) < MIN_REASON_WORDS:
                out.append(Refusal("D6", w, "the inferred reason has %d words; say which step needs what, and why no sentence gives it"
                                   % len(r.inferred.split())))
        cited = {k for r in d.requirements for k in r.cite}
        for e in d.exclude:
            why = plain_cell(e.get("why"))
            ks = as_keys(e.get("cites"))
            if not ks:
                out.append(Refusal("D9", "exclude", "an exclusion names no sentence"))
            if blank(why):
                out.append(Refusal("D9", "exclude %s" % ",".join(ks), "no reason: say why these sentences state no requirement"))
            for k in ks:
                if k not in src.by_key:
                    out.append(Refusal("D9", "exclude %s" % k, "not a sentence of the text"))
                elif k in cited:
                    out.append(Refusal("D9", "exclude %s" % k, "set aside AND cited by a requirement"))
        if blank(d.subject):
            out.append(Refusal("D7", "subject", "no name for the whole: a short name the producer will see in questions about the whole"))
        names = {plain_cell(a.get("name")).lower() for a in d.actors if isinstance(a, dict)}
        if not names:
            out.append(Refusal("D10", "actors", "no actor: who acts in this text?"))
        for g in d.goals:
            if self.actor_id(g.get("actor")) is None:
                out.append(Refusal("D10", "goal %s" % g.get("name"), "actor %r is not in actors" % g.get("actor")))
            for fld in ("name",) + FACETS:
                if isinstance(g.get(fld), str) and INTERNAL_ID.search(g[fld]):
                    out.append(Refusal("D12", "goal %s" % g.get("name"), "%s holds an internal id" % fld))
            for s in g.get("steps") or []:
                if isinstance(s, dict) and INTERNAL_ID.search(str(s.get("do") or "")):
                    out.append(Refusal("D12", "goal %s" % g.get("name"), "a step holds an internal id: %s" % s.get("do")))
            for e in g.get("ext") or []:
                if isinstance(e, dict) and INTERNAL_ID.search("%s %s" % (e.get("when") or "", e.get("do") or "")):
                    out.append(Refusal("D12", "goal %s" % g.get("name"), "an alternate path holds an internal id: %s" % (e.get("when") or e.get("do"))))
        return out

    def actor_id(self, ref):
        for i, a in enumerate(self.draft.actors, 1):
            if isinstance(a, dict) and (plain_cell(a.get("name")).lower() == plain_cell(ref).lower() or a.get("id") == ref):
                return a.get("id") or "A%d" % i
        return None

    # ---------------------------------------------------------------- the model
    def compile(self):
        """-> {file name: {top key: value}} for the model folder"""
        d, src = self.draft, self.source
        sid = "S-SRC"
        actors = []
        for i, a in enumerate(d.actors, 1):
            actors.append({"id": a.get("id") or "A%d" % i, "name": plain_cell(a.get("name")), "kind": a.get("kind") or "person",
                           "profile": plain_cell(a.get("profile")) or plain_cell(a.get("name"))})
        goals = []
        for gi, g in enumerate(d.goals, 1):
            uc = {"id": g.get("id") or "UC-%d" % gi, "name": plain_cell(g.get("name")), "level": g.get("level") or "sea",
                  "actor": self.actor_id(g.get("actor"))}
            for fld in FACETS:
                if not blank(g.get(fld)):
                    uc[fld] = plain_cell(g[fld])
            if g.get("gulf"):
                uc["gulf"] = copy.deepcopy(g["gulf"])
            if g.get("supporting"):
                uc["supporting"] = g["supporting"]
            uc["steps"] = [{k: v for k, v in (("do", plain_cell(s.get("do"))), ("reqs", as_keys(s.get("reqs"))), ("includes", as_keys(s.get("includes"))))
                            if v} for s in g.get("steps") or [] if isinstance(s, dict)]
            ext = []
            for e in g.get("ext") or []:
                x = {"at": e.get("at"), "when": plain_cell(e.get("when")), "do": e.get("do") if isinstance(e.get("do"), list) else [plain_cell(e.get("do"))],
                     "out": plain_cell(e.get("out")), "reqs": as_keys(e.get("reqs"))}
                if e.get("includes"):
                    x["includes"] = as_keys(e["includes"])
                ext.append(x)
            if ext:
                uc["ext"] = ext
            if g.get("parent"):
                uc["parent"] = g["parent"]
            goals.append(uc)
        epic_actor = self.actor_id(d.epic.get("actor")) if d.epic.get("actor") else next(
            (a["id"] for a in actors if a["kind"] == "person"), actors[0]["id"] if actors else None)
        epic = {"id": "EPIC-1", "name": d.subject or src.stem, "level": "kite", "actor": epic_actor}
        for fld in FACETS:
            if not blank(d.epic.get(fld)):
                epic[fld] = plain_cell(d.epic[fld])
        if d.epic.get("gulf"):
            epic["gulf"] = copy.deepcopy(d.epic["gulf"])
        top = [g for g in goals if not g.get("parent")]
        epic["steps"] = [{"do": g["name"], "includes": [g["id"]]} for g in top]
        for i, g in enumerate(top, 1):
            g["parent"] = {"uc": "EPIC-1", "step": i}
        if d.epic.get("ext"):
            epic["ext"] = copy.deepcopy(d.epic["ext"])
        carried = collections.defaultdict(list)
        for uc in goals + [epic]:
            for n, s in enumerate(uc["steps"], 1):
                for r in s.get("reqs") or []:
                    carried[r].append((uc, "step %d" % n))
        reqs = []
        for r in d.requirements:
            q = {"id": r.id, "kind": "inferred" if r.inferred else "authored", "text": r.says, "measure": r.check}
            if r.inferred:
                q["why"] = r.inferred
                if r.needs:
                    q["needs"] = r.needs
                if carried.get(r.id):
                    uc, where = carried[r.id][0]
                    q["at"] = "%s %s" % (uc["id"], where)
            else:
                q["sources"] = [sid]
                q["cite"] = list(r.cite)
                q["quote"] = r.quote
            reqs.append(q)
        cells = collections.OrderedDict((s.key, {"maps": []}) for s in src.sentences)
        for r in d.requirements:
            for k in r.cite:
                if k in cells:
                    cells[k]["maps"].append(r.id)
        for e in d.exclude:
            for k in as_keys(e.get("cites")):
                if k in cells and not cells[k]["maps"]:
                    cells[k] = {"excluded": plain_cell(e.get("why")), "by": "drafting agent"}
        story = next((s.text for s in src.sentences), "")
        files = collections.OrderedDict()
        files["meta.yaml"] = {"meta": {"subject": d.subject or src.stem, "mode": "greenfield", "story": {"text": story, "clauses": []},
                                       "grown_from": {"file": os.path.basename(src.origin), "kind": src.kind, "sha256_16": src.sha}}}
        files["people.yaml"] = {"actors": actors, "sources": [{
            "id": sid, "kind": "prose", "path": self.rel_source, "cite": "the text this model was grown from (%s)" % os.path.basename(src.origin)}]}
        files["requirements.yaml"] = {"requirements": reqs}
        files["usecases.yaml"] = {"usecases": [epic] + goals}
        files["spaces.yaml"] = {"spaces": [{"id": "S1", "title": "the sentences of %s" % os.path.basename(src.origin), "basis": "prose",
                                            "kind": "sentences", "source": sid, "cells": dict(cells)}]}
        files["answers.yaml"] = {"answers": []}
        return files

    def write(self, files):
        for name, body in files.items():
            if name == "answers.yaml" and os.path.exists(os.path.join(self.folder, name)):
                continue
            io.open(os.path.join(self.folder, name), "w", encoding="utf-8", newline="\n").write(dump(body))


def blank(v):
    return ucm.blank(v)


def dump(body):
    return yaml.safe_dump(body, default_flow_style=False, sort_keys=False, allow_unicode=True, width=1000)


# ---------------------------------------------------------------------- the brief and the skeleton

class Brief(object):
    """The text the agent in hand is given, and the skeleton it fills. BRIEF.md is the whole instruction; draft.yml carries one stub per sentence
    that states a rule, each already holding its sentence key and the sentence as the quote, so the agent edits words rather than copying keys."""

    def __init__(self, source, folder):
        self.source, self.folder = source, folder

    def skeleton(self):
        s = self.source
        subject = next(iter(s.headings.values()), "")      # a heading the text gives, else blank: the drafter names the whole, a file name would show up in questions
        lines = ["# DRAFT of the model for %s. Edit this file, then run the same command again." % os.path.basename(s.origin),
                 "# Read BRIEF.md first. Every requirement cites a sentence and quotes it, or is inferred with its reason; a requirement that does neither is refused.",
                 "# A FIELD of a goal the text does not state (trigger, pre, minimal, success, why, gulf) is LEFT OUT: it becomes a question, so write no guess. An INFERRED requirement is different: it is your reading of what a step needs, stated so the producer can confirm or correct it.",
                 "# Walk every step and write what it needs that no sentence gives as an INFERRED requirement (see BRIEF.md): that is where the useful questions come from.",
                 "subject: %s   # a short name for the whole; the producer sees it in questions about the whole" % json.dumps(subject),
                 "actors:   # who acts in the text. At least one person. kind: person | system | ai",
                 "  - {name: , kind: person}",
                 "requirements:   # one stub per sentence that carries a rule word. Rewrite `says` so a test could hold it; write `check`; keep cite and quote. Add the rest.",
                 ]
        n = 0
        for sent in s.sentences:
            if sent.rule:
                n += 1
                lines += ["  - id: R%d" % n, "    cite: [\"%s\"]" % sent.key, "    quote: %s" % json.dumps(sent.text), "    says: \"\"", "    check: \"\""]
        lines += ["  # an INFERRED requirement (nobody wrote it; a step needs it) looks like:",
                  "  # - {id: R%d, inferred: \"step 2 of 'Take a turn' needs <what>, and no sentence says where it comes from\", needs: \"<what, in plain words>\", says: \"...\", check: \"...\"}" % (n + 1),
                  "goals:   # 3 to 11 of them, each something a person does from start to finish, with 3 to 11 steps. Each step carries the requirement ids it realises (reqs).",
                  "  - name: ",
                  "    actor: ",
                  "    steps:",
                  "      - {do: , reqs: [R1]}",
                  "    ext:",
                  "      - {at: 1, when: , do: , out: \"rejoin 2\", reqs: []}   # an EXAMPLE line: fill it where the text or the chain shows an alternate path, else delete it. out: \"rejoin N\", \"end success\" or \"end minimal\"",
                  "exclude:   # sentences that state no requirement, each with its reason. Rule-bearing ones you set aside are asked about. The line below is an EXAMPLE: fill it or delete it.",
                  "  - {cites: [], why: }"]
        return "\n".join(lines) + "\n"

    def brief(self, command):
        s = self.source
        rel = "draft.yml"
        out = ["# Brief for the drafting agent: %s" % os.path.basename(s.origin), "",
               "The producer wrote the text below. Your job is to draft a requirements model from it WITHOUT inventing. The tool numbers the sentences, checks "
               "what you write, and refuses what it cannot ground. No model is called; you are the reader.", "",
               "## What to do", "",
               "1. Read the numbered sentences below (%s)." % s.describe(),
               "2. Edit `%s` in this folder (a skeleton is there, one stub per sentence that carries a rule word)." % rel,
               "3. Run `%s` again. Fix every refusal it prints. Repeat until it writes the model." % command, "",
               "## What the tool refuses (you cannot talk it round)", "",
               "- A requirement that cites no sentence and gives no inferred reason (D1). A key that is not a sentence below (D2).",
               "- A cite with no quote, a quote under two words, or words the cited sentence does not contain (D3, D4).",
               "- An inferred requirement whose reason is not a real sentence: say which step needs what, and why no sentence gives it (D6).",
               "- An exclusion with no reason, an actor nobody named, an internal id (R3, UC-2) in words a person will read (D9, D10, D12).", "",
               "## How to draft", "",
               "- One requirement per rule the text states. `says` is a statement a test could hold the finished work to. `check` is how anyone would see it hold or fail. "
               "Cite any sentence a requirement stands on, not only the ones marked [R].",
               "- **Then walk the chain and write what it needs.** For every step of every goal ask: does it use a number, a limit, a choice, an order, a condition or a "
               "word the text never defines ('a short way', 'small', 'slows')? Who decides? What if there is none, it is empty, it is blocked, it is already taken? What ends it? "
               "Each thing a step needs that NO sentence gives is an INFERRED requirement: `inferred` (the step and why no sentence gives it) and `needs` (the missing thing, in the "
               "producer's words). These are the questions the producer is asked first, so a draft with none has not been walked. Write the requirement as your best reading; the "
               "producer confirms or corrects it.",
               "- A vague word inside a sentence you cite ('small', 'several', 'quickly', 'as needed') is asked about by the tool on its own; you do not need to infer it.",
               "- The whole text needs 3 to 11 goals; a goal is something a person does from start to finish, with 3 to 11 steps. Each step carries the requirement ids it realises "
               "(`reqs`). Split or merge goals to fit, never pad a goal with a step the text does not support.",
               "- Give a goal an alternate path (`ext`: what happens when a step cannot complete, and where it ends) wherever the text or the chain shows one. A goal with none "
               "becomes a question for the producer; do not invent one to be rid of it.",
               "- A goal may also carry `trigger`, `pre` (what is true first), `minimal` (what holds if it fails), `success` and `why`. Fill one only from words in the text. If the "
               "text does not say, leave the field out: it becomes a question.",
               "- `subject` is a short name for the whole text; the producer will see it in questions about the whole.",
               "- Sentences that state no requirement go in `exclude` with a reason. A sentence with a rule word that you set aside is put to the producer, so do not set one aside "
               "because it is inconvenient.", "",
               "## The sentences (key, where it stands, [R] = carries a rule word)", ""]
        for sent in s.sentences:
            out.append("%-7s %s%s   (%s)" % (sent.key, "[R] " if sent.rule else "    ", sent.text, sent.where))
        if s.ignored:
            out += ["", "Fields in the export the tool did not read: %s." % ", ".join(s.ignored)]
        return "\n".join(out) + "\n"


# ---------------------------------------------------------------------- findings: the agent's errors and the producer's questions

DRAFTER_OK_RULES = {
    # (rule, message regex) -> becomes a question for the producer, not an error the drafter must fix
    ("U1", r"^no (trigger|pre|minimal|success|why)$"),
    ("U7", r"^unmapped"),
    ("U13", r"^no gulf of"),
    ("U17", r".*"),
    ("U4", r"^carries no requirement$"),
}


def is_question(finding):
    return any(finding.rule == r and re.search(rx, finding.message) for r, rx in DRAFTER_OK_RULES)


HINTS = {
    "U1": "a goal takes 3 to 11 steps, the whole takes 3 to 11 goals, a step must say something",
    "U2": "a goal below another needs a parent step that exists",
    "U3": "every step of the whole must lead to a goal",
    "U4": "an alternate path needs a step to leave from (at), a cause (when), what happens (do), where it ends (out) and a requirement (reqs)",
    "U5": "a requirement no step carries: put its id in the reqs of the step that realises it, or delete it",
    "U6": "an authored requirement cites a sentence; an inferred one gives its reason",
    "U8": "an id that is not in the draft: fix the spelling or add the requirement",
    "U9": "a person listed in actors who acts in no goal: use them as a goal's actor or take them out",
    "U10": "a requirement with no check",
    "U12": "a step must carry a requirement",
    "U15": "two things with one id",
    "U16": "a requirement only the whole's own steps carry: a goal's step must carry it",
}


def split_findings(findings):
    """-> (the drafter's errors, the producer's questions): the 19 rules' findings that only the drafter can fix, and those that are holes in the
    text itself"""
    mine = [f for f in findings if not is_question(f)]
    theirs = [f for f in findings if is_question(f)]
    return mine, theirs


def second_reader(source, folder, model):
    """source_rules.check run over the model's own cites: a rule sentence of the text that no requirement's line covers. A second, line-based
    reader of the same text, so the sentence-exact space and this one disagree out loud when they do."""
    sp = next((s for s in model.spaces.values() if s.kind == "sentences"), None)
    if sp is None:
        return None
    lines = []
    n = 0
    for c in sp.cells.values():
        for r in c.maps:
            req = model.requirements.get(r)
            if req is not None and req.kind != "inferred":
                n += 1
                lines.append("REQ-G-%d %s :%s" % (n, re.sub(r"\s+", " ", req.text or ""), c.key.split(".")[0]))
    # a run of set-aside lines is ONE range (source_rules reads a run of list items as one multi-line sentence, and an exclusion covers it only whole); a single blank line between two
    # set-aside lines is bridged. Measured on a public proposal: three false "uncited" hits became none.
    aside = sorted({int(c.key.split(".")[0]): c.excluded for c in sp.cells.values() if not c.maps and not blank(c.excluded)}.items())
    excluded = []
    for line, why in aside:
        if excluded and line - excluded[-1][1] <= 2:
            excluded[-1] = (excluded[-1][0], line, excluded[-1][2])
        else:
            excluded.append((line, line, why))
    d = tempfile.mkdtemp(prefix="grow_sr_")
    try:
        p = os.path.join(d, "reqs.txt")
        io.open(p, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        sents, holes = source_rules.check(os.path.join(folder, "source", source.stem + ".txt"), p, excluded)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return len(sents), [h[1] for h in holes]


def grow(args, out=print):
    try:
        source = SourceText.read(args.source, args.kind)
    except SourceError as e:
        out("grow: CANNOT READ THE TEXT: %s" % e)
        return 2
    except (OSError, ValueError, json.JSONDecodeError, csv.Error) as e:
        out("grow: CANNOT READ THE TEXT %s: %s" % (args.source, e))
        return 2
    folder = os.path.abspath(args.out)
    os.makedirs(folder, exist_ok=True)
    existing = os.path.join(folder, "answers.yaml")
    if os.path.exists(existing):
        done = (yaml.safe_load(io.open(existing, encoding="utf-8")) or {}).get("answers") or []
        if done and not args.force:
            out("grow: REFUSED: %s already holds %d recorded answer(s); growing again would discard them. Use `ask`, or grow into a new folder." % (folder, len(done)))
            return 1
    rel = source.write(folder)
    cmd = "closed_loop.py grow %s --out %s" % (args.source, args.out)
    draft_path = os.path.join(folder, "draft.yml")
    brief = Brief(source, folder)
    if not os.path.exists(draft_path):
        io.open(draft_path, "w", encoding="utf-8", newline="\n").write(brief.skeleton())
    io.open(os.path.join(folder, "BRIEF.md"), "w", encoding="utf-8", newline="\n").write(brief.brief(cmd))
    try:
        draft = Draft.load(draft_path)
    except SourceError as e:
        out("grow: %s" % e)
        return 1
    if not draft.filled() or not any(r.says for r in draft.requirements):
        out("GROW  %s  (%s)" % (os.path.basename(args.source), source.describe()))
        out("  wrote %s/%s, %s/draft.yml (a skeleton), %s/BRIEF.md" % (args.out, rel, args.out, args.out))
        out("")
        text = open(os.path.join(folder, "BRIEF.md"), encoding="utf-8").read()
        if len(source.sentences) <= 80:
            out(text.rstrip())
        else:
            out(text.split("## The sentences")[0].rstrip())
            out("")
            out("## The sentences: %d of them, in %s/BRIEF.md" % (len(source.sentences), args.out))
        out("")
        out("  Fill %s/draft.yml from the sentences, then run the same command again:" % args.out)
        out("      %s" % cmd)
        return 3
    g = Grower(source, draft, folder)
    refusals = g.refuse()
    out("GROW  %s  (%s)" % (os.path.basename(args.source), source.describe()))
    if refusals:
        out("  REFUSED: %d defect(s) in %s/draft.yml; nothing was written to the model." % (len(refusals), args.out))
        for r in refusals[:40]:
            out("    %s" % r)
        if len(refusals) > 40:
            out("    ... %d more" % (len(refusals) - 40))
        return 1
    files = g.compile()
    tmp = tempfile.mkdtemp(prefix="grow_try_")
    try:
        shutil.copytree(os.path.join(folder, "source"), os.path.join(tmp, "source"))
        for name, body in files.items():
            io.open(os.path.join(tmp, name), "w", encoding="utf-8", newline="\n").write(dump(body))
        model = ucm.UseCaseModel.load(tmp)
        findings = model.check()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    mine, theirs = split_findings(findings)
    if mine:
        out("  The draft cites and grounds, but the structure has %d error(s) only the drafter can fix; nothing was written to the model:" % len(mine))
        for f in mine[:40]:
            out("    %s" % f)
            if f.rule in HINTS:
                out("        -> %s" % HINTS[f.rule])
        return 1
    g.write(files)
    model = ucm.UseCaseModel.load(folder)
    out("  wrote the model to %s/ (people, requirements, usecases, spaces, answers; the text in source/)" % args.out)
    kinds = collections.Counter(r.kind for r in model.requirements.values())
    cites = sum(1 for r in draft.requirements if r.cite)
    out("  draft       %d requirement(s): %d cite a sentence, %d are inferred; %d goal(s), %d step(s), %d alternate path(s)" % (
        len(draft.requirements), cites, kinds.get("inferred", 0), len(draft.goals), sum(len(u.steps) for u in model.usecases.values()),
        sum(len(u.ext) for u in model.usecases.values())))
    sp = next(s for s in model.spaces.values() if s.kind == "sentences")
    n_cited = sum(1 for c in sp.cells.values() if c.maps)
    n_set = sum(1 for c in sp.cells.values() if not c.maps and not blank(c.excluded))
    out("  sentences   %d: %d cited, %d set aside with a reason, %d with neither" % (len(sp.cells), n_cited, n_set, len(sp.cells) - n_cited - n_set))
    warn = [r for r in draft.requirements if r.cite and not (set(source_rules.words(r.says)) & set(source_rules.words(r.quote)))]
    for r in warn:
        out("  WARN        %s shares no word with its own quote; a person should read it" % r.id)
    if not any(r.inferred for r in draft.requirements):
        out("  WARN        no requirement is INFERRED: every step of a chain needs something, and a draft that found nothing missing has not walked it (BRIEF.md, 'walk the chain')")
    sr = second_reader(source, folder, model)
    if sr is not None:
        out("  second reader (source_rules, by line): %d rule sentence(s) read, %d that no requirement's line covers" % (sr[0], len(sr[1])))
    import ask
    nq = len(ask.Queue(folder).open)
    out("  the 19 rules: %s" % ("PASS" if not findings else "%d open hole(s) (%s), none an error of the draft; they come to %d question(s) for the producer" % (
        len(findings), ", ".join("%s x%d" % (k, v) for k, v in sorted(collections.Counter(f.rule for f in findings).items(), key=lambda kv: int(kv[0][1:]))), nq)))
    out("")
    ask.show_next(folder, out=out, header="THE FIRST QUESTION", display=args.out)
    return 0        # the model is written; whether a question is open is `ask`'s exit code, not this one's


# ---------------------------------------------------------------------- the selftest: a small text, a clean draft, and one planted defect per refusal

FIXTURE_TEXT = """Seed Race is a game for two players.
The board has 12 pits, six for each player.
On a turn a player picks up every seed in one of their own pits and sows them one at a time into the pits that follow.
A player cannot pick up from an empty pit.
The player with the most seeds in their store when one side is empty wins.
"""


def fixture_draft():
    """a draft of FIXTURE_TEXT that grounds and passes the structure checks; its goals leave the facets blank, so the producer has questions"""
    return {
        "subject": "Seed Race",
        "actors": [{"name": "Player", "kind": "person"}],
        "requirements": [
            {"id": "R1", "cite": ["2.1"], "quote": "The board has 12 pits", "says": "The board has 12 pits, six for each player.",
             "check": "count the pits on a new board"},
            {"id": "R2", "cite": ["3.1"], "quote": "picks up every seed in one of their own pits",
             "says": "A turn picks up every seed in one of the player's own pits and sows them one at a time into the pits that follow.",
             "check": "play a turn and count the seeds in each pit"},
            {"id": "R3", "cite": ["4.1"], "quote": "cannot pick up from an empty pit", "says": "A player cannot pick up from an empty pit.",
             "check": "choose an empty pit: the choice is refused"},
            {"id": "R4", "cite": ["5.1"], "quote": "the most seeds in their store", "says": "When one side is empty the player with the most seeds in their store wins.",
             "check": "empty one side and compare the stores"},
            {"id": "R5", "cite": ["1.1"], "quote": "a game for two players", "says": "Two players take part.", "check": "start a game and count the players"},
            {"id": "R6", "inferred": "step 3 of 'Start a game' puts seeds in the pits, and no sentence says how many each pit starts with",
             "needs": "how many seeds each pit starts with", "says": "Every pit starts with the same number of seeds.", "check": "count the seeds in each pit of a new board"},
            {"id": "R7", "inferred": "step 4 of 'Start a game' picks who moves first, and no sentence says who that is or how it is decided",
             "needs": "who moves first", "says": "The game decides which player moves first.", "check": "start ten games and see who moves first"},
        ],
        "goals": [
            {"name": "Start a game", "actor": "Player", "steps": [
                {"do": "Two players sit down", "reqs": ["R5"]}, {"do": "The pits are laid out", "reqs": ["R1"]},
                {"do": "Seeds are placed in the pits", "reqs": ["R6"]}, {"do": "The first player is chosen", "reqs": ["R7"]}],
             "ext": [{"at": 4, "when": "nobody can decide who starts", "do": "the game decides again", "out": "rejoin 4", "reqs": ["R7"]}]},
            {"name": "Take a turn", "actor": "Player", "steps": [
                {"do": "The player chooses one of their own pits", "reqs": ["R3"]}, {"do": "The player picks up every seed in it", "reqs": ["R2"]},
                {"do": "The seeds are sown one at a time into the pits that follow", "reqs": ["R2"]}],
             "ext": [{"at": 1, "when": "the chosen pit is empty", "do": "the choice is refused", "out": "rejoin 1", "reqs": ["R3"]}]},
            {"name": "Finish the game", "actor": "Player", "steps": [
                {"do": "One side is found empty", "reqs": ["R4"]}, {"do": "The stores are compared", "reqs": ["R4"]},
                {"do": "The winner is announced", "reqs": ["R4"]}]},
        ],
    }


def write_fixture(folder, draft=None, text=FIXTURE_TEXT):
    """-> (text path, model folder): the fixture text on disk and, if a draft is given, draft.yml already in the model folder"""
    os.makedirs(folder, exist_ok=True)
    src = os.path.join(folder, "epic.txt")
    io.open(src, "w", encoding="utf-8", newline="\n").write(text)
    model = os.path.join(folder, "model")
    os.makedirs(model, exist_ok=True)
    if draft is not None:
        io.open(os.path.join(model, "draft.yml"), "w", encoding="utf-8", newline="\n").write(dump(draft))
    return src, model


def run_grow(src, model, **kw):
    """grow() with its words captured -> (exit code, text)"""
    lines = []
    ns = argparse.Namespace(source=src, out=model, kind=kw.get("kind"), force=kw.get("force", False))
    rc = grow(ns, out=lambda s="": lines.append(s))
    return rc, "\n".join(lines)


def selftest():
    tmp = tempfile.mkdtemp(prefix="grow_selftest_")
    ok = True

    def check(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print("selftest: %-78s %s%s" % (name, "PASS" if good else "FAIL", "" if good else "  " + str(detail)))

    def planted(name, code, mutate, text=FIXTURE_TEXT):
        d = copy.deepcopy(fixture_draft())
        mutate(d)
        src, model = write_fixture(os.path.join(tmp, re.sub(r"\W+", "_", name)), d, text)
        rc, said = run_grow(src, model)
        written = os.path.exists(os.path.join(model, "requirements.yaml"))
        check("PLANT %-34s -> refused %s, nothing written" % (name, code), rc == 1 and ("[%s " % code) in said and not written, said[-300:])

    # the control first: a gate that has only ever passed proves nothing, and one that only ever refuses is broken the other way
    src, model = write_fixture(os.path.join(tmp, "clean"), fixture_draft())
    rc, said = run_grow(src, model)
    check("CONTROL a clean draft is written (exit 0) and the first question printed", rc == 0 and os.path.exists(os.path.join(model, "requirements.yaml"))
          and "THE FIRST QUESTION" in said, said[-400:])
    m = ucm.UseCaseModel.load(model)
    mine, theirs = split_findings(m.check())
    check("CONTROL the 19 rules: no error left for the drafter, %d hole(s) left as questions" % len(theirs), not mine and theirs, "; ".join(map(str, mine[:3])))
    check("CONTROL every requirement carries its sentence or its chain reason",
          all((r.kind == "inferred" and r.why) or (r.kind == "authored" and "S-SRC" in r.sources) for r in m.requirements.values()))
    check("CONTROL the sentence space has one cell per sentence, all cited", len(m.spaces["S1"].cells) == 5 and all(c.maps for c in m.spaces["S1"].cells.values()))
    sr = second_reader(SourceText.read(src), model, m)
    check("CONTROL the second reader (source_rules) finds no rule sentence uncited", sr is not None and sr[1] == [], str(sr))
    io.open(os.path.join(model, "answers.yaml"), "w", encoding="utf-8").write("answers:\n- {n: 1, key: x, kind: absent}\n")
    rc, said = run_grow(src, model)
    check("CONTROL a second grow over recorded answers is refused, not silently overwritten", rc == 1 and "already holds" in said, said)

    # the plants: each refusal fires, and nothing is written
    planted("no citation and no reason", "D1", lambda d: d["requirements"][0].pop("cite"))
    planted("a sentence that is not there", "D2", lambda d: d["requirements"][0].update(cite=["9.9"]))
    planted("a cite with no quote", "D3", lambda d: d["requirements"][0].pop("quote"))
    planted("a one-word quote", "D3", lambda d: d["requirements"][0].update(quote="board"))
    planted("a quote the sentence lacks", "D4", lambda d: d["requirements"][0].update(quote="The board has 14 pits"))
    planted("cited AND inferred", "D5", lambda d: d["requirements"][5].update(cite=["1.1"], quote="a game for two players"))
    planted("an inferred reason of three words", "D6", lambda d: d["requirements"][5].update(inferred="the chain needs it"))
    planted("a requirement with no check", "D7", lambda d: d["requirements"][1].pop("check"))
    planted("two requirements, one id", "D8", lambda d: d["requirements"][1].update(id="R1"))
    planted("an exclusion with no reason", "D9", lambda d: (d["requirements"][4].update(inferred="step 1 of 'Start a game' seats two players and no sentence says how many"),
                                                           d["requirements"][4].pop("cite"), d["requirements"][4].pop("quote"), d.update(exclude=[{"cites": ["1.1"]}])))
    planted("a sentence set aside AND cited", "D9", lambda d: d.update(exclude=[{"cites": ["2.1"], "why": "background"}]))
    planted("an internal id in what a need is", "D12", lambda d: d["requirements"][5].update(needs="what R3 says about the start"))
    planted("an internal id in an alternate path", "D12", lambda d: d["goals"][1]["ext"][0].update(when="R3 is empty"))
    planted("no name for the whole", "D7", lambda d: d.pop("subject"))
    planted("an actor nobody named", "D10", lambda d: d["goals"][0].update(actor="Ghost"))
    planted("an internal id in words a person reads", "D12", lambda d: d["requirements"][1].update(says="R3 holds at every pit"))
    # a draft the structure rejects (a step carrying a requirement that does not exist): grounded, but not written
    d = fixture_draft()
    d["goals"][1]["steps"][0]["reqs"] = ["R99"]
    src2, model2 = write_fixture(os.path.join(tmp, "unknown_ref"), d)
    rc, said = run_grow(src2, model2)
    check("PLANT a step carrying an unknown requirement -> an error for the drafter, nothing written",
          rc == 1 and "U8" in said and "-> an id that is not in the draft" in said and not os.path.exists(os.path.join(model2, "requirements.yaml")), said[-300:])
    # a sentence nothing cites stays a hole and a question
    d = fixture_draft()
    d["requirements"][4].update(inferred="step 1 of 'Start a game' seats two players and no sentence says how many", needs="how many players")
    d["requirements"][4].pop("cite")
    d["requirements"][4].pop("quote")
    src3, model3 = write_fixture(os.path.join(tmp, "uncited"), d)
    rc, said = run_grow(src3, model3)
    holes = [f for f in ucm.UseCaseModel.load(model3).check() if f.rule == "U7"]
    check("PLANT a sentence nothing cites -> a U7 hole on exactly that sentence, and a question", rc == 0 and len(holes) == 1 and holes[0].where == "S1:1.1"
          and "THE FIRST QUESTION" in said, said[-300:])
    # a RULE sentence nothing cites is what the second reader (source_rules, by line) reports as well
    d = fixture_draft()
    d["requirements"][2].update(inferred="step 1 of 'Take a turn' needs the rule about empty pits and the agent could not find it", needs="what to do about an empty pit")
    d["requirements"][2].pop("cite")
    d["requirements"][2].pop("quote")
    src3b, model3b = write_fixture(os.path.join(tmp, "uncited_rule"), d)
    run_grow(src3b, model3b)
    m3b = ucm.UseCaseModel.load(model3b)
    sr3 = second_reader(SourceText.read(src3b), model3b, m3b)
    check("PLANT a rule sentence nothing cites -> the second reader (source_rules) reports it too, by its line", sr3 is not None and [h.first for h in sr3[1]] == [4]
          and [f.where for f in m3b.check() if f.rule == "U7"] == ["S1:4.1"], (sr3 and [h.first for h in sr3[1]], [f.where for f in m3b.check() if f.rule == "U7"]))
    # an agent that sets a rule-bearing sentence aside is asked about it
    d = fixture_draft()
    d["requirements"][2].update(inferred="step 1 of 'Take a turn' needs the rule about empty pits and the agent could not find it", needs="what to do about an empty pit")
    d["requirements"][2].pop("cite")
    d["requirements"][2].pop("quote")
    d["exclude"] = [{"cites": ["4.1"], "why": "background"}]
    src4, model4 = write_fixture(os.path.join(tmp, "setaside"), d)
    run_grow(src4, model4)
    import ask
    q4 = ask.Queue(model4)
    check("PLANT a rule sentence set aside as background -> put to the producer", any(h.key == "uncited:S1:4.1" for h in q4.holes), [h.key for h in q4.holes])

    # empty and unreadable text fails loudly
    for name, body, why in (("empty file", "", "is empty"), ("blank lines only", "\n\n  \n", "is empty"), ("headings only", "# Title\n## Section\n", "no sentences")):
        src5, model5 = write_fixture(os.path.join(tmp, re.sub(r"\W+", "_", name)), None, body)
        rc, said = run_grow(src5, model5)
        check("PLANT %-30s -> exit 2, loud (%s), no model" % (name, why), rc == 2 and "CANNOT READ" in said and why in said
              and not os.path.exists(os.path.join(model5, "requirements.yaml")), said)
    rc, said = run_grow(os.path.join(tmp, "no_such_file.txt"), os.path.join(tmp, "x"))
    check("PLANT a file that does not exist -> exit 2", rc == 2 and "CANNOT READ" in said, said)
    # a skeleton, then wait
    src6, model6 = write_fixture(os.path.join(tmp, "skeleton"), None)
    rc, said = run_grow(src6, model6)
    sk = io.open(os.path.join(model6, "draft.yml"), encoding="utf-8").read()
    check("CONTROL first run writes a skeleton with one stub per rule sentence and waits (exit 3)",
          rc == 3 and sk.count("- id: R") == 2 and os.path.exists(os.path.join(model6, "BRIEF.md")) and os.path.exists(os.path.join(model6, "source", "map.json")), said[-300:])
    rc, said = run_grow(src6, model6)
    check("CONTROL a skeleton left unfilled stays at exit 3, never an empty model", rc == 3 and not os.path.exists(os.path.join(model6, "requirements.yaml")), said[-200:])

    # the other kinds of text
    csvf = os.path.join(tmp, "tickets.csv")
    io.open(csvf, "w", encoding="utf-8", newline="").write(
        "Key,Summary,Description,Status,Assignee\n"
        "SR-1,Players take turns,\"A player picks up every seed in a pit.\nA player cannot pick up from an empty pit.\",Open,Sam\n"
        "SR-2,A winner is found,The player with the most seeds wins when one side is empty.,Open,Ana\n")
    s = SourceText.read(csvf)
    check("TICKETS a CSV export is read as tickets, each sentence labelled with its ticket and field",
          s.kind == "tickets" and len(s.sentences) == 5 and s.by_key["2.1"].where == "SR-1 description" and "status" in s.ignored, [(x.key, x.where) for x in s.sentences])
    jf = os.path.join(tmp, "tickets.json")
    io.open(jf, "w", encoding="utf-8").write(json.dumps({"issues": [{"key": "SR-9", "fields": {"summary": "Pits start full", "description": {"type": "doc", "content": [
        {"type": "paragraph", "content": [{"type": "text", "text": "Every pit starts with four seeds."}]}]}, "labels": ["x"]}}]}))
    s = SourceText.read(jf)
    check("TICKETS a JSON export with rich-text bodies is read", s.kind == "tickets" and [x.where for x in s.sentences] == ["SR-9 title", "SR-9 description"],
          [(x.key, x.where) for x in s.sentences])
    tf = os.path.join(tmp, "talk.txt")
    io.open(tf, "w", encoding="utf-8").write("[00:00:05] Ana: We want a game for two players. Nobody can pick up from an empty pit.\nSam: Right.\n[00:00:40] Ana: The one with most seeds wins.\n")
    s = SourceText.read(tf)
    check("TRANSCRIPT speaker turns are read as a transcript, each sentence labelled with its speaker and time",
          s.kind == "transcript" and s.by_key["1.2"].where == "Ana, at 00:00:05" and any(x.text == "Right." for x in s.sentences), [(x.key, x.where, x.text) for x in s.sentences])
    md = os.path.join(tmp, "prd.md")
    io.open(md, "w", encoding="utf-8").write("# PRD\n\n## Goals\n\n- Players must be able to pick up seeds.\n- [ ] The game ends when one side is empty.\n\n"
                                             "| Rule | Note |\n|---|---|\n| Empty pit | cannot be picked |\n\n```\ncode block ignored.\n```\n")
    s = SourceText.read(md)
    check("DOCUMENT a PRD: headings, fences and table rules are no sentences; lines keep their numbers; a bullet loses its marker",
          s.kind == "document" and [x.key for x in s.sentences] == ["5.1", "6.1", "10.1"] and s.by_key["5.1"].text.startswith("Players must") and "under \"Goals\"" in s.by_key["5.1"].where,
          [(x.key, x.text, x.where) for x in s.sentences])

    wrapped = os.path.join(tmp, "wrapped.md")
    io.open(wrapped, "w", encoding="utf-8").write("# KEP\n\n<!-- toc -->\n- [Summary](#summary)\n<!-- /toc -->\n\n[site]: https://example.org/\n\n"
                                                  "The current algorithm prefers to delete Pods that\nhave been running for the least time. This heuristic attempts to\nminimize disruption.\nIt states one rule per line.\nAnd another.\n\n"
                                                  "<!--\nNote to authors: this is guidance.\n-->\nA second paragraph states a rule.\n")
    s = SourceText.read(wrapped)
    check("DOCUMENT a hard-wrapped text is one sentence stream, keyed by the line each paragraph starts on; comments, link definitions and a table of contents are not sentences",
          [(x.key, x.text) for x in s.sentences] == [("9.1", "The current algorithm prefers to delete Pods that have been running for the least time."),
                                                      ("9.2", "This heuristic attempts to minimize disruption."), ("12.1", "It states one rule per line."), ("13.1", "And another."),
                                                      ("18.1", "A second paragraph states a rule.")],
          [(x.key, x.text) for x in s.sentences])

    runs = os.path.join(tmp, "runs.md")
    io.open(runs, "w", encoding="utf-8").write("# Notes\n\n- A pit must be filled\n- A pit cannot be emptied\n- A seed only moves forward\n\nA player wins when one side is empty.\n")
    sr_src = SourceText.read(runs)
    sr_model = os.path.join(tmp, "runs_model")
    os.makedirs(sr_model)
    sr_src.write(sr_model)
    for aside, label in (([k for k in sr_src.by_key if sr_src.by_key[k].line in (3, 4, 5)], "a run of set-aside list items is one exclusion"),
                         ([k for k in sr_src.by_key if sr_src.by_key[k].line in (3, 4)], "a run with one rule sentence left out of it")):
        cells = {k: ({"excluded": "a status list"} if k in aside else {"maps": [] if sr_src.by_key[k].line == 5 else ["R1"]}) for k in sr_src.by_key}
        for name, body in {"people.yaml": {"actors": [{"id": "A1", "name": "P", "kind": "person"}], "sources": [{"id": "S-SRC", "kind": "prose", "path": "source/runs.txt"}]},
                           "spaces.yaml": {"spaces": [{"id": "S1", "basis": "prose", "kind": "sentences", "source": "S-SRC", "cells": cells}]},
                           "requirements.yaml": {"requirements": [{"id": "R1", "kind": "authored", "text": "A player wins when one side is empty.", "measure": "m", "sources": ["S-SRC"]}]}}.items():
            io.open(os.path.join(sr_model, name), "w", encoding="utf-8").write(dump(body))
        res = second_reader(sr_src, sr_model, ucm.UseCaseModel.load(sr_model))
        want = 0 if len(aside) == 3 else 1
        check("SECOND READER %s -> %d rule sentence(s) uncited" % (label, want), res is not None and len(res[1]) == want, res)

    # the brief names every sentence, with its place
    br = Brief(SourceText.read(src), model).brief("closed_loop.py grow epic.txt --out model")
    check("BRIEF lists all 5 sentences with their keys and marks the rule-bearing ones", all(k in br for k in ("1.1", "2.1", "3.1", "4.1", "5.1")) and sum(1 for l in br.splitlines() if re.match(r"^\d+\.\d+\s+\[R\]", l)) == 2, br[-500:])
    shutil.rmtree(tmp, ignore_errors=True)
    print("selftest: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["grow"])
    ap.add_argument("source", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--kind", choices=KINDS)
    ap.add_argument("--force", action="store_true", help="grow over a folder that already holds recorded answers (discards them)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd != "grow" or not a.source or not a.out:
        ap.error("grow <text file> --out <model folder>, or --selftest")
    return grow(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
