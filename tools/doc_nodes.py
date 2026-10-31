#!/usr/bin/env python3
"""doc_nodes.py - the documents as nodes in the knowledge graph, so a change can say which documents it makes stale.

WHY: a requirement, a stat, a goal or a piece of code in the graph had no link to the documents that describe it. When a number or a
behaviour changed nothing said what to update: measured 2026-10-30, five public numbers went stale in one day and were found only by hand.

WHAT: each addressable piece of a document is a node of kind `doc` (a manual section, a slide, a concept page, a row of a list, a section
of a markdown file). A `documented in` edge runs from a requirement, stat, goal or code node to the piece that describes it.

NO GUESSING. A link exists for one of two reasons, and the reason is stored beside it:
  - the piece LITERALLY names the node: its requirement id, its qualified symbol, its stat name, its goal's title. The rule is exact tokens,
    what matched and at which file and line. A plain English word is never a name: a goal called `tests` or a stat called `killed` does not
    make every sentence with that word a link.
  - an AUTHORED line in the model's docs file says so, with the words the piece uses (`says`, which must still be in the piece) and why.
Never similarity. A node with no link is a finding, counted and named, not an error to hide.

THE CLASSES
  DocPiece      one addressable piece of a document, with its text and the way back to file and line
  DocReader     how one kind of document splits into pieces (HtmlSections, Markdown, Deck, Concepts, GoalRows, StatRows)
  DocSet        every document a model declares, read; refuses an empty set. `quoting` finds a literal string across them
  NameIndex     what a document could name in this graph, and the exact rule for each kind of name
  AuthoredLinks the links the text makes in other words, each re-verified against the piece on every build
  LinkLedger    the links last accepted (<model>/doc_links.lock), which is how a link that vanishes is seen as LOST
  DocOverlay    writes all of it into a graph store: entity, edge, literal
  DocGate       reads the graph back into holes for the closed loop's criterion
  DocIndex      answers "what must I update if this changes?" from the graph (`what-to-update`); the API a claim registry builds on

HOW A MODEL OPTS IN. Any .yaml in the model folder may hold (the one loader merges meta):
    meta:
      docs:
        sets: [path/to/docset.yaml]       # shared lists of documents; paths inside one are relative to that file
        documents:                        # this model's own; paths relative to the project root, as its code roots are
          - {as: guide, read: markdown, path: docs/guide.md}
        links:
          - {node: "req:W-1", doc: "doc:guide#paying", says: "the wallet refuses", why: "the section says this requirement in other words"}
        known:                            # a phantom found and not yet settled: listed with its why, failing again once it lapses
          - {names: W-9, why: "the guide still lists it; the model dropped it; its owner decides which is right"}
A model that declares nothing is told L20 was not measured. A model that declares documents that read as nothing FAILS: zero holes from zero
documents would be a lie.

    python -X utf8 doc_nodes.py attach <store> --model <folder> [--set docset.yaml] --out <copy>     never writes the store it reads
    python -X utf8 doc_nodes.py report <store> [--verify]
    python -X utf8 doc_nodes.py what-to-update <node id> --db <store> [--json]
    python -X utf8 doc_nodes.py quoting "<literal text>" --model <folder> [--set docset.yaml]
    python -X utf8 doc_nodes.py accept <model folder> --db <store> --why "<why>"
    python -X utf8 doc_nodes.py sample <store> [--n 20] [--seed 1]                                   links with their source line, for reading
    python -X utf8 doc_nodes.py --selftest
"""
import argparse
import ast
import collections
import contextlib
import html
import io
import json
import os
import random
import re
import shutil
import sqlite3
import sys
import tempfile

try:
    import yaml
except ImportError:  # as usecase_model.py: the loop must be able to say PyYAML is missing, and this is imported whenever it starts
    yaml = None

HERE = os.path.dirname(os.path.abspath(__file__))
# the project the model's paths are relative to: this repo, or a developer's own (CLOSED_LOOP_ROOT, set by the kit's installer)
ROOT = os.path.abspath(os.environ.get("CLOSED_LOOP_ROOT") or os.path.join(HERE, "..", ".."))

DOCUMENTED_IN, BUILT_FROM, LINKS_TO = 153, 154, 164
BECAUSE, NAMES_NOTHING, LINK_LOST, FOUND_AT, LINK_BROKEN, NAMES_OWNER, KNOWN_PHANTOM = 155, 156, 157, 158, 159, 165, 166
PROPS = [(DOCUMENTED_IN, "documented in", "described by a document"), (BUILT_FROM, "built from", "a page made from this section"),
         (BECAUSE, "documented because", "what matched and where"), (NAMES_NOTHING, "names nothing", "a document names what the model does not hold"),
         (LINK_LOST, "link lost", "the ledger holds it, the documents no longer make it"), (FOUND_AT, "found at", "file and line of a piece"),
         (LINK_BROKEN, "link broken", "an authored line that no longer resolves"), (LINKS_TO, "links to", "the document's own hyperlink to another document"),
         (NAMES_OWNER, "names the owner", "a piece names a type, a module or a use case, not the node itself"),
         (KNOWN_PHANTOM, "known to name nothing", "a phantom declared, with why, until its owner settles it")]
RELATIONS = (DOCUMENTED_IN, BUILT_FROM, LINKS_TO)                                      # edges: measurements of the documents, not of the model
LITERALS = (BECAUSE, NAMES_NOTHING, LINK_LOST, FOUND_AT, LINK_BROKEN, NAMES_OWNER, KNOWN_PHANTOM)     # so no fingerprint reads them and no walk crosses them
KINDS = ("doc", "stat", "goal")                                                        # the kinds this adds: a reader of the model leaves them out
FAMILY_DOCUMENT, FAMILY_AUTHORED = 64, 16     # the edge's evidence family: the document's own words, or the model author's (usecase_kg.MODEL)
NOT_DECLARED = "not declared"
UNREADABLE = "UNREADABLE: "
LEDGER = "doc_links.lock"
BLOCK_TAGS = {"p", "li", "tr", "dd", "dt", "h1", "h2", "h3", "h4", "h5", "h6", "div", "pre", "section", "ul", "ol", "table", "aside", "details",
              "summary", "blockquote", "figure", "figcaption"}
SNAKE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+")      # a stat or goal id no sentence uses by accident
PLAIN_NAME = re.compile(r"[A-Za-z_]\w*")


class DocSetError(RuntimeError):
    """A declared document could not be read, or read as nothing. Never swallowed: it becomes a failing hole and a non-zero exit."""


class DocPiece(object):
    """One addressable piece of a document: a manual section, a slide, a concept page, a row, a markdown section. Its id comes from what it
    is (`doc:<document>#<key>`), never a position, so a rebuild keeps it. It keeps its readable text and the way back to file and line, so a
    link can say where it matched and a reader can go and look. Evidence: the file at `path`."""

    def __init__(self, document, key, sub, title, path, line=None, raw="", text=None, offsets=None, spans=(), blocks=(), built_from=(), hrefs=(), row=None):
        self.document, self.key, self.sub, self.title, self.path, self.line = document, key, sub, title, path, line
        self.raw, self.text = raw, raw if text is None else text
        self.offsets, self.spans, self.blocks = offsets, list(spans), list(blocks)
        self.built_from, self.hrefs, self.row = list(built_from), list(hrefs), row

    @property
    def id(self):
        return "doc:%s#%s" % (self.document, self.key)

    def line_at(self, i):
        """the file line of the plain text's offset i"""
        return self.line + self.raw.count("\n", 0, self.offsets[i] if self.offsets is not None else i)

    def where(self, i=0):
        """file and line of a place in the piece. A row's words are put together from its fields, so no line holds them: a row says so"""
        if self.row:
            return "%s, row %s" % (self.path, self.row)
        return "%s:%d" % (self.path, self.line_at(i) if self.text else self.line)

    def found_at(self):
        """where the piece starts, as the graph records it"""
        return "%s:%d" % (self.path, self.line) if self.line is not None else self.where()

    def block_at(self, i):
        for a, b in self.blocks:
            if a <= i < b:
                return self.text[a:b]
        return self.text

    def says(self, words):
        """the piece holds these words in this order, whatever the white space between them; returns the offset or None"""
        m = re.search(r"\s+".join(re.escape(w) for w in str(words).split()), self.text) if str(words).split() else None
        return m.start() if m else None


class Plain(object):
    """Turns a document's source into what a reader sees, keeping for each character the offset it came from. The block boundaries (a
    paragraph, a list item, a table row) and the code-formatted spans survive, because the naming rules read them."""

    @staticmethod
    def of_html(raw):
        text, offs, spans, blocks = [], [], [], []
        pos, code_at, block_at, size = 0, None, 0, 0

        def emit(seg, at):
            nonlocal size
            u = html.unescape(seg)
            same = len(u) == len(seg)
            text.append(u)
            offs.extend(at + i if same else at for i in range(len(u)))
            size += len(u)
        for m in re.finditer(r"<!--.*?-->|<(/?)([A-Za-z][A-Za-z0-9]*)\b[^>]*>", raw, re.S):
            if m.start() < pos:
                continue                     # inside a script or a style
            emit(raw[pos:m.start()], pos)
            pos, tag = m.end(), (m.group(2) or "").lower()
            if tag in ("script", "style") and not m.group(1):
                close = re.compile(r"</%s\s*>" % tag, re.I).search(raw, pos)
                if not close:
                    raise DocSetError("an unclosed <%s>" % tag)
                pos = close.end()
            if tag == "code":
                if m.group(1) and code_at is not None:
                    spans.append(("".join(text)[code_at:size].strip(), code_at))
                    code_at = None
                elif not m.group(1):
                    code_at = size + 1       # after the space the tag becomes
            if tag in BLOCK_TAGS:
                if size > block_at:
                    blocks.append((block_at, size))
                block_at = size
            emit(" ", m.start())             # a tag separates words; a word never runs into the next block
        emit(raw[pos:], pos)
        if size > block_at:
            blocks.append((block_at, size))
        return "".join(text), offs, spans, blocks

    @staticmethod
    def of_markdown(raw):
        """markdown reads as itself: offsets are identity. A block is a paragraph, a list item or a table row; a span is `backticked`"""
        spans = [(m.group(1).strip(), m.start(1)) for m in re.finditer(r"(?<!`)`+([^`\n]+)`+", raw)]
        cuts = sorted({0, len(raw)} | {m.start() + 1 for m in re.finditer(r"\n(?=\s*\n|\s*[-*|]\s|\s*\d+\.\s|#)", raw)})
        return raw, None, spans, [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]


class DocReader(object):
    """How one kind of document splits into pieces. `decl` is the document's line in the docs file: `as` its short name (the first part of
    every piece id), `read` this reader's name, `path` the file, relative to `base`. A subclass returns at least one piece or raises."""

    sub = "section"

    def __init__(self, decl, base):
        self.decl, self.document = decl, decl.get("as")
        declared = str(decl.get("path") or "")
        if not self.document or not declared:
            raise DocSetError("a document needs `as` and `path`: %s" % decl)
        self.file = declared if os.path.isabs(declared) else os.path.normpath(os.path.join(base, declared))
        try:
            rel = os.path.relpath(self.file, ROOT)
        except ValueError:
            rel = declared                   # another drive: no path from the root exists, so the declared one is what can be said
        self.path = (declared if rel.startswith("..") else rel).replace("\\", "/")

    def source(self):
        if not os.path.isfile(self.file):
            raise DocSetError("document `%s`: no such file: %s" % (self.document, self.path))
        return io.open(self.file, encoding="utf-8").read()

    def pieces(self):
        raise NotImplementedError

    def nodes(self):
        """nodes this document's rows stand for: (id, kind, label, sub), each with the row that is its own (see GoalRows)"""
        return []


class HtmlSections(DocReader):
    """A page whose addressable pieces are its <section id="..."> elements. Sections must not nest, or a regular walk would cut them wrong:
    measured on the manual, 54 opened and 54 closed; a page that breaks that refuses instead of being half read."""

    sub = "section"

    def pieces(self):
        src, out = self.source(), []
        for m in re.finditer(r'<section id="([^"]+)"[^>]*>.*?</section>', src, re.S):
            text, offs, spans, blocks = Plain.of_html(m.group(0))
            head = re.search(r"<h[1-6][^>]*>(.*?)</h[1-6]>", m.group(0), re.S)
            title = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", head.group(1)))).strip() if head else m.group(1)
            hrefs = re.findall(r'href="[^"#]+#([A-Za-z0-9_-]+)"', m.group(0))
            out.append(DocPiece(self.document, m.group(1), self.decl.get("sub") or self.sub, title, self.path, src.count("\n", 0, m.start()) + 1,
                                m.group(0), text, offs, spans, blocks, hrefs=hrefs))
        opened = len(re.findall(r"<section\b", src))
        if opened != len(out):
            raise DocSetError("%s: %d <section> tags but %d readable sections (nested, or one with no id)" % (self.path, opened, len(out)))
        # the page's own text outside its sections is a piece too: measured on the manual, its contents list quoted a count on no section at all
        rest = re.sub(r'<section id="[^"]+"[^>]*>.*?</section>', lambda m: "\n" * m.group(0).count("\n"), src, flags=re.S)
        text, offs, spans, blocks = Plain.of_html(rest)
        if out and text.strip():
            out.append(DocPiece(self.document, "outside-any-section", self.decl.get("sub") or self.sub, "Outside any section", self.path, 1, rest, text, offs, spans, blocks))
        return out


class Markdown(DocReader):
    """A markdown file: each heading starts a piece, whatever comes before the first heading is the piece `top`. The key is the heading's
    slug; a repeated heading takes ~2, ~3, the same mark the code ids use for overloads. A # inside a fenced code block is not a heading."""

    sub = "markdown section"

    def pieces(self):
        src = self.source()
        heads, fence, at = [], False, 0
        for line in src.splitlines(True):
            if line.lstrip().startswith("```"):
                fence = not fence
            m = None if fence else re.match(r"#{1,6}\s+(.*\S)\s*$", line)
            if m:
                heads.append((at, m.group(1)))
            at += len(line)
        marks = ([(0, None)] if not heads or heads[0][0] > 0 else []) + heads
        seen, out = collections.Counter(), []
        for i, (start, heading) in enumerate(marks):
            end = marks[i + 1][0] if i + 1 < len(marks) else len(src)
            raw = src[start:end]
            if not raw.strip():
                continue
            slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-") if heading else "top"
            seen[slug] += 1
            key = slug if seen[slug] == 1 else "%s~%d" % (slug, seen[slug])
            text, offs, spans, blocks = Plain.of_markdown(raw)
            out.append(DocPiece(self.document, key, self.decl.get("sub") or self.sub, heading or os.path.basename(self.file), self.path, src.count("\n", 0, start) + 1,
                                raw, text, offs, spans, blocks))
        return out


class Deck(DocReader):
    """A slide deck: deck.json's `order` lists the slides shown, each one slides/<id>.html. A slide in the order with no file refuses; a file
    not in the order is not shown, so it is no piece and is listed (`extras`). Speaker notes count: a number in them goes stale the same way."""

    sub = "slide"

    def pieces(self):
        deck = json.loads(self.source())
        self.order = list(deck.get("order") or [])
        folder = os.path.join(os.path.dirname(self.file), "slides")
        out = []
        for sid in self.order:
            path = os.path.join(folder, sid + ".html")
            if not os.path.isfile(path):
                raise DocSetError("deck order names slide %s: no such file in %s" % (sid, self.path))
            raw = io.open(path, encoding="utf-8").read()
            text, offs, spans, blocks = Plain.of_html(raw)
            head = re.search(r"<h[1-6][^>]*>(.*?)</h[1-6]>", raw, re.S)
            title = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", head.group(1)))).strip() if head else sid
            rel = "%s/slides/%s.html" % (os.path.dirname(self.path), sid)
            out.append(DocPiece(self.document, sid, self.sub, title, rel, 1, raw, text, offs, spans, blocks, hrefs=re.findall(r'href="[^"#]+#([A-Za-z0-9_-]+)"', raw)))
        self.extras = sorted(f[:-5] for f in os.listdir(folder) if f.endswith(".html") and f[:-5] not in set(self.order)) if os.path.isdir(folder) else []
        return out


class Concepts(DocReader):
    """The concept pages: a Python list `CONCEPTS = [(slug, title, [section ids])]` in the page builder. A concept page has no words of its
    own, it is built from manual sections, so it carries `built from` edges and no text: what makes a section stale makes its pages stale.
    Read by parsing the file, never by importing it: importing a page builder could run its gates. `pages_of` names the document it reads."""

    sub = "concept page"

    def pieces(self):
        src = self.source()
        tree = ast.parse(src)
        for n in tree.body:
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CONCEPTS" for t in n.targets):
                rows = ast.literal_eval(n.value)
                of = self.decl.get("pages_of")
                if not of:
                    raise DocSetError("%s: a concepts document needs `pages_of`, the document its sections live in" % self.path)
                return [DocPiece(self.document, slug, self.sub, title, self.path, n.lineno, text="", built_from=["doc:%s#%s" % (of, s) for s in sections])
                        for slug, title, sections in rows]
        raise DocSetError("%s: no CONCEPTS list" % self.path)


class GoalRows(DocReader):
    """The Goals List: a YAML `goals:` table, one row per thing the learned parts are trained for. Each row is a piece (its row on the public
    page) and a node of kind `goal`, whose own row documents it. Its words are the row's, so a row can name another node like any piece."""

    sub = "goals row"

    def rows(self):
        rows = (yaml.safe_load(self.source()) or {}).get("goals") or []
        if not rows:
            raise DocSetError("%s: no goals" % self.path)
        return rows

    def pieces(self):
        src, out = self.source(), []
        for g in self.rows():
            m = re.search(r"^\s*-\s*id:\s*%s\s*$" % re.escape(str(g["id"])), src, re.M)
            if not m:
                raise DocSetError("%s: goal %s has no `- id:` line to point at" % (self.path, g["id"]))
            words = " ".join(str(g.get(k) or "") for k in ("trained_to", "learns", "asked_by", "note"))
            out.append(DocPiece(self.document, g["id"], self.sub, g.get("trained_to") or g["id"], self.path, src.count("\n", 0, m.start()) + 1, raw=words, text=words, row=g["id"]))
        return out

    def nodes(self):
        return [("goal:%s" % g["id"], "goal", g.get("trained_to") or g["id"], g.get("state"), "doc:%s#%s" % (self.document, g["id"])) for g in self.rows()]


class StatRows(DocReader):
    """The Stats List: the generated `stats` table, one row per stat a record holds. A row marked IDENTITY says what the row is (a time, a
    seed, a model's name), not a number anyone quotes, so it is no node: measured 2026-10-30, those are the likes of `time`, `n` and `t`.
    A stat's id carries its record because one name can mean different things in two records: `stat:<record>/<name>`."""

    sub = "stats row"

    def rows(self):
        rows = [s for s in (json.loads(self.source()).get("stats") or []) if s.get("verdict") != "IDENTITY"]
        if not rows:
            raise DocSetError("%s: no stats" % self.path)
        return rows

    def pieces(self):
        return [DocPiece(self.document, "%s/%s" % (s["record"], s["name"]), self.sub, s["name"], self.path, raw=s["name"], text=s["name"],
                         row="%s/%s" % (s["record"], s["name"])) for s in self.rows()]

    def nodes(self):
        return [("stat:%s/%s" % (s["record"], s["name"]), "stat", s["name"], s["record"], "doc:%s#%s/%s" % (self.document, s["record"], s["name"])) for s in self.rows()]


READERS = {"html_sections": HtmlSections, "markdown": Markdown, "deck": Deck, "concepts": Concepts, "goal_rows": GoalRows, "stat_rows": StatRows}


class DocSet(object):
    """Every document a model declares, read into pieces. An empty set, a missing file, or a file that yields nothing raises DocSetError: a
    report of no holes must never come from no documents. Evidence: the model's meta.docs, and the files it names."""

    def __init__(self, docs_meta, root=ROOT):
        self.meta, self.root = docs_meta or {}, root
        self.readers, self.pieces, self.by_id, self.nodes = [], [], {}, []
        self.read()

    def declarations(self):
        out = []
        for s in self.meta.get("sets") or []:
            path = s if os.path.isabs(s) else os.path.normpath(os.path.join(self.root, s))
            if not os.path.isfile(path):
                raise DocSetError("no such document set: %s" % s)
            rows = (yaml.safe_load(io.open(path, encoding="utf-8")) or {}).get("documents") or []
            out += [(d, os.path.dirname(path)) for d in rows]
        out += [(d, self.root) for d in self.meta.get("documents") or []]
        return out

    def read(self):
        if yaml is None:
            raise DocSetError("PyYAML is not installed, so the document set cannot be read")
        try:
            decls = self.declarations()
        except (OSError, ValueError, AttributeError, yaml.YAMLError) as e:
            raise DocSetError("the document set cannot be read: %s" % e)
        if not decls:
            raise DocSetError("the document set is empty: nothing to link to, so no count of holes could mean anything")
        for d, base in decls:
            cls = READERS.get(d.get("read"))
            if cls is None:
                raise DocSetError("document `%s`: no reader called %s (there are %s)" % (d.get("as"), d.get("read"), ", ".join(sorted(READERS))))
            r = cls(d, base)
            try:
                got = r.pieces()
            except (ValueError, OSError, SyntaxError, KeyError, TypeError, AttributeError, yaml.YAMLError) as e:
                raise DocSetError("document `%s` (%s) cannot be read: %s" % (r.document, r.path, e))
            if not got:
                raise DocSetError("document `%s` (%s) read as nothing" % (r.document, r.path))
            self.readers.append(r)
            for p in got:
                if p.id in self.by_id:
                    raise DocSetError("two pieces with one id: %s (two documents with one `as`?)" % p.id)
                self.by_id[p.id] = p
                self.pieces.append(p)
            self.nodes += r.nodes()

    def cross_links(self):
        """[(piece, target piece id)], [(piece, fragment)]: a document's own hyperlinks to a piece of ANOTHER document, read off its href
        fragment, and those that land nowhere. `#slide-N` is the deck's order; anything else a section's id. A link inside one document is
        a cross-reference, not a second place that says the thing, so it is no edge."""
        order = next((r.order for r in self.readers if isinstance(r, Deck)), [])
        decks = {r.document for r in self.readers if isinstance(r, Deck)}
        pages = {r.document for r in self.readers if isinstance(r, HtmlSections)}
        keys = collections.defaultdict(list)
        for p in self.pieces:
            if p.document in pages:
                keys[p.key].append(p.id)
        out, dangling = [], []
        for p in self.pieces:
            for frag in p.hrefs:
                m = re.fullmatch(r"slide-(\d+)", frag)
                if m and decks:
                    n = int(m.group(1))
                    if 1 <= n <= len(order):
                        out.append((p, "doc:%s#%s" % (sorted(decks)[0], order[n - 1])))
                    else:
                        dangling.append((p, frag))
                elif keys.get(frag):
                    out += [(p, t) for t in keys[frag] if self.by_id[t].document != p.document]
                else:
                    dangling.append((p, frag))
        return out, dangling

    def quoting(self, literal):
        """[(piece, where)] every place a literal string stands, white space aside. What a claim registry asks when a number moves"""
        out = []
        for p in self.pieces:
            for m in re.finditer(r"\s+".join(re.escape(w) for w in str(literal).split()), p.text):
                out.append((p, p.where(m.start())))
        return out


class Hit(object):
    """One place a piece names a node: the node, the exact words, where in the piece's text, and which naming rule said so"""

    def __init__(self, node, matched, offset, rule):
        self.node, self.matched, self.offset, self.rule = node, matched, offset, rule


class NameIndex(object):
    """What a document could name in this graph, and the exact rule for each kind. Nothing here measures likeness; each rule is a token that
    either stands in the text or does not:
      a requirement   its id, whole. An id-shaped token of the same shape the model does not hold is a PHANTOM: `CL-99` names nothing
      code            `Type.Member` or the full dotted id, whole. A bare member only inside a code-formatted span and only with its type named
                      in the same paragraph, list item or row (one long section can discuss three classes). Two such types: ambiguous, no link
      a stat          a snake_case name anywhere. A plain word (`killed`) only as a code span in a document declared `names_stats`
      a goal          its title (`trained_to`) as a whole phrase, or a snake_case id; `owner` or `tests` in prose are English
    A type, a module or a use case named without the node itself is an OWNER: recorded apart, shown apart, never a link."""

    GUARD_L, GUARD_R = r"(?<![\w.-])", r"(?![\w-]|\.\d)"

    def __init__(self, requirements, code, stats, goals, use_cases=()):
        self.use_cases = sorted(use_cases)
        self.uc_re = re.compile(self.GUARD_L + "(" + "|".join(re.escape(u) for u in self.use_cases) + ")" + self.GUARD_R) if self.use_cases else None
        self.reqs = {r: "req:%s" % r for r in requirements}
        shapes = sorted({re.sub(r"\d+", r"\\d+", re.escape(r)) for r in requirements}, key=len, reverse=True)
        self.req_re = re.compile(self.GUARD_L + "(" + "|".join(shapes) + ")" + self.GUARD_R) if shapes else None
        self.code_keys, self.members, self.types = collections.defaultdict(set), collections.defaultdict(set), set()
        for c in code:
            base = re.split(r"[#@]", c[5:])[0]
            base = re.sub(r"~\d+$", "", base)
            parts = base.split(".")
            self.code_keys[base].add(c)
            if len(parts) >= 2:
                self.code_keys[".".join(parts[-2:])].add(c)
                if PLAIN_NAME.fullmatch(parts[-1]) and not parts[-1].startswith("__") and parts[-1] != parts[-2]:     # a constructor's name is the class's: naming the class is not naming it
                    self.members[parts[-1]].add((parts[-2], c))
                self.types.add(parts[-2])
        self.code_re = re.compile(r"(?<!\w)(" + "|".join(re.escape(k) for k in sorted(self.code_keys, key=len, reverse=True)) + r")(?!\w)") if self.code_keys else None
        self.type_re = {t: re.compile(r"(?<!\w)%s(?!\w)" % re.escape(t)) for t in self.types}
        self.stat_names, self.plain_stats = collections.defaultdict(set), collections.defaultdict(set)
        for sid, name, record in stats:
            if SNAKE.fullmatch(name):
                self.stat_names[name].add(sid)
            elif re.fullmatch(r"[a-z]+", name):
                self.plain_stats[(record, name)].add(sid)
        self.stat_re = re.compile(r"(?<![\w.-])(" + "|".join(re.escape(s) for s in sorted(self.stat_names, key=len, reverse=True)) + r")(?![\w-])") if self.stat_names else None
        self.goal_phrases, self.goal_ids = {}, {}
        for gid, title in goals:
            if title:
                self.goal_phrases[re.sub(r"\s+", " ", title.lower())] = "goal:%s" % gid
            if SNAKE.fullmatch(gid):
                self.goal_ids[gid] = "goal:%s" % gid
        alts = [r"\s+".join(re.escape(w) for w in t.split()) for t in sorted(self.goal_phrases, key=len, reverse=True)]
        self.goal_re = re.compile(r"(?<!\w)(" + "|".join(alts) + r")(?!\w)", re.I) if alts else None
        self.goal_id_re = re.compile(r"(?<![\w.-])(" + "|".join(sorted(self.goal_ids)) + r")(?![\w-])") if self.goal_ids else None

    def phantoms(self, piece):
        """[(token, offset)] ids of a requirement's shape that the model does not hold"""
        return [(m.group(1), m.start(1)) for m in self.req_re.finditer(piece.text) if m.group(1) not in self.reqs] if self.req_re else []

    def hits(self, piece, plain_stat_record=None):
        """(hits, ambiguous, owners): the nodes this piece names, the bare members two named types could own, and the types it names"""
        text, hits, ambiguous = piece.text, [], []
        if self.req_re:
            hits += [Hit(self.reqs[m.group(1)], m.group(1), m.start(1), "its id") for m in self.req_re.finditer(text) if m.group(1) in self.reqs]
        if self.code_re:
            for m in self.code_re.finditer(text):
                hits += [Hit(c, m.group(1), m.start(1), "its symbol") for c in self.code_keys[m.group(1)]]
        for span, at in piece.spans:
            bare = re.match(r"([A-Za-z_]\w*)\s*(?:[(<].*)?$", span)
            if not bare or bare.group(1) not in self.members:
                continue
            block = piece.block_at(at)
            owners = sorted({t for t, _ in self.members[bare.group(1)] if self.type_re[t].search(block)})
            if len(owners) > 1:
                ambiguous.append((bare.group(1), owners, piece.where(at)))
            elif owners:
                hits += [Hit(c, "%s (with its type %s in the same block)" % (bare.group(1), owners[0]), at, "its member and type")
                         for t, c in self.members[bare.group(1)] if t == owners[0]]
        if self.stat_re:
            for m in self.stat_re.finditer(text):
                hits += [Hit(s, m.group(1), m.start(1), "its name") for s in self.stat_names[m.group(1)]]
        for span, at in piece.spans:
            for s in self.plain_stats.get((plain_stat_record, span), ()):     # a document not declared `names_stats` passes None: no record is called that
                hits.append(Hit(s, span, at, "its name, in a document that names stats"))
        if self.goal_re:
            for m in self.goal_re.finditer(text):
                hits.append(Hit(self.goal_phrases[re.sub(r"\s+", " ", m.group(1).lower())], m.group(1), m.start(1), "its title"))
        if self.goal_id_re:
            hits += [Hit(self.goal_ids[m.group(1)], m.group(1), m.start(1), "its id") for m in self.goal_id_re.finditer(text)]
        owners = {t for t in self.types if self.type_re[t].search(text)}
        if self.uc_re:
            owners |= {m.group(1) for m in self.uc_re.finditer(text)}
        return hits, ambiguous, sorted(owners)


class Link(object):
    """A `documented in` edge and why it exists: `how` is names (the document's own words), row (the node's own row) or authored"""

    def __init__(self, node, doc, how, matched, reason):
        self.node, self.doc, self.how, self.matched, self.reason = node, doc, how, matched, reason

    def key(self):
        return (self.node, self.doc)


class AuthoredLinks(object):
    """The links a document makes in other words: meta.docs.links, each {node, doc, says, why}. Authored once, re-verified on every build: the
    node must be in the graph, the piece must exist and still hold the words `says`, and a `why` is required. A line that fails is BROKEN, a
    failing hole, so an authored link cannot rot into a claim nobody can check. This is the only way to link without a literal name."""

    def __init__(self, rows):
        self.rows = rows or []

    def check(self, nodes, docset):
        links, broken = [], []
        for r in self.rows:
            node, doc, says, why = r.get("node"), r.get("doc"), r.get("says"), r.get("why")
            subject = node or "the links file"
            if not why or not str(why).strip():
                broken.append((subject, "an authored link with no reason: %s -> %s" % (node, doc)))
            elif node not in nodes:
                broken.append((subject, "an authored link names a node the graph does not hold: %s -> %s" % (node, doc)))
            elif doc not in docset.by_id:
                broken.append((subject, "an authored link points at a piece that no longer exists: %s (it said: %s)" % (doc, says)))
            else:
                at = docset.by_id[doc].says(says) if says else None
                if at is None:
                    broken.append((subject, "%s no longer says %r, the words this link stands on" % (doc, says)))
                else:
                    links.append(Link(node, doc, "authored", str(says), "authored: it says %r at %s: %s" % (says, docset.by_id[doc].where(at), why)))
        return links, broken


class LinkLedger(object):
    """The links last accepted, one per line in <model>/doc_links.lock, so a link that vanishes is seen. Without it a renamed section would
    only ever show as fewer links, which reads as nothing at all. Accepting is a person's or an agent's statement that the documents as they
    stand are right, and is refused while a document says something false. The same idea as loop.lock, for documents."""

    def __init__(self, model_root):
        self.path = os.path.join(model_root, LEDGER) if model_root else None

    def read(self):
        if not self.path or not os.path.exists(self.path):
            return None
        out = {}
        for line in io.open(self.path, encoding="utf-8"):
            if line.startswith("#") or not line.strip():
                continue
            node, doc, how, matched = line.rstrip("\n").split("\t", 3)
            out[(node, doc)] = (how, matched)
        return out

    def write(self, rows, why):
        lines = ["# doc_nodes.py accept: the links as last accepted, so a link that vanishes shows as lost. Never edit. Accepted because: %s" % why]
        lines += ["\t".join(r) for r in sorted(rows)]
        io.open(self.path, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")

    def lost(self, now):
        """[(node, doc, how, matched)] held by the ledger, made by no document now; None when there is no ledger to compare with"""
        held = self.read()
        if held is None:
            return None
        return sorted((n, d, h, m) for (n, d), (h, m) in held.items() if (n, d) not in now)


class DocOverlay(object):
    """Writes the documents into a graph store: pieces and the stat and goal rows as nodes, `documented in`, `built from` and `links to` as
    edges, and every reason, phantom, broken line and lost link as a literal, so the graph is the record and the criterion reads only the
    graph. Idempotent. Deterministic: sorted, with ids from what things are, so two builds export the same triples."""

    def __init__(self, db, meta, model_root, root=ROOT):
        self.db, self.meta, self.model_root, self.root = db, meta or {}, model_root, root
        self.docset, self.links, self.ambiguous, self.dangling = None, [], [], []

    def clear(self):
        marks = ",".join("?" * len(KINDS))
        self.db.execute("DELETE FROM entity WHERE kind IN (%s)" % marks, KINDS)
        self.db.execute("DELETE FROM edge WHERE p IN (%s)" % ",".join(map(str, RELATIONS)))
        self.db.execute("DELETE FROM literal WHERE p IN (%s)" % ",".join(map(str, LITERALS)))

    def state(self, value):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES ('docs', ?)", (value,))

    def attach(self):
        self.db.executemany("INSERT OR IGNORE INTO prop VALUES (?, ?, ?)", PROPS)
        self.db.execute("INSERT OR IGNORE INTO family VALUES (?, 'document')", (FAMILY_DOCUMENT,))
        self.clear()
        if "docs" not in self.meta:
            self.state(NOT_DECLARED)
            self.db.commit()
            return self
        try:
            self.docset = DocSet(self.meta["docs"], self.root)
        except DocSetError as e:
            self.state(UNREADABLE + str(e))
            self.db.commit()
            return self
        self.write()
        self.db.commit()
        return self

    def write(self):
        ds = self.docset
        ents = {p.id: (p.id, "doc", p.title, p.sub) for p in ds.pieces}
        edges, lits = set(), set()
        for node, kind, label, sub, _ in ds.nodes:
            ents[node] = (node, kind, label, sub)
        known = dict(self.db.execute("SELECT id, kind FROM entity"))
        names = self.names(known, ds)
        by = collections.defaultdict(dict)
        rows_of = {(n[0], n[4]) for n in ds.nodes}
        known_phantoms = {str(k.get("names")): k.get("why") for k in self.meta["docs"].get("known") or []}
        seen_phantoms = set()
        for p in ds.pieces:
            lits.add((p.id, FOUND_AT, p.found_at()))
            for t in p.built_from:
                if t in ds.by_id:
                    edges.add((p.id, BUILT_FROM, t, FAMILY_AUTHORED))
                else:
                    lits.add((p.id, NAMES_NOTHING, "this page is built from %s, which no document holds" % t))
            if not p.text or p.sub == StatRows.sub:
                continue                     # a built page has no words; a stats row's only word is its own name, which describes nothing else
            for token, at in names.phantoms(p):
                seen_phantoms.add(token)
                if str(known_phantoms.get(token) or "").strip():
                    lits.add((p.id, KNOWN_PHANTOM, "%s at %s: %s" % (token, p.where(at), known_phantoms[token])))
                else:
                    lits.add((p.id, NAMES_NOTHING, "%s at %s: the model holds no such requirement" % (token, p.where(at))))
            hits, ambiguous, owners = names.hits(p, self.names_stats(p))
            self.ambiguous += ambiguous
            for t in owners:
                lits.add((p.id, NAMES_OWNER, t))
            for h in hits:
                if (h.node, p.id) in rows_of:
                    continue                 # a row naming its own node says nothing new: it is linked as that node's row
                by[(h.node, p.id)].setdefault(h.matched, Link(h.node, p.id, "names", h.matched, "names %r (%s) at %s" % (h.matched, h.rule, p.where(h.offset))))
        for node, _, _, _, row in ds.nodes:
            by[(node, row)].setdefault("(row)", Link(node, row, "row", "(row)", "its own row, %s" % ds.by_id[row].where()))
        nodes = set(known) | {n[0] for n in ds.nodes}
        authored, broken = AuthoredLinks(self.meta["docs"].get("links")).check(nodes, ds)
        for a in authored:
            by[a.key()][a.matched] = a
        for (node, doc), found in sorted(by.items()):
            for link in found.values():
                edges.add((node, DOCUMENTED_IN, doc, FAMILY_AUTHORED if link.how == "authored" else FAMILY_DOCUMENT))
                lits.add((node, BECAUSE, "%s | %s" % (doc, link.reason)))
                self.links.append(link)
        for subject, message in broken:
            lits.add((subject, LINK_BROKEN, message))
        for token in sorted(set(known_phantoms) - seen_phantoms):
            lits.add(("the known list", LINK_BROKEN, "%s is declared known, but no document names it as a phantom any more: drop the line" % token))
        cross, self.dangling = ds.cross_links()
        for p, target in cross:
            edges.add((p.id, LINKS_TO, target, FAMILY_DOCUMENT))
        ledger = LinkLedger(self.model_root)
        lost = ledger.lost(set(by))
        for node, doc, how, matched in lost or []:
            lits.add((node, LINK_LOST, "%s (%s %s) is gone: was it renamed, or does it no longer say this?" % (doc, how, matched)))
        self.db.executemany("INSERT OR IGNORE INTO entity VALUES (?, ?, ?, ?)", [ents[k] for k in sorted(ents)])
        self.db.executemany("INSERT OR IGNORE INTO edge VALUES (?, ?, ?, ?)", sorted(edges))
        self.db.executemany("INSERT OR IGNORE INTO literal VALUES (?, ?, ?)", sorted(lits))
        self.state("read %d pieces of %d documents; ledger %s" % (len(ds.pieces), len(ds.readers), "none" if lost is None else "held"))

    def names_stats(self, piece):
        for r in self.docset.readers:
            if r.document == piece.document:
                return r.decl.get("names_stats")
        return None

    @staticmethod
    def names(known, ds):
        reqs = [i[4:] for i, k in known.items() if k == "requirement"]
        code = [i for i, k in known.items() if k == "code"]
        stats = [(n, label, sub) for n, kind, label, sub, _ in ds.nodes if kind == "stat"]
        goals = [(n[5:], label) for n, kind, label, _, _ in ds.nodes if kind == "goal"]
        return NameIndex(sorted(reqs), sorted(code), sorted(stats), sorted(goals), [i[3:] for i, k in known.items() if k == "use case"])


def attach(db, meta, model_root=None, root=ROOT):
    """the one call a graph build makes. `db` an open store with usecase_kg's tables; `meta` the model's meta. A model that declares no
    documents gets none and L20 says it was not measured"""
    return DocOverlay(db, meta, model_root, root).attach()


class DocGate(object):
    """The closed loop's view of the documents, read from the graph alone (criterion L20 DOCUMENTED, usecase_kg.LoopGate.documented). Returns
    (entity, message, failing). Failing is kept for what is provably FALSE in a document: a requirement id the model does not hold, an authored
    line that no longer resolves, a declared set that reads as nothing. Everything else is listed: a requirement no document describes is a
    finding about the documents, and most models have never had any, so failing on it would turn every model red for a thing that was never
    asked of them; a lost link is listed because a section is renamed far more often than a promise is broken, and the ledger takes it back."""

    def __init__(self, db):
        self.db = db

    def state(self):
        row = self.db.execute("SELECT value FROM meta WHERE key = 'docs'").fetchone()
        return row[0] if row else NOT_DECLARED

    def literals(self, prop):
        return self.db.execute("SELECT s, v FROM literal WHERE p = ? ORDER BY s, v", (prop,)).fetchall()

    def unlinked(self, kind):
        """the nodes of a kind with no `documented in` edge"""
        return [i for (i,) in self.db.execute("SELECT id FROM entity e WHERE kind = ? AND NOT EXISTS (SELECT 1 FROM edge x WHERE x.s = e.id AND x.p = ?) ORDER BY id", (kind, DOCUMENTED_IN))]

    def total(self, kind):
        return self.db.execute("SELECT COUNT(*) FROM entity WHERE kind = ?", (kind,)).fetchone()[0]

    def holes(self):
        state = self.state()
        if state == NOT_DECLARED:
            return [("the documents", "not measured: this model declares no document set (meta.docs)", False)]
        if state.startswith(UNREADABLE):
            return [("the documents", state, True)]
        out = [(s, "names what does not exist: %s" % v, True) for s, v in self.literals(NAMES_NOTHING)]
        out += [(s, v, True) for s, v in self.literals(LINK_BROKEN)]
        out += [(s, "lost link: %s" % v, False) for s, v in self.literals(LINK_LOST)]
        out += [(s, "names what does not exist, declared known: %s" % v, False) for s, v in self.literals(KNOWN_PHANTOM)]
        out += [(r, "no document describes it", False) for r in self.unlinked("requirement")]
        gone = self.unlinked("code")         # a stat or a goal always has its own row: only code can be left with no document, once requirements are named above
        if gone:
            out.append(("the code nodes", "%d of %d named by no document" % (len(gone), self.total("code")), False))
        if state.endswith("ledger none"):
            out.append(("the ledger", "no %s yet: a link that vanishes cannot be seen until one is accepted" % LEDGER, False))
        return out


class DocRef(object):
    """One document piece that describes a node, as DocIndex returns it: its id, title, file and line, and why the link exists"""

    def __init__(self, doc, title, at, because):
        self.doc, self.title, self.at, self.because = doc, title, at, because

    def as_dict(self):
        return {"doc": self.doc, "title": self.title, "at": self.at, "because": self.because}


class DocIndex(object):
    """Reads the links back out of a store and answers the owner's question: which documents describe this node, so which do I update. A node
    the graph does not hold is UNKNOWN, never "no documents". This is the API the claim registry builds on: `documents_of`, `pages_from`,
    `linked_from`, `through`."""

    # what a node with no words of its own hangs from: it is described by what its relation reaches (usecase_kg.PROPS)
    IS_AT, ASSERTS, MAPS, CHECKED_BY, BINDS, INFERS = 152, 134, 122, 149, 107, 108

    def __init__(self, db):
        self.db = db
        self.kind = dict(db.execute("SELECT id, kind FROM entity"))

    def resolve(self, name):
        """the node id for a name: exact, or with its kind's prefix. Nothing else: a near miss is no answer"""
        if name in self.kind:
            return name
        hits = [p + name for p in ("req:", "code:", "stat:", "goal:", "doc:", "test:") if p + name in self.kind]
        return hits[0] if len(hits) == 1 else None

    def ref(self, doc, node=None):
        title = self.db.execute("SELECT label FROM entity WHERE id = ?", (doc,)).fetchone()
        at = self.db.execute("SELECT v FROM literal WHERE s = ? AND p = ?", (doc, FOUND_AT)).fetchone()
        why = []
        if node:
            why = [v.split(" | ", 1)[1] for (v,) in self.db.execute("SELECT v FROM literal WHERE s = ? AND p = ? AND v LIKE ? ORDER BY v", (node, BECAUSE, doc + " | %"))]
        return DocRef(doc, title[0] if title else "", at[0] if at else "", why)

    def documents_of(self, node):
        return [self.ref(d, node) for (d,) in self.db.execute("SELECT DISTINCT o FROM edge WHERE s = ? AND p = ? ORDER BY o", (node, DOCUMENTED_IN))]

    def pages_from(self, doc):
        """the pages built from this piece: they change when it does"""
        return [p for (p,) in self.db.execute("SELECT s FROM edge WHERE o = ? AND p = ? ORDER BY s", (doc, BUILT_FROM))]

    def linked_from(self, doc):
        """the pieces of other documents this one's own hyperlinks join it to, either way: they tend to say the same thing"""
        rows = self.db.execute("SELECT o FROM edge WHERE s = ? AND p = ? UNION SELECT s FROM edge WHERE o = ? AND p = ? ORDER BY 1", (doc, LINKS_TO, doc, LINKS_TO))
        return [p for (p,) in rows]

    def through(self, node):
        """[(requirement or code node, how it joins)]: what this node hangs from, by the graph's own edges and never by likeness"""
        q = lambda sql, *a: [x for (x,) in self.db.execute(sql, a)]
        kind, out = self.kind.get(node), []
        if kind == "code":
            for r in q("SELECT DISTINCT r.s FROM edge r JOIN edge a ON a.s = r.o WHERE r.p = ? AND a.p = ? AND a.o = ? ORDER BY 1", self.ASSERTS, self.IS_AT, node):
                out.append((r, "its effect sits in this code"))
            for r in q("SELECT DISTINCT m.o FROM edge c JOIN edge m ON m.s = c.s WHERE c.p = ? AND c.o = ? AND m.p = ? ORDER BY 1", self.IS_AT, node, self.MAPS):
                if not any(x == r for x, _ in out):
                    out.append((r, "it claims this code's branches"))
        elif kind in ("mutant", "assertion", "cell"):
            out += [(c, "this sits at that code") for c in q("SELECT DISTINCT o FROM edge WHERE s = ? AND p = ? ORDER BY 1", node, self.IS_AT)]
        elif kind == "test":
            out += [(r, "names this test as its check") for r in q("SELECT DISTINCT s FROM edge WHERE o = ? AND p = ? ORDER BY 1", node, self.CHECKED_BY)]
        elif kind in ("step", "extension"):
            out += [(r, "this carries it") for r in q("SELECT DISTINCT o FROM edge WHERE s = ? AND p IN (?, ?) ORDER BY 1", node, self.BINDS, self.INFERS)]
        return out

    def owner_named_in(self, node):
        """pieces that name what this node belongs to but not the node: code's type or module, or the use case carrying a requirement. A weaker
        tie, shown apart and never counted as a link"""
        kind, owners = self.kind.get(node), []
        if kind == "code":
            parts = re.sub(r"~\d+$", "", re.split(r"[#@]", node[5:])[0]).split(".")
            owners = parts[-2:-1]
        elif kind == "requirement":
            owners = [u[3:] for (u,) in self.db.execute("SELECT DISTINCT own.s FROM edge own JOIN edge x ON x.s = own.o WHERE own.p IN (100, 116) AND x.p IN (?, ?) AND x.o = ? ORDER BY 1", (self.BINDS, self.INFERS, node))]
        return [(s, o) for o in owners for (s,) in self.db.execute("SELECT s FROM literal WHERE p = ? AND v = ? ORDER BY s", (NAMES_OWNER, o))]

    def what_to_update(self, name):
        """the whole answer as data; None when the graph does not hold the node. An edge is direct; `through` and `owner` are shown as what they are"""
        node = self.resolve(name)
        if node is None:
            return None
        def full(refs):
            return [dict(r.as_dict(), pages=self.pages_from(r.doc), linked=self.linked_from(r.doc)) for r in refs]
        return {"node": node, "kind": self.kind[node], "direct": full(self.documents_of(node)),
                "through": [{"via": v, "how": how, "documents": full(self.documents_of(v))} for v, how in self.through(node)],
                "owner": [{"doc": d, "names": t} for d, t in self.owner_named_in(node)]}


class DocReport(object):
    """The counts and the names, from the graph: how many of each kind are linked, how, and which are not. Printed by `attach` and `report`"""

    def __init__(self, db):
        self.db, self.gate = db, DocGate(db)

    def counts(self):
        q = lambda sql, *a: self.db.execute(sql, a).fetchone()[0]
        out = {"pieces": dict(self.db.execute("SELECT sub, COUNT(*) FROM entity WHERE kind = 'doc' GROUP BY sub ORDER BY sub").fetchall()),
               "links": q("SELECT COUNT(*) FROM (SELECT DISTINCT s, o FROM edge WHERE p = ?)", DOCUMENTED_IN),
               "by evidence": {"the document's own words": q("SELECT COUNT(*) FROM edge WHERE p = ? AND fam = ?", DOCUMENTED_IN, FAMILY_DOCUMENT),
                               "authored": q("SELECT COUNT(*) FROM edge WHERE p = ? AND fam = ?", DOCUMENTED_IN, FAMILY_AUTHORED)}, "kinds": {}}
        for kind in ("requirement", "code", "stat", "goal"):
            total, gone = self.gate.total(kind), self.gate.unlinked(kind)
            beyond = q("SELECT COUNT(DISTINCT x.s) FROM edge x JOIN entity e ON e.id = x.s JOIN entity d ON d.id = x.o WHERE e.kind = ? AND x.p = ? AND d.sub NOT IN ('goals row', 'stats row')", kind, DOCUMENTED_IN)
            out["kinds"][kind] = {"total": total, "linked": total - len(gone), "unlinked": gone, "named beyond its own row": beyond}
        return out

    def print(self, limit=40):
        state = self.gate.state()
        print("DOCUMENTS IN THE GRAPH: %s" % state)
        if state == NOT_DECLARED or state.startswith(UNREADABLE):
            return
        c = self.counts()
        print("  pieces: %s" % ", ".join("%d %s" % (n, k) for k, n in c["pieces"].items()))
        print("  %d links: %s" % (c["links"], ", ".join("%d %s" % (n, k) for k, n in c["by evidence"].items())))
        for kind, k in c["kinds"].items():
            print("  %-12s %4d nodes, %4d linked, %4d UNLINKED; %d named beyond its own row" % (kind, k["total"], k["linked"], len(k["unlinked"]), k["named beyond its own row"]))
        for kind in ("requirement", "code", "goal"):
            gone = c["kinds"][kind]["unlinked"]
            if gone:
                print("  unlinked %s: %s%s" % (kind, ", ".join(g.split(":", 1)[1] for g in gone[:limit]), "" if len(gone) <= limit else " ... %d more" % (len(gone) - limit)))
        for e, m, failing in self.gate.holes():
            if failing or m.startswith("lost link"):
                print("  %s %s: %s" % ("HOLE" if failing else "listed", e, m))


class Sample(object):
    """A reproducible sample of links with the source line each stands on, so a person can read them and say how many are right. It only prints.
    `verify` is the mechanical half: for every link by name, the words must stand on the line it cites"""

    def __init__(self, db, root=ROOT):
        self.db, self.root = db, root

    def rows(self, kinds=None):
        out = []
        for node, v in self.db.execute("SELECT s, v FROM literal WHERE p = ? ORDER BY s, v", (BECAUSE,)):
            doc, because = v.split(" | ", 1)
            kind = self.db.execute("SELECT kind FROM entity WHERE id = ?", (node,)).fetchone()
            if kind and (not kinds or kind[0] in kinds):
                out.append((node, kind[0], doc, because))
        return out

    def source_line(self, because):
        m = re.search(r" at ([^ ]+?):(\d+)(?:$|:)", because)
        if not m:
            return None
        path = m.group(1) if os.path.isabs(m.group(1)) else os.path.join(self.root, m.group(1))
        if not os.path.isfile(path):
            return None
        lines = io.open(path, encoding="utf-8").read().split("\n")
        return lines[int(m.group(2)) - 1].strip() if 0 < int(m.group(2)) <= len(lines) else None

    def verify(self):
        """every `names 'x' ... at file:line`: the text x is literally on that line. A row cites no line, so it is not read here"""
        bad, checked = [], 0
        for node, kind, doc, because in self.rows():
            m = re.match(r"names '(.+?)' \((.+?)\) at ", because) or re.match(r'names "(.+?)" \((.+?)\) at ', because)
            if not m or ", row " in because:
                continue
            word = re.sub(r" \(with its type .*$", "", m.group(1))
            line = self.source_line(because)
            checked += 1
            if line is None or word.lower() not in html.unescape(re.sub(r"\s+", " ", line)).lower():
                bad.append((node, doc, because))
        return checked, bad


# ---------------------------------------------------------------------- the selftest

SCHEMA = """CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT); CREATE TABLE family(bit INTEGER PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE prop(id INTEGER PRIMARY KEY, label TEXT NOT NULL, aliases TEXT); CREATE TABLE entity(id TEXT PRIMARY KEY, kind TEXT NOT NULL, label TEXT, sub TEXT);
CREATE TABLE edge(s TEXT NOT NULL, p INTEGER NOT NULL, o TEXT NOT NULL, fam INTEGER NOT NULL, PRIMARY KEY (s, p, o, fam));
CREATE TABLE literal(s TEXT NOT NULL, p INTEGER NOT NULL, v TEXT NOT NULL, PRIMARY KEY (s, p, v));"""
GUIDE = """<html><body><nav>The contents quote AB-3.</nav>
<section id="alpha"><h2>Alpha</h2><p>The rule AB-1 holds here, and <code>Ledger.Post</code> does it.</p>
<p>The owner of the tests was killed in prose; no goal or stat is named by those words, nor is AB-10x a name.</p>
<p>See <a href="https://example.test/deck#slide-1">the slide</a>.</p></section>
<section id="beta"><h2>Beta</h2><p>Who owns new code: the phrase of a goal. The stat graph_nodes rose. <code>Close</code> belongs to Ledger here.</p>
<p>The wallet refuses a second payment. The class <code>Ledger</code> holds it.</p></section>
<section id="gamma"><h2>Gamma</h2><p><code>Close</code> with no type near it names nothing.</p><p>And Ledger.Postpone is another method.</p></section>
<section id="delta"><h2>Delta</h2><p><code>Close</code> is named here beside both Ledger and Other, so it could be either.</p><p>The goal id break_order stands here.</p></section>
</body></html>"""
SLIDE = """<section id="one"><h2>One slide</h2><p>Quotes AB-2.</p><aside>Notes quote AB-3.</aside><a href="https://example.test/guide#beta">more</a></section>"""
README_MD = "# Kit\n\nThe kit.\n\n## Install\n\nRun it. `killed` here is not a stat: this file does not name stats.\n\n```\n# not a heading\n```\n\n## Limits\n\nCites AB-2 as well.\n"
STATS_MD = "# Stats\n\n| stat | why |\n|---|---|\n| `killed` | breaks caught |\n| `graph_nodes` | nodes |\n"
CONCEPTS_PY = 'CONCEPTS = [("first", "The first concept", ["alpha", "beta"])]\n'
GOALS_YAML = "goals:\n  - id: owner\n    trained_to: Who owns new code\n    state: trained\n  - id: tests\n    trained_to: Which tests catch a break\n    state: trained\n  - id: break_order\n    trained_to: Which breaks will get past the tests\n    state: trained\n"
STATS_JSON = json.dumps({"stats": [{"name": "killed", "record": "morph", "verdict": "DECIDES"}, {"name": "graph_nodes", "record": "morph", "verdict": "TRIGGERED"},
                                   {"name": "time", "record": "morph", "verdict": "IDENTITY"}, {"name": "seconds", "record": "morph", "verdict": "WATCHED"}, {"name": "break_order/x", "record": "morph", "verdict": "WATCHED"}]})


class SelftestFixture(object):
    """A small document set and a small graph in a temp folder, written fresh for each plant. Nothing here touches a real document"""

    def __init__(self):
        self.d = tempfile.mkdtemp(prefix="doc_nodes_selftest_")
        self.model = os.path.join(self.d, "model")
        os.makedirs(self.model)
        os.makedirs(os.path.join(self.d, "deck", "slides"))
        for rel, text in (("guide.html", GUIDE), ("deck/deck.json", json.dumps({"order": ["one"]})), ("deck/slides/one.html", SLIDE), ("README.md", README_MD),
                          ("STATS.md", STATS_MD), ("pages.py", CONCEPTS_PY), ("goals.yaml", GOALS_YAML), ("goals.json", STATS_JSON)):
            io.open(os.path.join(self.d, rel), "w", encoding="utf-8", newline="\n").write(text)
        self.n = 0

    def docs(self):
        p = lambda rel: os.path.join(self.d, rel)
        return {"documents": [{"as": "guide", "read": "html_sections", "path": p("guide.html")}, {"as": "slide", "read": "deck", "path": p("deck/deck.json")},
                              {"as": "concept", "read": "concepts", "path": p("pages.py"), "pages_of": "guide"}, {"as": "goals", "read": "goal_rows", "path": p("goals.yaml")},
                              {"as": "stats", "read": "stat_rows", "path": p("goals.json")}, {"as": "readme", "read": "markdown", "path": p("README.md")},
                              {"as": "statsmd", "read": "markdown", "path": p("STATS.md"), "names_stats": "morph"}],
                "links": [{"node": "req:AB-4", "doc": "doc:guide#beta", "says": "The wallet refuses", "why": "the section says this requirement in other words"}]}

    def store(self, meta=None, reqs=("AB-1", "AB-2", "AB-3", "AB-4", "AB-5", "AB-10"), code=("Ledger.Post", "Ledger.Close", "Ledger.Open~1", "Ledger.Open~2", "Other.Close", "Ledger.Ledger")):
        self.n += 1
        db = sqlite3.connect(os.path.join(self.d, "g%d.sqlite" % self.n))
        db.executescript(SCHEMA)
        db.executemany("INSERT INTO entity VALUES (?, 'requirement', ?, 'authored')", [("req:" + r, r) for r in reqs])
        db.executemany("INSERT INTO entity VALUES (?, 'code', ?, NULL)", [("code:" + c, c) for c in code])
        attach(db, {"docs": self.docs()} if meta is None else meta, self.model, self.d)
        return db

    def write(self, rel, text):
        io.open(os.path.join(self.d, rel), "w", encoding="utf-8", newline="\n").write(text)


def edges_of(db, node):
    return sorted(o for (o,) in db.execute("SELECT DISTINCT o FROM edge WHERE s = ? AND p = ?", (node, DOCUMENTED_IN)))


def selftest():
    results = []

    def check(name, good, said=""):
        results.append(bool(good))
        print("selftest: %-74s %s%s" % (name, "PASS" if good else "FAIL", "" if good else "  " + str(said)))
    f = SelftestFixture()
    db = f.store()
    gate = DocGate(db)
    holes = gate.holes()
    # the clean set: what must link, with what reason
    check("a piece that names a requirement links it", edges_of(db, "req:AB-1") == ["doc:guide#alpha"], edges_of(db, "req:AB-1"))
    check("a slide's words and its speaker notes both link", "doc:slide#one" in edges_of(db, "req:AB-2") and "doc:slide#one" in edges_of(db, "req:AB-3"))
    check("two documents naming one requirement give two links", edges_of(db, "req:AB-2") == ["doc:readme#limits", "doc:slide#one"], edges_of(db, "req:AB-2"))
    check("a qualified symbol links its code node", "doc:guide#alpha" in edges_of(db, "code:Ledger.Post"))
    check("text outside every section links too", "doc:guide#outside-any-section" in edges_of(db, "req:AB-3"), edges_of(db, "req:AB-3"))
    check("the reason says what matched and where", any("names 'AB-1'" in v and "guide.html:2" in v for (v,) in db.execute("SELECT v FROM literal WHERE s = 'req:AB-1' AND p = ?", (BECAUSE,))))
    check("a bare member in a code span, its type in the same block, links", edges_of(db, "code:Ledger.Close") == ["doc:guide#beta"], edges_of(db, "code:Ledger.Close"))
    check("PLANT a class named in a code span does not link its constructor", not edges_of(db, "code:Ledger.Ledger"), edges_of(db, "code:Ledger.Ledger"))
    check("a bare member with no type in its block does not link", "doc:guide#gamma" not in edges_of(db, "code:Ledger.Close") and not edges_of(db, "code:Other.Close"))
    check("a goal's title links it, as well as its own row", edges_of(db, "goal:owner") == ["doc:goals#owner", "doc:guide#beta"], edges_of(db, "goal:owner"))
    check("a snake_case stat links anywhere, with its own row and STATS", "doc:guide#beta" in edges_of(db, "stat:morph/graph_nodes") and "doc:stats#morph/graph_nodes" in edges_of(db, "stat:morph/graph_nodes"))
    check("a plain-word stat links only in a code span of a stats document", "doc:statsmd#stats" in edges_of(db, "stat:morph/killed") and "doc:readme#install" not in edges_of(db, "stat:morph/killed"))
    check("PLANT a stats row names nothing but itself: a goal's id inside a stat's name is no link", "doc:stats#morph/break_order/x" not in edges_of(db, "goal:break_order"), edges_of(db, "goal:break_order"))
    check("an IDENTITY row is no stat node", db.execute("SELECT COUNT(*) FROM entity WHERE id = 'stat:morph/time'").fetchone()[0] == 0)
    check("an authored line links, carrying its why", "doc:guide#beta" in edges_of(db, "req:AB-4") and any("in other words" in v for (v,) in db.execute("SELECT v FROM literal WHERE s = 'req:AB-4' AND p = ?", (BECAUSE,))))
    check("a concept page is built from its sections", sorted(o for (o,) in db.execute("SELECT o FROM edge WHERE s = 'doc:concept#first' AND p = ?", (BUILT_FROM,))) == ["doc:guide#alpha", "doc:guide#beta"])
    check("a hyperlink to another document is an edge, both ways", {("doc:guide#alpha", "doc:slide#one"), ("doc:slide#one", "doc:guide#beta")} <= set(db.execute("SELECT s, o FROM edge WHERE p = ?", (LINKS_TO,))))
    check("a page's text outside its sections is a piece of its own", "doc:guide#outside-any-section" in DocIndex(db).kind)
    check("a # in a fenced code block is not a heading", not any("not-a-heading" in i for i in DocIndex(db).kind))
    check("the clean set has no failing hole", not [h for h in holes if h[2]], [h for h in holes if h[2]])

    def said_why(node, doc):
        return [v.split(" | ", 1)[1] for (v,) in db.execute("SELECT v FROM literal WHERE s = ? AND p = ? AND v LIKE ?", (node, BECAUSE, doc + " | %"))]
    for node, doc, want in (("code:Ledger.Post", "doc:guide#alpha", "names 'Ledger.Post' (its symbol) at "),
                            ("code:Ledger.Close", "doc:guide#beta", "names 'Close (with its type Ledger in the same block)' (its member and type) at "),
                            ("stat:morph/graph_nodes", "doc:guide#beta", "names 'graph_nodes' (its name) at "), ("goal:owner", "doc:guide#beta", "names 'Who owns new code' (its title) at "),
                            ("stat:morph/killed", "doc:statsmd#stats", "names 'killed' (its name, in a document that names stats) at "),
                            ("goal:break_order", "doc:guide#delta", "names 'break_order' (its id) at ")):
        check("the reason of %s says which rule matched" % node, any(w.startswith(want) for w in said_why(node, doc)), said_why(node, doc))
    check("a plain stat's reason cites its own line, in markdown", any(w.endswith("STATS.md:5") for w in said_why("stat:morph/killed", "doc:statsmd#stats")), said_why("stat:morph/killed", "doc:statsmd#stats"))
    line = GUIDE[:GUIDE.index("The goal id break_order")].count("\n") + 1
    check("a snake_case goal id links where it stands, and cites that line", any(w.endswith("guide.html:%d" % line) for w in said_why("goal:break_order", "doc:guide#delta")), said_why("goal:break_order", "doc:guide#delta"))
    amb = DocOverlay(db, {"docs": f.docs()}, f.model, f.d).attach().ambiguous
    check("PLANT a bare member two named types could own links neither, and is recorded", "doc:guide#delta" not in edges_of(db, "code:Ledger.Close") and not edges_of(db, "code:Other.Close") and [(m, o) for m, o, _ in amb] == [("Close", ["Ledger", "Other"])] and amb[0][2].endswith(":%d" % (GUIDE[:GUIDE.index('<section id="delta">')].count("\n") + 1)), amb)
    no_reqs = f.store({"docs": dict(f.docs(), links=[])}, reqs=())
    check("a graph holding no requirement reads the documents all the same", not [h for h in DocGate(no_reqs).holes() if h[2]] and "doc:guide#alpha" in edges_of(no_reqs, "code:Ledger.Post"))
    check("code no document names is counted, exactly", ("the code nodes", "4 of 6 named by no document", False) in holes, holes)
    check("PLANT no code left unnamed: the count's line goes", not [h for h in DocGate(f.store(code=("Ledger.Post",))).holes() if h[0] == "the code nodes"])
    check("no ledger yet is said, and lists", ("the ledger", "no %s yet: a link that vanishes cannot be seen until one is accepted" % LEDGER, False) in holes, holes)
    # English is not a name
    check("PLANT English: `owner` and `tests` in prose link no goal", "doc:guide#alpha" not in edges_of(db, "goal:owner") and "doc:guide#alpha" not in edges_of(db, "goal:tests"))
    check("PLANT English: `killed` in prose links no stat", "doc:guide#alpha" not in edges_of(db, "stat:morph/killed"))
    check("PLANT a longer token: AB-10x links neither AB-1 nor AB-10", edges_of(db, "req:AB-10") == [])
    check("PLANT a longer symbol: Ledger.Postpone does not link Ledger.Post", "doc:guide#gamma" not in edges_of(db, "code:Ledger.Post"))
    # a requirement with no document is counted, named, and is no failure
    rep = DocReport(db).counts()
    check("PLANT no document: AB-5 is counted and named as unlinked, and lists", "req:AB-5" in rep["kinds"]["requirement"]["unlinked"] and ("req:AB-5", "no document describes it", False) in holes)
    check("PLANT the count of unlinked requirements is exact", len(rep["kinds"]["requirement"]["unlinked"]) == 2, rep["kinds"]["requirement"]["unlinked"])
    # the ledger: a section renamed
    good = DocSet(f.docs(), f.d)
    ledger = LinkLedger(f.model)
    ledger.write({(n, d, ("authored" if fam == FAMILY_AUTHORED else "names"), "x") for n, d, fam in db.execute("SELECT s, o, fam FROM edge WHERE p = ?", (DOCUMENTED_IN,))}, "the selftest's clean set")
    f.write("guide.html", GUIDE.replace('id="alpha"', 'id="alpha2"'))
    moved = f.store()
    lost = [(s, v) for s, v in moved.execute("SELECT s, v FROM literal WHERE p = ?", (LINK_LOST,))]
    check("PLANT a section renamed: the link it made shows as LOST", any(s == "req:AB-1" and "doc:guide#alpha " in v for s, v in lost), lost)
    check("PLANT a section renamed: the same name now links the new piece", edges_of(moved, "req:AB-1") == ["doc:guide#alpha2"])
    check("with a ledger held, the line saying there is none goes", not [h for h in DocGate(moved).holes() if h[0] == "the ledger"])
    check("PLANT a section renamed: a lost link lists, it does not fail", any(h[1].startswith("lost link") and not h[2] for h in DocGate(moved).holes()))
    f.write("guide.html", GUIDE.replace('id="beta"', 'id="beta2"'))
    gone = f.store()
    check("PLANT the piece an authored line points at renamed: BROKEN, failing", any(h[0] == "req:AB-4" and h[2] and "no longer exists" in h[1] for h in DocGate(gone).holes()), DocGate(gone).holes())
    f.write("guide.html", GUIDE.replace("The wallet refuses", "The purse declines"))
    words = f.store()
    check("PLANT the words an authored line stands on rewritten: BROKEN, failing", any(h[0] == "req:AB-4" and h[2] and "no longer says" in h[1] for h in DocGate(words).holes()))
    f.write("guide.html", GUIDE)
    os.remove(ledger.path)
    # the fabricated id
    f.write("guide.html", GUIDE.replace("The rule AB-1 holds", "The rule AB-99 holds"))
    fake = f.store()
    check("PLANT a fabricated id: AB-99 is flagged and fails, naming the piece", any(h[0] == "doc:guide#alpha" and h[2] and "AB-99" in h[1] for h in DocGate(fake).holes()), DocGate(fake).holes())
    check("PLANT a fabricated id makes no edge and no node", db.execute("SELECT COUNT(*) FROM entity WHERE id = 'req:AB-99'").fetchone()[0] == 0)
    f.write("guide.html", GUIDE)
    f.write("guide.html", GUIDE.replace("The rule AB-1 holds", "The rule AB-99 holds"))
    known = f.store({"docs": dict(f.docs(), known=[{"names": "AB-99", "why": "the guide still lists it; its owner decides"}])})
    check("PLANT a fabricated id declared known, with why: lists, does not fail", not [h for h in DocGate(known).holes() if h[2]] and any("declared known" in h[1] and "AB-99" in h[1] for h in DocGate(known).holes()), DocGate(known).holes())
    f.write("guide.html", GUIDE)
    lapsed = f.store({"docs": dict(f.docs(), known=[{"names": "AB-99", "why": "stale"}])})
    check("PLANT a known phantom no document names any more: it lapses, failing", any(h[2] and "AB-99" in h[1] for h in DocGate(lapsed).holes()), DocGate(lapsed).holes())
    meta = {"docs": dict(f.docs(), links=[{"node": "req:AB-1", "doc": "doc:guide#alpha", "says": "The rule"}, {"node": "req:AB-404", "doc": "doc:guide#alpha", "says": "The rule", "why": "x"}])}
    bad = f.store(meta)
    held = DocGate(bad).holes()
    check("PLANT an authored line with no why: BROKEN, failing", any(h[2] and "no reason" in h[1] for h in held), held)
    blank = DocGate(f.store({"docs": dict(f.docs(), links=[{"node": "req:AB-1", "doc": "doc:guide#alpha", "says": "The rule", "why": "   "}])})).holes()
    check("PLANT an authored line whose why is only spaces: BROKEN, failing", any(h[0] == "req:AB-1" and h[2] and "no reason" in h[1] for h in blank), blank)
    nameless = DocGate(f.store({"docs": dict(f.docs(), links=[{"doc": "doc:guide#alpha", "says": "The rule", "why": "x"}])})).holes()
    check("PLANT an authored line naming no node at all: failing, on the links file", any(h[0] == "the links file" and h[2] for h in nameless), nameless)
    check("PLANT an authored line naming no node: BROKEN, failing", any(h[2] and "does not hold" in h[1] for h in held), held)
    # the empty set: failing, never zero
    for what, m in (("documents: []", {"docs": {"documents": []}}), ("docs: {}", {"docs": {}}), ("docs with only links", {"docs": {"links": []}}),
                    ("a missing file", {"docs": {"documents": [{"as": "x", "read": "markdown", "path": os.path.join(f.d, "gone.md")}]}}),
                    ("no reader", {"docs": {"documents": [{"as": "x", "read": "telepathy", "path": os.path.join(f.d, "README.md")}]}})):
        e = f.store(m)
        h = DocGate(e).holes()
        check("PLANT an empty or unreadable set (%s): fails, does not report zero" % what, len(h) == 1 and h[0][0] == "the documents" and h[0][2] and h[0][1].startswith(UNREADABLE), h)
    f.write("empty.html", "<html><body><p>no sections at all</p></body></html>")
    h = DocGate(f.store({"docs": {"documents": [{"as": "x", "read": "html_sections", "path": os.path.join(f.d, "empty.html")}]}})).holes()
    check("PLANT a document that reads as no pieces fails", len(h) == 1 and h[0][2] and "read as nothing" in h[0][1], h)
    f.write("bad.yaml", "documents: [{as: x")
    h = DocGate(f.store({"docs": {"sets": [os.path.join(f.d, "bad.yaml")]}})).holes()
    check("PLANT a set file that cannot be parsed fails", len(h) == 1 and h[0][2] and "cannot be read" in h[0][1], h)
    f.write("bad.json", "{not json")
    h = DocGate(f.store({"docs": {"documents": [{"as": "x", "read": "deck", "path": os.path.join(f.d, "bad.json")}]}})).holes()
    check("PLANT a document that cannot be parsed fails, naming it", len(h) == 1 and h[0][2] and "cannot be read" in h[0][1] and "`x`" in h[0][1], h)
    f.write("nest.html", '<section id="a"><section id="b"></section></section>')
    h = DocGate(f.store({"docs": {"documents": [{"as": "x", "read": "html_sections", "path": os.path.join(f.d, "nest.html")}]}})).holes()
    check("PLANT nested sections refuse rather than half read", len(h) == 1 and h[0][2] and "readable sections" in h[0][1], h)
    h = DocGate(f.store({})).holes()
    check("a model that declares nothing is told it was not measured, and does not fail", h == [("the documents", "not measured: this model declares no document set (meta.docs)", False)], h)
    # what-to-update
    ix = DocIndex(db)
    a = ix.what_to_update("req:AB-1")
    check("what-to-update names the piece, its file and line, and why", a["direct"][0]["doc"] == "doc:guide#alpha" and a["direct"][0]["at"].endswith("guide.html:2") and "names 'AB-1'" in a["direct"][0]["because"][0], a)
    check("what-to-update names the pages built from the piece", a["direct"][0]["pages"] == ["doc:concept#first"], a["direct"][0])
    check("what-to-update names the piece's hyperlinked slide", a["direct"][0]["linked"] == ["doc:slide#one"], a["direct"][0])
    check("what-to-update takes a bare id", ix.what_to_update("AB-1")["node"] == "req:AB-1")
    check("what-to-update on a node the graph lacks is UNKNOWN, not `no documents`", ix.what_to_update("AB-404") is None)
    check("what-to-update on an unlinked node answers with an empty list", ix.what_to_update("req:AB-5")["direct"] == [])
    db.execute("INSERT INTO entity VALUES ('assert:AB-1#1', 'assertion', 't', 'calls')")
    db.execute("INSERT INTO edge VALUES ('req:AB-1', ?, 'assert:AB-1#1', 16)", (DocIndex.ASSERTS,))
    db.execute("INSERT INTO edge VALUES ('assert:AB-1#1', ?, 'code:Ledger.Open~1', 32)", (DocIndex.IS_AT,))
    t = ix.what_to_update("code:Ledger.Open~1")
    check("code with no document of its own answers through its requirement, labelled so", t["direct"] == [] and t["through"][0]["via"] == "req:AB-1" and t["through"][0]["documents"][0]["doc"] == "doc:guide#alpha", t)
    check("a piece that names only the type is shown apart, not as a link", ("doc:guide#beta", "Ledger") in ix.owner_named_in("code:Ledger.Open~1") and "doc:guide#beta" not in edges_of(db, "code:Ledger.Open~1"))
    f.write("guide.html", GUIDE.replace("Beta</h2>", "UC-1 Beta</h2>"))
    uc = f.store(reqs=("AB-1", "AB-5"))
    uc.execute("INSERT INTO entity VALUES ('uc:UC-1', 'use case', 'x', 'sea')")
    uc.execute("INSERT INTO edge VALUES ('uc:UC-1', 100, 'step:UC-1#1', 16)")
    uc.execute("INSERT INTO edge VALUES ('step:UC-1#1', 107, 'req:AB-5', 16)")
    attach(uc, {"docs": f.docs()}, f.model, f.d)
    f.write("guide.html", GUIDE)
    check("a requirement no piece names shows the piece naming its use case, apart", ("doc:guide#beta", "UC-1") in DocIndex(uc).owner_named_in("req:AB-5") and edges_of(uc, "req:AB-5") == [], DocIndex(uc).owner_named_in("req:AB-5"))
    check("a quoted literal is found across documents", {p.id for p, _ in good.quoting("AB-2")} == {"doc:readme#limits", "doc:slide#one"})
    # stability, and a re-attach
    def triples(d):
        return sorted(map(str, d.execute("SELECT * FROM entity").fetchall())) + sorted(map(str, d.execute("SELECT * FROM edge").fetchall())) + sorted(map(str, d.execute("SELECT * FROM literal").fetchall()))
    one, two = f.store(), f.store()
    check("two attaches of one set are identical", triples(one) == triples(two))
    attach(one, {"docs": f.docs()}, f.model, f.d)
    check("attaching twice to one store changes nothing", triples(one) == triples(two))
    no = f.store({})
    check("declaring nothing adds no node, edge or literal", not no.execute("SELECT COUNT(*) FROM entity WHERE kind IN ('doc', 'stat', 'goal')").fetchone()[0] and not no.execute("SELECT COUNT(*) FROM edge").fetchone()[0])
    # the command line's own refusals: its exit codes are what a person or a tick acts on
    def said(*argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(list(argv))
        return code, buf.getvalue()
    f.write("guide.html", GUIDE.replace("The rule AB-1 holds", "The rule AB-99 holds"))
    f.store()
    store = os.path.join(f.d, "g%d.sqlite" % f.n)
    f.write("guide.html", GUIDE)
    check("PLANT accept is refused while a document says what is not so", said("accept", f.model, "--db", store, "--why", "x")[0] == 1 and not os.path.exists(os.path.join(f.model, LEDGER)))
    f.store()
    store = os.path.join(f.d, "g%d.sqlite" % f.n)
    check("accept on a clean set writes the ledger", said("accept", f.model, "--db", store, "--why", "x")[0] == 0 and os.path.exists(os.path.join(f.model, LEDGER)))
    os.remove(os.path.join(f.model, LEDGER))
    check("the command answers what-to-update and exits 0", said("what-to-update", "req:AB-1", "--db", store)[0] == 0 and "doc:guide#alpha" in said("what-to-update", "req:AB-1", "--db", store)[1])
    check("the command says UNKNOWN and exits 2 on a node the graph lacks", said("what-to-update", "AB-404", "--db", store) [0] == 2 and "UNKNOWN" in said("what-to-update", "AB-404", "--db", store)[1])
    f.store({})
    empty = os.path.join(f.d, "g%d.sqlite" % f.n)
    check("on a model that declares no documents it says it cannot say, and exits 2", said("what-to-update", "req:AB-1", "--db", empty)[0] == 2 and "cannot say" in said("what-to-update", "req:AB-1", "--db", empty)[1])
    # the mechanical check of the reasons
    # the citations are written relative to ROOT when the file is under it, and a stranger's temp folder IS under it (ROOT defaults to the home folder and the temp folder is
    # in the home folder on Windows): resolve them the way the code that wrote them does, not against the fixture's folder (MEASURED 10-31 on a clean home: 93 of 94 checks
    # passed and a clean install read INSTALLED WITH FAILURES)
    v = Sample(db)
    checked, wrong = v.verify()
    check("every link by name stands on the line it cites (%d read)" % checked, checked > 5 and not wrong, wrong[:3])
    db.execute("UPDATE literal SET v = replace(v, 'guide.html:2', 'guide.html:5') WHERE s = 'req:AB-1' AND p = ?", (BECAUSE,))
    _, wrong = Sample(db, f.d).verify()
    check("PLANT a reason citing the wrong line is refused by the check", any(n == "req:AB-1" for n, _, _ in wrong), wrong[:3])
    # the graph's own invariants: nothing the loop already reads moves
    ok = selftest_invariants(check)
    print("selftest: %s (%d of %d)" % ("PASS" if all(results) and ok else "FAIL", sum(results), len(results)))
    return 0 if all(results) and ok else 1


def selftest_invariants(check):
    """Documents must not move what the loop already measures: no requirement's fingerprint, no distance between nodes, no anchor count, no
    gap. A plant shows each guard fires: the same build with the relation treated as model content does move the fingerprint."""
    try:
        import usecase_kg as kgm
        import usecase_model as ucm
    except ImportError as e:
        print("selftest: the graph's invariants not run: %s" % e)
        return False
    if DOCUMENTED_IN not in kgm.MEASURED_EDGES:
        print("selftest: usecase_kg does not hold the document relations yet: invariants not run")
        return True
    t = tempfile.mkdtemp(prefix="doc_nodes_inv_")
    prose = os.path.join(t, "src.txt")
    io.open(prose, "w", encoding="utf-8").write("The reader wants to see the flow. The view must not lie.\n")
    meta, data = ucm._clean_data(prose)
    doc = os.path.join(t, "guide.md")
    io.open(doc, "w", encoding="utf-8").write("# Guide\n\nR-1 and R-2 and IR-1 are all here.\n")
    bare = kgm.KnowledgeGraph.build(ucm.UseCaseModel.from_data(meta, data, root=t), os.path.join(t, "bare.sqlite"))
    with_docs = dict(meta, docs={"documents": [{"as": "guide", "read": "markdown", "path": doc}]})
    full = kgm.KnowledgeGraph.build(ucm.UseCaseModel.from_data(with_docs, data, root=t), os.path.join(t, "full.sqlite"))
    good = True

    def hold(name, cond, said=""):
        nonlocal good
        good &= bool(cond)
        check(name, cond, said)
    a, b = bare.fingerprints(), full.fingerprints()
    hold("the build with documents really has links", full.db.execute("SELECT COUNT(*) FROM edge WHERE p = ?", (DOCUMENTED_IN,)).fetchone()[0] >= 3)
    hold("no requirement's fingerprint moves when documents are attached", all(a[k] == b[k] for k in a if k.startswith("req:")), [k for k in a if a[k] != b[k]][:3])
    saved = kgm.MEASURED_EDGES
    kgm.MEASURED_EDGES = tuple(p for p in saved if p not in RELATIONS)
    c = full.fingerprints()
    kgm.MEASURED_EDGES = saved
    hold("PLANT a document relation read as model content moves a fingerprint: the guard is real", any(a[k] != c[k] for k in a if k.startswith("req:")))
    hold("no distance shrinks through a document", full.distance("req:R-1", "req:IR-1") == bare.distance("req:R-1", "req:IR-1"), (full.distance("req:R-1", "req:IR-1"), bare.distance("req:R-1", "req:IR-1")))
    open_walk = full.distance("req:R-1", "req:IR-1", skip=())
    hold("PLANT a walk that crosses documents does get shorter: the skip is real", open_walk is not None and open_walk <= 2, open_walk)
    hold("no anchor count moves", bare.anchor_k() == full.anchor_k())
    gaps = lambda kg: sorted(str(g) for g in kg.schema_gaps() + kg.graph_gaps())
    hold("no gap appears or goes", gaps(bare) == gaps(full), set(gaps(full)) ^ set(gaps(bare)))
    for tool, constant in (("graph_gnn.py", "NOT_THE_MODEL"), ("graph_flythrough.py", "NOT_DRAWN")):
        path = os.path.join(HERE, tool)
        if os.path.exists(path):
            said = [ast.literal_eval(n.value) for n in ast.parse(io.open(path, encoding="utf-8").read()).body
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == constant for t in n.targets)]
            hold("%s leaves out exactly the kinds this adds (%s)" % (tool, constant), said == [KINDS], said)
    try:
        import graph_gnn
    except ImportError:
        print("selftest: the network's view not read: it needs torch")
    else:
        same = graph_gnn.GraphView(os.path.join(t, "bare.sqlite"), saved_view=False), graph_gnn.GraphView(os.path.join(t, "full.sqlite"), saved_view=False)
        hold("the network's view of the graph is the same with documents as without", same[0].ids == same[1].ids and sorted(same[0].edges) == sorted(same[1].edges), len(same[1].ids) - len(same[0].ids))
        saved, graph_gnn.NOT_THE_MODEL = graph_gnn.NOT_THE_MODEL, ()
        leaked = graph_gnn.GraphView(os.path.join(t, "full.sqlite"), saved_view=False)
        graph_gnn.NOT_THE_MODEL = saved
        hold("PLANT with the documents let in, the network does see them: the guard is real", len(leaked.ids) > len(same[1].ids))
    bare.db.close()
    full.db.close()
    return good


# ---------------------------------------------------------------------- the command line

def model_meta(folder):
    import usecase_model as ucm
    return ucm.UseCaseModel.load(folder).meta


def set_meta(a):
    """the model's docs, with `--set` added or standing in, so a graph can be tried against a document set before its model declares one"""
    meta = dict(model_meta(a.model)) if a.model else {}
    if a.set:
        docs = dict(meta.get("docs") or {})
        docs["sets"] = list(docs.get("sets") or []) + [os.path.abspath(a.set)]
        meta["docs"] = docs
    return meta


def text_answer(a):
    print("%s (%s)" % (a["node"], a["kind"]))

    def show(rows, pad="  "):
        for r in rows:
            print("%s%s  %s  [%s]" % (pad, r["doc"], r["title"], r["at"]))
            for b in r["because"]:
                print("%s    because %s" % (pad, b))
            for p in r["pages"]:
                print("%s    rebuilds %s" % (pad, p))
            for p in r["linked"]:
                print("%s    linked by its own hyperlink: %s" % (pad, p))
    if a["direct"]:
        print("  described by %d piece(s):" % len(a["direct"]))
        show(a["direct"])
    else:
        print("  no document names it directly")
    for t in a["through"]:
        print("  through %s (%s): %d piece(s)" % (t["via"], t["how"], len(t["documents"])))
        show(t["documents"], "    ")
    if a["owner"]:
        print("  names what it belongs to, not it: read these if that changes: %s" % ", ".join(sorted({"%s (%s)" % (o["doc"], o["names"]) for o in a["owner"]})))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["attach", "report", "what-to-update", "quoting", "accept", "sample"])
    ap.add_argument("arg", nargs="?", help="the store, the node id, the text, or the model folder, as the command says")
    ap.add_argument("--db")
    ap.add_argument("--model")
    ap.add_argument("--set", help="a document set file, added to what the model declares")
    ap.add_argument("--out", help="attach: the copy to write; the store read is never changed")
    ap.add_argument("--why")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--kinds", help="sample: only these node kinds, comma separated")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--verify", action="store_true", help="report: every link by name must stand on the line it cites")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.cmd or not a.arg:
        ap.error("a command and its argument, or --selftest")
    if a.cmd == "attach":
        if not a.out or os.path.abspath(a.out) == os.path.abspath(a.arg):
            ap.error("attach needs --out, a copy: it never writes the store it reads")
        shutil.copyfile(a.arg, a.out)
        db = sqlite3.connect(a.out)
        over = DocOverlay(db, set_meta(a), a.model).attach()
        DocReport(db).print()
        if over.ambiguous:
            print("  ambiguous bare members, not linked: %d (%s)" % (len(over.ambiguous), "; ".join("%s in %s" % (m, "/".join(o)) for m, o, _ in over.ambiguous[:5])))
        if over.dangling:
            print("  hyperlinks that land on no piece: %d (%s)" % (len(over.dangling), ", ".join("%s #%s" % (p.id, f) for p, f in over.dangling[:5])))
        return 2 if DocGate(db).state().startswith(UNREADABLE) else 0
    if a.cmd == "quoting":
        for p, where in DocSet(set_meta(a).get("docs"), ROOT).quoting(a.arg):
            print("%s  [%s]" % (p.id, where))
        return 0
    store = a.db or (a.arg if a.cmd in ("report", "sample") else None)
    if not store or not os.path.exists(store):
        print("doc_nodes: no store at %s" % store, file=sys.stderr)
        return 2
    db = sqlite3.connect(store)
    if a.cmd == "report":
        DocReport(db).print()
        if a.verify:
            checked, bad = Sample(db).verify()
            print("  verify: %d links by name read, %d do not stand on the line they cite" % (checked, len(bad)))
            for n, d, b in bad[:20]:
                print("    %s -> %s: %s" % (n, d, b))
            return 1 if bad else 0
        return 0
    if a.cmd == "what-to-update":
        answer = DocIndex(db).what_to_update(a.arg)
        if answer is None:
            print("%s: UNKNOWN, the graph holds no such node, which says nothing about the documents" % a.arg)
            return 2
        if DocGate(db).state() == NOT_DECLARED:
            print("%s: this model declares no document set, so the graph cannot say (meta.docs)" % answer["node"])
            return 2
        print(json.dumps(answer, indent=1)) if a.json else text_answer(answer)
        return 0
    if a.cmd == "sample":
        s = Sample(db)
        rows = [r for r in s.rows(set(a.kinds.split(",")) if a.kinds else None) if not r[3].startswith("its own row")]     # a node's own row cannot be wrong
        pick = random.Random(a.seed).sample(rows, min(a.n, len(rows)))
        print("%d of %d links, seed %d" % (len(pick), len(rows), a.seed))
        for node, kind, doc, because in pick:
            text = db.execute("SELECT v FROM literal WHERE s = ? AND p = 133", (node,)).fetchone()
            print("\n%s (%s) -> %s\n  node says: %s\n  because %s\n  line: %s" % (node, kind, doc, (text[0] if text else "")[:160], because[:220], (s.source_line(because) or "")[:300]))
        return 0
    if a.cmd == "accept":
        if not a.why:
            ap.error("accept needs --why")
        holes = [h for h in DocGate(db).holes() if h[2]]
        if holes:
            print("doc_nodes accept: REFUSED, %d document hole(s) stand:" % len(holes))
            for h in holes[:12]:
                print("  %s: %s" % (h[0], h[1]))
            return 1
        rows, seen = set(), set()
        for node, doc, fam in db.execute("SELECT s, o, fam FROM edge WHERE p = ?", (DOCUMENTED_IN,)):
            why = db.execute("SELECT v FROM literal WHERE s = ? AND p = ? AND v LIKE ? ORDER BY v", (node, BECAUSE, doc + " | %")).fetchone()
            how = "authored" if fam == FAMILY_AUTHORED else "row" if why and "its own row" in why[0] else "names"
            m = re.search(r"names '(.+?)' \(", why[0]) if why else None
            rows.add((node, doc, how, m.group(1) if m else how))
            seen.add((node, doc))
        LinkLedger(a.arg).write(rows, a.why)
        print("doc_nodes accept: %d links held in %s" % (len(rows), os.path.join(a.arg, LEDGER).replace("\\", "/")))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
