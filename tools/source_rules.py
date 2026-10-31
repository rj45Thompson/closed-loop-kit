#!/usr/bin/env python3
"""source_rules.py - every RULE the reference source states must be cited by a REQUIREMENT, in the source's own subject.

WHY THIS EXISTS (2026-10-24, Rampart): a key game mechanic was missing from the requirements, and the use-case gap check did
not catch it. A ring of wall with no piece missing encloses a castle, and a player with no enclosed castle when time runs out
loses the round. This file is the repair: a rule sentence of the source that no requirement cites is now a reported hole.

What failed was not the walk, it was what the walk was given. interrogate.py proved every row of the closed space
(research/super_rampart_space.json) mapped to a use case, and it did: row S15 "repair is timed; no enclosed castle at the end
loses the round" mapped to UC-10. But S15 was MY PARAPHRASE of the manual's ":675-677 if YOU do not have at least one Castle
completely enclosed when time runs out, YOU lose the round" - the paraphrase dropped the subject, the requirement written from
it said "the TEAM loses one life", and from then on every gate agreed with every other gate about a rule the source never
states. Same for S17 "cannon award: home castle 2, every other castle 1" against ":691-692 based on the number of Castles YOU
surround". A space row is a summary; a gate that checks only summaries certifies the summariser.

So this check reads the SOURCE, not the summary:
  1. RuleSentence - every sentence of the source that states a rule (a modal or rule verb: must, cannot, only, lose, win,
     become, award, will ...), with its line numbers. Mechanical: nobody gets to decide which sentences matter.
  2. Requirement  - every line of the requirements file, with the source lines it cites (":675-677", ":752-758, :1089-1094").
  3. A sentence is COVERED when a live (not SUPERSEDED) requirement cites a range overlapping it. A WIDE cite (more than
     BLOCK lines) covers a sentence only if the requirement also shares content words with it - a cite of :654-698 is a
     pointer at a page, not a statement of each rule on it.
  4. A sentence is EXCLUDED only by a recorded reason (the space's "source_excluded" list, or a space row whose copy cell
     is 'no' - two-player-only rules in a co-op game, secret codes the product owner ruled out).
  5. SUBJECT DRIFT: a sentence whose subject is the PLAYER ("you", "your", "he", "his") cited only by requirements that
     speak of the TEAM ("team", "shared") and never of a player. This is the exact paraphrase that lost Rampart its death rule.

An uncovered sentence and a drift are QUESTIONS (interrogate.py --source prints them and fails). Measured on the plant: the
requirements file as it stood before the 10-24 fix leaves :675-677 and :689-698 uncovered.
"""
import io
import re

# a sentence states a rule when it carries one of these. Over-inclusion is the safe side: an extra sentence costs one citation
# or one recorded exclusion; a missed one is the failure this file exists to prevent.
RULE_WORDS = re.compile(
    r"\b(must|cannot|can not|can't|can only|only|will|won't|lose|loses|lost|win|wins|won|become|becomes|award|awarded|"
    r"earns?|receive|lasts?|ends?|appear|appears|destroy\w*|sink|sinks|placed|enclos\w*|surround\w*|turn into|revert\w*|"
    r"impass\w*|recruit\w*|disable|strike|leaves?|cumulative|unless|if)\b", re.I)
PERSONAL = re.compile(r"\b(you|your|yours|he|his|him|player's|each player|one player)\b", re.I)
TEAM = re.compile(r"\b(team|team's|shared|together)\b", re.I)
# an individual in a domain that is not a game is the reader, the person, the user - measured 2026-10-27: "The reader sees the
# whole flow and a readable detail together" was read as team-only ("together") because no word here named an individual
PLAYER_WORDS = re.compile(r"\b(player|players|each|every|own|their|you|your|his|her|seat|slot|reader|person|user)\b", re.I)
CITE = re.compile(r":(\d+)(?:-(\d+))?")
REQ_LINE = re.compile(r"^(REQ-[A-Z]+-\d+)\s+(.*)$")
STOP = set("the a an and or of to in on for with that this it its is are was were be been must can will shall should would "
           "could not no at by from as if then when what which who how all any each every one two some more less than you your "
           "he his him they them their there these those such also do does done has have had only so".split())
BLOCK = 25       # a cite wider than this many lines is a page pointer, not a statement


def words(text):
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in STOP}


class RuleSentence:
    """One sentence of the reference source that states a rule, with the source lines it spans (1-based, inclusive)."""

    def __init__(self, first, last, text):
        self.first, self.last, self.text = first, last, re.sub(r"\s+", " ", text).strip()

    @property
    def personal(self):
        return bool(PERSONAL.search(self.text))

    def where(self):
        return ":%d" % self.first if self.first == self.last else ":%d-%d" % (self.first, self.last)


class Requirement:
    """One line of the requirements file and the source line ranges it cites."""

    def __init__(self, rid, text, line):
        self.id, self.text, self.line = rid, text, line
        self.ranges = [(int(a), int(b or a)) for a, b in CITE.findall(text)]
        self.live = "SUPERSEDED" not in text

    def covers(self, s):
        for a, b in self.ranges:
            # the cite must hold at least half the sentence's lines: OCR sentences share their boundary lines, and a cite of
            # :673-675 ("cannot be moved again") once "covered" :675-677 ("you lose the round") by its last line alone, which
            # let a team-worded death rule citing :675-677 pass the drift plant below
            shared = min(b, s.last) - max(a, s.first) + 1
            if shared < max(1, (s.last - s.first + 2) // 2):
                continue
            if b - a + 1 <= BLOCK or len(words(self.text) & words(s.text)) >= 2:
                return True
        return False

    @property
    def speaks_of_team_only(self):
        return bool(TEAM.search(self.text)) and not PLAYER_WORDS.search(TEAM.sub("", self.text))


def extract(source_path, first=1, last=None):
    """Every rule sentence in lines first..last of a plain-text source (OCR is fine: a heading, a blank line or a bullet ends a
    sentence as well as its full stop)."""
    lines = io.open(source_path, encoding="utf-8", errors="replace").read().splitlines()
    last = last or len(lines)
    out, buf, start = [], [], None

    def flush(end):
        nonlocal buf, start
        text = " ".join(t for _, t in buf).strip()
        if text and RULE_WORDS.search(text) and len(words(text)) >= 3:
            out.append(RuleSentence(start, end, text))
        buf, start = [], None

    for n in range(first, last + 1):
        raw = lines[n - 1].rstrip()
        t = raw.strip()
        heading = t and t.upper() == t and re.search(r"[A-Z]{3}", t)
        if not t or heading or re.match(r"^[•■*\-]\s*", t) or re.match(r"^\d+[.,]\s", t):
            if buf:
                flush(n - 1)
            if not t or heading:
                continue
            t = re.sub(r"^[•■*\-]\s*|^\d+[.,]\s", "", t)
        # split at sentence ends inside the line, keeping line numbers
        parts = re.split(r"(?<=[.!?;])\s+", t)
        for i, p in enumerate(parts):
            if start is None:
                start = n
            buf.append((n, p))
            if i < len(parts) - 1 or re.search(r"[.!?;]$", p):
                flush(n)
    if buf:
        flush(last)
    return out


def requirements(path):
    reqs = []
    for i, line in enumerate(io.open(path, encoding="utf-8", errors="replace").read().splitlines(), 1):
        m = REQ_LINE.match(line.strip())
        if m:
            reqs.append(Requirement(m.group(1), m.group(2), i))
    return reqs


def parse_ranges(spec):
    """'752-758,1089-1094' or ':752-758, :1089-1094' -> [(752, 758), (1089, 1094)]"""
    return [(int(a), int(b or a)) for a, b in re.findall(r"(\d+)(?:-(\d+))?", spec or "")]


def extract_every(source_path, first=1, last=None):
    """EVERY sentence of lines first..last, rule word or not: for an essay, which states what it wants without modal verbs.
    Measured 2026-10-27 on the Blueprint view's source, an essay on visual scripting: extract() read 13 rule sentences and
    left 22 of 44 sentences unchecked ("Blueprints require dragging data lines across the screen" carries no rule word and is
    a requirement all the same). Split as usecase_model.py's sentences() splits, so the two count the same source the
    same way."""
    lines = io.open(source_path, encoding="utf-8", errors="replace").read().splitlines()
    last = last or len(lines)
    out = []
    for n in range(first, last + 1):
        t = lines[n - 1].strip()
        if not t or re.fullmatch(r"[=\-]{3,}", t) or (t.upper() == t and re.search(r"[A-Z]{3}", t)):
            continue
        t = re.sub(r"^[*\-]\s*", "", t)
        start = 0
        for m in re.finditer(r"[.!?](?=\s+[A-Z(\"])", t):
            piece = t[start:m.end()].strip()
            if re.search(r"\b(?:vs|e\.g|i\.e|etc)\.$", piece, re.I):
                continue
            out.append(RuleSentence(n, n, piece))
            start = m.end()
        if t[start:].strip():
            out.append(RuleSentence(n, n, t[start:].strip()))
    return out


def check(source_path, reqs_path, excluded, first=1, last=None, every=False):
    """-> (sentences, holes). `excluded` is a list of (first, last, why). Each hole is (kind, sentence, detail).
    every=True holds every sentence to a requirement, not only those with a rule word (an essay; see extract_every)."""
    sents = (extract_every if every else extract)(source_path, first, last)
    reqs = [r for r in requirements(reqs_path) if r.live]
    holes = []
    for s in sents:
        if any(a <= s.first and s.last <= b for a, b, _ in excluded):
            continue
        citing = [r for r in reqs if r.covers(s)]
        if not citing:
            holes.append(("source", s, "no live requirement cites " + s.where()))
        elif s.personal and all(r.speaks_of_team_only for r in citing):
            holes.append(("drift", s, "the source's subject is the player; %s speak of the team only"
                          % ", ".join(r.id for r in citing)))
    return sents, holes


def exclusions_from_space(space):
    """A space's recorded exclusions: its "source_excluded" list, and the source lines of every row whose copy cell is 'no'."""
    out = []
    for e in space.get("source_excluded", []):
        for a, b in parse_ranges(e.get("lines")):
            out.append((a, b, e.get("why", "")))
    for f in space.get("features", []):
        copy = (f.get("cells") or {}).get("copy") or {}
        if copy.get("v") == "no":
            src = ((f.get("cells") or {}).get("snes") or {}).get("src", "")
            for a, b in parse_ranges(src.split(":", 1)[1] if ":" in src.replace("://", "") else ""):
                out.append((a, b, "%s copy=no: %s" % (f.get("id"), copy.get("basis", ""))))
    return out


def selftest():
    """The two failures this file exists for, planted: an uncited per-player rule, and the same rule cited in the team's words.
    Then the honest file must pass. A gate that has only ever passed is indistinguishable from one that is switched off."""
    import os, tempfile
    d = tempfile.mkdtemp()
    src, req = os.path.join(d, "manual.txt"), os.path.join(d, "reqs.txt")
    io.open(src, "w", encoding="utf-8").write(
        "REPAIR PHASE\n\nWall Sections can only be placed on open ground.\n"
        "If you do not have at least one Castle\ncompletely enclosed when time runs out, you lose the round.\n")
    good = ("REQ-A-01  A piece goes only on open ground.  [:3]\n"
            "REQ-A-02  Every player must hold a castle inside their own walls or the round is lost.  [:4-5]\n")
    cases = [("uncited", good.splitlines()[0] + "\n", {"source"}),
             ("drift", good.splitlines()[0] + "\nREQ-A-02  If no castle is enclosed the team loses a life.  [:4-5]\n", {"drift"}),
             ("superseded", good.splitlines()[0] + "\nREQ-A-02  SUPERSEDED. Every player must hold a castle.  [:4-5]\n", {"source"}),
             ("honest", good, set())]
    ok = True
    for name, text, want in cases:
        io.open(req, "w", encoding="utf-8").write(text)
        _, holes = check(src, req, [])
        got = {k for k, _, _ in holes}
        print("  selftest %-10s want %-10s got %-10s %s" % (name, sorted(want), sorted(got), "ok" if got == want else "FAIL"))
        ok &= got == want
    # an essay sentence with no rule word: invisible to the rule filter, a hole when every sentence must be cited
    io.open(src, "a", encoding="utf-8").write("The wire cuts horizontally across everything else.\n")
    io.open(req, "w", encoding="utf-8").write(good)
    for every, want in ((False, set()), (True, {"source"})):
        _, holes = check(src, req, [], every=every)
        got = {k for k, _, _ in holes}
        print("  selftest essay every=%-5s want %-10s got %-10s %s" % (every, sorted(want), sorted(got), "ok" if got == want else "FAIL"))
        ok &= got == want
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse, json, sys
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", required=True)
    ap.add_argument("--reqs", required=True)
    ap.add_argument("--space", help="closed space json holding source_excluded and copy=no rows")
    ap.add_argument("--from", dest="first", type=int, default=1)
    ap.add_argument("--to", dest="last", type=int, default=0)
    ap.add_argument("--every-sentence", action="store_true",
                    help="hold EVERY sentence to a requirement, not only rule-word sentences (an essay, not a rulebook)")
    a = ap.parse_args()
    ex = exclusions_from_space(json.load(io.open(a.space, encoding="utf-8"))) if a.space else []
    sents, holes = check(a.source, a.reqs, ex, a.first, a.last or None, every=a.every_sentence)
    print("SOURCE RULES  %d %s sentence(s) read, %d excluded range(s), %d hole(s)" % (
        len(sents), "" if a.every_sentence else "rule", len(ex), len(holes)))
    for kind, s, detail in holes:
        print("  [%-6s] %-10s %s\n           %s" % (kind, s.where(), s.text[:150], detail))
    sys.exit(1 if holes else 0)
