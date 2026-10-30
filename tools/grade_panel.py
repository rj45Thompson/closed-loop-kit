#!/usr/bin/env python3
"""grade_panel.py - more than one blind grader on a graded pair, and whether they agree.

WHY (measured 10-30, PK-5e): with one grader per pair, 3 of 5 repeated tasks reversed from one round to the next, which read as if the
loop's effect were noise. With three, 1 reversed and 1 split, and all three agreed on 7 of 11 pairs: the noise was the single grader.

Each extra grader gets its own coin (ab_loop.Grading.brief) and a brief under a random name, so nothing in its paths says which run or
round it is. The map from name to run and coin stays beside the briefs under a name no grader is told. The first grader's own key and
brief are put back: the board was unblinded with them. Its verdict is the panel's first vote, read from ab_loop's board.

    py -X utf8 grade_panel.py setup <run> [<run> ...]    a brief per extra grader; prints the names; each grader writes <name>.json beside its brief
    py -X utf8 grade_panel.py tally                      each pair's votes and majority, into ~/.claude/ab/agreement.json (run_plan and status_data read it)
    py -X utf8 grade_panel.py --selftest
"""
import io
import json
import os
import sys
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
AB = os.path.join(os.path.expanduser("~"), ".claude", "ab")
SPLIT = "split"   # no two graders agree: a verdict for nobody


class GradePanel(object):
    """The graders of a set of pairs: where their briefs and answers live, and what their votes come to."""

    def __init__(self, ab=AB):
        self.ab = ab
        self.out = os.path.join(ab, "grading", "more")
        self.map = os.path.join(self.out, "map.json")
        self.result = os.path.join(ab, "agreement.json")

    def made(self):
        return json.load(io.open(self.map, encoding="utf-8")) if os.path.exists(self.map) else {}

    def setup(self, runs, extra=2):
        import ab_loop
        os.makedirs(self.out, exist_ok=True)
        made = self.made()
        names = []
        for run in runs:
            path = os.path.join(self.ab, "runs", run)
            key_file, brief_file = os.path.join(path, "grade_key.json"), os.path.join(path, "grade_brief.txt")
            kept = [io.open(f, encoding="utf-8").read() if os.path.exists(f) else None for f in (key_file, brief_file)]
            for n in range(extra):
                ab_loop.Grading.brief(path)
                name = uuid.uuid4().hex[:12]
                io.open(os.path.join(self.out, name + ".txt"), "w", encoding="utf-8").write(io.open(brief_file, encoding="utf-8").read())
                made[name] = {"run": run, "grader": n + 2, "key": json.load(io.open(key_file, encoding="utf-8"))}
                names.append(name)
            for f, text in zip((key_file, brief_file), kept):
                if text is not None:
                    io.open(f, "w", encoding="utf-8").write(text)
        json.dump(made, io.open(self.map, "w", encoding="utf-8"), indent=1)
        return names

    def first_votes(self):
        path = os.path.join(self.ab, "results.jsonl")
        rows = [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()] if os.path.exists(path) else []
        return {r["run"]: r.get("better", "tie") for r in rows}

    @staticmethod
    def majority(votes):
        """the verdict at least half the votes share, or SPLIT"""
        top = max(set(votes), key=votes.count)
        return top if votes.count(top) * 2 > len(votes) else SPLIT

    def tally(self):
        first = self.first_votes()
        votes = {}
        for name, m in self.made().items():
            votes.setdefault(m["run"], [first.get(m["run"])] if m["run"] in first else [])
            path = os.path.join(self.out, name + ".json")
            if os.path.exists(path):
                answer = json.load(io.open(path, encoding="utf-8"))
                votes[m["run"]].append(m["key"].get(answer.get("better"), "tie"))
        rows = [{"run": run, "votes": v, "majority": self.majority(v), "agree": len(set(v)) == 1 and len(v) > 1} for run, v in sorted(votes.items())]
        json.dump(rows, io.open(self.result, "w", encoding="utf-8"), indent=1)
        return rows


def agreement(ab=AB):
    """{run: its panel's row}, empty when no panel has graded"""
    path = os.path.join(ab, "agreement.json")
    return {r["run"]: r for r in json.load(io.open(path, encoding="utf-8"))} if os.path.exists(path) else {}


def selftest():
    import tempfile
    t = tempfile.mkdtemp(prefix="grade_panel_selftest_")
    os.makedirs(os.path.join(t, "grading", "more"))
    panel = GradePanel(t)
    good = panel.majority(["on", "off", "on"]) == "on" and panel.majority(["on", "off", "tie"]) == SPLIT and panel.majority(["tie", "tie", "on"]) == "tie"
    print("selftest: MAJORITY two of three carry the verdict, and three different votes are a split %s" % ("PASS" if good else "FAIL"))
    ok = good
    io.open(os.path.join(t, "results.jsonl"), "w", encoding="utf-8").write(json.dumps({"run": "R", "better": "off"}) + "\n")
    json.dump({"a": {"run": "R", "grader": 2, "key": {"X": "on", "Y": "off"}}, "b": {"run": "R", "grader": 3, "key": {"X": "off", "Y": "on"}}}, io.open(panel.map, "w", encoding="utf-8"))
    io.open(os.path.join(panel.out, "a.json"), "w", encoding="utf-8").write(json.dumps({"better": "X"}))     # X is on, by its own coin
    io.open(os.path.join(panel.out, "b.json"), "w", encoding="utf-8").write(json.dumps({"better": "Y"}))     # Y is on, by its own coin
    row = panel.tally()[0]
    good = row["votes"] == ["off", "on", "on"] and row["majority"] == "on" and not row["agree"] and agreement(t)["R"]["majority"] == "on"
    print("selftest: TALLY each grader is unblinded by its own coin, the board's verdict is the first vote %s" % ("PASS" if good else "FAIL"))
    ok = ok and good
    print("selftest: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv):
    if not argv or argv[0] == "--selftest":
        return selftest()
    panel = GradePanel()
    if argv[0] == "setup":
        print(json.dumps(panel.setup(argv[1:])))
        return 0
    if argv[0] == "tally":
        rows = panel.tally()
        for r in rows:
            print("%-18s %-22s majority %-6s%s" % (r["run"], ",".join(r["votes"]), r["majority"], " ALL AGREE" if r["agree"] else ""))
        print("grade_panel: %d pairs; all agree on %d" % (len(rows), sum(r["agree"] for r in rows)))
        return 0
    sys.exit("grade_panel: setup <run>... | tally | --selftest")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
