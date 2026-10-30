#!/usr/bin/env python3
"""gnn_goals.py - everything the learned parts are trained for, how each is doing, and how that list has changed.

WHY (the project owner, 10-30: show "a list of everything they're trained for and how that mutates over time and grows and adjusts itself ...
because the goals will be ever changing"). The goals used to live in a docstring and a board, and a list written by hand goes stale the day a
goal moves. So the goals are DATA, gnn_goals.yaml, and everything said about them is read from the records: a score from the run that made it,
a date from git or the row's own time. A job in the records that no goal claims fails the check, so the list cannot fall behind.

  GoalBook    the goals, as data, with the check that every job in the records belongs to one
  Scores      each goal's mean over the seeds run, per project, against the plain rule it has to beat
  Candidates  what the graph could be trained for next: relations with enough links that nothing predicts, the list growing by itself
  StateLog    one row whenever a goal's state moves, so what changed is on file from now on, not reconstructed
  Timeline    every dated change, from git, the records and the log

    py -X utf8 gnn_goals.py                   the list and the timeline
    py -X utf8 gnn_goals.py --write F.json    the same as data, for a page; also logs a row if anything moved
    py -X utf8 gnn_goals.py --check           exit 1 on a job no goal claims or a trained goal with no record
    py -X utf8 gnn_goals.py --selftest
"""
import argparse
import collections
import datetime
import glob
import io
import json
import os
import sqlite3
import statistics
import subprocess
import sys
import tempfile

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BOOK = os.path.join(HERE, "gnn_goals.yaml")
LOG = os.path.join(HERE, "gnn_goals_log.jsonl")
PROJECTS = {"shops": os.path.join(HERE, "..", "shops"), "selfmodel": os.path.join(HERE, "selfmodel")}
SEEDS = os.path.join(HERE, "tick_runs")
MARGIN = 0.02      # run_plan's bar: a learned part is on only if it wins by more than this
COMBINED = os.path.join(HERE, "gnnmodels", "combined", "combined.sqlite")


def rows_of(path):
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in io.open(path, encoding="utf-8") if line.strip()]


class GoalBook(object):
    """The goals as written. Every job a record names must belong to a goal or a declared variant: that is what keeps the list honest"""

    def __init__(self, path=BOOK):
        data = yaml.safe_load(io.open(path, encoding="utf-8"))
        self.goals, self.variants, self.structure, self.candidate_links = data["goals"], data.get("variants") or [], data.get("structure") or {}, data.get("candidate_links", 20)
        self.project_names, self.rule_names = data.get("projects") or {}, data.get("rules") or {}
        self.rule_stats, self.record_names = data.get("rule_stats") or {}, data.get("records") or {}
        self.patterns_found, self.attempts = data.get("patterns_found") or [], data.get("attempts") or []
        self.jobs = {j: g["id"] for g in self.goals for j in g.get("jobs") or []}
        self.variant_jobs = {v["job"]: v for v in self.variants}

    def unclaimed(self, seen):
        return sorted(j for j in seen if j not in self.jobs and j not in self.variant_jobs)

    def learned_relations(self):
        return {g.get("learns") for g in self.goals}


class Scores(object):
    """Means over seeds, because one seed flatters: measured 10-30, the first seed of break order was the best of ten"""

    def __init__(self, projects=PROJECTS, seeds=SEEDS):
        self.by = collections.defaultdict(lambda: collections.defaultdict(dict))     # job -> project -> seed -> (learned, plain, wins, rule)
        self.sources = []
        for name, folder in projects.items():
            for row in rows_of(os.path.join(folder, "kg", "gnn_tasks.jsonl")):
                self._jobs(row["jobs"], name, row.get("seed", 0))
                self.sources.append(("gnn_tasks", name, row.get("time")))
            for row in rows_of(os.path.join(folder, "kg", "survivor_rank.jsonl")):
                self._break(row["recall20"], name, row.get("seed", 0), None)
                self.sources.append(("survivor_rank", name, row.get("time")))
        for path in sorted(glob.glob(os.path.join(seeds, "*_gnn_seeds.jsonl"))):
            for row in rows_of(path):
                self._jobs(row["jobs"], row["model"], row.get("seed", 0))
        for path in sorted(glob.glob(os.path.join(seeds, "*_cheap_seeds.jsonl"))):
            for row in rows_of(path):
                self._break(row["break_order"], row["model"], row.get("seed", 0), row.get("break_order_on"))
                e = row["equivalent"]
                self.by["equivalent"][row["model"]][row.get("seed", 0)] = (e["learned"], e["plain"], e["wins"], e["best_plain"])

    def _jobs(self, jobs, project, seed):
        for job, v in jobs.items():
            self.by[job][project][seed] = (v["learned"], v["plain"], v["wins"], v["best_plain"])

    def _break(self, recall, project, seed, on):
        learned, plain = recall["learned"], recall["operator prior"]
        self.by["break order"][project][seed] = (learned, plain, on if on is not None else learned - plain > MARGIN, "operator prior")

    def jobs_seen(self):
        return set(self.by)

    def summary(self, job, project):
        seeds = self.by.get(job, {}).get(project)
        if not seeds:
            return None
        rows = list(seeds.values())
        wins = sum(1 for r in rows if r[2])
        return {"learned": round(statistics.mean(r[0] for r in rows), 3), "plain": round(statistics.mean(r[1] for r in rows), 3), "wins": wins, "seeds": len(rows),
                "low": min(r[0] for r in rows), "high": max(r[0] for r in rows), "rule": rows[0][3], "achieved": wins > len(rows) / 2.0, "flips": 0 < wins < len(rows)}


class Candidates(object):
    """What the graph holds that nothing predicts: a relation with enough links is a goal nobody has set yet. It proposes; a person or a tick decides"""

    def __init__(self, book, store):
        self.book, self.store = book, store

    def list(self):
        if not os.path.exists(self.store):
            return []
        db = sqlite3.connect(self.store)
        names = dict(db.execute("select id,label from prop"))
        count = collections.Counter(names.get(p) for (p,) in db.execute("select p from edge"))
        db.close()
        known = self.book.learned_relations()
        return [{"relation": r, "links": n} for r, n in count.most_common() if r and n >= self.book.candidate_links and r not in known and r not in self.book.structure]


class Patterns(object):
    """What the results have in common: trying for a goal is not reaching it, and the point of the list is to see which kinds get there. It only counts, and it says how little each count rests on"""

    ANECDOTE_UNDER = 8

    def __init__(self, goals):
        self.cells = [(g, p, s) for g in goals for p, s in g["scores"].items()]
        self.goals = goals

    def label(self, support, projects):
        if support < self.ANECDOTE_UNDER:
            return "an anecdote: %d results" % support
        return "a pattern to test: %d results from %d projects" % (support, projects)

    def list(self):
        out = []
        projects = len({p for _, p, _ in self.cells})
        by = collections.defaultdict(list)
        for g, p, s in self.cells:
            by[g.get("model") or "unnamed"].append(s)
        if len(by) > 1:
            parts = []
            for name, cells in sorted(by.items()):
                reached = [s for s in cells if s["achieved"]]
                doubted = sum(1 for s in reached if s.get("doubted"))
                parts.append("%s reached its goal on %d of %d results%s" % (name, len(reached), len(cells), " (%d of them doubted)" % doubted if doubted else ""))
            out.append({"text": "; ".join(parts) + ".", "rests_on": self.label(len(self.cells), projects)})
        flips = [(g, p, s) for g, p, s in self.cells if s["flips"]]
        if flips:
            out.append({"text": "%d of %d results change with the random seed alone, so no single run settles one." % (len(flips), len(self.cells)), "rests_on": self.label(len(self.cells), projects)})
        extra = [(g, v, p, s) for g in self.goals for v in g.get("variants") or [] for p, s in v["scores"].items()]
        if extra:
            gain = [s["learned"] - g["scores"][p]["learned"] for g, v, p, s in extra if p in g["scores"]]
            helped = sum(1 for x in gain if x > MARGIN)
            out.append({"text": "Giving a goal more of the graph to read helped by more than the bar in %d of %d tries." % (helped, len(gain)), "rests_on": self.label(len(gain), projects)})
        return out


class StatList(object):
    """The second list: every stat and piece of metadata on file, what the audit says of it, and which goals use it. The goals are what we try for; these are what we could try with"""

    RECORD_KEYS = {"morph log (stats.jsonl)": "morph", "gnn_tasks.jsonl": "gnn_tasks", "survivor_rank.jsonl": "survivor_rank", "loop_ticks.jsonl": "loop_ticks", "gnn_curve.jsonl": "gnn_curve", "gnn_curve_1029.jsonl": "gnn_curve"}

    def __init__(self, book, combined=COMBINED, audit_out=None):
        self.book, self.combined, self.audit_out = book, combined, audit_out

    def audited(self):
        if self.audit_out is not None:
            return self.audit_out
        import telemetry_audit
        return telemetry_audit.audit(os.path.join(HERE, ".."), say=lambda s: None)[0]

    def linked(self):
        """stat -> the goals that use it and whether a requirement reads it, from the joined graph; empty until the subgraphs exist"""
        if not os.path.exists(self.combined):
            return {}
        db = sqlite3.connect(self.combined)
        prop = dict(db.execute("select id,label from prop"))
        out = collections.defaultdict(lambda: {"goals": set(), "read": False})
        for s, p, o in db.execute("select s,p,o from edge"):
            name = prop.get(p)
            if name == "uses stat":
                out[o.split(":", 1)[-1]]["goals"].add(s.split(":", 1)[-1])
            elif name == "reads stat":
                out[o.split(":", 1)[-1]]["read"] = True
        db.close()
        return out

    def rows(self):
        links, rows, seen = self.linked(), [], set()
        for record, fields in self.audited().items():
            key = self.RECORD_KEYS.get(record, record.split(".")[0])
            for field, verdict in fields.items():
                path = key + "/" + field.replace(".", "/")
                hit = [k for k in links if k == path or k.startswith(path + "/")]
                seen.update(hit)
                goals = sorted({g for k in hit for g in links[k]["goals"]})
                rows.append({"record": key, "name": field, "verdict": verdict.split(" for ")[0].split(" (")[0], "goals": goals, "read": any(links[k]["read"] for k in hit)})
        for k, v in sorted(links.items()):
            if k not in seen:
                rows.append({"record": k.split("/")[0], "name": k.split("/", 1)[-1], "verdict": None, "goals": sorted(v["goals"]), "read": v["read"]})
        return rows


class Asks(object):
    """How the two lists work on each other, through the graph: a goal that has not got there asks for what beat it, a stat no goal reads asks for a goal, and the graph's unclaimed relations offer goals"""

    def __init__(self, book, goals, stats, candidates):
        self.book, self.goals, self.stats, self.candidates = book, goals, stats, candidates

    def goals_ask_stats(self):
        out, seen = [], set()
        for g in self.goals:
            for project, s in g["scores"].items():
                stat = self.book.rule_stats.get(s["rule"])
                if not s["achieved"] and stat and (g["id"], stat) not in seen:
                    seen.add((g["id"], stat))
                    out.append({"goal": g["id"], "trained_to": g["trained_to"], "rule": self.book.rule_names.get(s["rule"], s["rule"]), "stat": stat})
        return out

    def stats_ask_goals(self):
        return [r for r in self.stats if not r["goals"] and r["verdict"] not in (None, "IDENTITY") and r["record"] == "morph"]

    def used_unread(self):
        return [r for r in self.stats if r["goals"] and not r["read"] and r["verdict"] != "IDENTITY"]

    def data(self):
        return {"goals_ask_stats": self.goals_ask_stats(), "stats_ask_goals": [r["name"] for r in self.stats_ask_goals()], "used_unread": [r["record"] + "/" + r["name"] for r in self.used_unread()], "graph_offers": self.candidates}


class StateLog(object):
    """One row whenever anything about a goal moves, so the list's history is kept from now on rather than worked out later"""

    def __init__(self, path=LOG):
        self.path = path

    def rows(self):
        return rows_of(self.path)

    @staticmethod
    def state_of(goals, candidates):
        return {"goals": {g["id"]: {"state": g["state"], "scores": {p: {k: s[k] for k in ("learned", "plain", "wins", "seeds")} for p, s in g["scores"].items()}} for g in goals}, "candidates": [c["relation"] for c in candidates]}

    def record(self, goals, candidates, now=None):
        state = self.state_of(goals, candidates)
        rows = self.rows()
        if rows and {k: rows[-1][k] for k in state} == state:
            return False
        state["time"] = now or datetime.datetime.now().isoformat(timespec="seconds")
        with io.open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(state, sort_keys=True) + "\n")
        return True

    def changes(self):
        """(time, goal, what moved), from each row against the one before"""
        out = []
        rows = self.rows()
        for before, after in zip(rows, rows[1:]):
            for gid, g in after["goals"].items():
                old = before["goals"].get(gid)
                if old is None:
                    out.append((after["time"], gid, "Added to the list"))
                    continue
                if old["state"] != g["state"]:
                    out.append((after["time"], gid, "From %s to %s" % (old["state"], g["state"])))
                for project, s in g["scores"].items():
                    was = old["scores"].get(project)
                    if was and (s["wins"] > s["seeds"] / 2.0) != (was["wins"] > was["seeds"] / 2.0):
                        out.append((after["time"], gid, "On %s: now %s the plain rule" % (project, "beats" if s["wins"] > s["seeds"] / 2.0 else "loses to")))
            for gid in before["goals"]:
                if gid not in after["goals"]:
                    out.append((after["time"], gid, "Removed from the list, which should never happen: a goal that stops is marked retired"))
            for r in after["candidates"]:
                if r not in before["candidates"]:
                    out.append((after["time"], None, "The graph now proposes a new goal: predict `%s`" % r))
        return out


def born(goal, cwd=HERE):
    """when the goal's code first appeared; ABSENT outside a git checkout, never a guess"""
    if not goal.get("symbol") or not goal.get("file"):
        return None
    try:
        out = subprocess.run(["git", "log", "--reverse", "--format=%aI", "-S" + goal["symbol"], "--", goal["file"]], cwd=cwd, capture_output=True, text=True, timeout=120).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return None
    return out[0][:16] if out else None


class Timeline(object):
    """Every dated change, read from where it happened"""

    def __init__(self, goals, projects=PROJECTS, seeds=SEEDS, log=None, names=None):
        self.goals, self.projects, self.seeds, self.log, self.names = goals, projects, seeds, log, names or {}

    def events(self):
        out = []
        for g in self.goals:
            if g.get("born"):
                out.append((g["born"], g["id"], "Trained to: %s" % g["trained_to"] if g["state"] != "planned" else "Its recorder written: %s" % g["trained_to"]))
        for project, folder in self.projects.items():
            name = self.names.get(project, project)
            passes = [r for r in rows_of(os.path.join(folder, "stats.jsonl")) if r.get("gnn_mrr")]
            for r in passes:
                m = r["gnn_mrr"]
                best = max(v for k, v in m.items() if k != "gnn")
                out.append((str(r.get("time") or "")[:16] or None, "owner", "%s, pass %s: %.2f against the best plain rule's %.2f" % (name, r.get("n"), m["gnn"], best)))
            for r in rows_of(os.path.join(folder, "gnn_tune.jsonl")):
                out.append((r["time"][:16], "owner", "%s: reshaped what it reads, %d variants: %s" % (name, len(r.get("variants") or []), r.get("summary", ""))))
        for path in sorted(glob.glob(os.path.join(self.seeds, "*_seeds.jsonl"))):
            stamp = datetime.datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="minutes")
            rows = rows_of(path)
            out.append((stamp, None, "More seeds behind the verdicts: %d more runs" % len(rows)))
        if self.log:
            out.extend((t[:16], g, what) for t, g, what in self.log.changes())
        return sorted(out, key=lambda e: e[0] or "")


def report(book=None, scores=None, projects=PROJECTS, seeds=SEEDS, log=None, git=True, combined=COMBINED, audit_out=None):
    book = book or GoalBook()
    scores = scores or Scores(projects, seeds)
    log = log or StateLog()
    goals = []
    for g in book.goals:
        row = dict(g)
        row["scores"] = {}
        for job in g.get("jobs") or []:
            for p in projects:
                s = scores.summary(job, p)
                if s:
                    s["doubted"] = any(d.get("project") in (None, p) for d in g.get("doubts") or [])
                    row["scores"][p] = s
        row["born"] = born(g) if git else None
        row["variants"] = [dict(v, scores={p: scores.summary(v["job"], p) for p in projects if scores.summary(v["job"], p)}) for v in book.variants if v["of"] == g["id"]]
        goals.append(row)
    stores = [os.path.join(f, "kg", os.path.basename(os.path.normpath(f)) + ".sqlite") for f in projects.values()]
    candidates = Candidates(book, stores[0]).list() if stores else []
    stats = StatList(book, combined, audit_out).rows()
    return {"goals": goals, "candidates": candidates, "patterns": Patterns(goals).list() + book.patterns_found, "attempts": book.attempts, "projects": book.project_names, "rules": book.rule_names, "records": book.record_names, "stats": stats,
            "asks": Asks(book, goals, stats, candidates).data(), "unclaimed": book.unclaimed(scores.jobs_seen()),
            "no_record": [g["id"] for g in goals if g["state"] == "trained" and not g["scores"]]}, log


def check(rep):
    problems = ["a job in the records that no goal claims: %s" % j for j in rep["unclaimed"]]
    problems += ["a goal marked trained with no record: %s" % g for g in rep["no_record"]]
    return problems


def line(s):
    return "%.2f vs %.2f (%s), wins %d of %d seeds" % (s["learned"], s["plain"], s["rule"], s["wins"], s["seeds"])


def show(rep, log):
    for g in rep["goals"]:
        print("%-16s %-42s %s%s" % (g["id"], g["trained_to"], g["state"], "  since " + g["born"] if g.get("born") else ""))
        for p, s in g["scores"].items():
            print("    %-10s %s" % (p, line(s)))
        for v in g["variants"]:
            for p, s in v["scores"].items():
                print("    %-10s with %s: %s" % (p, v["adds"], line(s)))
    print("the graph proposes:", ", ".join("%s (%d)" % (c["relation"], c["links"]) for c in rep["candidates"]) or "nothing")
    for p in rep["patterns"]:
        print("pattern: %s (%s)" % (p["text"], p["rests_on"]))
    for t, g, what in Timeline(rep["goals"], log=log, names=rep["projects"]).events():
        print("  %-16s %-14s %s" % (t or "(no time)", g or "", what[:110]))


def selftest():
    ok = True
    t = tempfile.mkdtemp(prefix="gnn_goals_selftest_")
    book_path = os.path.join(t, "book.yaml")
    io.open(book_path, "w", encoding="utf-8").write(yaml.safe_dump({"goals": [
        {"id": "a", "trained_to": "A", "learns": "maps to", "jobs": ["job a"], "state": "trained"},
        {"id": "b", "trained_to": "B", "learns": "other", "jobs": [], "state": "planned"}], "variants": [{"job": "job a + x", "of": "a", "adds": "x"}], "structure": {"in space": "bookkeeping"}, "candidate_links": 2}))
    project = os.path.join(t, "p")
    os.makedirs(os.path.join(project, "kg"))
    tasks = os.path.join(project, "kg", "gnn_tasks.jsonl")
    io.open(tasks, "w", encoding="utf-8").write(json.dumps({"time": "2026-10-29T10:00:00", "seed": 0, "jobs": {"job a": {"learned": .5, "plain": .6, "wins": False, "best_plain": "rule"}, "job a + x": {"learned": .4, "plain": .6, "wins": False, "best_plain": "rule"}}}) + "\n")
    store = os.path.join(project, "kg", "p.sqlite")
    db = sqlite3.connect(store)
    db.executescript("create table prop(id integer,label text); create table edge(s text,p integer,o text);")
    db.executemany("insert into prop values(?,?)", [(1, "maps to"), (2, "in space"), (3, "checked by"), (4, "rare")])
    db.executemany("insert into edge values('s',?,'o')", [(1,), (1,), (2,), (2,), (3,), (3,), (4,)])
    db.commit()
    db.close()
    projects, seeds = {"p": project}, os.path.join(t, "none")
    log_path = os.path.join(t, "log.jsonl")

    joined = os.path.join(t, "joined.sqlite")
    db = sqlite3.connect(joined)
    db.executescript("create table prop(id integer,label text); create table edge(s text,p integer,o text);")
    db.executemany("insert into prop values(?,?)", [(1, "uses stat"), (2, "reads stat")])
    db.executemany("insert into edge values(?,?,?)", [("job:a", 1, "stat:morph/killed"), ("a::req:R", 2, "stat:morph/killed"), ("job:a", 1, "stat:gnn_tasks/seed")])
    db.commit()
    db.close()
    audit = {"morph log (stats.jsonl)": {"killed": "DECIDES", "idle": "WATCHED", "time": "IDENTITY"}, "gnn_tasks.jsonl": {"seed": "DECIDES"}}

    def rep():
        return report(GoalBook(book_path), Scores(projects, seeds), projects, seeds, StateLog(log_path), git=False, combined=joined, audit_out=audit)
    r0 = rep()[0]
    good = r0["asks"]["stats_ask_goals"] == ["idle"] and r0["asks"]["used_unread"] == ["gnn_tasks/seed"] and [x["name"] for x in r0["stats"] if x["goals"] == ["a"] and x["read"]] == ["killed"]
    print("selftest: TWO LISTS a stat no goal reads asks for one, a stat a goal uses with no requirement reading it is named, an identity field is neither %s" % ("PASS" if good else "FAIL"))
    ok = ok and good
    r, log = rep()
    good = not check(r) and r["goals"][0]["scores"]["p"]["learned"] == .5 and r["goals"][0]["variants"][0]["scores"]["p"]["learned"] == .4
    print("selftest: LIST a goal reads its score from the record, a variant stays under its goal %s" % ("PASS" if good else "FAIL"))
    ok = ok and good
    cells = [{"id": "x", "model": "m1", "scores": {"p": {"achieved": True, "flips": False, "learned": .5}}, "variants": []}, {"id": "y", "model": "m2", "scores": {"p": {"achieved": False, "flips": True, "learned": .5}}, "variants": []}]
    found = Patterns(cells).list()
    good = "m1 reached its goal on 1 of 1" in found[0]["text"] and "anecdote" in found[0]["rests_on"] and "1 of 2 results change" in found[1]["text"]
    print("selftest: PATTERN it counts by kind of model and calls a small count an anecdote %s" % ("PASS" if good else "FAIL"))
    ok = ok and good
    cells[0]["scores"]["p"]["doubted"] = True
    good = "(1 of them doubted)" in Patterns(cells).list()[0]["text"]
    print("selftest: DOUBTED a result that reached the bar but is doubted is counted as doubted %s" % ("PASS" if good else "FAIL"))
    ok = ok and good
    good = [c["relation"] for c in r["candidates"]] == ["checked by"]
    print("selftest: GROWS a relation with enough links and no goal is proposed; bookkeeping, a goal's own and a rare one are not %s" % ("PASS" if good else "FAIL"))
    ok = ok and good
    with io.open(tasks, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": "2026-10-30T10:00:00", "seed": 1, "jobs": {"job new": {"learned": .5, "plain": .6, "wins": False, "best_plain": "rule"}}}) + "\n")
    r2, _ = rep()
    good = any("job new" in p for p in check(r2))
    print("selftest: UNCLAIMED a new job in the records with no goal fails the check %s" % ("PASS" if good else "FAIL"))
    ok = ok and good
    io.open(tasks, "w", encoding="utf-8").write("")
    r3, _ = rep()
    good = any("trained with no record" in p for p in check(r3))
    print("selftest: NO RECORD a goal marked trained with nothing behind it fails the check %s" % ("PASS" if good else "FAIL"))
    ok = ok and good
    io.open(tasks, "w", encoding="utf-8").write(json.dumps({"seed": 0, "jobs": {"job a": {"learned": .5, "plain": .6, "wins": False, "best_plain": "rule"}}}) + "\n")
    r, log = rep()
    first = log.record(r["goals"], r["candidates"], "2026-10-30T10:00:00")
    again = log.record(r["goals"], r["candidates"], "2026-10-30T11:00:00")
    good = first and not again
    print("selftest: LOG a row is written when something moved, and not when nothing did %s" % ("PASS" if good else "FAIL"))
    ok = ok and good
    io.open(tasks, "w", encoding="utf-8").write(json.dumps({"seed": 0, "jobs": {"job a": {"learned": .9, "plain": .6, "wins": True, "best_plain": "rule"}}}) + "\n")
    r, log = rep()
    log.record(r["goals"], r["candidates"], "2026-10-30T12:00:00")
    good = any("now beats" in what for _, _, what in log.changes())
    print("selftest: MUTATES a verdict that flips shows in the timeline as a dated change %s" % ("PASS" if good else "FAIL"))
    ok = ok and good
    print("selftest: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv):
    if "--selftest" in argv:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    rep, log = report()
    problems = check(rep)
    if a.check:
        for p in problems:
            print("gnn_goals: FAIL - " + p)
        print("gnn_goals: %d goals, %d problems" % (len(rep["goals"]), len(problems)))
        return 1 if problems else 0
    if a.write:
        moved = log.record(rep["goals"], rep["candidates"])
        rep["timeline"] = [{"time": t, "goal": g, "what": w} for t, g, w in Timeline(rep["goals"], log=log, names=rep["projects"]).events()]
        rep["problems"] = problems
        io.open(a.write, "w", encoding="utf-8", newline="\n").write(json.dumps(rep, indent=1, sort_keys=True))
        print("gnn_goals: %s, %s" % (a.write, "a new row in the log" if moved else "nothing moved"))
        return 1 if problems else 0
    show(rep, log)
    for p in problems:
        print("gnn_goals: FAIL - " + p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
