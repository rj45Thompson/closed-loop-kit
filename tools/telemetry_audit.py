#!/usr/bin/env python3
"""telemetry_audit.py - is every stat the loop records actually USED? A stat gathered and never read is cost with no return.

WHY (2026-10-29): a stat nobody looks at is data gathered and not used, so the process is reviewed for what it is missing. The
closed loop's own rule, turned on its telemetry: every recorded value is traced to what reads it.

Each stat, in each record the loop writes, gets one verdict:
  DECIDES   code reads it and it changes what runs next (the stats' pick of the next area, a run_plan line a pass acts on, the idle rule)
  WATCHED   it is trended or printed, and nothing acts on it
  BANKED    training data kept for a learner that does not exist yet; it must name the board item that will read it
  UNREAD    nothing reads it - exit 1: give it a reader, or stop recording it
  IDENTITY  what a row IS (its time, model, pass number, note), not a measurement
A consumer is found by READING a tool's source for the stat's quoted name, never by a list here that could drift; only BANKED and
IDENTITY are declared, each with its reason. A name counts as read only in a tool that names the record's FILE (a word another record
happens to share is no reader), and never in a selftest's fixture or where the code WRITES it (ReaderCode): 10-30 the audit reported
6 UNREAD while 11 were, because its own plants, the recorder and an area's name all passed for readers. Read by run_plan is one hop
short: it counts only for a line of the plan a pass acts on (PlanReaders). A switch no pass reads fails the audit.

    python -X utf8 telemetry_audit.py [--root DIR]     the audit (exit 1 while any stat is UNREAD)
    python -X utf8 telemetry_audit.py --selftest
"""
import argparse
import ast
import collections
import glob
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
IDENTITY = {"time", "model", "n", "note", "tool_commit", "backfilled_from", "graph", "seed", "folds", "epochs", "by", "fullness",
            "run", "t", "mode", "loop",
            # the graph a curve row measured, described beside its score (its size is read by run_plan's floor as `requirements`)
            "nodes", "edges", "links", "linked_cells", "methods", "baselines"}
# training data with no learner yet: each names the item that will read it (the board, MORE FOR THE GNN)
BANKED = {"cochange.jsonl": "GN-6 changed together", "impact.jsonl": "GN-3 impact before the run"}
# the tools a record's reader can be: those that act on it, and those that only trend or print it (a tool counts for a record only
# when it names that record's file)
ACTING, SHOWING = ("run_plan.py", "loop_tick.py", "improvement.py"), ("loop_stats.py", "status_data.py")
# what a pass runs: a line of the plan changes a pass only if one of these reads it (a status page showing a line changes nothing)
PASS_TOOLS = ("closed_loop.py", "cs_effect_check.py", "py_effect_check.py", "usecase_kg.py", "usecase_model.py", "buzzsaw.py")


def improvement_registry(p=None):
    """(the stats an improvement requirement with a trigger covers, the banked records) from improvement/stat_requirements.yaml"""
    p = p or os.path.join(HERE, "improvement", "stat_requirements.yaml")
    if not os.path.exists(p):
        return set(), [{"record": k, "for": v, "until": "9999-12-31"} for k, v in BANKED.items()]
    import yaml
    d = yaml.safe_load(io.open(p, encoding="utf-8")) or {}
    reqs = [r for r in d.get("requirements") or [] if r.get("trigger") and r.get("action")]
    return {r["stat"] for r in reqs}, d.get("banked") or []


def morph_fields(root):
    keys = collections.Counter()
    for f in glob.glob(os.path.join(root, "**", "stats.jsonl"), recursive=True):
        for l in io.open(f, encoding="utf-8"):
            if l.strip():
                keys.update(json.loads(l).keys())
    return keys


def json_fields(root, name, nested=()):
    keys = collections.Counter()
    for f in glob.glob(os.path.join(root, "**", name), recursive=True):
        for l in io.open(f, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            keys.update(k for k in r if k not in nested)
            for n in nested:
                if isinstance(r.get(n), dict):
                    keys.update("%s.%s" % (n, k) for k in r[n])
    return keys


def source(name):
    """a tool's CODE: its docstrings, comments and selftests stripped (10-29: run_plan's docstring named gnn_curve.jsonl, and the first
    audit counted that sentence as a reader while no code read the file; 10-30: `effects` and two GNN jobs were DECIDES only because a
    selftest's fixture spelled them)"""
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return ""
    tree = ast.parse(io.open(p, encoding="utf-8").read())
    tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and "selftest" in n.name)]
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            body[0] = ast.Pass()
    return ast.unparse(tree)


class ReaderCode(ast.NodeTransformer):
    """A tool's code with the places it WRITES a field blanked: a dict display's keys and `row["x"] = ...`. The recorder always spells
    what it records, so before this every stat loop_stats writes passed for WATCHED by loop_stats itself and no stat of the morph log
    could ever be UNREAD (measured 10-30: `survived`, `effects`). An area's NAME goes too: it is a label that can share a stat's name
    (the area `assumptions`, the stat `assumptions`). What is left is what the tool reads: r["x"], r.get("x"), a name in a list it
    loops over."""

    def visit_Dict(self, node):
        self.generic_visit(node)
        node.keys = [ast.Constant("") if isinstance(k, ast.Constant) and isinstance(k.value, str) else k for k in node.keys]
        return node

    def visit_Subscript(self, node):
        self.generic_visit(node)
        if isinstance(node.ctx, ast.Store) and isinstance(node.slice, ast.Constant):
            node.slice = ast.Constant("")
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if getattr(node.targets[0], "id", "") == "AREAS":
            for area in node.value.elts:
                area.elts[0] = ast.Constant("")
        return node


def reading(name):
    """the text a stat's quoted name must appear in to count as read in this tool"""
    code = source(name)
    return ast.unparse(ReaderCode().visit(ast.parse(code))) if code else ""


def names_file(code, base):
    return any(q in code for q in ("'%s.jsonl'" % base, '"%s.jsonl"' % base, "'%s*.jsonl'" % base, '"%s*.jsonl"' % base))


def areas_and_kinds():
    """(the stats NextImprovement decides on, the stats loop_stats trends) - read from its source, not imported (no side effects). An
    area's NAME is no stat (10-30: the area `tests` made the stat `tests` DECIDES, the area `assumptions` the stat `assumptions`): only
    each area's third element, its stats, counts"""
    s = source("loop_stats.py")
    a = set()
    for node in ast.walk(ast.parse(s)):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "AREAS":
            for area in node.value.elts:
                a |= {c.value for c in area.elts[2].elts}
    if not a:
        raise ValueError("loop_stats.py: no NextImprovement.AREAS to read: the audit cannot say what decides")
    j = s.find("KINDS = {")
    kinds = s[j:s.find("}", j)] if j >= 0 else ""
    k = set(re.findall(r'"(\w+)"\s*:', kinds)) | set(re.findall(r"'(\w+)'\s*:", kinds))
    # a stat DERIVED from fields (value(): gnn from gnn_mrr, pinned_share from decisions_pinned ...) passes its verdict to them
    v = s.find("def value(")
    body = s[v:s.find("\ndef ", v + 1)] if v >= 0 else ""
    inputs = set(re.findall(r"get\(['\"](\w+)['\"]", body))
    derived = set(re.findall(r"stat\s*==\s*['\"](\w+)['\"]", body))
    if derived & a:
        a |= inputs
    elif derived & k:
        k |= inputs
    return a, k


class PlanReaders(object):
    """Which lines of run_plan's plan a pass acts on, found by reading the pass's tools, never from a list. MEASURED 10-30: all 17 stats of
    gnn_tasks and survivor_rank read DECIDES because run_plan reads them, while no tool of a pass named the plan: 6 of its 7 lines were
    written for a status page, and two of them flipped by seed alone with no effect on any pass. A reader names the plan AND quotes the
    line, in its code proper (not a docstring, a selftest, or where it writes the name). Its breadcrumb: run_plan.PlanLine."""

    def __init__(self, lines=None, tools=PASS_TOOLS):
        if lines is None:
            import run_plan
            lines = run_plan.LINES
        self.lines = lines
        self.code = {t: reading(t) for t in tools}

    def readers(self, name):
        quoted = ('"%s"' % name, "'%s'" % name)
        return sorted(os.path.basename(t) for t, c in self.code.items() if "run_plan" in c and any(q in c for q in quoted))

    def reached(self):
        return [l.name for l in self.lines if self.readers(l.name)]

    def faults(self):
        """a switch nobody reads (its on/off is a claim about a pass that no pass honours), or a line still marked as waiting that a pass
        now reads (it would read None)"""
        return ["%s: a switch no tool of a pass reads" % l.name for l in self.lines if not l.waits and not self.readers(l.name)] + \
               ["%s: read by %s, yet still marked as waiting" % (l.name, ", ".join(self.readers(l.name))) for l in self.lines if l.waits and self.readers(l.name)]

    def decides(self, record, field):
        """True when the field feeds a line a pass acts on; False when it feeds only lines nobody acts on, or names no line"""
        return any(l.rests_on(record, field) and self.readers(l.name) for l in self.lines)


def audit(root, say=print, registry=None, plan=None):
    areas, kinds = areas_and_kinds()
    acting, showing = {t: reading(t) for t in ACTING}, [reading(t) for t in SHOWING]
    plan = plan or PlanReaders()
    # covered by an IMPROVEMENT requirement (improvement/stat_requirements.yaml): a trigger improvement.py evaluates every tick
    covered, banked = improvement_registry(registry)
    out = collections.OrderedDict()

    def judge(record, base, field, morph=False):
        leaf = field.split(".", 1)[-1]
        quoted = ('"%s"' % leaf, "'%s'" % leaf)
        # a reader names the FILE and each field it uses; improvement.py reads the morph log only through its registry (`covered`
        # below): a word in its code is the registry's (10-30: its d.get("requirements") made the stat `requirements` DECIDES)
        acts = "".join(c for t, c in acting.items() if t != "run_plan.py" and names_file(c, base) and not (morph and t == "improvement.py"))
        shows = "".join(c for c in showing if names_file(c, base))
        in_plan = names_file(acting["run_plan.py"], base) and any(q in acting["run_plan.py"] for q in quoted)
        if field.split(".")[0] in IDENTITY:
            v = "IDENTITY"
        elif (morph and field in areas) or any(q in acts for q in quoted) or (in_plan and plan.decides(base, field)):
            v = "DECIDES"
        elif leaf in covered:      # the morph log's stat, or the same stat described in an evidence row (a trend of it reads the log)
            v = "TRIGGERED"
        elif (morph and field in kinds) or in_plan or any(q in shows for q in quoted):
            v = "WATCHED"
        else:
            v = "UNREAD"
        out.setdefault(record, collections.OrderedDict())[field] = v

    for f in sorted(morph_fields(root)):
        judge("morph log (stats.jsonl)", "stats", f, morph=True)
    # the evidence files
    for name, nested in (("gnn_tasks.jsonl", ("shape", "jobs")), ("survivor_rank.jsonl", ("recall20",)), ("loop_ticks.jsonl", ()),
                         ("gnn_curve.jsonl", ()), ("gnn_curve_1029.jsonl", ())):
        fields = json_fields(root, name, nested)
        for f in sorted(fields):
            judge(name, name.split("_1029")[0].replace(".jsonl", ""), f)
    today = __import__("time").strftime("%Y-%m-%d")
    for b in banked:
        if glob.glob(os.path.join(root, "**", b["record"]), recursive=True):
            past = str(b.get("until", "")) < today
            out.setdefault(b["record"], collections.OrderedDict())["(every row)"] = (
                "UNREAD (banked for %s until %s: past its date)" % (b["for"], b["until"]) if past else "BANKED for %s until %s" % (b["for"], b["until"]))
    unread = 0
    for record, fields in out.items():
        by = collections.defaultdict(list)
        for f, v in fields.items():
            by[v.split(" for ")[0].split(" (")[0]].append(f)
        unread += len(by.get("UNREAD", []))
        say("%s: %s" % (record, "; ".join("%s %d" % (k, len(v)) for k, v in sorted(by.items()))))
        for k in ("UNREAD", "WATCHED"):
            if by.get(k):
                say("  %-8s %s" % (k, ", ".join(sorted(by[k]))))
    reached = plan.reached()
    say("the plan: %d of %d lines reach a pass%s; %d measured only%s" % (
        len(reached), len(plan.lines), " (%s)" % ", ".join(reached) if reached else "",
        sum(1 for l in plan.lines if l.waits), " (%s)" % ", ".join(l.name for l in plan.lines if l.waits) if any(l.waits for l in plan.lines) else ""))
    faults = plan.faults()
    for f in faults:
        say("  UNREAD   the plan's line %s" % f)
    unread += len(faults)
    say("telemetry_audit: %d stat(s) UNREAD%s" % (unread, " - each needs a reader that acts on it, or the loop stops recording it" if unread else ""))
    return out, unread


def plan_lines_checked(t):
    """a plan's line is reached only by a tool of the pass that names the plan and quotes the line in its code; a switch nobody reads fails
    the audit, a stat that only feeds a line nobody reads is WATCHED, and one that feeds a line a pass reads DECIDES"""
    import run_plan
    d = os.path.join(t, "plan_probe")
    os.makedirs(d)
    line = run_plan.PlanLine("plant switch", {"gnn_tasks": ("jobs.planted",)})
    wait = run_plan.PlanLine("plant waits", {"gnn_tasks": ("jobs.waited",)}, waits="a reader")

    def tool(name, code):
        p = os.path.join(d, name)
        io.open(p, "w", encoding="utf-8").write(code)
        return p
    no_plan = tool("a.py", "def f(r):\n    return r['plant switch']\n")                                   # quotes the line, never names the plan
    fixture = tool("b.py", "import run_plan\ndef selftest():\n    return {}['plant switch']\n")             # only in a selftest
    written = tool("c.py", "import run_plan\ndef f(p):\n    p['plant switch'] = 1\n")                      # only writes it
    doc = tool("d.py", "import run_plan\ndef f():\n    \"\"\"reads 'plant switch'\"\"\"\n    return 1\n")    # only says so
    real = tool("e.py", "import run_plan\ndef f(k):\n    return run_plan.decide(k)['plant switch']['on']\n")

    def audited(tools, lines):
        plan = PlanReaders(lines, tools)
        rows = os.path.join(t, "plan_root")
        os.makedirs(rows, exist_ok=True)
        io.open(os.path.join(rows, "gnn_tasks.jsonl"), "w", encoding="utf-8").write(json.dumps({"jobs": {"planted": {}, "waited": {}}}) + "\n")
        return plan, audit(rows, say=lambda s: None, registry=os.path.join(t, "reg.yaml"), plan=plan)[1]
    nobody, unread_nobody = audited((no_plan, fixture, written, doc), (line,))
    reader, unread_real = audited((real,), (line,))
    stale_wait = PlanReaders((run_plan.PlanLine("plant switch", {}, waits="a reader"),), (real,)).faults()
    good = nobody.reached() == [] and len(nobody.faults()) == 1 and "no tool of a pass reads" in nobody.faults()[0] and unread_nobody == unread_real + 1
    good = good and reader.reached() == ["plant switch"] and not reader.faults() and reader.decides("gnn_tasks", "jobs.planted") \
        and not reader.decides("gnn_tasks", "jobs.waited") and not nobody.decides("gnn_tasks", "jobs.planted")
    good = good and len(stale_wait) == 1 and "still marked as waiting" in stale_wait[0]
    waiting = PlanReaders((wait,), (real,))
    good = good and not waiting.faults() and waiting.reached() == [] and not waiting.decides("gnn_tasks", "jobs.waited")
    # the verdict itself: the same real job, read by run_plan either way, DECIDES only when a tool of the pass reads the line it feeds
    owner = [l for l in run_plan.LINES if l.name == "gnn owner"][0]
    reads_owner = tool("f.py", "import run_plan\ndef f(k):\n    return run_plan.decide(k)['gnn owner']['on']\n")
    rows = os.path.join(t, "plan_root")
    io.open(os.path.join(rows, "gnn_tasks.jsonl"), "w", encoding="utf-8").write(json.dumps({"jobs": {"owner as today": {}}}) + "\n")

    def verdict(tools):
        return audit(rows, say=lambda s: None, registry=os.path.join(t, "reg.yaml"), plan=PlanReaders((owner,), tools))[0]["gnn_tasks.jsonl"]["jobs.owner as today"]
    good = good and verdict((reads_owner,)) == "DECIDES" and verdict((no_plan,)) == "WATCHED"
    # on the real tools: a pass reads every switch, and no line still marked as waiting
    actual = PlanReaders()
    lines = {l.name: l for l in run_plan.LINES}
    good = good and all((actual.readers(n) != []) == (not l.waits) for n, l in lines.items())
    print("selftest: PLAN a line is reached only by a tool of the pass naming the plan and quoting the line in its code; a switch nobody reads fails the "
          "audit, a line still waiting once read fails it too, and a stat counts as deciding only through a line a pass reads %s" % ("PASS" if good else "FAIL"))
    return good


def selftest():
    """a stat a decision reads is DECIDES, one only trended is WATCHED, one nothing reads is UNREAD and fails the audit, a row's
    identity is never a gap, and an evidence file nothing reads is UNREAD field by field"""
    import shutil
    import tempfile
    t = tempfile.mkdtemp(prefix="telemetry_audit_selftest_")
    io.open(os.path.join(t, "stats.jsonl"), "w", encoding="utf-8").write(json.dumps(
        {"time": "x", "holes_ours": 1, "graph_nodes": 5, "never_read_anywhere": 3,
         # three ways the first audit took a name for a reader (MEASURED 10-30: it said 6 UNREAD while 11 were): `effects` is named
         # only in a selftest's fixture, `wins` is read by run_plan from ANOTHER record, `coverage` is an area's NAME, not a stat
         "effects": 1, "wins": 1, "coverage": 1, "requirements": 1,
         # what a pass records about the public pages against their records (claims.py): read by NextImprovement's `documents` area
         "doc_claims_stale": 0, "doc_numbers_unsourced": 488}) + "\n")
    io.open(os.path.join(t, "nobody.jsonl"), "w", encoding="utf-8").write("{}\n")
    io.open(os.path.join(t, "gnn_curve.jsonl"), "w", encoding="utf-8").write(json.dumps({"gnn": 0.5, "graph": "g"}) + "\n")
    io.open(os.path.join(t, "gnn_tasks.jsonl"), "w", encoding="utf-8").write(json.dumps({"graph": "g", "shape": {
        "requirements": 1, "covered_by_a_trigger": 1, "never_in_a_reader": 1}, "jobs": {"owner as today": {}, "only_a_plant_job": {}}}) + "\n")
    reg = os.path.join(t, "reg.yaml")
    io.open(reg, "w", encoding="utf-8").write("requirements:\n  - {id: IMP-T, stat: covered_by_a_trigger, trigger: {above: 0}, action: {area: tests}}\n"
                                              "banked:\n  - {record: old.jsonl, for: GN-X, until: '2000-01-01'}\n")
    io.open(os.path.join(t, "stats.jsonl"), "a", encoding="utf-8").write(json.dumps({"covered_by_a_trigger": 1}) + "\n")
    io.open(os.path.join(t, "old.jsonl"), "w", encoding="utf-8").write("{}\n")
    out, unread = audit(t, say=lambda s: None, registry=reg)
    probe = ("def f():\n    \"\"\"reads gnn_curve.jsonl\"\"\"\n    return 1\n"
             "def write(row):\n    row[\"written_only\"] = 1\n    return {\"dict_key_only\": 2}\n"
             "def read(r):\n    return r.get(\"really_read\"), r[\"read_too\"]\n"
             "def selftest():\n    return \"plant_only\"\n")
    pf = os.path.join(t, "probe.py")
    io.open(pf, "w", encoding="utf-8").write(probe)
    doc_is_no_reader = "gnn_curve.jsonl" not in source(pf)
    rd = reading(pf)
    names_only = "plant_only" not in rd and "written_only" not in rd and "dict_key_only" not in rd and "really_read" in rd and "read_too" in rd
    stats_of_areas, _ = areas_and_kinds()
    areas_right = "holes_ours" in stats_of_areas and "tests" not in stats_of_areas and {"doc_claims_stale", "doc_numbers_unsourced"} <= stats_of_areas
    m = out["morph log (stats.jsonl)"]
    good = m["holes_ours"] == "DECIDES" and m["graph_nodes"] == "WATCHED" and m["never_read_anywhere"] == "UNREAD" and m["time"] == "IDENTITY"
    good = good and unread >= 1 and doc_is_no_reader and m.get("covered_by_a_trigger") == "TRIGGERED" \
        and "past its date" in out.get("old.jsonl", {}).get("(every row)", "")
    cur = out.get("gnn_curve.jsonl", {})
    good = good and (cur.get("gnn") in ("UNREAD", "DECIDES")) and cur.get("graph") != "UNREAD"
    ev = out.get("gnn_tasks.jsonl", {})
    # none of these may pass for read: a plant's fixture, another record's reader, an area's name
    false_readers = {f: m.get(f) for f in ("effects", "wins", "coverage") if m.get(f) != "UNREAD"}
    good2 = not false_readers and names_only and areas_right and m.get("doc_claims_stale") == "DECIDES" and m.get("doc_numbers_unsourced") == "DECIDES" and m.get("requirements") == "WATCHED" and ev.get("shape.requirements") == "DECIDES" \
        and ev.get("shape.covered_by_a_trigger") == "TRIGGERED" and ev.get("shape.never_in_a_reader") == "UNREAD" \
        and ev.get("jobs.owner as today") == "DECIDES" and ev.get("jobs.only_a_plant_job") == "UNREAD"
    good3 = plan_lines_checked(t)
    good = good and good3
    print("selftest: READER a name in a selftest, a name where the code writes it, a name another record's reader reads and an area's name are "
          "no readers; a stat trended under its own name is covered in an evidence file too %s%s" % (
              "PASS" if good2 else "FAIL", "" if good2 else " %s" % ([false_readers, names_only, areas_right, ev],)))
    good = good and good2
    shutil.rmtree(t, ignore_errors=True)
    print("selftest: AUDIT a stat a decision reads DECIDES, a trended one is WATCHED, one an improvement requirement's trigger covers is "
          "TRIGGERED, an unread one fails, a banked record past its date fails, identity is never a gap %s" % ("PASS" if good else "FAIL"))
    print("selftest: %s" % ("PASS" if good else "FAIL"))
    return 0 if good else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(HERE, ".."))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    _, unread = audit(os.path.abspath(a.root))
    return 1 if unread else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
