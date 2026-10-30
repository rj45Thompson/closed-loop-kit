#!/usr/bin/env python3
"""closed_loop.py - the closed loop as ONE command: requirements as data, held to the code, to the tests, and to each change's written
argument (a chain of claims, each with its evidence).

WHY (2026-10-29): the closed loop comes packaged with a check of the argument chains an agent writes, to confirm the intention on each branch.
(A chain is a FILE the agent writes on purpose, claims with their evidence. It is never the model's private thinking, and the tool does not try to read that.) The pieces existed apart - the rules (usecase_model.py), the knowledge graph and its 14 criteria (usecase_kg.py), a code
adapter (cs_effect_check.py for C#, the Blueprint view's logic_map.py for Blueprints) and a chain check (the Blueprint view's
chain_check.py, Blueprint traces only). This runs them as one, over a C# model, and adds what none did:

  BRANCHES   every branch of the code the model's tests cover (each if / else if / while / for / foreach / switch case) is a cell:
             claimed by the requirement whose effect matched it, or excluded with its reason (meta.loop.branches.excluded). A branch
             no requirement claims is behaviour nobody asked for - written as a generated space (<model>/generated_branches.yaml),
             so the EXISTING rule U7 and the loop's L9 fail on it. This is "intention on each branch".
  CHAINS     an argument chain (YAML, below), held to the model and the code: every step must cite what it stands on (a
             requirement id, a question id, a Type.Method place) and every cite must exist - a step resting on nothing is a
             hallucinated ground and FAILS; a step whose words the cited requirement does not share is listed (a new finding or a
             mistake: a person decides); every owner question no step reached is a BLIND SPOT the chain could not see.

A chain: chain: {id, by, when, question, steps: [{says, cites: [REQ-ID | Q-ID | Type.Method ...]}]}

    python -X utf8 closed_loop.py run <model> [--chain C.yaml ...] [--no-kill] [--out REPORT.md]
    python -X utf8 closed_loop.py --selftest

Exit 0 = the loop holds on every hole of ours and every chain is grounded; 1 = a hole of ours or an ungrounded chain step; 2 = could
not run. The owner's questions never fail it - they are counted and printed: only the owner closes those.
"""
import argparse
import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
# the project the model's paths are relative to: this repo, or a developer's own (CLOSED_LOOP_ROOT, set by the kit's installer)
ROOT = os.path.abspath(os.environ.get("CLOSED_LOOP_ROOT") or os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import usecase_model as ucm  # noqa: E402
import cs_effect_check as cs  # noqa: E402
import usecase_kg as kgm  # noqa: E402

STOP = set("a an the is it its of to in on at and or not what when who how why does do did which where this that with be are was for "
           "from by as if then else so no one every each any can must here only than more".split())


def words(s):
    out = set()
    for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", s or ""):
        for part in [w] + re.split(r"_|(?<=[a-z0-9])(?=[A-Z])", w):
            p = part.lower()
            if len(p) > 2 and p not in STOP:
                out.add(p)
    return out


# ── BRANCHES: every branch of the covered code claimed or excluded ───────────────────────────────

def read_claims_lock(model):
    p = os.path.join(model.root, "claims.lock")
    if not os.path.exists(p):
        return None
    out = {}
    for line in io.open(p, encoding="utf-8"):
        # TAB-separated (MEASURED 10-29: an equivalent's key holds spaces - "Store.Buy@6cda91: boundary > to >= (1)" - and a space
        # split read it back as "Store.Buy@6cda91:", so its pin was never found and it lapsed); a space line is an older lock's
        sep = "\t" if "\t" in line else " "
        parts = line.rstrip("\n").split(sep, 2)
        if len(parts) >= 2 and not line.startswith("#"):
            out[parts[0]] = (parts[1], parts[2] if len(parts) > 2 else "")
    return out


def write_claims_lock(model, fps, why):
    """the lock rewritten with these entries (method claims, or `equiv:<mutant>` for an equivalent's rests_on), every other kept"""
    keep = {k: v for k, v in (read_claims_lock(model) or {}).items() if k not in fps}
    lines = ["# closed_loop.py: each method claim's fingerprint of its method's statements, and each equivalent's of the methods it",
             "# rests on, when they were pinned, and why. Never edit: --pin-claims \"<why>\" re-pins after a person read them."]
    rows = {k: (v[0], v[1]) for k, v in keep.items()}
    rows.update({m: (fp, why) for m, fp in fps.items()})
    lines += ["%s\t%s\t%s" % (m, fp, w) for m, (fp, w) in sorted(rows.items())]
    io.open(os.path.join(model.root, "claims.lock"), "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")


def branch_space(model, verdicts, code, pin=None):
    """the generated space S-BRANCH: one cell per branch statement in every covered file (cs_effect_check's own walk, so the keys
    are the mutants' keys), mapped to the requirements whose effects matched it or whose method claims it; excluded ones carry
    the model's reason"""
    covered = []
    for t in ucm.as_list((model.loop.get("tests") or [])):
        for c in ucm.as_list(t.get("covers")):
            covered.append(cs.src_path(c))
    excluded = {e["ref"]: e["why"] for e in ucm.as_list((model.loop.get("branches") or {}).get("excluded")) if isinstance(e, dict)}
    # a METHOD-level claim (meta.loop.branches.claims: [{method: Type.Method, reqs: [...], why}]): the requirement owns every branch
    # of that method - coarser than an effect's statement, so the report counts the branches each claim covers
    by_method = {c["method"]: (list(ucm.as_list(c.get("reqs"))), c.get("why")) for c in ucm.as_list((model.loop.get("branches") or {}).get("claims"))
                 if isinstance(c, dict) and c.get("method")}
    claims = {}
    for r in model.requirements.values():
        for a in r.effect:
            v = verdicts["assertions"].get(a.id) or {}
            for m in v.get("matched") or []:
                claims.setdefault(re.sub(r"\s+", " ", m)[:120], set()).add(r.id)
    files = [f for f in code.files if any(f == c or f.startswith(c + os.sep) for c in covered)]
    decisions = code.branches(files)
    # A METHOD CLAIM IS PINNED to its method's decisions (REVIEW_1 #2, #3, 10-29): unpinned, it claimed any branch added to the
    # method later - an intruder `if (price > 500) return NotOnSale` was claimed on sight - and deleting a claimed branch made the
    # score better. claims.lock holds each claim's fingerprint of its method's decisions; a method whose decisions moved has a
    # STALE claim, its branches are claimed by nobody (L9 fails) until someone reads them and pins again (--pin-claims "<why>").
    fps = {}
    for key, method, st in decisions:
        fps.setdefault(method, []).append(re.sub(r"\s+", " ", st.text))
    fps = {m: cs.sha("\n".join(ts)) for m, ts in fps.items()}
    lock = read_claims_lock(model)
    # REVIEW_2 #6 (10-29): a missing lock no longer pins itself - a deleted claims.lock had re-pinned every claim, silently.
    # Without one, every claim is unpinned (its branches claimed by nobody) until a person runs --pin-claims "<why>".
    if pin is not None:
        write_claims_lock(model, {m: fps.get(m, "absent") for m in by_method}, pin)
    lock = read_claims_lock(model) or {}
    cells = {}
    for key, method, st in decisions:
        text = re.sub(r"\s+", " ", st.text)[:120]
        if key in excluded:
            cells[key] = {"excluded": excluded[key]}
        elif text in claims:
            cells[key] = {"text": text, "maps": sorted(claims[text])}
        elif method in by_method and lock.get(method, ("",))[0] == fps.get(method):
            cells[key] = {"text": text + "  [claimed with its method: %s]" % by_method[method][1], "maps": sorted(by_method[method][0])}
        elif method in by_method:
            cells[key] = {"text": text + "  [its method's claim is STALE: %s's decisions changed since claims.lock pinned it - read them, then "
                                         "--pin-claims]" % method, "maps": []}
        else:
            cells[key] = {"text": text, "maps": []}
    return {"spaces": [{"id": "S-BRANCH", "title": "every branch of the code the tests cover (generated by closed_loop.py - do not edit)",
                        "basis": "code", "kind": "list", "cells": cells}]}


# ── CHAINS: a change's written argument held to the model, the code and the graph ──────────────────────────

def check_chain(model, code, chain, kg=None):
    """(rows, fails, reached, blind): each step's cites resolved; the owner questions no step reached. With the graph, each pair a
    step cites is measured apart in it: a step that joins what the graph does not join (no path in 4 edges) is listed - a new link
    the model lacks, or a mistake; a person decides"""
    rows, fails, reached = [], 0, set()
    for i, st in enumerate(ucm.as_list(chain.get("steps")), 1):
        cites, bad, ground = ucm.as_list(st.get("cites")), [], set()
        for c in cites:
            c = str(c)
            if c in model.requirements:
                reached.add(c)
                ground |= words(model.requirements[c].text)
            elif any(q.id == c for q in model.questions):
                ground |= words(next(q.text for q in model.questions if q.id == c))
            elif "." in c:
                try:
                    code.method(c)
                    ground |= words(c)
                except KeyError:
                    bad.append(c)
            else:
                bad.append(c)
        if not cites or bad:
            fails += 1
            verdict = "FAIL - " + ("cites nothing: a claim with no ground" if not cites else "cites what is not there: %s" % ", ".join(bad))
        else:
            shared = words(st.get("says")) & ground
            verdict = "grounded" + ("" if shared else " - its words are not the cited requirement's: a new finding, or a mistake (a person decides)")
            if kg is not None and len(cites) > 1:
                node = lambda c: "req:%s" % c if c in model.requirements else "q:%s" % c if any(q.id == c for q in model.questions) else "code:%s" % c
                apart = [(a, b, kg.distance(node(str(a)), node(str(b)))) for j, a in enumerate(cites) for b in cites[j + 1:]]
                loose = ["%s ~ %s" % (a, b) for a, b, d in apart if d is None]
                verdict += ("; the graph joins every pair it cites (%s)" % ", ".join("%s-%s %d" % (a, b, d) for a, b, d in apart)
                            if not loose else "; NOT joined in the graph: %s - a link the model lacks, or a mistake" % ", ".join(loose))
        rows.append((i, st.get("says") or "", cites, verdict))
    owner = sorted(r.id for r in model.requirements.values() if r.kind == "inferred")
    return rows, fails, reached, [r for r in owner if r not in reached]


# ── run ─────────────────────────────────────────────────────────────────────────────────────────

def py(*args):
    r = subprocess.run([sys.executable, "-X", "utf8"] + list(args), cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, r.stdout + r.stderr


def test_checks(tests):
    """({requirement id: [Fixture.Test]}, [tests naming none]) from the `/// CHECKS: W-1, W-2` line written above each [Test].
    REVIEW_2 #3 (10-29): the mapping from a test to what it checks belongs WITH the test, written when the test is - a table in the
    model, filled in after the mutants showed which tests kill what, can be fitted to pass L16"""
    out, orphans = {}, []
    sources = []
    for t in tests:
        p = cs.src_path(t["test"].split()[0])
        if p.endswith(".csproj"):           # a dotnet test project: every .cs it compiles (bin and obj are build output)
            sources += [os.path.join(d, f) for d, ds, fs in os.walk(os.path.dirname(p)) if not {"bin", "obj"} & set(d.replace("\\", "/").split("/"))
                        for f in fs if f.endswith(".cs")]
        else:
            sources.append(p)
    for path in sources:
        src = io.open(path, encoding="utf-8-sig").read()
        if path.endswith(".py"):
            py_checks(src, out, orphans)
            continue
        fixture = re.search(r"\bclass\s+(\w+)", cs.blank_out(src))
        fixture = fixture.group(1) if fixture else "Tests"
        for m in re.finditer(r"((?:[ \t]*///[^\n]*\n)*)[ \t]*\[(?:Test|TestCase|TestMethod|DataTestMethod|Fact|Theory)\b[^\n]*\n(?:[ \t]*\[[^\n]*\n)*[ \t]*public\s+(?:async\s+)?(?:void|Task|ValueTask)\s+(\w+)\s*\(", src):
            ids = [x.strip() for line in re.findall(r"CHECKS:\s*([^\n]+)", m.group(1)) for x in line.split(",") if x.strip()]
            if not ids:
                orphans.append("%s.%s" % (fixture, m.group(2)))
            for rid in ids:
                out.setdefault(rid, []).append("%s.%s" % (fixture, m.group(2)))
    return out, orphans


def py_checks(src, out, orphans):
    """a Python test's checks: a `# CHECKS: W-1, W-2` comment names the requirements of the checks that follow it (until the next
    such comment); each check is named as the runner reports it - the literal of `check("...")` or of `print("selftest: ...")`.
    Python check lines were never read until P5-C and P5-D (10-29): L16 listed every Python requirement as naming no test"""
    ids = []
    for line in src.splitlines():
        m = re.match(r"\s*#\s*(?:///\s*)?CHECKS:\s*(.+)", line)
        if m:
            ids = [x.strip().split(" ")[0] for x in m.group(1).split(" - ")[0].split(",") if x.strip()]
            continue
        n = re.search(r"\bcheck\(\s*([\"'])(.+?)\1", line) or re.search(r"print\(\s*([\"'])selftest: (.+?)(?: %s|\1)", line)
        if not n:
            continue
        name = re.sub(r"\s+", " ", n.group(2)).strip()[:90]
        if not ids:
            orphans.append(name)
        for rid in ids:
            out.setdefault(rid, []).append(name)


def engine_sha():
    """the adapter's own source: a change to how mutants are made or judged re-runs them, as a change to the code does"""
    return cs.sha("".join(io.open(os.path.join(HERE, f), encoding="utf-8").read() for f in ("cs_effect_check.py", "py_effect_check.py")))


def hole_keys(gate):
    """{"ours": [...], "owner": [...]}: every hole the gate's last run found, as "<criterion> <entity>" - stable across runs (an
    entity id, never a message that carries counts)"""
    out = {"ours": set(), "owner": set()}
    for cid, gs in (getattr(gate, "last_holes", None) or {}).items():
        for g in gs or []:
            if g.owner:
                out["owner"].add("%s %s" % (cid, g.entity))
            elif g.failing:
                out["ours"].add("%s %s" % (cid, g.entity))
    return {k: sorted(v) for k, v in out.items()}


def hole_delta(before, now):
    """(added, closed, standing): the holes of ours THIS change made, the ones it closed, and those that were already there
    (AB-4, 10-29: the loop-on arm of R2 started in a model with 337 holes of ours it had not caused, had to script this split
    itself, and still drifted into fixing old ones - scope creep the blind grader marked)"""
    b, n = set(before), set(now)
    return sorted(n - b), sorted(b - n), sorted(n & b)


def self_improve_on(model_dir, flag=True):
    """Self-improvement (each pass's row in the morph log and the stats' picks of what to improve next) is ON by default. Off: the run's
    `--no-self-improve`, or `self_improve: false` in the model's loop block. Measured 10-30 on the shop feature: the row and the picks
    cost about a third of a second and three printed lines, so the larger saving is stopping the standing tick (loop_tick.py)."""
    if not flag:
        return False
    path = os.path.join(model_dir, "meta.yaml")
    if not os.path.exists(path):
        return True
    meta = yaml.safe_load(io.open(path, encoding="utf-8")) or {}
    return ((meta.get("meta") or {}).get("loop") or {}).get("self_improve", True) is not False


def run(model_dir, chains, kill, out, reuse=True, workers=2, pin=None, gnn=False, note=None, accept=None, since=None, self_improve=True):
    lines, worst = [], 0
    started = time.time()
    model = ucm.UseCaseModel.load(model_dir)
    if not model.closed_loop:
        print("closed_loop: %s is not a closed-loop model (meta.loop.closed)" % model_dir, file=sys.stderr)
        return 2
    # the code adapter by the model's language (meta.loop.code.language): C# (cs_effect_check, the default) or Python
    # (py_effect_check - the loop run on the loop: the tool runs on itself and improves from what it finds)
    language = str((model.loop.get("code") or {}).get("language") or "csharp").lower()
    if language == "python":
        import py_effect_check as pe
        code = pe.PyCode(ucm.as_list((model.loop.get("code") or {}).get("roots")), ucm.as_list((model.loop.get("code") or {}).get("scope")))
    else:
        code = cs.CsCode(ucm.as_list((model.loop.get("code") or {}).get("roots")))
    tests = [t for t in ucm.as_list(model.loop.get("tests")) if isinstance(t, dict)]
    kgdir = os.path.join(model_dir, "..", "kg")
    name = os.path.basename(os.path.dirname(os.path.abspath(model_dir)))
    # the verdicts live beside the model from its FIRST run (MEASURED 10-29 on the kit's example: with no kg/ yet they went to a temp
    # folder, so the morph log had nothing to read and the next run could not reuse them)
    os.makedirs(kgdir, exist_ok=True)
    vpath = os.path.join(kgdir, "verdicts.json")
    code_sha = cs.sha("".join(sorted(code.files.values())))
    effects = cs.sha(json.dumps(sorted(a.text() for r in model.requirements.values() for a in r.effect)))
    # an equivalence that RESTS ON other methods (REVIEW_2 #2: Press's `MoveCursor(2)` is MoveCursor(1) only while MoveCursor takes
    # Math.Sign) is pinned to their statements in claims.lock and LAPSES when they change - a survivor again, until re-pinned
    eq, lapsed, pins = {}, {}, {}
    lock0 = read_claims_lock(model) or {}
    for e in ucm.as_list(model.loop.get("equivalent")):
        if not (isinstance(e, dict) and e.get("mutant")):
            continue
        rests = [str(x) for x in ucm.as_list(e.get("rests_on"))]
        if rests:
            try:
                fp = cs.sha("\n".join(re.sub(r"\s+", " ", st.text) for m in rests for st in code.method(m)))
            except KeyError as err:
                lapsed[e["mutant"]] = "rests on %s, which names nothing now (%s)" % (", ".join(rests), err)
                continue
            pins["equiv:" + e["mutant"]] = fp
            if pin is None and lock0.get("equiv:" + e["mutant"], ("",))[0] != fp:
                lapsed[e["mutant"]] = "rests on %s, changed since it was pinned (or never pinned): read it again, then --pin-claims" % ", ".join(rests)
                continue
        eq[e["mutant"]] = e.get("why")
    if pin is not None and pins:
        write_claims_lock(model, pins, pin)
    # the tests are in the stamp too: a new test kills what the old ones let survive, so reusing their verdicts would hide it
    test_text = "".join(io.open(cs.src_path(t["test"].split()[0]), encoding="utf-8-sig").read() for t in tests)   # a command: its script
    # and what they READ (REVIEW_2 #5): a new row in a sheet a test reads changes what it kills, so it re-runs the mutants
    for t in tests:
        for r in ucm.as_list(t.get("reads")):
            p = cs.src_path(r)
            files = [p] if os.path.isfile(p) else sorted(os.path.join(d, f) for d, _, fs in os.walk(p) for f in fs if not f.endswith(".meta"))
            if not files:
                raise RuntimeError("meta.loop.tests reads %s, which holds no file" % r)
            test_text += "".join(io.open(f, "rb").read().decode("utf-8", "replace") for f in files)
    stamp = {"code": code_sha, "effects": effects, "engine": engine_sha(), "equivalent": cs.sha(json.dumps(sorted(eq.items()))),
             "tests_sha": cs.sha(test_text)}
    old = json.load(io.open(vpath, encoding="utf-8")) if os.path.exists(vpath) else {}
    if reuse and kill and all(old.get(k) == v for k, v in stamp.items()) and "mutants" in old:
        verdicts = old          # the same code, effects, engine and equivalents: every mutant's verdict stands
        lines.append("_(verdicts reused: the code, the effects, the engine and the equivalents are unchanged since %s was written)_" % os.path.basename(vpath))
    else:
        if language == "python":
            verdicts = pe.PyEffectCheck(code, pe.PyRunner(tests) if tests else None, eq, max(workers, 3)).run(model_dir, vpath, kill=kill)
        else:
            verdicts = cs.CsEffectCheck(code, cs.Runner(tests) if tests else None, eq, workers).run(model_dir, vpath, kill=kill)
        verdicts.update(stamp)
        io.open(vpath, "w", encoding="utf-8").write(json.dumps(verdicts, indent=1, sort_keys=True))
    verdicts["equivalent_lapsed"] = lapsed
    va, ms, bs = verdicts["assertions"], verdicts.get("mutants") or {}, verdicts.get("branches") or {}
    killed = sum(1 for v in va.values() if (v.get("kill") or {}).get("killed") is True)
    survived = [k for k, v in va.items() if (v.get("kill") or {}).get("killed") is False]
    lines.append("## 2. The code (cs_effect_check.py): %d effects, %d hold; behaviour: %d killed outright, %d with a survivor, %d not killable"
                 % (len(va), sum(1 for v in va.values() if v.get("holds")), killed, len(survived),
                    sum(1 for v in va.values() if v.get("holds") and (v.get("kill") or {}).get("killed") is None)))
    lines.append("Every mutant of every effect's statement and of every covered branch: %d made, %d killed, %d SURVIVED, %d no verdict "
                 "(a file no test covers, or a copy that did not compile); %d tests ran."
                 % (len(ms), sum(1 for m in ms.values() if m.get("killed") is True), sum(1 for m in ms.values() if m.get("killed") is False),
                    sum(1 for m in ms.values() if m.get("killed") is None), len(verdicts.get("tests") or [])))
    if verdicts.get("not_walked"):
        lines.append("NOT WALKED, so no break was made in them: %s" % ", ".join(verdicts["not_walked"]))
    for k in survived:
        lines.append("- SURVIVED %s: %s" % (k, va[k]["kill"].get("why")))
    gen = branch_space(model, verdicts, code, pin)
    gpath = os.path.join(model_dir, "generated_branches.yaml")
    io.open(gpath, "w", encoding="utf-8", newline="\n").write(
        "# GENERATED by closed_loop.py - every branch of the covered code, claimed or excluded. Never edit.\n"
        + yaml.safe_dump(gen, sort_keys=False, allow_unicode=True, width=200))
    cells = gen["spaces"][0]["cells"]
    open_b = [k for k, c in cells.items() if isinstance(c, dict) and "maps" in c and not c["maps"]]
    pinned = sum(1 for b in bs.values() if b.get("killed") is True)
    lines.append("## 3. Branches: %d in the covered code, %d claimed by a requirement, %d excluded with a reason, %d claimed by NONE; "
                 "%d PINNED (every mutant killed), %d with a survivor"
                 % (len(cells), sum(1 for c in cells.values() if c.get("maps")), sum(1 for c in cells.values() if "excluded" in c), len(open_b),
                    pinned, sum(1 for b in bs.values() if b.get("killed") is False)))
    for k in open_b:
        lines.append("- %s: %s" % (k, cells[k]["text"]))
    # ONE QUESTION PER DECISION (prior art: dominator mutants, Ammann et al. 2014): the survivors grouped by the statement they
    # broke, with who claims it - so a boundary broken three ways is one missing test, not three
    groups = {}
    for key, m in sorted(ms.items()):
        if m.get("killed") is False and key not in (verdicts.get("equivalent_declared") or {}):
            groups.setdefault((m["place"], m["line"], m["statement"]), []).append(m["what"])
    if groups:
        owner = {}
        for k, c in cells.items():
            if c.get("maps"):
                owner.setdefault(k.split("#b")[0], set()).update(c["maps"])
        lines.append("Survivors, one question per decision (%d decision(s), %d mutant(s)):" % (len(groups), sum(len(v) for v in groups.values())))
        for (place, line, stmt), whats in sorted(groups.items()):
            lines.append("- %s line %d `%s` - %s; claimed by %s" % (place, line, stmt[:80], "; ".join(whats),
                                                                    ", ".join(sorted(owner.get(place, []))) or "nobody"))
    # the model is checked AFTER the branch space is written: that space is one of its cell spaces (a check before it read the
    # last run's space - MEASURED 10-29: "HOLES" reported for a model that passed)
    rc, said = py(os.path.join(HERE, "usecase_model.py"), model_dir)
    lines[0:0] = ["## 1. The model (usecase_model.py, 19 rules, with the generated branch space): %s" % ("PASS" if rc == 0 else "HOLES"),
                  "```\n" + "\n".join(said.strip().splitlines()[-6:]) + "\n```"]
    worst = max(worst, 1 if rc == 1 else rc)
    # THE GRAPH is the record: built where the model keeps it, with every
    # verdict, mutant, test and chain as nodes, exported as N-Triples - the loop's criteria and the chains' walks read it
    model = ucm.UseCaseModel.load(model_dir)         # again: the generated branch space is one of its spaces
    declared, orphans = test_checks(tests)
    unknown = sorted(rid for rid in declared if rid not in model.requirements)
    if unknown:
        lines.append("- TESTS NAME REQUIREMENTS THE MODEL DOES NOT HOLD: %s" % ", ".join(unknown))
        worst = max(worst, 1)
    for rid, names in declared.items():
        if rid in model.requirements:
            r = model.requirements[rid]
            r.checks = sorted(set(r.checks) | set(names))
    lines.append("Tests: %d name what they check beside themselves (/// CHECKS:), %d name nothing%s" % (
        len({n for v in declared.values() for n in v}), len(orphans),
        "" if not orphans else " - " + ", ".join(o.split(".")[-1] for o in orphans[:6]) + (" ..." if len(orphans) > 6 else "")))
    os.makedirs(kgdir, exist_ok=True)
    dbp = os.path.join(kgdir, "%s.sqlite" % name)
    kg = kgm.KnowledgeGraph.build(model, dbp, verdicts)
    loaded = [(c, (yaml.safe_load(io.open(c, encoding="utf-8")) or {}).get("chain") or {}) for c in chains]
    for _, ch in loaded:
        kg.add_chain(model, ch)
    ntriples = kg.export_ntriples(os.path.join(kgdir, "%s.nt" % name))
    gate = kgm.LoopGate(kg, model, verdicts)
    if accept:
        # the lock taken HERE, on the graph this run built (each test's /// CHECKS: merged in) - a lock taken by usecase_kg.py on
        # its own graph would fingerprint the requirements without their checks and read STALE on the next run
        blocking = gate.accept(accept)
        lines.append("Lock: %s" % ("REFUSED - %d hole(s) of ours stand (%s)" % (len(blocking), "; ".join(str(g) for g in blocking[:3]))
                                   if blocking else "taken (%s)" % cs.shown(gate.lock_path)))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = gate.run(True, 8)
    said = buf.getvalue()
    c = kg.counts()
    lines.append("## 4. The closed loop (usecase_kg.py, %d criteria), on the graph: %d nodes, %d edges, %d literals -> %s (%d triples)"
                 % (len(kgm.LOOP), c["entities"], c["edges"], c["literals"], cs.shown(dbp), ntriples))
    lines.append("```\n" + "\n".join(l for l in said.splitlines() if l.strip()) + "\n```")
    # THE MODAL READING (prior art: Modal Transition Systems, Larsen & Thomsen 1988): an authored or derived requirement is a MUST,
    # an inferred one a MAY only its owner settles; the requirements are closed when no may is left - relative to what the
    # mutants could refute (mutation disproves closure, it never proves it)
    must = sorted(r.id for r in model.requirements.values() if r.kind != "inferred")
    may = sorted(r.id for r in model.requirements.values() if r.kind == "inferred")
    lines.append("Modal status: %d MUST (authored or derived), %d MAY (inferred: each an owner's question) -> %s" % (
        len(must), len(may), "closed: no may is left" if not may else "NOT closed until the owner settles: " + ", ".join(may)))
    worst = max(worst, 1 if rc == 1 else rc)
    for cpath, chain in loaded:
        rows, fails, reached, blind = check_chain(model, code, chain, kg)
        worst = max(worst, 1 if fails else 0)
        lines.append("## 5. Chain %s (%s): %d steps, %d ungrounded; %d owner question(s) the chain never reached"
                     % (chain.get("id"), chain.get("by"), len(rows), fails, len(blind)))
        for i, says, cites, verdict in rows:
            lines.append("- %d. %s [%s] -> %s" % (i, says[:110], ", ".join(map(str, cites)), verdict))
        if blind:
            lines.append("- BLIND SPOTS (the model holds them, the chain never looked): " + ", ".join(blind))
    kg.db.close()
    if gnn:
        # THE PROPOSER (graph_gnn.py), measured before it speaks: whole methods hidden, against random / words / same-method
        import graph_gnn as gg
        v = gg.GraphView(dbp)
        res = gg.evaluate(v, 5, 0, 300, "method")
        text, beats = gg.report(res, "method")
        lines.append("## 6. Who owns each branch: the GNN proposer, measured first\n```\n" + text + "\n```")
        unowned = [c for c in v.cells if not any(cc == c for cc, _ in v.links)]
        if beats and unowned:
            p = gg.LinkProposer(v, 0, 300).fit(v.links, [])
            lines.append("Proposals (questions, never claims - a person or an exact check accepts one):")
            sure = gg.SURE
            for c in unowned:
                conf = p.confidence(c)
                lines.append("- %s: %s" % (c[14:], ("%s (%.0f%% sure)" % (", ".join(r[4:] for r in p.rank(c)[:3]), 100 * conf)) if conf >= sure
                                           else "no proposal: the GNN is %.0f%% sure, below its %.0f%% bar (it abstains)" % (100 * conf, 100 * sure)))
        elif beats:
            lines.append("Every branch is claimed: nothing to propose.")
    # THE HOLES, kept beside the report so a later run can say what IT changed (--since)
    keys = hole_keys(gate)
    hpath = (out or os.path.join(kgdir, "CLOSED_LOOP.md")) + ".holes.json"
    io.open(hpath, "w", encoding="utf-8", newline="\n").write(json.dumps(keys, indent=1) + "\n")
    if since:
        spath = since if since.endswith(".holes.json") else since + ".holes.json"
        if not os.path.exists(spath):
            raise RuntimeError("--since %s: no holes file beside it (%s) - only a report this version wrote has one" % (since, spath))
        added, closed, standing = hole_delta(json.load(io.open(spath, encoding="utf-8"))["ours"], keys["ours"])
        delta = ["## 0. This change, against %s: %d hole(s) of ours ADDED, %d CLOSED, %d already there" % (
            cs.shown(spath), len(added), len(closed), len(standing)),
                 "The ones to close are the ADDED: they are this change's. Those already there are someone else's work - leave them "
                 "unless the task asks (fixing them is scope creep)."]
        delta += ["- ADDED: %s" % h for h in added[:40]] + (["- ... %d more added" % (len(added) - 40)] if len(added) > 40 else [])
        delta += ["- CLOSED: %s" % h for h in closed[:20]] + (["- ... %d more closed" % (len(closed) - 20)] if len(closed) > 20 else [])
        lines.insert(0, "\n".join(delta))
        print("closed_loop: since %s: %d hole(s) of ours added, %d closed, %d already there" % (cs.shown(spath), len(added), len(closed), len(standing)))
    report = "# Closed loop: %s\n\n" % model.meta.get("subject", model_dir) + "\n\n".join(lines) + "\n"
    if out:
        io.open(out, "w", encoding="utf-8", newline="\n").write(report)
    verdict = "closed_loop: %s" % {0: "HOLDS (the owner's questions stay open)", 1: "BROKEN - a hole of ours or an ungrounded step"}.get(worst, "could not run")
    if out:
        io.open(out, "a", encoding="utf-8", newline="\n").write("\n" + verdict + "\n")
    print(report)
    print(verdict)
    # TELEMETRY, a pillar of the loop: every pass a row in the morph log, read from what this pass wrote - with or without --out
    # (P5-D, 10-29: a pass run without --out left no row, and its stats' pick went unrecorded); the report it reads is the one it wrote
    if not out:
        out = os.path.join(kgdir, "CLOSED_LOOP.md")
        io.open(out, "w", encoding="utf-8", newline="\n").write(report + "\n" + verdict + "\n")
    if kill and not self_improve_on(model_dir, self_improve):
        print("closed_loop: self-improvement is off: no row in the morph log, no picks")
    elif kill:
        import loop_stats
        row = loop_stats.MorphLog(model_dir).add(loop_stats.PassStats(model_dir, out, time.time() - started, note).collect())
        print("closed_loop: pass %d recorded in %s" % (row["n"], cs.shown(loop_stats.MorphLog(model_dir).path)))
        for line in loop_stats.summary(model_dir):     # the stats choose what to improve next, and every 20 passes a report
            print("closed_loop: " + line)
    return worst


def selftest():
    """a grounded chain passes; a step citing nothing, or a cite that names nothing, fails; an owner question no step reaches is a blind
    spot; a branch no requirement claims is a cell with no mapping"""
    ok = True
    o, orph = {}, []
    py_checks("# CHECKS: W-1 - the wallet pays\ncheck(\"W-1 paying empties it\", ok)\n# CHECKS: W-2, W-3\nprint(\"selftest: W-2 and 3 hold %s\" % x)\n", o, orph)
    good = o == {"W-1": ["W-1 paying empties it"], "W-2": ["W-2 and 3 hold"], "W-3": ["W-2 and 3 hold"]} and not orph
    ok &= good
    print("selftest: PYCHECKS a Python test's # CHECKS comment names the requirements of the checks after it, each named as the runner "
          "reports it %s" % ("PASS" if good else "FAIL"))
    d = tempfile.mkdtemp(prefix="closed_loop_checks_")
    t = os.path.join(d, "Mixed.cs")
    io.open(t, "w", encoding="utf-8").write("""
public class Mixed {
    /// CHECKS: A-1
    [Test]
    public void Nunit() { }
    /// CHECKS: A-2
    [TestCase(1)]
    [TestCase(2)]
    public void Cases(int a) { }
    /// CHECKS: A-3
    [TestMethod]
    public void MsTest() { }
    /// CHECKS: A-4
    [TestMethod]
    [DataRow(1)]
    public async Task MsTestAsync(int a) { }
    /// CHECKS: A-5
    [Fact]
    public void Xunit() { }
    [Theory]
    public void Named() { }
}
""")
    named, orph = test_checks([{"test": t}])
    good = named == {"A-1": ["Mixed.Nunit"], "A-2": ["Mixed.Cases"], "A-3": ["Mixed.MsTest"], "A-4": ["Mixed.MsTestAsync"], "A-5": ["Mixed.Xunit"]} and orph == ["Mixed.Named"]
    ok &= good
    shutil.rmtree(d, ignore_errors=True)
    print("selftest: FRAMEWORKS a test's CHECKS are read under NUnit, MSTest and xUnit, async too, so L16 does not invent holes on them %s" % ("PASS" if good else "FAIL: %s %s" % (named, orph)))
    added, closed, standing = hole_delta(["L13 req:A", "L15 cell:x", "L2 req:B"], ["L13 req:A", "L15 cell:x", "L13 req:NEW"])
    good = added == ["L13 req:NEW"] and closed == ["L2 req:B"] and standing == ["L13 req:A", "L15 cell:x"]
    ok &= good
    print("selftest: SINCE the holes a change added and closed come apart from those already there %s" % ("PASS" if good else "FAIL"))
    tmp = tempfile.mkdtemp(prefix="closed_loop_selftest_")
    p = os.path.join(tmp, "Wallet.cs")
    io.open(p, "w", encoding="utf-8").write(cs.SELF_CS)
    code = cs.CsCode([p])

    class R(object):
        def __init__(self, rid, kind, text):
            self.id, self.kind, self.text, self.effect = rid, kind, text, []

    class Q(object):
        def __init__(self, qid, text):
            self.id, self.text = qid, text

    class M(object):
        requirements = {"R-1": R("R-1", "authored", "Paying takes the price from the money"), "IR-1": R("IR-1", "inferred", "the owner's refund")}
        questions = [Q("Q1", "what does paying take?")]
        loop = {}
    m = M()
    cases = [
        ("a grounded step", {"steps": [{"says": "paying takes the money", "cites": ["R-1", "Wallet.Pay"]}]}, 0, ["IR-1"]),
        ("a step citing nothing", {"steps": [{"says": "it pays"}]}, 1, ["IR-1"]),
        ("a cite naming nothing", {"steps": [{"says": "it refunds", "cites": ["Wallet.Refund"]}]}, 1, ["IR-1"]),
        ("the owner question reached", {"steps": [{"says": "the refund", "cites": ["IR-1"]}]}, 0, []),
    ]
    for what, chain, fails, blind in cases:
        rows, f, reached, b = check_chain(m, code, chain)
        good = f == fails and b == blind
        ok &= good
        print("selftest: CHAIN %-28s -> %d ungrounded, blind %s %s" % (what, f, b, "PASS" if good else "FAIL"))
    # THE CLAIMS LOCK (REVIEW_1 #2): a method claim pinned to its decisions; an intruder branch stales it, a re-pin restores it
    m.root = tmp
    m.loop = {"tests": [{"covers": [p]}], "branches": {"claims": [{"method": "Wallet.Pay", "reqs": ["R-1"], "why": "paying"}]}}
    v = {"assertions": {}}
    unpinned = branch_space(m, v, code)["spaces"][0]["cells"]           # no lock and no --pin-claims: nobody's claim holds
    first = branch_space(m, v, code, pin="the first pin")["spaces"][0]["cells"]
    io.open(p, "w", encoding="utf-8").write(cs.SELF_CS.replace("Money -= price;", "if (price > 500) return;\n        Money -= price;"))
    intruded = branch_space(m, v, cs.CsCode([p]))["spaces"][0]["cells"]
    repinned = branch_space(m, v, cs.CsCode([p]), pin="read the new guard")["spaces"][0]["cells"]
    pay = lambda cells: [c for k, c in cells.items() if k.startswith("Wallet.Pay#")]
    good = (all(c["maps"] == [] for c in pay(unpinned)) and
            all(c["maps"] == ["R-1"] for c in pay(first)) and all(c["maps"] == [] and "STALE" in c["text"] for c in pay(intruded))
            and len(pay(intruded)) == len(pay(first)) + 1 and all(c["maps"] == ["R-1"] for c in pay(repinned)))
    ok &= good
    print("selftest: CLAIMS an intruder branch stales Wallet.Pay's claim (%d -> %d decisions, none claimed), a re-pin restores it %s" % (
        len(pay(first)), len(pay(intruded)), "PASS" if good else "FAIL"))
    sw = tempfile.mkdtemp(prefix="closed_loop_switch_")
    good = self_improve_on(sw) and not self_improve_on(sw, False)
    io.open(os.path.join(sw, "meta.yaml"), "w", encoding="utf-8").write("meta:\n  loop:\n    closed: true\n")
    good = good and self_improve_on(sw)
    io.open(os.path.join(sw, "meta.yaml"), "w", encoding="utf-8").write("meta:\n  loop:\n    self_improve: false\n")
    good = good and not self_improve_on(sw)
    ok &= good
    print("selftest: SWITCH self-improvement is on by default, and off by the flag or by the model's loop block %s" % ("PASS" if good else "FAIL"))
    print("selftest: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["run"])
    ap.add_argument("model", nargs="?")
    ap.add_argument("--chain", action="append", default=[])
    ap.add_argument("--no-kill", action="store_true")
    ap.add_argument("--accept", help="take the lock with this reason once no hole of ours stands (refused otherwise)")
    ap.add_argument("--note", help="what this pass changed - written to the morph log beside its numbers")
    ap.add_argument("--no-self-improve", action="store_true", help="this pass records no row in the morph log and prints no picks (self-improvement is on by default)")
    ap.add_argument("--since", help="an earlier report (or its .holes.json): list the holes of ours this change ADDED and CLOSED, "
                                    "apart from those already there")
    ap.add_argument("--gnn", action="store_true", help="measure the GNN proposer on the graph and show its proposals when it wins")
    ap.add_argument("--out")
    ap.add_argument("--fresh", action="store_true", help="re-run every mutant even when the code and the effects are unchanged")
    ap.add_argument("--workers", type=int, default=2, help="mutants run at once")
    ap.add_argument("--overlay", help="a sandbox laid out as the Unity project (DIR/Assets/...): the model's code and tests are read "
                    "from it where it holds them, and the mutants compile over it - the live project is never touched")
    ap.add_argument("--pin-claims", help="re-pin every method claim to its method's decisions now, with this reason (after reading them)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.overlay:
        cs.OVERLAY = os.path.abspath(a.overlay)
    if a.cmd != "run" or not a.model:
        ap.error("run <model> [--chain C.yaml ...], or --selftest")
    return run(a.model, a.chain, not a.no_kill, a.out, reuse=not a.fresh, workers=a.workers, pin=a.pin_claims, gnn=a.gnn, note=a.note, self_improve=not a.no_self_improve,
               accept=a.accept, since=a.since)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
