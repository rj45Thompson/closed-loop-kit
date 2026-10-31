#!/usr/bin/env python
"""record_checks.py - name the tests that check each requirement, read from the verdicts of a pass.

A requirement that names no test as its check leaves the kill rule (a break's catching tests vote for the requirements they check)
with nothing to count, and L16 listing every requirement. Writing the names by hand is guessing; this reads them: for each
requirement, every break of its code (its effects' statements and the branches it claims) that a test caught, and WHICH tests
caught it, exactly as the pass recorded them. Nothing is added that no verdict names.

Four outcomes per requirement, each said in words:
  checked           at least one named test caught a break of its code
  caught by a crash a break was caught, but only as a test process dying, which names no check
  nothing catches   the code has breaks and every one survived: a finding about the tests
  no code / not in the record   there is nothing to break, or the pass pre-dates the requirement

The names are the printed labels of a pass, so they move when a label's numbers move: write again after each full pass
(--check says when the file no longer matches the verdicts). They are fitted to the record by construction, so the clause of
L16 that asks a named check to catch the code's breaks cannot fail on them; what stays live is the break a requirement's
named checks miss (caught only by a crash) and the requirement with a break nothing catches.

    python -X utf8 record_checks.py <model folder> [--verdicts V.json] [--write] [--check]
    python -X utf8 record_checks.py --selftest
"""
import argparse
import collections
import io
import json
import os
import re
import sys
import tempfile

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import usecase_model as ucm   # noqa: E402

FILE = "record_checks.json"
CRASH = re.compile(r"^\S+ exited -?\d+$")      # PyRunner's name for a test process that died with no check line: it names no check


def is_check(name):
    """a name in a mutant's `by` that names a check: not a hang notice `(...)`, not a process death"""
    return bool(name) and not name.startswith("(") and not CRASH.match(name)


class RecordChecks(object):
    """What the verdicts of a pass show about each requirement's checks. `aims[rid]`: the ids a break is made of for that requirement
    (its effects' ids and the branch cells it claims), the same set L16 reads; `mutants`: the pass's mutants, each with `of`, `killed`, `by`."""

    def __init__(self, aims, mutants):
        self.aims, self.mutants = aims, mutants

    @classmethod
    def from_model(cls, model, verdicts, cells):
        aims = {rid: {a.id for a in r.effect} for rid, r in model.requirements.items()}
        for key, c in cells.items():
            for rid in c.get("maps") or []:
                if rid in aims:
                    aims[rid].add(key)
        return cls(aims, verdicts.get("mutants") or {})

    def breaks(self, rid):
        return [m for m in self.mutants.values() if self.aims.get(rid, set()) & set(m.get("of") or [])]

    def catchers(self, rid):
        """{test name as recorded: breaks of this requirement's code it caught}"""
        out = collections.Counter()
        for m in self.breaks(rid):
            if m.get("killed") is True:
                for t in set(m.get("by") or []):
                    if is_check(t):
                        out[t] += 1
        return out

    def status(self, rid):
        ms = self.breaks(rid)
        if not self.aims.get(rid):
            return "no code", "nothing to break: its effect is absent or it claims no branch"
        if not ms:
            return "not in the record", "it has code to break, but this pass made no break of it (the model changed after the pass)"
        killed = [m for m in ms if m.get("killed") is True]
        if not killed:
            return "nothing catches", "%d break(s) of its code, every one survived every test" % len(ms)
        if not self.catchers(rid):
            return "caught by a crash", "%d break(s) caught, each only by a test process dying: no check is named" % len(killed)
        crash = sum(1 for m in killed if not any(is_check(t) for t in m.get("by") or []))
        return "checked", "%d of %d break(s) caught by %d named test(s)%s" % (
            len(killed), len(ms), len(self.catchers(rid)), "; %d of them only by a crash" % crash if crash else "")

    def document(self, stamp):
        """the file's content: names only from verdicts, sorted by how many breaks each caught"""
        checks, said = {}, {}
        for rid in sorted(self.aims):
            st, why = self.status(rid)
            said[rid] = {"status": st, "why": why}
            c = self.catchers(rid)
            if c:
                checks[rid] = [t for t, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))]
        return {"format": "record-checks/1", "from": stamp, "checks": checks, "status": said}


def stamp_of(verdicts):
    return dict({k: verdicts.get(k) for k in ("code", "effects", "engine", "tests_sha")}, mutants=len(verdicts.get("mutants") or {}))


def cells_of(model_dir):
    p = os.path.join(model_dir, "generated_branches.yaml")
    if not os.path.isfile(p):
        return {}
    d = yaml.safe_load(io.open(p, encoding="utf-8")) or {}
    out = {}
    for sp in d.get("spaces") or []:
        if sp.get("id") == "S-BRANCH":
            out.update(sp.get("cells") or {})
    return out


def run(folder, verdicts_path, write, check):
    vp = verdicts_path or os.path.join(folder, "..", "kg", "verdicts.json")
    verdicts = json.load(io.open(vp, encoding="utf-8"))
    model = ucm.UseCaseModel.load(folder)
    rc = RecordChecks.from_model(model, verdicts, cells_of(folder))
    doc = rc.document(stamp_of(verdicts))
    counts = collections.Counter(v["status"] for v in doc["status"].values())
    for rid in sorted(doc["status"], key=lambda r: (r.startswith("IR"), len(r), r)):
        print("%-8s %-18s %s" % (rid, doc["status"][rid]["status"], doc["status"][rid]["why"]))
    print("record_checks: %d requirement(s): %s" % (len(doc["status"]), ", ".join("%d %s" % (n, k) for k, n in sorted(counts.items()))))
    path = os.path.join(folder, FILE)
    if write:
        io.open(path, "w", encoding="utf-8", newline="\n").write(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print("record_checks: wrote %s (%d requirement(s) name checks)" % (path, len(doc["checks"])))
    if check:
        held = json.load(io.open(path, encoding="utf-8")) if os.path.isfile(path) else {}
        stale = held.get("checks") != doc["checks"] or held.get("from") != doc["from"]
        unknown = sorted(r for r in (held.get("checks") or {}) if r not in model.requirements)
        print("record_checks: %s%s" % ("STALE: the file does not match these verdicts - write it again" if stale else "current",
                                       "; names requirements the model no longer holds: %s" % ", ".join(unknown) if unknown else ""))
        return 1 if stale or unknown else 0
    return 0


def selftest():
    ok = True
    mut = lambda of, killed, by: {"of": of, "killed": killed, "by": by}
    ms = {"a": mut(["R-1#1"], True, ["CLAIMS Wallet.Pay stales (3 -> 4)", "R-1 exited 1"]),
          "b": mut(["R-1#1"], True, ["R-1 exited 1"]),                                    # caught only by a crash
          "c": mut(["G#b1"], True, ["CLAIMS Wallet.Pay stales (3 -> 4)", "One"]),          # a claimed branch, two checks
          "d": mut(["R-2#1"], False, []), "e": mut(["R-2#1"], False, []),                   # nothing catches R-2
          "f": mut(["R-5#1"], True, ["(One hung past 300 s)"])}                            # a hang notice names no check
    rc = RecordChecks({"R-1": {"R-1#1", "G#b1"}, "R-2": {"R-2#1"}, "R-3": set(), "R-4": {"R-4#1"}, "R-5": {"R-5#1"}}, ms)
    got = {r: rc.status(r)[0] for r in rc.aims}
    good = got == {"R-1": "checked", "R-2": "nothing catches", "R-3": "no code", "R-4": "not in the record", "R-5": "caught by a crash"}
    ok &= good
    print("selftest: STATUS checked, nothing catches, no code, not in the record and caught by a crash are told apart %s" % ("PASS" if good else "FAIL %s" % got))
    doc = rc.document({})
    good = doc["checks"] == {"R-1": ["CLAIMS Wallet.Pay stales (3 -> 4)", "One"]} and "R-2" not in doc["checks"]
    ok &= good
    print("selftest: NAMES only the labels a verdict gives for the requirement's own breaks, most breaks first; a crash and a hang name none %s" % ("PASS" if good else "FAIL %s" % doc["checks"]))
    # the plant: remove what caught R-1's breaks and its names go; no name survives that no verdict holds
    gone = RecordChecks(rc.aims, {k: dict(m, by=[]) for k, m in ms.items()}).document({})
    good = gone["checks"] == {}
    ok &= good
    print("selftest: PLANT with the catching tests removed from the verdicts no requirement names a check %s" % ("PASS" if good else "FAIL %s" % gone["checks"]))
    # the file reaches the model: written beside the requirements, merged into what a requirement names, an unknown id ignored
    tmp = tempfile.mkdtemp(prefix="record_checks_")
    io.open(os.path.join(tmp, "requirements.yaml"), "w", encoding="utf-8").write(
        "requirements:\n  - {id: R-1, kind: authored, text: t, measure: m, checks: [Own.Test]}\n  - {id: R-2, kind: authored, text: t, measure: m}\n")
    io.open(os.path.join(tmp, FILE), "w", encoding="utf-8").write(json.dumps(dict(doc, checks=dict(doc["checks"], **{"R-9": ["X"]}))))
    m = ucm.UseCaseModel.load(tmp)
    good = m.requirements["R-1"].checks == ["CLAIMS Wallet.Pay stales (3 -> 4)", "One", "Own.Test"] and m.requirements["R-2"].checks == [] and "R-9" not in m.requirements
    ok &= good
    print("selftest: LOAD the recorded names join the ones a requirement names itself; a requirement with none stays with none %s" % ("PASS" if good else "FAIL %s" % m.requirements["R-1"].checks))
    return ok


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?")
    ap.add_argument("--verdicts")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        ok = selftest()
        print("selftest: %s" % ("PASS" if ok else "FAIL"))
        return 0 if ok else 1
    if not a.folder:
        ap.error("the model folder, or --selftest")
    return run(a.folder, a.verdicts, a.write, a.check)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
