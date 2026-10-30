#!/usr/bin/env python3
"""survivor_rank.py - learn from the verdicts which breaks will get past the tests, and test that learning fairly.

WHY (2026-10-29): can the stat data be used to train something that improves each pass, and what test would show it? The novelty
review: learning pays most on RANKING WHICH HOLES TO SHOW FIRST,
predictive mutation testing). Every pass leaves a verdict per break (kg/verdicts.json): the operator, the statement, the method, and
whether a test caught it. That is training data. A model that ranks the breaks most likely to SURVIVE first lets a pass show its
holes first (a fast pass), and tells a code-writing agent where its new code's tests are likely to be weak, BEFORE the breaks run.

THE TEST THAT LETS IT IMPROVE (and keeps it honest): it is scored only on code it never saw.
  BY METHOD   five folds, each holding out whole methods (new code), trained on the rest of the same project
  ACROSS      trained on one project's verdicts, tested on another's (the shops against the tool's own code)
The score is RECALL AT K: of the real survivors, the share found in the first K% of the ranking (K = 10, 20). Random order finds
K% of them; the OPERATOR PRIOR (each operator kind's survival rate in the training data) is the plain ranker it must beat.

    python -X utf8 survivor_rank.py eval <verdicts.json> [more ...]      the folds by method, then every pair across
    python -X utf8 survivor_rank.py --selftest
"""
import argparse
import collections
import io
import json
import os
import random
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from loop_stats import op_kind  # noqa: E402

KS = (10, 20)


def rows(path):
    """(features, survived, method) per break with a verdict: killed, or every test passed. A break that did not compile, ran no
    test, or was proven equivalent is no example of either and is left out"""
    m = json.load(io.open(path, encoding="utf-8"))["mutants"]
    out = []
    for r in m.values():
        why = str(r.get("why") or "")
        if r.get("killed"):
            y = 0
        elif why.startswith("every test passed"):
            y = 1
        else:
            continue
        out.append((features(r), y, str(r.get("place"))))
    return out


def hint(verdicts, source, top=10):
    """WR-1, the code-writing side: the statements of `source` whose breaks
    are most likely to get past the tests, BEFORE any break is run - trained on the project's own verdicts, each statement scored
    for the kinds of break the history shows apply to its kind of statement. A hint for where to add a test, never a verdict"""
    from sklearn.feature_extraction import FeatureHasher
    from sklearn.linear_model import LogisticRegression
    m = json.load(io.open(verdicts, encoding="utf-8"))["mutants"]
    data, kinds, example = [], collections.defaultdict(set), {}
    for r in m.values():
        why = str(r.get("why") or "")
        if r.get("killed") or why.startswith("every test passed"):
            data.append((features(r), 0 if r.get("killed") else 1))
            k = op_kind(r.get("what"))
            kinds[statement_kind(str(r.get("statement") or ""))].add(k)
            example.setdefault(k, r.get("what"))
    if len({y for _, y in data}) < 2:
        return []
    h = FeatureHasher(n_features=2 ** 14, input_type="string")
    clf = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced").fit(h.transform([f for f, _ in data]), [y for _, y in data])
    lines = io.open(source, encoding="utf-8-sig").read().splitlines()
    method, out = "", []
    for i, line in enumerate(lines, 1):
        t = line.strip()
        d = re.match(r"(?:def\s+(\w+)|.*?\b(?:public|private|internal|protected|static)\b[^=;(]*?\b(\w+)\s*\()", t)
        if d:
            method = d.group(1) or d.group(2) or method
            continue
        if not t or t.startswith(("#", "//", "/*", "*", "{", "}", "using ", "import ", "from ", "[")) or t in ("else", "else:"):
            continue
        sk = statement_kind(t)
        cands = [{"what": example[k], "statement": t, "place": "X." + method, "file": os.path.basename(source)} for k in kinds.get(sk, ())]
        if not cands:
            continue
        p = clf.predict_proba(h.transform([features(c) for c in cands]))[:, 1]
        j = max(range(len(cands)), key=lambda x: p[x])
        out.append((p[j], i, t, op_kind(cands[j]["what"])))
    return sorted(out, reverse=True)[:top]


def statement_kind(s):
    s = (s or "").strip()
    for k, pat in (("return", r"^return\b"), ("if", r"^(if|else if|elif)\b"), ("throw", r"^(throw|raise)\b"), ("loop", r"^(for|foreach|while)\b"),
                   ("assign", r"^[\w\.\[\]]+\s*[\+\-\*/]?=[^=]"), ("call", r"\w\s*\(")):
        if re.search(pat, s):
            return k
    return "other"


def features(r):
    """words a model may use: the operator's kind, the statement's kind and its words, the method's and file's names - never the
    verdict, the tests that caught it, or anything a run decides"""
    st = str(r.get("statement") or "")
    f = ["op=" + op_kind(r.get("what")), "st=" + statement_kind(st), "file=" + os.path.basename(str(r.get("file"))),
         "op*st=%s*%s" % (op_kind(r.get("what")), statement_kind(st))]
    f += ["w=" + w.lower() for w in re.findall(r"[A-Za-z_]{3,}", st)][:30]
    f += ["m=" + w.lower() for w in re.findall(r"[A-Z][a-z]+|[a-z]+", str(r.get("place")).split(".")[-1])]
    return f


def recall_at(scores, ys, k):
    n = max(1, int(round(len(ys) * k / 100.0)))
    order = sorted(range(len(ys)), key=lambda i: -scores[i])
    found = sum(ys[i] for i in order[:n])
    return found / float(max(1, sum(ys)))


def fit_predict(train, test, seed=0):
    """(learned scores, operator-prior scores) for the test rows"""
    from sklearn.feature_extraction import FeatureHasher
    from sklearn.linear_model import LogisticRegression
    h = FeatureHasher(n_features=2 ** 14, input_type="string")
    Xtr, ytr = h.transform([f for f, _, _ in train]), [y for _, y, _ in train]
    Xte = h.transform([f for f, _, _ in test])
    if len(set(ytr)) < 2:
        learned = [0.0] * len(test)
    else:
        clf = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced", random_state=seed).fit(Xtr, ytr)
        learned = list(clf.predict_proba(Xte)[:, 1])
    rate = collections.defaultdict(lambda: [0, 0])
    for f, y, _ in train:
        rate[f[0]][0] += y
        rate[f[0]][1] += 1
    base = sum(ytr) / float(max(1, len(ytr)))
    prior = [(rate[f[0]][0] + base) / (rate[f[0]][1] + 1.0) for f, _, _ in test]
    return learned, prior


def score(test, learned, prior, rnd):
    ys = [y for _, y, _ in test]
    rand = [rnd.random() for _ in ys]
    return {k: {"learned": recall_at(learned, ys, k), "operator prior": recall_at(prior, ys, k), "random": recall_at(rand, ys, k)} for k in KS}


def method_folds(data, folds=5, seed=0):
    """[(train, test)]: each fold holds out whole methods - new code - and trains on the rest"""
    methods = sorted({m for _, _, m in data})
    random.Random(seed).shuffle(methods)
    out = []
    for k in range(folds):
        hold = set(methods[k::folds])
        out.append(([d for d in data if d[2] not in hold], [d for d in data if d[2] in hold]))
    return out


def by_method(data, folds=5, seed=0):
    learned_all, prior_all, test_all = [], [], []
    for train, test in method_folds(data, folds, seed):
        if not test or not train:
            continue
        lr, pr = fit_predict(train, test, seed)
        learned_all += lr
        prior_all += pr
        test_all += test
    return score(test_all, learned_all, prior_all, random.Random(seed + 99)), len(test_all), sum(y for _, y, _ in test_all)


def temporal(old, new, seed=0, only_new_code=True):
    """the INCREMENTAL test: trained on an earlier pass's verdicts, scored on a later pass's - only on the breaks of statements the
    earlier pass did not hold (new or changed code), so remembering a verdict earns nothing"""
    seen = {(m, tuple(sorted(x for x in f if x.startswith("w=")))) for f, _, m in old}
    test = [d for d in new if (d[2], tuple(sorted(x for x in d[0] if x.startswith("w=")))) not in seen] if only_new_code else new
    return across(old, test, seed)


def across(train, test, seed=0):
    lr, pr = fit_predict(train, test, seed)
    return score(test, lr, pr, random.Random(seed + 99)), len(test), sum(y for _, y, _ in test)


def line(name, res, n, s):
    return "  %-34s %5d breaks, %4d survived: %s" % (name, n, s, "; ".join(
        "top %d%%: learned %.0f%%, operator prior %.0f%%, random %.0f%%" % (k, 100 * res[k]["learned"], 100 * res[k]["operator prior"],
                                                                           100 * res[k]["random"]) for k in KS))


def evaluate(paths, say=print):
    data = {p: rows(p) for p in paths}
    out = {}
    say("survivor_rank: of the real survivors, the share each ranking finds in its first 10% and 20% (tested only on code it never saw)")
    for p, d in data.items():
        res, n, s = by_method(d)
        out[("method", p)] = res
        say(line("%s (by method)" % shown(p), res, n, s))
        # the EVIDENCE row run_plan.py reads to decide whether a pass orders its breaks by this model (beside the verdicts it learned from)
        io.open(os.path.join(os.path.dirname(os.path.abspath(p)), "survivor_rank.jsonl"), "a", encoding="utf-8").write(json.dumps({
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "breaks": n, "survived": s,
            "recall20": {k: round(v, 3) for k, v in res[20].items()}}) + "\n")
    for a in paths:
        for b in paths:
            if a != b:
                res, n, s = across(data[a], data[b])
                out[(a, b)] = res
                say(line("%s -> %s" % (shown(a), shown(b)), res, n, s))
    return out


def shown(p):
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(p))))


def selftest():
    """the test can tell learning from luck: a planted operator that always survives is found first by both the learner and the
    prior; a planted WORD that always survives (the operator says nothing) is found only by the learner; labels with no signal leave
    the learner near random; the folds never train on a method they test"""
    import tempfile
    rnd = random.Random(3)
    ok = True

    def mk(n, signal):
        ms = {}
        for i in range(n):
            op = rnd.choice(["equality != to == (1)", "the statement removed", "Pay's argument 1 made default", "the number 0 made 1"])
            word = rnd.choice(["Clear", "Total", "Price", "Count", "Level"])
            meth = "T.M%d" % (i % 40)
            surv = (signal == "op" and op == "the statement removed") or (signal == "word" and word == "Clear") or \
                   (signal == "none" and rnd.random() < 0.25)
            ms["%s@x: %d" % (meth, i)] = {"place": meth, "file": "a.cs", "line": i, "statement": "x.%s = y;" % word, "what": op,
                                         "killed": not surv, "why": "every test passed" if surv else "failed: T"}
        d = tempfile.mkdtemp(prefix="survivor_rank_")
        p = os.path.join(d, "verdicts.json")
        io.open(p, "w", encoding="utf-8").write(json.dumps({"mutants": ms}))
        return rows(p)
    r_op, _, _ = by_method(mk(800, "op"))
    r_w, _, _ = by_method(mk(800, "word"))
    r_none, _, _ = by_method(mk(800, "none"))
    good = r_op[20]["learned"] > 0.75 and r_op[20]["operator prior"] > 0.75
    good = good and r_w[20]["learned"] > 0.75 and r_w[20]["operator prior"] < 0.4
    good = good and abs(r_none[20]["learned"] - 0.20) < 0.12
    ok &= good
    print("selftest: SIGNAL an operator that always survives is found by both; a word that always survives only by the learner; "
          "no signal leaves it near random %s" % ("PASS" if good else "FAIL"))
    data = mk(200, "op")
    fs = method_folds(data)
    leak = any({m for _, _, m in tr} & {m for _, _, m in te} for tr, te in fs)
    covered = sorted(m for _, te in fs for _, _, m in te) == sorted(m for _, _, m in data)
    good = not leak and covered and all(te for _, te in fs)
    ok &= good
    print("selftest: FOLDS no held-out method is ever trained on, and every break is held out exactly once %s" % ("PASS" if good else "FAIL"))
    print("selftest: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", choices=["eval", "temporal", "hint"])
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "hint" and len(a.paths) == 2:
        rows_ = hint(a.paths[0], a.paths[1], a.top)
        print("survivor_rank hint: the statements of %s whose breaks this project's history says are most likely to get past the "
              "tests (a hint for where a test is missing, never a verdict):" % os.path.basename(a.paths[1]))
        for p, i, t, k in rows_:
            print("  line %4d  %3.0f%%  %-26s %s" % (i, 100 * p, k, t[:90]))
        return 0
    if a.cmd == "temporal" and len(a.paths) == 2:
        res, n, s = temporal(rows(a.paths[0]), rows(a.paths[1]))
        print(line("earlier -> later, new code only", res, n, s))
        res, n, s = temporal(rows(a.paths[0]), rows(a.paths[1]), only_new_code=False)
        print(line("earlier -> later, every break", res, n, s))
        return 0
    if a.cmd == "eval" and a.paths:
        evaluate(a.paths)
        return 0
    ap.error("eval <verdicts.json> [...] | --selftest")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
