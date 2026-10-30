# Closed loop kit - release notes

## 0.1.2 (2026-10-30): first public release

- MIT licence, and a README written for someone arriving at the repository.
- The tools' comments and docstrings now state each rule, its reason and its measured evidence. Five user-facing strings changed with them; no behaviour did.
- A clean install on Python and PyYAML alone no longer reads INSTALLED WITH FAILURES: `run_plan.py` and `survivor_rank.py` need PyTorch and scikit-learn, and their selftests are now named as skipped without them. scikit-learn is named as an optional piece.

## 0.1.1 (2026-10-30)

- The C# adapter no longer skips code silently: overloads, operators, generic methods and property bodies are walked, and the report names what it still does not walk. It runs on read-only trees such as a Perforce workspace, and reads MSTest and xUnit check lines. Found by running it on Unreal's own C# and on osu!.
- `gnn_goals.py`: the goals list and the stats list, generated from the records, with a check that every job belongs to a goal.
- `graph_flythrough.py`: any graph the loop wrote, as one page you can fly through, with a tour along one real thread and a view per question. `--hide` keeps names you choose off the page.
- `--no-self-improve`, or `self_improve: false` in the model's `loop:` block: a pass records no row and prints no picks. Self-improvement stays on by default.
- `improvement.py` and its requirements now ship, so the stats' picks hear the triggers as well as the trends.
- The standing tick now ships too: `loop_tick.py`, its runbook `TICK.md`, `run_plan.py` (which parts run on this project), `telemetry_audit.py` (every stat has a reader), `survivor_rank.py` and the learned jobs `gnn_tasks.py`, `gnn_more.py`, `gnn_curve.py`. Nothing starts a tick for you.
- `grade_panel.py`: more than one blind grader per pair, and whether they agree. One grader per pair was measured to be too noisy.
- `ab_loop.py leaks`: scan an arm's own tool calls for what it should not have read; a blind diff now hides what names the arm.

## 0.1 (2026-10-29): first shared prototype

What is in it:
- the use case model as data and its 19 rules (`usecase_model.py`);
- the knowledge graph and its 17 criteria (`usecase_kg.py`), written to SQLite and N-Triples on every run;
- the one command (`closed_loop.py`): the model, every mutant of the code, the generated branch space, the graph, the criteria, the
  reasoning chains, the morph log, and the GNN's section;
- the C# adapter (`cs_effect_check.py`) with a `dotnet test` runner, and the Python adapter (`py_effect_check.py`);
- the GNN link proposer (`graph_gnn.py`): typed relations, a pair scorer over the graph's exact signals, calibration and abstain;
- the on/off harness (`ab_loop.py`) and the telemetry (`loop_stats.py`, `STATS.md`);
- the skill for Claude Code, an installer that runs every selftest and the whole loop on the example, and the example.

Known limits:
- a prototype: its own requirements are a model the loop runs on itself, and that model still names holes (the installer was the
  newest; the accuracy and token targets are unmeasured);
- a thin model gives verdicts that mean little; how thin is too thin is not known yet;
- mutation can show requirements are NOT closed; it can never prove they are - "closed" is relative to the operators it tries;
- visuals, timing and feel are not checked: a person looks.

What we would most like back: benchmarks of agents with and without it on your own systems (the on/off harness's board), the
failure piles you see, and any criterion you can make pass for the wrong reason.
