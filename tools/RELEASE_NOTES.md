# Closed loop kit - release notes

## Not released yet

- Run with no clone: `pyproject.toml` and a console command, `closed-loop-kit` (alias `closed-loop`). `uvx --from git+https://github.com/rj45Thompson/closed-loop-kit closed-loop-kit try` (or `pipx run --spec ...`) runs the Python example through the whole loop in a temporary folder and must end HOLDS; `grow`, `ask`, `run`, `prove`, `install` and `tool NAME` are its subcommands. Not published to a package index: that needs the owner's account, and the name `closed-loop-kit` was free when last asked (HTTP 404 from the index, 2026-10-31). To publish, from a built kit: `uv build`, then `uv publish` with a token.
- `install.py --agent NAME` writes the skill's front door, `run` and `prove` in the file layout another coding agent reads (`AGENTS.md` for Codex and OpenCode, `.cursor/rules/closed-loop.mdc`, `GEMINI.md`, `.github/copilot-instructions.md`, `.windsurf/rules/closed-loop.md`) into the current project and touches nothing else. The layouts are generated at build time from `skill/SKILL.md` and carry its hash; a layout cut from an older SKILL.md is refused. No agent other than Claude Code has been tried.
- `closed_loop.py prove` (`prove.py`): every selftest and example of the install, several at once at below-normal priority, ending in one result: what passed, what failed, what gave no answer, what was not run and why. A selftest that hangs is killed and named, one that exits 0 having checked nothing is named and is not a pass, and a selftest that needs PyTorch or scikit-learn on a machine without them is NOT RUN with that said. `install.py` now proves through it, so a full install no longer runs 17 selftests and 2 examples one after another; `--quick` is `prove --quick`.
- A clean install on a Windows home folder no longer reads INSTALLED WITH FAILURES: `doc_nodes.py`'s selftest resolved its fixtures' citations against the fixture folder, which is inside the project root whenever the temp folder is inside the home folder (ROOT defaults to the home folder). `usecase_model.py`'s selftest printed no check line in the form every other tool uses, so a runner could not tell it from one that checked nothing; each of its checks now ends in PASS or FAIL.
- `install.py` no longer imports PyTorch to see whether it is there (it asks for it by name): on a machine that had it, that was about 50 seconds of a loaded first run.
- The README says where the tools folder is, how to set `CLOSED_LOOP_ROOT` in PowerShell and cmd (the bash form fails there), the options of `ask`, and what a first run costs.
- `doc_nodes.py`: the documents as nodes in the graph, `what-to-update <node>`, and criterion L20 DOCUMENTED, which makes 18 criteria.
- `loop_tick.py finish` sets the records going: it rewrites the Goals List and Stats List, rebuilds the growth page when that tool and a site are present, and prints the documents that describe the stat in `--targets`; `start` prints what the two lists ask. A step whose tool is not installed says so and is skipped; a step that fails makes `finish` exit 1.
- `improvement.py`: a requirement can be about one score inside a stat that is a dict (`derived: {key: ...}`), in its trigger and in the outcome of a tick that targeted it. Totalling the dict moved with every rival rule's score and said nothing about the learned part's own.

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
