# Closed loop

Test the requirements an AI acts on, then hold them to the code, the tests and the written argument for each change, in one command.

**This is a prototype**, by RJ Thompson, shared to be tried and benchmarked. Its own requirements are a model the loop runs on itself,
and that model still names holes. Read [the limits](#honest-limits) before you trust a green report.

- **The manual:** <https://rj45thompson.github.io/anchor-demo/harness/fly/>
- **Results on code anyone can check:** <https://rj45thompson.github.io/anchor-demo/harness/public-code/>

## What it is

An AI that writes code from prompts drifts from what was asked, and reports work done that is not. The requirements it acts on are
prose nobody can check, so a mistake is found by a person, late, if at all.

The closed loop keeps the requirements as data and compiles them into a knowledge graph. Then it holds them to the code with checks
that can fail: every statement of the covered code is broken on purpose, every way the tool knows, and a test must notice each break,
namely the test of the requirement that owns that code. What no test notices, what no requirement owns and what only the owner can
decide are all counted, so "is it done?" becomes a number that has to reach zero.

It also tests the requirements themselves. It walks each use case and asks what each step needs. A need that nothing provides is a
failure of the requirements, so it becomes a new requirement, marked inferred, and goes to the owner as a question.

What is in the box:

- **The model:** use cases and requirements as YAML, checked by 19 rules (`usecase_model.py`).
- **The graph:** requirements, steps, code places, breaks, tests and argument steps as nodes in SQLite, held by 17 criteria, each proved
  by a planted mistake in its selftest (`usecase_kg.py`).
- **The one command:** `closed_loop.py run` does all of it and writes a report.
- **Code checks** for C# (`cs_effect_check.py`, any .NET project through `dotnet test`) and Python (`py_effect_check.py`). A survivor
  counts as no hole only when compiled code proves it the same program, or a reason says why, resting on code that has not changed.
- **Argument chains:** a chain is a file an agent writes on purpose, listing its claims and their evidence. Each step is walked through the
  graph, and a citation of something the graph does not hold fails the run.
- **A graph neural network** (`graph_gnn.py`, optional) that proposes which requirement owns new code, and only speaks once it beats
  the plain rules on links it never saw.
- **An on/off harness** (`ab_loop.py`, `grade_panel.py`) that measures agents with the loop against agents without it, graded blind.
- **Telemetry** (`loop_stats.py`, `STATS.md`) whose trends pick the next improvement, and a page you can fly through to see any graph
  the loop wrote (`graph_flythrough.py`).
- **A skill for Claude Code** (`skill/SKILL.md`). The tools are plain command-line Python and run without it.

## When to use it

Use it for **building larger systems**: code many changes will touch, where a requirement must stay true after the AI's tenth edit as
well as its first, and where you want to know which line exists because of which requirement.

Do not use it for a one-off prompt, a quick script or exploratory work. The loop costs setup and run time, and it pays that back only
when the same requirements are held across many changes.

## Install

Needs Python 3.10 or newer and PyYAML. The rest is optional, and the installer names whatever is absent:

- the **.NET SDK**, 8 or newer, for the C# adapter and the C# example. The example's first build restores NUnit from NuGet, so it needs
  the internet once.
- **PyTorch** for the graph neural network.
- **scikit-learn** for `survivor_rank.py`, which ranks which holes to show first.

```
git clone <this repository> closed-loop-kit
cd closed-loop-kit
pip install pyyaml
python install.py
```

By default the skill goes in `~/.claude/skills/closed-loop` and the tools in `~/.claude/closed-loop`. `--dest DIR` and `--skills DIR`
move them, `--check` only checks what is needed, and `--uninstall` removes what an install put down.

The installer proves itself: it runs every selftest, then runs the whole loop on the examples, each of which must print HOLDS. On Python
and PyYAML alone, 14 of the 20 selftests and the Python example run; what needs an optional piece is named as skipped, never counted as passed.

## Try it

`example/` is a small wallet and store in C#; `example_py/` is the same store in Python. Each holds a complete model in `model/*.yaml`, and
each test names what it checks on the line above it (`/// CHECKS: W-1`). The whole brief, from the example's imaginary owner: "The wallet pays
only what it holds; a limited line sells out and an unlimited one never does; the display holds eight digits." After an install, from the
tools folder:

```
CLOSED_LOOP_ROOT=. python -X utf8 closed_loop.py run example_py/model --out example_py/CLOSED_LOOP.md
python -X utf8 loop_stats.py show example_py/model
```

`CLOSED_LOOP_ROOT` is the folder the model's paths are relative to (on Windows, set it the way your shell does). Exit 1 means a hole of
yours is open; the owner's questions never fail a run. Read the report: survivors one per decision, the criteria table, the owner's
questions.

## Use it on your code

1. **Model your system.** The problem story first, in your words and with sources, then requirements, then use cases (an epic and its
   sea-level use cases, 3 to 11 steps each), then the `loop:` block that points at your code and your tests.
   `python -X utf8 usecase_model.py <model>` checks the 19 rules.
2. **Seed it first.** The loop learns from what is already written, so give it as many sourced requirements as you can before you lean on
   what it proposes, the pillars first: the rules that own the most code and run through the most methods.
3. **Run the loop.** `CLOSED_LOOP_ROOT=<your repo> python -X utf8 closed_loop.py run <model> --out REPORT.md`. On a model's first run add
   `--pin-claims "<why>"`, which writes `<model>/claims.lock`, holding each method claim to a fingerprint of its method's decisions. Pin
   again only after you have read a claimed method whose decisions changed; the report marks its claim STALE until then.
4. **Close what is yours**: a missing test, an unclaimed line, an equivalent to prove or explain. Leave the inferred requirements to their
   owner. Take the lock with `--accept "<why>"` once nothing of yours stands.
5. **Each pass**, run `loop_stats.py show <model>` to read the morph log, and have a fresh agent review the pass for ideas not yet tried.

Self-improvement is on by default: each pass records its own stats and prints what to improve next, in about a third of a second and three
lines. `--no-self-improve`, or `self_improve: false` in the `loop:` block, turns it off. The larger spend is the standing tick
(`loop_tick.py`, `TICK.md`), a fresh agent improving one thing at a time. It ships here, and nothing starts it for you.

Other languages need an adapter that writes the `effect-verdicts/1` document `cs_effect_check.py` writes: per requirement effect a shape
verdict, and per break its verdict and the tests that caught it. The model, the graph, the criteria and the chains are language-neutral.

## What has been measured

**On code anyone can check** ([the full results](https://rj45thompson.github.io/anchor-demo/harness/public-code/)), measured 30 October 2026
with no test or product code written for any target. Unreal's source needs an Unreal licence to read; osu! is fully open.

- Unreal Engine 5.7 `SemVer.cs`: the coverage tool says 90% of lines. Of 236 live breaks, 206 were caught (87.3%) and 30 got past every
  test, 14 of them on lines marked as covered.
- Unreal Engine 5.7 `JsonObject.cs`: 77% of lines. Of 429 live breaks, 299 were caught (69.7%) and 130 got past, 67 on covered lines.
- osu! scoring, against 151 of its tests: 72% of lines. Of 536 live breaks, 286 were caught (53.4%) and 250 got past, 145 on covered lines.

**Against careful reading**, the same page's control run. Two agents per class, one with the loop and one without, on the two Unreal
classes. The loop added no defect, and it cost more: 1.4 times the tokens and 1.7 times the minutes on `SemVer`, 1.2 times the tokens on
`JsonObject`. An agent reading carefully found everything the loop arm found. That is one run per class, a sign and not a measure, and it
tested finding defects, not how much the tests hold, which is what breaks are for. The defects listed on that page were found by an agent
reading the code against each rule while writing the model. The breaks did not find them; they measured what the tests hold and showed
where no test looks.

**On one C# game feature**, a shop system of 488 lines of code, on 2026-10-29:

- With one break per requirement the loop read "15 caught, 0 got past"; with every statement broken every way it read 89 of 340 getting
  past, on the same code. A loop that breaks code one way flatters.
- Over five passes, breaks made went 340 -> 806, breaks no test caught 89 -> 29 (most of the rest proven or explained equivalent), and
  statements held by a test 31 -> 169 of 173.
- Walking the use cases found 10 requirements nobody had written, beside the 17 already written.
- The graph neural network, on whole methods it never saw, scored MRR 0.76 against 0.67 for the best plain ranker. It beat the plain rule
  only on the largest model tried (27 requirements over 41 methods) and only once about half of its code was claimed. On the 4-requirement
  examples the plain rule won at every fullness. Until your model is that full, read the plain rule's suggestions.

## Honest limits

- **It is a prototype.** Real code broke it in six ways on the public trials, including silently skipping overloads, operators, generic
  methods and property bodies while the report read as if the whole file was held. Those are fixed, each with a planted case, and the report
  now names what it does not walk, such as a field's value or an indexer. Expect more of this.
- **It refutes, it never proves.** Breaking code can show requirements are not closed. It can never show they are: "closed" means closed
  against the breaks it tried and what the tests observe.
- **A thin model misleads.** With few requirements, few tests and a small graph, a green report reads as proof and is not, and that can be
  **worse than not using the loop at all**. How many nodes is enough is not known yet.
- **Setup comes first.** It needs a problem story, requirements, use cases and tests a machine can run before it helps at all.
- **It has not been shown to save anything yet.** The two targets, 95% accuracy or better and up to 90% fewer tokens per closed
  requirement, are targets. The measurements so far go the other way on tokens: 1.2 to 1.4 times more above, and 58% more on one small,
  precisely specified task. The target is about larger systems held across many changes, which has not been measured.
- **Visuals, timing and feel are not checked.** A person looks.
- **It is not networked.** Each checkout keeps its own graph.
- **The comments are working notes.** They keep dated measurements and short ids from the project's own record (`KT-8`, `PK-5`) that do
  not ship, and a few paths name a Unity runner that is not in this repository and stays inert without its settings.
- **What it reads:** `ab_loop.py harvest` reads your own Claude Code transcripts, locally, and keeps what it finds outside any repository.
  A page from `graph_flythrough.py` loads three.js from a CDN and fonts from Google when opened.

## What we would like back

Benchmarks of agents with and without it on your own systems (`ab_loop.py board`), the failure piles you see, and any criterion you can make
pass for the wrong reason.

## Licence

MIT. See `LICENSE`.
