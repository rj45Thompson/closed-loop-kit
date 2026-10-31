# Closed loop

Test the requirements an AI acts on, then hold them to the code, the tests and the written argument for each change, in one command.

**This is a prototype**, by RJ Thompson, shared to be tried and benchmarked. Its own requirements are a model the loop runs on itself,
and that model still names holes. Read [the limits](#honest-limits) before you trust a green report.

**Where the evidence stands.** On the one end-to-end test so far, agents building a complete program from a written brief, with the loop and without it, graded against tests the
agents never saw, **the loop did not beat a plain agent asking good questions**: the scores were the same within the margin and the loop cost more tokens and minutes, a sign and not a measure. It did what it
is built to do, and that was not enough. Its tests held the code to each agent's own reading of the brief, wrong parts included, so a requirement that came in wrong was defended as
firmly as a right one; what moved the score most was how complete and how correct the requirements were that the agent was given, and no check that runs later can see a requirement that came in wrong. So the kit now asks for the rules
first, from the person who holds them (`elicit.py`), and closes the loop last and once. The narrower claims it can back are further down: it shows what your tests hold and where no
test looks, and it keeps a record of where each requirement came from. Whether it pays on a large system held across many changes has not been measured.

## Run it with no clone

Python 3.10 or newer and nothing else installed first. This runs the shipped Python example through the whole loop in a temporary folder and must end `HOLDS`:

```
uvx --from git+https://github.com/rj45Thompson/closed-loop-kit closed-loop-kit try
pipx run --spec git+https://github.com/rj45Thompson/closed-loop-kit closed-loop-kit try
```

`closed-loop-kit` (alias `closed-loop`) is the package's one command. `grow`, `ask`, `run` and `prove` are its subcommands (the same as `closed_loop.py` below), `install` is the
installer in [Install](#install), and `tool NAME` runs any other shipped tool. It is not on a package index yet: that needs the owner's account, so there is no bare `uvx closed-loop-kit`
until it is published (from a built kit, `uv build`, then `uv publish`). The git address works once this repository carries the files; until then the same command takes a path to
a built kit or a wheel in place of the address.

- **The manual:** <https://rj45thompson.github.io/anchor-demo/harness/fly/>
- **Results on code anyone can check:** <https://rj45thompson.github.io/anchor-demo/harness/public-code/>

## Start here: an epic in, the first question out

You have a rough epic, a PRD, a ticket export or a call transcript, and you want the requirements nobody wrote down. Five minutes, Python and PyYAML, no .NET, no key:

```
python install.py --quick                                                  # under a minute: the front door only, proved on a shipped example (python install.py proves the whole loop: about a minute on Python alone, minutes more with the .NET SDK)
cd ~/.claude/closed-loop                                                   # the tools folder (on Windows: cd %USERPROFILE%\.claude\closed-loop); the same files are in tools/ of this clone
python -X utf8 closed_loop.py grow my_epic.md --out my_model              # numbers every sentence, writes my_model/BRIEF.md and a skeleton draft.yml, exits 3
#   your agent reads the numbered sentences and fills my_model/draft.yml; then the same command again:
python -X utf8 closed_loop.py grow my_epic.md --out my_model              # refuses what it cannot ground, or writes the model and prints the FIRST QUESTION
python -X utf8 closed_loop.py ask my_model --answer "their words" --check "how it would be seen to hold"   # or --confirm (the producer agrees with the reading offered), --absent "reason", --later "reason"
```

**What a program can do about prose is small and exact, so the work is split.** The tool numbers each sentence (`4.1` is line 4, first sentence; for tickets it also says which ticket and field, for a
transcript who said it and when), prints a brief, and then checks what comes back. The agent that runs the command is the reader; no model is called. Every requirement in the draft must quote at least
two words of a sentence it cites, or be marked INFERRED with the reason a step of the chain needs it. A requirement that cites nothing, cites a sentence that is not in the text, quotes words the
sentence lacks, or is inferred with no real reason, is refused and nothing is written. An empty text, or one with no sentences, stops the run.

**The questions come from holes in the chain, not from a script.** `ask` takes the model the draft became and asks one question at a time, in plain words with no id in them, about: two statements that
cannot both hold (a number that changed, a rule and its negation), a step that needs something nobody authored, a sentence of your text that no requirement captures, a goal with no alternate path (what
happens when a step cannot complete?), and the fields of a goal the text never states. They are ranked: what an answer can change comes first. A reply is one of four things: the producer's words
(they become a requirement with the producer as its source), ABSENT with the reason, CONFIRMED, or LATER. The answer is written into the model and the next question comes from the new state; an answered
question never returns, a session that stops resumes where it was, and a stop with questions open says how many (`ask my_model --status`; `--list` shows every open question in order). `grow` exits 0
once it has written the model and 1 when it refused the draft; `ask` exits 1 while a question is open and 0 when none is. The order is: statements that disagree (any answer built on either is suspect),
then a need nobody authored, a vague word or an uncaptured sentence of your text, a goal with no alternate path, and last the fields of a goal; inside a kind, the answer more of the chain rests on first. The shipped `example_grow/` holds the wallet brief with a draft, and a PRD, a ticket export and a
transcript with a draft each.

**If someone can be asked, ask first: `elicit.py`.** `grow` and `ask` start from text. When the rules are in a person's head, start from the person. `elicit.py` crosses every act, thing and stage the text and the interface name with what must be known about each (who, when, when not, what follows, may it be passed, what changes near the end, what counts, what is equal, what if it is claimed and is not there), and asks in batches. It holds each rule to two hearings in different forms, keeps a rule open while an answer disagrees with it, and does not take one "not specified" as final. `check` says what is open. The reason is measured and it is a limit of the loop itself: a closed loop holds a wrong requirement as tightly as a right one, and tests written from the same reading as the code cannot tell them apart. So the rules have to be right before the loop closes. The closing then comes last, as one pass: `elicit.py breaks` runs every break of the finished code against the tests, with no model, and a break they miss is read as a question. The skill's first section gives the order.

It borrows ideas, none of its code: Spec Kit's clarify step (MIT), BMAD-METHOD's analyst brainstorming (MIT), Superpowers' one-question-at-a-time brainstorming (MIT), OpenSpec's explore step (MIT), Agent OS's shaping
questions (MIT), Task Master's parse of a requirements document into tasks (MIT with the Commons Clause, so the idea only), SpecTrace's import of requirements from specs (MIT). Kiro and Claude Code's question tool
are proprietary: idea only. What this adds is that a requirement carries its sentence or its reason and a program checks it.

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
- **The graph:** requirements, steps, code places, breaks, tests and argument steps as nodes in SQLite, held by 18 criteria, each proved
  by a planted mistake in its selftest (`usecase_kg.py`).
- **Asking** (`elicit.py`): the rules from the person who holds them, widely and twice, before any code. Above.
- **The front door** (`grow.py`, `ask.py`): from text someone already wrote to a first model and the questions that grow it, with no key and no .NET. Above.
- **The one command:** `closed_loop.py run` does all of it and writes a report.
- **The documents as nodes** (`doc_nodes.py`): declare your documents and each section becomes a node, linked from what it names literally or from a line
  you author with the words it stands on. `doc_nodes.py what-to-update <node>` says which documents a change makes stale.
- **Code checks** for C# (`cs_effect_check.py`, any .NET project through `dotnet test`) and Python (`py_effect_check.py`). A survivor
  counts as no hole only when compiled code proves it the same program, or a reason says why, resting on code that has not changed.
- **Argument chains:** a chain is a file an agent writes on purpose, listing its claims and their evidence. Each step is walked through the
  graph, and a citation of something the graph does not hold fails the run.
- **A graph neural network** (`graph_gnn.py`, optional) that proposes which requirement owns new code, and only speaks once it beats
  the plain rules on links it never saw.
- **An on/off harness** (`ab_loop.py`, `grade_panel.py`) that measures agents with the loop against agents without it, graded blind.
- **Telemetry** (`loop_stats.py`, `STATS.md`) whose trends pick the next improvement, and a page you can fly through to see any graph
  the loop wrote (`graph_flythrough.py`).
- **A skill for Claude Code** (`skill/SKILL.md`), and the same front door in the file layouts of other coding agents ([below](#other-coding-agents)). The tools are plain command-line Python and run without either.

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

The installer proves itself with the same command you can run again at any time, from the tools folder: `python -X utf8 closed_loop.py prove` (`prove --quick` is the front door only).
It runs every selftest and the whole loop on the examples, several at once at below-normal priority so the machine stays usable, and ends in one result: what passed, what failed, what gave no
answer, and what was not run and why. A selftest that needs a piece the machine lacks (PyTorch, scikit-learn) says so and is never counted as passed; so is an example that needs the .NET SDK. A
selftest that hangs is killed and named; one that exits 0 having checked nothing is named and is not a pass. On Python and PyYAML alone the Python example and every selftest that needs nothing else
run. Measured on a clean Windows home folder, on a machine other work was using: 28 to 62 seconds with Python and PyYAML alone, and 5 to 8 minutes the first time with PyTorch and the .NET SDK, most of it the C# example, whose first build downloads about 250 MB of packages.

### Other coding agents

The skill is written for Claude Code. `python install.py --agent NAME` (or `closed-loop-kit install --agent NAME`) writes the same front door (`grow`, `ask`), `run` and `prove` in the
file another agent reads, into the current project, and touches nothing else: no tools, no skill folder, no proof run.

- `codex`, `opencode`, `agents`: `AGENTS.md`, read by Codex, OpenCode and others that read AGENTS.md.
- `cursor`: `.cursor/rules/closed-loop.mdc`, a Cursor rule the agent applies when its description fits.
- `gemini`: `GEMINI.md`, read by Gemini CLI.
- `copilot`: `.github/copilot-instructions.md`, read by GitHub Copilot.
- `windsurf`: `.windsurf/rules/closed-loop.md`, a Windsurf model-decision rule. Windsurf's documentation now lists that folder as legacy but imported by default.

A file you already own (`AGENTS.md`, `GEMINI.md`, `copilot-instructions.md`) gets one block between two marker comments, replaced when it is there and appended when not; the rest is
left alone. A Cursor or Windsurf file of that name that is not ours is refused. The files are generated at build time from `skill/SKILL.md`, the one source, and carry its hash: `install.py --agent` refuses a file cut from an older SKILL.md. Each layout was
checked against that tool's own documentation (the pages, and the date they were read, are in `agents/layouts.json`). **No agent other than Claude Code has been tried:** these are the files the
documentation says each reads, valid and in the right place, and nobody has watched one of those agents follow them. The idea of one source written out in many agents' layouts is
seen in OpenSpec and Superpowers (both MIT); no code is taken from either.

## Try it

`example/` is a small wallet and store in C#; `example_py/` is the same store in Python. Each holds a complete model in `model/*.yaml`, and
each test names what it checks on the line above it (`/// CHECKS: W-1`). The whole brief, from the example's imaginary owner: "The wallet pays
only what it holds; a limited line sells out and an unlimited one never does; the display holds eight digits." After an install, from the
tools folder (`cd ~/.claude/closed-loop`):

```
python -X utf8 closed_loop.py prove                       # every selftest and both examples, together, one result
```

or one example by hand. `CLOSED_LOOP_ROOT` is the folder the model's paths are relative to, so it is set for that one command, in the way your shell sets a variable:

```
CLOSED_LOOP_ROOT=. python -X utf8 closed_loop.py run example_py/model --out example_py/CLOSED_LOOP.md          # bash, zsh, Git Bash
$env:CLOSED_LOOP_ROOT="."; python -X utf8 closed_loop.py run example_py/model --out example_py/CLOSED_LOOP.md   # PowerShell
set "CLOSED_LOOP_ROOT=." && python -X utf8 closed_loop.py run example_py/model --out example_py/CLOSED_LOOP.md  # cmd
python -X utf8 loop_stats.py show example_py/model
```

Exit 1 means a hole of yours is open; the owner's questions never fail a run. Read the report: survivors one per decision, the criteria table, the owner's
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

**On one C# game feature of ours**, on 2026-10-29:

- With one break per requirement the loop read "15 caught, 0 got past"; with every statement broken every way it read 89 of 340 getting
  past, on the same code. A loop that breaks code one way flatters.
- Over five passes, breaks made went 340 -> 806, breaks no test caught 89 -> 29 (most of the rest proven or explained equivalent), and
  statements held by a test 31 -> 169 of 173.
- Walking the use cases found 10 requirements nobody had written, beside the 17 already written.
- The graph neural network proposes owners for new code and abstains below 70% sure. On the tool's own code a plain rule beats it, MRR 0.65 against 0.72
  (a mean over five seeds). It beat the no-test-results rule only on the largest model tried, a game feature of ours that is not public, and only once about
  half of its code was claimed. On the 4-requirement examples the plain rule won at every fullness. Until your model is that full, read the plain rule's suggestions.

## Honest limits

- **The drafter is the reader, and the tool cannot read.** It checks that a citation exists and that the quoted words are in the sentence. It cannot check that a requirement means what its sentence means,
  and a thin draft makes thin questions: the queue is only as long as the chain the draft built. The questions that reach what a producer left out come from the needs the drafting agent infers while it
  walks each step, and from vague words in the text; the holes the tool finds by itself (a goal with no alternate path, the fields of a goal) are mostly questions a producer of a rough epic has little to say about.
  On two invented games, an agent told to ask clarifying questions, with no kit, reached more of the producer's rules than the queue did (one run each, a sign and not a measure). What the queue adds
  is the record (every requirement has its sentence or its reason, every answer its source, nothing is asked twice, a session resumes), and that is all it claims.
- **The loop cannot tell a wrong requirement from a right one.** Every break caught means the tests hold the code to the requirements as the agent read them. If the reading is wrong, the tests defend it. That is why asking comes first, and asking only moves the risk: a rule that two hearings agree on can still be wrong, and `elicit.py` cannot judge whether a person's answer is true, only whether their answers agree.
- **Two holes cannot be answered ABSENT in the rules.** The 19 rules have no place for "this goal has no alternate path, and here is why" or "there is no separate way to see the result"; those answers are
  recorded and the question closes, and the rules still list the hole, which the tool says.

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
