---
name: closed-loop
description: "Hold an AI's work to its owner's requirements: the requirements kept as a use case model (data, 19 rules), compiled into a knowledge graph that is the record (requirements, steps, code places, every mutant, every test, every step of each change's written argument), and held by 18 criteria - every statement of the covered code broken every way and caught by a test (the requirement's OWN test), equivalents proven, every line claimed by a requirement, each change's written argument (a chain of claims with their evidence) walked through the graph, a GNN proposing owners for new code once it beats the baselines, and an on/off harness measuring agents with and without it. For building larger systems where requirements must stay true across many changes - not for one-off prompts. Use when asked to grow requirements from an epic, a PRD, a ticket export or a transcript, to find what the requirements are missing, what to ask the producer, to close the loop, prove requirements are complete, check intention on each branch, check a change's argument against the graph, link code to requirements, compare an agent with and without the loop, or '/closed-loop'."
---

# Closed loop

Tools: `{TOOLS}` (installed by the kit's install.py). Read `{TOOLS}/README.md` once; it says WHEN this pays and when it does not.

## Start here: you have text and no model (an epic, a PRD, a ticket export, a transcript)

Two commands take it to the first question for the person who owns the text. No key, no .NET, five minutes.

```
python -X utf8 {TOOLS}/closed_loop.py grow <the text> --out <model folder>      # 1st run: numbers every sentence, writes BRIEF.md and a skeleton draft.yml, exits 3
# YOU are the reader. Edit <model folder>/draft.yml from the numbered sentences (BRIEF.md says how), then run the SAME command again:
python -X utf8 {TOOLS}/closed_loop.py grow <the text> --out <model folder>      # refuses a draft it cannot ground, or writes the model and prints the first question
python -X utf8 {TOOLS}/closed_loop.py ask <model folder>                        # the next question (exit 1), or "nothing is open" (exit 0)
python -X utf8 {TOOLS}/closed_loop.py ask <model folder> --confirm | --answer "<their words>" --check "<how it would be seen to hold>" | --absent "<reason>" | --later "<reason>"
```

- **Do not invent.** Every requirement in the draft quotes at least two words of a numbered sentence and cites its key, or is INFERRED with the reason a step of the chain needs it.
  A requirement that cites nothing, cites a sentence that is not there, quotes words the sentence lacks, or is inferred with a one-line reason, is REFUSED and nothing is written.
  A field the text does not state (a goal's trigger, pre, minimal, success, why) is left out of the draft; it becomes a question.
- **Ask the producer the sentence the tool prints, in those words.** It is plain words with no id in it. Then give the tool the reply exactly as the producer gave it. Never answer a question yourself.
  ABSENT needs a reason ("nothing there, because ..."). "Probably fine" records nothing.
- **Keep asking until it says nothing is open.** Each answer changes the model, and the next question comes from the new state; it never repeats one that has an answer on record,
  and a session that stops resumes where it was (all state is in `<model folder>/answers.yaml`). If you must stop early, `ask <model folder> --status` says how many are open: say so.
- The questions are ranked: two statements that disagree, then a need nobody authored, then a sentence of the text nothing captures, then a goal with no alternate path, then the fields of a goal.
- Then `run` (below) holds code to the requirements this grew. A grown model has no code yet; it is the requirements an agent will write to.

## Run it

```
python -X utf8 {TOOLS}/closed_loop.py run <model folder> [--chain C.yaml ...] [--out REPORT.md] [--gnn]
       [--pin-claims "<why>"] [--accept "<why>"] [--since EARLIER.md] [--overlay SANDBOX] [--note "<what this pass changed>"]
python -X utf8 {TOOLS}/loop_stats.py show <model folder>          # the morph log: every pass's numbers
python -X utf8 {TOOLS}/closed_loop.py prove [--quick]             # every selftest and example of this install, together; one result: passed, failed, unknown, not run and why
python -X utf8 {TOOLS}/graph_gnn.py eval <model>/../kg/<name>.sqlite --group method
python -X utf8 {TOOLS}/ab_loop.py harvest|sandbox|brief|outcome|grade|record|board
python -X utf8 {TOOLS}/doc_nodes.py what-to-update <node id> --db <model>/../kg/<name>.sqlite     # which documents describe this node
```
Set `CLOSED_LOOP_ROOT` to the project the model's paths are relative to. Exit 1 = a hole of ours; the owner's questions never fail it.

## The model

A folder of YAML: `meta.yaml` (the problem story, the loop block), `people.yaml` (actors, sources, persona questions),
`requirements.yaml`, `usecases.yaml` (an epic and its sea-level use cases), `spaces.yaml` (two closed spaces of different shapes).
`{TOOLS}/example/model` is a complete one. The loop block:

```yaml
  loop:
    closed: true
    code: {language: csharp | python, roots: [...], scope: [place prefixes, python only]}
    tests: [{runner: dotnet | py_selftest, test: <tests .csproj | "script --selftest">, root: <dotnet: folder>, covers: [...], reads: [...]}]
    entry: [{name, why}]                  # conditions set outside the model
    equivalent: [{mutant, why, rests_on: [Type.Method]}]   # mutant: the id under mutants in <model>/../kg/verdicts.json, Method@hash: what
    branches: {claims: [{method, reqs, why}], excluded: [{ref, why}]}
```
A method with overloads is `Type.Method~N` (the Nth in source order), an operator `Type.operator<`, a property with a body `Type.Name`. Anything the walk cannot reach (a field's value, an indexer) is named under NOT WALKED in the report.
A requirement: `{id, kind: authored|derived|inferred, sources, text, measure, effect: [{sets|calls|branches|reads: <name>, at: <place>}]}`.
A test names what it checks on the line above it: `/// CHECKS: W-1, W-2` (C#) - the mapping written WITH the test.

## What the verdicts mean

- **L14 KILLED / L15 PINNED** - every compiled mutant of a requirement's code, and of every statement, fails a test. One survivor
  is a hole: a missing test, or an equivalent to prove or explain.
- **L16 OWN-CHECK** - the break fails the requirement's OWN checks; a named check that catches none of its breaks is a hole.
- **L19 EQUIVALENT** - a survivor is no hole only when proven the same program (identical IL / bytecode) or explained with a reason
  whose ground (`rests_on`) has not changed.
- **L9 / claims.lock** - every statement claimed; a method claim is pinned to its statements and stales when they change.
- **Modal status** - authored requirements are musts, inferred ones mays only the owner settles; closed when no may is left.
- **The GNN** proposes only after beating the plain baselines on links it never saw, and only when it is sure enough.

## Rules

1. Mutation refutes closure; it never proves it. "Closed" is relative to the operators and to what the tests observe.
2. Never mutate the live project: schema copies and sandboxes only.
3. A widened check is proved by a plant it must catch, in the selftest, before it is believed.
4. Each pass: a fresh agent reviews it for ideas not yet tested; the morph log records it.
5. Visuals, timing and feel stay with a person.
6. **A change answers for the holes it ADDED.** Run the loop once before you change anything (`--out before.md`), then after with
   `--since before.md`: it lists the holes of ours your change added and closed, apart from those already there. Close the
   added ones; the ones already there are not your task (10-29: an agent without this drifted into fixing old holes, and
   the blind grader marked it scope creep).
7. **Numbers belong in the graph, never in the model.** Do not write run counts (tests passed, mutants killed) into requirements,
   cells or use cases: the loop measures them and keeps them in the graph and the morph log; a count in the model is a
   self-report nobody re-measures.
8. **Seed it first, then it flies.** Write as many sourced requirements as you can before relying on what the loop proposes,
   the pillars first: the rules that own the most code and run through the most methods. 10-29: the GNN beat the plain "effects
   in this method" ranker only on a 27-requirement model and only once about half its code was claimed; on 4-requirement models
   the plain ranker won throughout. A proposal below that fullness is the plain ranker's, whatever the GNN says.

## Requirements make requirements; self-improvement is on

- The loop extracts new requirements from existing ones: a step needing what nothing provides is an inferred requirement, and it goes to the owner as a question. Never settle one yourself.
- The more complete the requirements, the more accurate the prompt built from them. Where a feature joins two others, read both models side by side for the places they meet: the walk and the graph do not name those yet (one pair, not blind: the graph named 2 of 23, a reader 20).
- Self-improvement is on by default: each pass records a row and prints what the stats pick. `--no-self-improve`, or `self_improve: false` in the model's `loop:` block, turns it off.
