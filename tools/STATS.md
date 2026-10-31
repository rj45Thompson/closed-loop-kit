# The closed loop's telemetry: what each stat is, why it measures growth, how it is computed

Telemetry is a pillar of the loop. Any stat that measures growth or improvement is measured, each stat says what it is, why and how,
and each loop brainstorms what else to measure and adjusts the set (2026-10-29).

Every `closed_loop.py run --out R.md` appends one row to `<model>/../stats.jsonl` (the MORPH LOG), written by
`loop_stats.py` (`PassStats` reads, `MorphLog` keeps). Every number is READ from what the pass wrote (verdicts.json, the
graph, the model, the report); a number the pass did not write is absent from the row, never 0. `loop_stats.py show <model>` prints
the curve. Rows 0-5 of the shops' log were backfilled from the earlier reports (each says so in `backfilled_from`).

## Growth comes in two directions, and both must be measured

A loop can look better by WIDENING less, and look worse by widening more. So every stat below is one of:
- **NET** - how hard the pass tried to break things (should grow, then hold);
- **CLOSURE** - how much of what it tried is caught, claimed or answered (should grow towards the net);
- **OPEN** - what is still not closed, and whose it is (should shrink, the owner's part only by the owner);
- **COST** - what a pass costs (should not grow faster than the net);
- **TOOL** - the tool itself: its size and how much of it its own plants prove.

A pass where NET grew and OPEN grew too is progress (the net found more), not regress. Read the two together.

## The stats

| stat | kind | why it measures growth | how |
|---|---|---|---|
| `mutants` | NET | how many ways the pass broke the code | verdicts.json `mutants` |
| `operator_kinds` | NET | how many KINDS of break (a wider net, not a bigger one) | distinct `op_kind(what)` |
| `decisions` | NET | how much code is held to a requirement (cells of the branch space) | verdicts.json `branches` |
| `killed` | CLOSURE | breaks a test caught | mutants with `killed: true` |
| `survived_unexplained` | OPEN | breaks nothing caught and nobody explained: missing tests | survivors not declared equivalent |
| `equivalent_tce` | CLOSURE | survivors PROVEN the same program (same IL) | `equivalent: TCE` |
| `equivalent_declared` | OPEN-ish | survivors explained by a person's reason - a claim, not a proof | meta.loop.equivalent |
| `equivalent_lapsed` | OPEN | explanations whose ground changed | rests_on fingerprints moved |
| `no_verdict` | OPEN | breaks that did not compile or that no test covers | `killed: null`, not TCE |
| `kill_rate` | CLOSURE | caught / (caught + unexplained survivors) | from the above |
| `decisions_pinned` | CLOSURE | decisions every one of whose breaks was caught | branches `killed: true` |
| `effects_killed` / `effects_shape_only` | CLOSURE / OPEN | requirements held by behaviour vs by shape only | assertions' `kill` |
| `pillar_top3_share` / `pillars_80` | TOOL | how few requirements own the code: a run of rises is code piling onto the same few | owner_pillars: branch-to-requirement links in the graph |
| `survivors_by_operator` | where gaps are | which operators find the holes: where tests are weakest | survivors per `op_kind` |
| `kills_by_operator` | where tests bite | which operators the tests already catch | kills per `op_kind` |
| `tests`, `tests_idle`, `test_kills_top` | CLOSURE | how many tests, how many catch nothing, the strongest five | verdicts `tests` + kills by test |
| `requirements`, `musts`, `mays` | OPEN | the modal reading: closed when no may is left | model kinds (inferred = may) |
| `use_cases`, `claims` | NET | how much of the product the model covers | the model |
| `graph_nodes`, `graph_edges`, `graph_by_kind` | NET | the record's size, by what it holds | the sqlite store; the two counts leave out document, stat and goal nodes and their links, `graph_by_kind` shows them |
| `holes` (per criterion), `holes_ours` | OPEN | what still lets a mistake through, per criterion | the report's LOOP table |
| `gnn_mrr` (gnn and each baseline) | CLOSURE | whether the proposer links new code better than the baselines | the report's section 6 |
| `chain` (steps, ungrounded, blind) | CLOSURE / OPEN | the AI's reasoning held to the graph | the report's section 5 |
| `loop` | - | HOLDS or BROKEN | the report's last line |
| `seconds` | COST | how long a pass took | the run's wall clock |
| `tool_lines`, `tool_commit` | TOOL | how big the tool is, at which commit | the autobot files, git |

## Interesting so far (the shops, passes 0-6)

- The first upgrade read **"15 killed, 0 survived"**: one break per effect. Every break of every decision read **89 of 340
  survived** - the same code, a wider net. A single-mutant loop flatters.
- Survivors went **89 -> 13 -> 69 -> 28 -> 29** while mutants went **340 -> 340 -> 631 -> 772 -> 806**: each widening found more,
  each round of tests closed most of it. The curve is a saw, and that is what a loop that keeps widening looks like.
- Effects checked by SHAPE only fell **15 -> 1** in one move (the view's decisions moved into the screen's own class): structure, not tests, was
  the limit.
- The operators that found the most real gaps were the ones added last (call arguments: culture; text: messages; vacuity:
  guards forced true). Each new operator kind found something the older ones could not.

## Brainstorm: stats not measured yet (each pass adds to this list, and moves what it built up into the table)

- **Time to close**: passes between a survivor first appearing and its kill - how fast the loop closes what it finds.
- **Mutant subsumption**: dominator mutants per decision (the minimal set of questions) - prior art: Ammann et al. 2014.
- **Test specificity**: how many requirements' code each test kills - a test that kills everything checks nothing in particular.
- **Owner latency**: how long each may (inferred requirement) waits for its owner; how many the owner settled per week.
- **Requirement churn**: requirements added / changed / retired per pass, and the lock's STALE count over time.
- **GNN calibration**: precision of its top proposal at each confidence; how often a proposal was accepted.
- **A/B piles per pass**: failure piles, on vs off, as more pairs are graded.
- **Tokens per pass** and per closed hole (needs the agents' usage numbers; the "up to 90% saved" goal is a TARGET until measured).
- **Self-model stats**: the same rows for the tool's own model (its selfmodel folder) - the loop measured on itself.
- **Equivalent share**: declared vs proven equivalents over time - a rising declared share is a warning, not progress.
