# The standing tick: self-improvement you schedule

Every pass already records its own stats and prints what to improve next. A tick goes one step further: a fresh agent, with no memory
of the last one, takes ONE item, measures the same stat before and after, and records whether it was worth it. Nothing starts a tick
for you. Schedule one (a scheduled task, cron, your own harness) when you want the tool to keep improving, and expect it to cost
real tokens: our first two ticks took about 200,000 and 290,000 by the app's count. Stop scheduling it, or create `.loop_tick.STOP`
beside these tools, and it stops.

Give the agent this file and run these in order. `<model>` is your model folder, `<tools>` this folder.

## 1. Start

    py -X utf8 <tools>/loop_tick.py start

Exit 3 BUSY (another tick holds the lock), 4 STOPPED or 5 IDLE (the last three ticks were worth "no" and nothing changed since): stop at
once. MODE LIGHT means a pass or a tune is running: change no tool this tick. Between steps run `loop_tick.py beat`.

## 2. Health

    py -X utf8 <tools>/<tool>.py --selftest                for each tool
    py -X utf8 <tools>/telemetry_audit.py --root <tools>   every stat must have a reader

A red selftest is the item. So is a stat nothing reads: give it a reader or stop recording it. Never weaken a planted mistake to make a
selftest pass.

## 3. Choose one item

    py -X utf8 <tools>/improvement.py check <model>        the triggers that fire on your latest passes
    py -X utf8 <tools>/loop_stats.py trends <model>        what the stats pick

In this order: a red selftest, an unread stat, a fired trigger, what the stats pick. The start of an epoch (every 12 ticks, `start` says
so) is a brainstorm instead: write ranked ideas down, each with the number that would prove it.

## 4. Do it

Measure the stat with one command, change one thing, measure with the same command, and look at what it made. A long run goes through
`loop_tick.py launch -- <command>`, so it outlives this session.

## 5. Benchmark, at the end of an epoch

    py -X utf8 <tools>/gnn_tasks.py eval <graph.sqlite>    needs PyTorch: does each learned part beat its plain rule here?
    py -X utf8 <tools>/run_plan.py <model> --write         which learned parts a pass runs here, on this project's own evidence, and which are only measured
    py -X utf8 <tools>/improvement.py graph <model>        did the improvements work?

## 6. Finish

    py -X utf8 <tools>/loop_tick.py finish --item <id> --minutes <n> --found "<one measured line>" --worth yes|no|unknown --targets <stat>

Say `worth no` honestly: it is what keeps an idle loop from spending. Commit only the files you changed. Send nothing to anyone.
