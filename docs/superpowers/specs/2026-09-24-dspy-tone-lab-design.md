# DSPy tone lab design

## Outcome

`/lab` publicly explains and demonstrates how Tone Your Mind improves its
Granite prompt against Jev. Production reads the same versioned prompt program
that the lab emits.

## Two loops

- The live loop remains bounded: source score, then at most three Granite
  rewrites, each rescored by Jev. Distance controls five feedback strengths and
  every later attempt multiplies the insistence.
- The offline loop uses DSPy MIPROv2 on fixed, non-user examples. Jev measures
  target distance and meaning retention in one structured review. The metric is
  85% tone accuracy and 15% meaning retention, making the target the clear
  priority without discarding the original proposition.

## Publication

The repository includes the fixed dataset, optimiser, selected prompt program,
and a compact result artifact. The lab page publishes the methodology,
baseline/selected aggregate scores, five distance bands, iteration escalation,
and representative trajectories. No user input or production log is retained.

## Production contract

The selected JSON prompt program is imported by the Worker. It contains the
system instruction, five distance bands, three retry multipliers, metric
weights, provenance, and version. Tests reject incomplete prompt programs.
