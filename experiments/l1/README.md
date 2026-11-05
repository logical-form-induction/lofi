# L1 known-grammar logical-form induction

This experiment asks whether inference judgments identify a lexical logical-form
assignment when the grammar and logic are fixed. The source declaration owns a
directed application grammar, a typed propositional language, and a lexical
interpretation used only to synthesize labeled pairs and held-out gold forms.
The learner receives pair tokens and inference labels. It receives no semantic
constraint source, so the word-to-form assignment is induced through the
declared inference grounding.

`induction.yaml` crosses the theorem-prover likelihood treatment with the
standard sentence-vector NLI treatment at two matched parameter budgets and
four seeds. It contains 16 points. Both treatments use the same form decoder,
known grammar, and known logic. The difference is the grounding: the primary
treatment proves decoded form pairs, while `nli-baseline` predicts from sentence
vectors without using a decoded form to produce its label.

Logical-form evaluation fits a many-to-one symbol alignment on development and
freezes it for test. It reports exact normal-form match, aligned accuracy, the
random-relabeling floor, recovery relative to the logical vocabulary's fiber,
fixed-vocabulary accuracy, and fiber size. This distinguishes failure to recover
a meaning from a relabeling that inference judgments cannot identify. The same
runs report supertagging, labeled dependencies, and aligned lexical categories
from the published category readout.

`lm-prover.yaml` and `logic-lm.yaml` are the two LM-plus-prover baselines. They
replay repository-recorded raw formalizations through the common truth-table
proof service. The LINC treatment votes over three independent candidates; the
Logic-LM treatment consumes solver feedback before selecting a repaired
formalization. The committed traces contain no scores. LoFI parses, type-checks,
proves, aligns, and scores them at execution time, while the prompt, data, and
generation files are identified by content digests.

The reduced freeze comprises two neural points and one point from each
LM-plus-prover pipeline. The neural points are in `reduced-freeze.yaml`; the
replayed baselines are in `lm-prover-reduced.yaml` and
`logic-lm-reduced.yaml`. All four use the same declared grammar, logical
vocabulary, and truth-table semantics; the replayed baselines share one fixed
six-pair evaluation population. Thus the complete reduced treatment runs
without an external prover. After freeze 1 has been minted, launch the neural
comparison with:

```bash
uv run lofi launch \
  --experiment experiments/l1/reduced-freeze.yaml \
  --freeze freezes/1/FREEZE.json --out launch.json
```

Run the two serial baseline-fit documents through the same freeze, then ingest
the three experiment result roots. The resulting table places
`fiber-relative-recovery` for theorem-prover induction and the NLI arm beside
the logical-form and inference measurements computed for LINC and Logic-LM.
