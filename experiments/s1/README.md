# Known-grammar lexical induction

The CCGbank campaign holds the packaged grammar fixed and varies how its
lexicon is learned. The executable learner treatments are unsupervised
induction, labeled outside reconstruction, a type-level tag dictionary, and
per-occurrence lexical supervision with either a recursive or Prange-style
factored linear decoder. Each treatment is crossed with tree-RNN,
holographic, and structurally biased Transformer term encoders at 250,000 and
500,000 matched parameters under four point seeds.

Held-out evaluation has three declared routes. Lexical diagnostics report
supertag accuracy overall and within exhaustive training-frequency and gold
ambiguity strata. Structural diagnostics report labeled directed dependency
F-score. Lexical-type diagnostics fit a one-to-one label alignment on the
development partition and freeze that alignment before scoring the test
partition. The resulting primitive and constructor recovery rows are paired
with empirical random-relabeling floors.

The synthetic campaign samples strings and lexical categories from the same
packaged CCGbank grammar used by the learner. It crosses corpus sizes of 512,
2,048, and 8,192 sentences with uniform short and negative-binomial long-tail
length laws, the same three encoders, the same two capacity targets, and four
point seeds. Thus its 144 points isolate data scale and length distribution
without changing the grammar supplied to the learner.

The Bisk--Hockenmaier campaign uses the package's independent baseline-fit
boundary. Its production document expects licensed CCGbank AUTO files and the
dependency reference data under `data/bisk-hockenmaier/`. The repository's
reduced document substitutes a one-sentence fixture and exercises the same
fit and scoring implementation.

`reduced-freeze.yaml` is the local neural contraction. It runs one seed of
each executable CCGbank learner treatment against the repository CCGbank
fixture and requires every declared evaluator row to reach the run store.
`bisk-hockenmaier-reduced.yaml` separately executes the published induction
baseline on its repository fixture.
