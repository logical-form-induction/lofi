# Lexical distribution and ambiguity campaigns

These experiments ask how the distribution from which a synthetic categorial
grammar is sampled changes lexical-type recovery. `lexical-distribution.yaml`
crosses the sentence-length law, form inventory size, category-inventory
diversity, category token-share law, form rank-frequency law, and the signed
coupling between form frequency and category class size. The category-share,
form-frequency, and ambiguity marginals remain fixed across the coupling
intervention. Every point uses the grammar drawn with the data as the learner
language, and a frozen capacity pad keeps total model size fixed when an
inventory changes the natural parameter count.

The power-law category-share cells identify the signed coupling intervention.
Uniform category shares are retained as a distributional control: because a
constant category-share marginal has no rank variation, its achieved coupling
may be undefined. In every cell, the analysis reads the measured coupling and
concordance rather than substituting the requested level.

`ambiguity.yaml` holds the form and category inventories, category shares,
rank-frequency law, length law, and parser fixed. It crosses (i) mean
categories per form, (ii) dispersion from many mildly ambiguous forms to a
few highly ambiguous forms, and (iii) the signed placement of ambiguous forms
in the frequency order. Thus the ambiguity treatments are changes to the
synthetic lexicon itself, not labels attached to otherwise equal points.

Both campaigns score development-fitted, test-frozen many-to-one primitive
alignment. The emitted observed row is paired with the evaluator's empirical
random-relabeling floor. Corpus metrics record the realized vocabulary,
category inventory, length histogram, ambiguity histogram,
ambiguity-frequency correlation, rank-frequency exponent, fitted category
coupling, and rank-class concordance. Analyses should use those achieved
values rather than treating the requested settings as observations.

`reduced-freeze.yaml` is the executable local contraction. It runs both signs
of the ambiguity-frequency treatment under four independent point seeds. Its
test executes the production point path, reads the evaluator's observed and
floor rows, and derives achieved-factor records from each closed metrics
record.
