# Syntactic anchoring campaign

This campaign asks whether clamping a small set of lexical categories to their
gold values improves recovery of a synthetic categorial lexicon.
`anchoring.yaml` compares four selection mechanisms at budgets of a quarter, a
half, and three quarters. Rank policies take the most frequent share of lexical
forms. Quantile policies cover the corresponding share of observed token mass.
Closed-category and open-category policies order the observed categories by how
many forms bear them, smallest first for the closed role and largest first for
the open role, and take the forms observed under the stated share of that
order. A form observed under several categories is clamped to their union.

Each selection policy is crossed with the coupling between form frequency and
category class size in the generating lexicon, at -0.75, 0, and 0.75. The
source grammar is a directed application grammar over four primitives, with a
power-law category share, a power-law rank-frequency law, and a mean of 1.5
categories per form. The mechanisms need not attain identical corpus mass at
the same nominal budget. Analyses therefore read the achieved selection record
of every point, which gives the types pinned, the tokens covered, the mass
share, and the frequency-class concordance, rather than substituting the
nominal budget.

Every point learns its categories under one fixed parameter budget, with the
gold lexicon pinning the selected forms through a pinning objective beside the
grammaticality and coherence terms. The lexical-recovery evaluator reports
lexical-type F1 and many-to-one primitive accuracy, fitted on the development
split and scored on the test split, together with its random-relabeling floor.
`anchoring.json` preregisters one contrast: primitive accuracy against that
floor. The production design expands to 144 points under four seeds.

`reduced-freeze.yaml` is the executable local contraction. It fits the rank,
token-mass-quantile, and closed-category mechanisms at the half-share budget
under both signs of the coupling, on sixteen sentences and one seed. Its test
trains all six points through the production executor and reads each closed
point's achieved selection budget beside its recovery and floor rows.
