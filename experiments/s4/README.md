# Supervision and noise campaigns

These experiments ask what three string-supervision regimes contribute to
lexical-category recovery and where recovery breaks down as observations are
corrupted. The regimes use the current synthetic-corpus boundary. A
positive-only point contains only strings observed as grammatical. A
certified-negative point adds strings proved non-derivable by the generating
grammar at a declared proportion. A labeled-string point presents the same
binary grammaticality labels after independent symmetric flips.

The three regime-specific run documents extend the packaged CCG supervision
fragments and apply one shared capacity-matched S4 model and synthetic source.
This keeps the regime semantics at the package boundary while making the
lexical interventions comparable across campaigns.

`positive-only.yaml` varies the share of certified bad strings that are
observed as positive, from a clean sample through 50 percent contamination.
`certified-negatives.yaml` varies the share of bad strings whose certified
negative labels are retained. `labeled-strings.yaml` holds that share fixed
and varies the symmetric label-flip rate through 40 percent. Thus each noise
level changes the realized training data rather than serving as an analysis
label.

Every ladder is crossed with the same six contractions of the S3 design. The
profiles cover inventory size and diversity, both signs and zero of the
fixed-marginal frequency-to-category coupling intervention, and concentrated
high-frequency ambiguity. Analyses read the achieved corruption, per-class
flip rates, category coupling, concordance, and ambiguity-frequency
correlation from the closed metrics records. The first noise level at which
primitive recovery no longer exceeds the empirical relabeling floor is the
breakdown point for that lexical profile.

`reduced-freeze.yaml` is the executable local contraction. It fits a clean
positive-only point, a contaminated positive-only point, a certified-negative
point, and clean and noisy labeled points. Its integration test reads the real
lexical-recovery rows and the achieved corruption and noise reports from all
five closed records.
