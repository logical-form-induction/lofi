# Logical-form anchoring campaign

This campaign asks whether a small set of lexical logical-form clamps improves
held-out meaning recovery. `anchoring.yaml` compares three selection
mechanisms. Rank policies take the most frequent quarter, half, or three
quarters of lexical forms. Quantile policies cover the corresponding shares of
observed token mass. Constant-class policies select cumulative semantic
classes at the same nominal budgets: constants; constants and predicates; or
constants, predicates, and connectives.

The source interpretation also contains an identity class, so none of the
class treatments selects every logical form. The three selection mechanisms
need not attain identical corpus mass. Analyses therefore use the achieved
`selection.mass_share` and the class-wise selection record emitted for every
point rather than substituting the nominal budget.

Every point learns both category and logical-form term spaces under one fixed
parameter budget. The semantic gold source anchors constrained chart cells to
the complete terms encoded through the `forms` readout's semantic space. The
logical-form evaluator reports normal-form matching, fitted alignment, its
random-relabeling floor, and fiber-relative recovery on the held-out split.

`reduced-freeze.yaml` is the executable local contraction. It fits the rank,
token-mass-quantile, and constant-class mechanisms at the half-share budget,
then reads each closed point's achieved logical-form budget beside its recovery
rows.
