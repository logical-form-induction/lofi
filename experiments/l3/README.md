# L3: Logical-form distributions and ambiguity

L3 measures a **semantic-factor response** (SFR): the extent to which logical-form recovery changes when the data-generating semantics changes while the learner, grammar, and logic remain fixed. The suite separates two questions.

First, `distribution.yaml` crosses premise and hypothesis length laws with the size and diversity of the logical-form inventory, the coupling between logical-form class size and token frequency, and the within-class rank law. This design distinguishes an inventory-size effect from a frequency effect. Second, `ambiguity.yaml` crosses the amount and spread of lexical polysemy with whether extra meanings fall on rare or frequent forms and with a scope-reading target.

Every point learns lexical meanings from inference labels under the shared grammar and truth-table logic in `l3-run.yaml`. Recovery is reported relative to an alignment floor, and the run record reports the semantic inventory and its achieved class-frequency and polysemy measurements. `reduced-freeze.yaml` exercises four yoked semantic treatments through the same production training and reporting path.

The files have the following roles: `l3-run.yaml` owns the learner and source boundary; `distribution.yaml` and `ambiguity.yaml` own the two production designs; their JSON files preregister the corresponding factors and contrasts; and `reduced-freeze.yaml` supplies the executable reduced comparison.
