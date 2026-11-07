# L4: Inference supervision and label confusion

L4 measures a **supervision-regime response** (SRR): the extent to which logical-form recovery depends on which inference relations training observes and which labels it is given. The suite separates relation composition from annotation noise.

First, `regimes.yaml` begins with entailment-only training and increases the non-entailment share through four declared contamination levels. Contradiction and neutrality split that share equally, so the experiment changes the amount of negative evidence without changing its composition. Second, `label-noise.yaml` holds the relation mixture fixed and compares clean labels, two uniform flip rates, and an explicit asymmetric confusion matrix.

Every point uses the known relational grammar and truth-table logic in `l4-run.yaml`, whose fixed logical forms make entailment, contradiction, and neutrality reachable while the supervision treatment changes. The learner and matched-capacity boundary remain those of L3. The confusion report records achieved labels conditional on oracle labels rather than inferring noise from the requested rate. `reduced-freeze.yaml` executes four representative treatments and reports logical-form recovery beside the achieved relation proportions and confusion matrix.

The files have the following roles: `l4-run.yaml` owns the fixed relational source and learner; `regimes.yaml` owns the relation-composition design; `label-noise.yaml` owns the confusion design; their JSON files preregister the corresponding contrasts; and `reduced-freeze.yaml` supplies the executable reduced comparison.
