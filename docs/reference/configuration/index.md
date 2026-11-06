# Configuration documents

This page maps each section of a run, experiment, baseline fit, judgment fit
and active learning document, and of a site profile, onto the declaration class
that validates it and the reference page that renders that class.

Each root class is closed, so a key it does not declare is refused. A tagged
field selects its variant by `kind`, and the variants of each family are
the names registered in its [entry point group](../entry-points.md). The
[configuration guide](../../how-to/setup/configuration.md) explains how a
document is composed from packaged groups; `lofi configs` lists those groups and
`lofi show-config` prints the resolved document (see
[Command line](../cli.md)).

| Document | Root class | Rendered on | Checked-in example |
| --- | --- | --- | --- |
| Run document | `RunSpec` | [Spec](../spec.md) | `docs/examples/synthetic-parser.yaml`, `lofi/configs/run/` |
| Experiment document | `ExperimentDocument` | [Run](../run.md) | `experiments/s1/bisk-hockenmaier.yaml` |
| Baseline fit document | `BaselineFitSpec` | [Run](../run.md) | `experiments/s1/bisk-hockenmaier-fit.json` |
| Judgment fit document | `JudgmentFitSpec` | [Judgment](../judgment.md) | `docs/examples/models-training/active-learning/fit.yaml` |
| Active learning document | `ActiveLearningSpec` | [Training](training.md) | `docs/examples/models-training/active-learning/active-learning.yaml` |
| Site profile | `SiteProfile` | [Run](../run.md) | `docs/examples/site-profile.toml` |

An experiment document's `config` names the document its points are composed
over, and its `kind` (`run`, `baseline-fit`, `judgment-fit` or
`active-learning`) selects which of the four point documents that is.

## Run document

`RunSpec` validates the top level of a run document.

| Section | Required | Class | Rendered on |
| --- | --- | --- | --- |
| `model` | yes | `ModelSection` | [Spec](../spec.md); variants on [Model](model.md) |
| `loss` | no | `LossSection` | [Spec](../spec.md); variants on [Training](training.md) |
| `data` | yes | `DataSection` | [Spec](../spec.md); variants on [Datasets](datasets.md) |
| `trainer` | yes | `TrainerSpec` | [Training](training.md) |
| `release` | yes | `ReleaseSpec` | [Spec](../spec.md) |
| `parse` | yes | `ParseSpec` | [Spec](../spec.md) |
| `approximation` | no | `ApproximationSpec` | [Spec](../spec.md) |
| `type_sampler` | no | `TypeSamplerSpec` | [Training](training.md) |
| `pair_sampler` | no | `PairSamplerSpec` | [Training](training.md) |
| `optimizer` | yes | `OptimizerSpec` | [Training](training.md) |
| `device_policy` | yes | `DevicePolicySpec` | [Spec](../spec.md) |
| `scheduler` | no | `SchedulerSpec` | [Training](training.md) |
| `sinks` | no, keyed by role | `SinkSpec` | [Evaluation](evaluation.md) |
| `probes` | no, keyed by role | `ProbeSpec` | [Spec](../spec.md) |
| `counterfactuals` | no | `CounterfactualsSpec` | [Spec](../spec.md) |
| `evaluators` | no, keyed by role | `EvaluatorSpec` | [Evaluation](evaluation.md) |
| `comparison` | no | `SourceComparisonPointSpec` | [Spec](../spec.md) |
| `signature_assignment` | no | `SignatureAssignmentsSpec` | [Spec](../spec.md) |

### The `model` section

| Field | Required | Class | Rendered on |
| --- | --- | --- | --- |
| `language` | yes | `ModelLanguageSpec` | [Spec](../spec.md); languages on [Logical form language](language.md) |
| `capacity` | no | `ParameterCapacitySpec` | [Spec](../spec.md) |
| `parser` | no | `ParserSpec` | [Model](model.md#parsers) |
| `token_encoder` | no | `TokenEncoderSpec` | [Model](model.md#tokens) |
| `subword_pooler` | no | `SubwordPoolerSpec` | [Model](model.md#poolers) |
| `controller` | no | `ControllerSpec` | [Model](model.md#controllers) |
| `grounding` | no | `GroundingSpec` | [Model](model.md#groundings) |
| `encoders` | no, keyed by role | `TermEncoderSpec` | [Model](model.md#encoders) |
| `decoders` | no, keyed by role | `TermDecoderSpec` | [Model](model.md#decoders) |
| `spaces` | no, keyed by sort | `SpaceSpec` | [Spec](../spec.md) |
| `readouts` | no, keyed by role | `ReadoutSpec` | [Model](model.md#readouts) |
| `span_decoders` | no, keyed by role | `SpanDecoderSpec` | [Model](model.md#span-decoders) |
| `actions` | no | `ActionsSpec` | [Spec](../spec.md) |

### The `data` section

`DataSection` has one field, `datasets`, which maps each training dataset role
to a `DatasetSection` and must name at least one.

| Field of a dataset role | Required | Class | Rendered on |
| --- | --- | --- | --- |
| `dataset` | yes | `DatasetSpec` | [Datasets](datasets.md#datasets) |
| `featurizer` | no | `FeaturizerSpec` | [Datasets](datasets.md#featurizers) |
| `constraint_source` | no | `ConstraintSourceSpec` | [Spec](../spec.md) |
| `constraint_policy` | no | `ConstraintPolicySection` | [Spec](../spec.md) |

A dataset that declares a grammar does so in its `grammar` field, which
[Grammar declarations](grammar.md) renders.

### The `loss` section

`LossSection` has one field, `terms`: a list of `WeightedLossSpec` values, at
least one and no two under one name. Each term's objective is a `LossSpec`
variant, rendered on [Training](training.md#losses).

## Experiment document

`ExperimentDocument` validates an experiment document. Paths in it are relative
to the directory holding the document. `lofi.run` exports the class and every
section class the table names.

| Section | Required | Class | Rendered on |
| --- | --- | --- | --- |
| `role` | yes | a nonblank name | |
| `freeze_id` | yes | a string of the form `freeze/<n>` | |
| `config` | yes | `ExperimentConfigSpec` | [Run](../run.md) |
| `launch` | yes | `LaunchSpec` | [Run](../run.md) |
| `failure` | yes | `FailurePolicySpec` | [Run](../run.md) |
| `sweep` | yes | `Sweep` | [Run](../run.md) |
| `signature_twins` | no | `SignatureTwinsSpec` | [Run](../run.md) |
| `seed_policy` | yes | `ExperimentSeedSpec` | [Run](../run.md) |
| `rows` | yes | `RowReference` list | [Run](../run.md) |
| `reports` | no | `ReportSpec` list | [Run](../run.md) |
| `reducers` | no | `ReducerSpec` list | [Spec](../spec.md) |
| `primary` | yes | `RowReference` | [Run](../run.md) |
| `preregistration` | yes | a path | |
| `factors` | no, keyed by factor | `FactorSpan` | [Run](../run.md) |
| `compute` | yes | `ComputeSpec`: `UnpricedCompute` or `ExperimentCompute` | [Run](../run.md) |
| `pricing` | no | `PricingSpec`: `StackBootstrapPricing` or `LengthBatchPricing` | [Run](../run.md) |
| `contraction` | yes | `ExperimentContraction`, whose rungs are `RungSpec` variants | [Run](../run.md) |
| `stop_rules` | yes | `StopRuleSpec` list, each `when` a `PredicateSpec` | [Run](../run.md) |
| `decision_point` | yes | `ExperimentDecision`, whose `signal` is a `PredicateSpec` | [Run](../run.md) |
| `evaluation` | yes | `EvaluationSpec`, whose `correction` is a `CorrectionSpec` | [Run](../run.md) |

## Baseline fit document

`BaselineFitSpec` has one field, `system`, a `BaselineSystemSpec` variant
rendered on [Evaluation](evaluation.md#baselines). The system must declare at
least one named metric and no metric twice.

## Judgment fit document

`JudgmentFitSpec` validates a judgment fit document. Every field is required;
a field typed `| None` is written as `null` when it is absent.

| Field | Class | Rendered on |
| --- | --- | --- |
| `dataset` | a nonblank dataset identifier | |
| `response` | `ResponseDeclaration`, written under one constructor key: `ResponseDeclaration_responsespec` (`ResponseSpec`) or `ResponseDeclaration_boundedresponsespec` (`BoundedResponseSpec`) | [Judgment](../judgment.md) |
| `inference` | `FitInference`: `NutsInference`, `VariationalInference` (with a required `learning_rate`) or `MapInference` | [Judgment](../judgment.md) |
| `elicitation` | `Elicitation` | [Judgment](../judgment.md) |
| `calibration` | `JudgmentFitCalibration` (with a required `guide_learning_rate`) or `null` | [Judgment](../judgment.md) |
| `comparison` | `ComparisonSpec` or `null` | [Judgment](../judgment.md) |
| `dynamic_range` | `DynamicRangeSpec` or `null` | [Judgment](../judgment.md) |
| `ablation` | a list of `TableAblationSpec` | [Judgment](../judgment.md) |

## Site profile

`SiteProfile` validates the site profile that the `LOFI_SITE_PROFILE`
environment variable names.

| Field | Class | Rendered on |
| --- | --- | --- |
| `submission` | `SiteSubmission` | [Run](../run.md) |
| `placement` | `SitePlacement` | [Run](../run.md) |
| `storage` | `SiteStorage` | [Run](../run.md) |
| `environment` | `SiteEnvironment` | [Run](../run.md) |
| `interpreter` | `SiteInterpreter` | [Run](../run.md) |
| `transport` | `SiteTransport` | [Run](../run.md) |
| `allocation` | `SiteAllocation` | [Run](../run.md) |
