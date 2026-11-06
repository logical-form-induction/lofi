# Configuration documents

To change a run's objective, locate `loss` in the [run document table](#run-document)
and follow its link to [Training](training.md#losses). The document key names
the setting; its declaration class specifies accepted values. This table
connects those configuration and Python views.

“Required” means the resolved document must contain the field. “Nullable”
means its value may be `null`; that field can still be required. A default
supplies an omitted value where the declaration allows it.
[Composition](../../explanation/documents-are-the-dsl.md#composition) can supply
fields from fragments. Many variants use `kind` to select an alternative;
[grammar declarations](grammar.md) use `formalism`. The selected alternative
determines which other fields are accepted.

This page maps each section of a run, experiment, baseline fit, judgment fit
and [active learning](../../how-to/training/acquire-labels-actively.md) document, and of a site profile, onto the declaration class
that validates it and the reference page that renders that class.

Each root class is [closed](../../explanation/boundaries.md#where-declared-values-enter), so a key it does not declare is refused. A [tagged
field](../../explanation/swappable-assumptions.md#every-choice-is-a-declared-value) selects its variant by `kind`, and the variants of each family are
the names registered in its [entry point group](../entry-points.md). The
[configuration guide](../../how-to/setup/configuration.md) explains how a
document is composed from packaged groups; [`lofi configs`](../cli.md#configs) lists those groups and
[`lofi show-config`](../cli.md#show-config) prints the resolved document (see
[Command line](../cli.md)).

| Document | Root class | Rendered on | Checked-in example |
| --- | --- | --- | --- |
| Run document | [`RunSpec`](../spec.md#lofi.spec.RunSpec) | [Spec](../spec.md) | [`docs/examples/synthetic-parser.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/synthetic-parser.yaml), [`lofi/configs/run/`](https://github.com/logical-form-induction/lofi/tree/main/lofi/configs/run/) |
| Experiment document | [`ExperimentDocument`](../run.md#lofi.run.ExperimentDocument) | [Run](../run.md) | [`experiments/s1/bisk-hockenmaier.yaml`](https://github.com/logical-form-induction/lofi/blob/main/experiments/s1/bisk-hockenmaier.yaml) |
| Baseline fit document | [`BaselineFitSpec`](../run.md#lofi.run.BaselineFitSpec) | [Run](../run.md) | [`experiments/s1/bisk-hockenmaier-fit.json`](https://github.com/logical-form-induction/lofi/blob/main/experiments/s1/bisk-hockenmaier-fit.json) |
| Judgment fit document | [`JudgmentFitSpec`](../judgment.md#lofi.judgment.JudgmentFitSpec) | [Judgment](../judgment.md) | [`docs/examples/models-training/active-learning/fit.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/models-training/active-learning/fit.yaml) |
| Active learning document | [`ActiveLearningSpec`](training.md#lofi.spec.acquisition.ActiveLearningSpec) | [Training](training.md) | [`docs/examples/models-training/active-learning/active-learning.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/models-training/active-learning/active-learning.yaml) |
| Site profile | [`SiteProfile`](../run.md#lofi.run.SiteProfile) | [Run](../run.md) | [`docs/examples/site-profile.toml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/site-profile.toml) |

An experiment document's `config` names the document its points are composed
over, and its `kind` (`run`, `baseline-fit`, `judgment-fit` or
`active-learning`) selects which of the four point documents that is.

## Run document

[`RunSpec`](../spec.md#lofi.spec.RunSpec) validates the top level of a run document.

| Section | Required | Class | Rendered on |
| --- | --- | --- | --- |
| `model` | yes | [`ModelSection`](../spec.md#lofi.spec.ModelSection) | [Spec](../spec.md); variants on [Model](model.md) |
| `loss` | no | [`LossSection`](../spec.md#lofi.spec.LossSection) | [Spec](../spec.md); variants on [Training](training.md) |
| `data` | yes | [`DataSection`](../spec.md#lofi.spec.DataSection) | [Spec](../spec.md); variants on [Datasets](datasets.md) |
| `trainer` | yes | [`TrainerSpec`](components.md#lofi.spec.components.trainer.TrainerSpec) | [Training](training.md) |
| `release` | yes | [`ReleaseSpec`](../spec.md#lofi.spec.ReleaseSpec) | [Spec](../spec.md) |
| `parse` | yes | [`ParseSpec`](../spec.md#lofi.spec.ParseSpec) | [Spec](../spec.md) |
| `approximation` | no | [`ApproximationSpec`](../spec.md#lofi.spec.ApproximationSpec) | [Spec](../spec.md) |
| `type_sampler` | no | [`TypeSamplerSpec`](components.md#lofi.spec.components.trainer.TypeSamplerSpec) | [Training](training.md) |
| `pair_sampler` | no | [`PairSamplerSpec`](components.md#lofi.spec.components.trainer.PairSamplerSpec) | [Training](training.md) |
| `optimizer` | yes | [`OptimizerSpec`](components.md#lofi.spec.components.trainer.OptimizerSpec) | [Training](training.md) |
| `device_policy` | yes | [`DevicePolicySpec`](../spec.md#lofi.spec.DevicePolicySpec) | [Spec](../spec.md) |
| `scheduler` | no | [`SchedulerSpec`](components.md#lofi.spec.components.trainer.SchedulerSpec) | [Training](training.md) |
| `sinks` | no, keyed by role | [`SinkSpec`](components.md#lofi.spec.components.run.SinkSpec) | [Evaluation](evaluation.md) |
| `probes` | no, keyed by role | [`ProbeSpec`](components.md#lofi.spec.components.run.ProbeSpec) | [Spec](../spec.md) |
| `counterfactuals` | no | [`CounterfactualsSpec`](../spec.md#lofi.spec.CounterfactualsSpec) | [Spec](../spec.md) |
| `evaluators` | no, keyed by role | [`EvaluatorSpec`](components.md#lofi.spec.components.run.EvaluatorSpec) | [Evaluation](evaluation.md) |
| `comparison` | no | [`SourceComparisonPointSpec`](../spec.md#lofi.spec.SourceComparisonPointSpec) | [Spec](../spec.md) |
| `signature_assignment` | no | [`SignatureAssignmentsSpec`](../spec.md#lofi.spec.SignatureAssignmentsSpec) | [Spec](../spec.md) |

### The `model` section

| Field | Required | Class | Rendered on |
| --- | --- | --- | --- |
| `language` | yes | [`ModelLanguageSpec`](../spec.md#lofi.spec.ModelLanguageSpec) | [Spec](../spec.md); languages on [Logical form language](language.md) |
| `capacity` | no | [`ParameterCapacitySpec`](../spec.md#lofi.spec.ParameterCapacitySpec) | [Spec](../spec.md) |
| `parser` | no | [`ParserSpec`](components.md#lofi.spec.components.model.ParserSpec) | [Model](model.md#parsers) |
| `token_encoder` | no | [`TokenEncoderSpec`](components.md#lofi.spec.components.model.TokenEncoderSpec) | [Model](model.md#tokens) |
| `subword_pooler` | no | [`SubwordPoolerSpec`](components.md#lofi.spec.components.model.SubwordPoolerSpec) | [Model](model.md#poolers) |
| `controller` | no | [`ControllerSpec`](components.md#lofi.spec.components.model.ControllerSpec) | [Model](model.md#controllers) |
| `grounding` | no | [`GroundingSpec`](components.md#lofi.spec.components.model.GroundingSpec) | [Model](model.md#groundings) |
| `encoders` | no, keyed by role | [`TermEncoderSpec`](components.md#lofi.spec.components.model.TermEncoderSpec) | [Model](model.md#encoders) |
| `decoders` | no, keyed by role | [`TermDecoderSpec`](components.md#lofi.spec.components.model.TermDecoderSpec) | [Model](model.md#decoders) |
| `spaces` | no, keyed by sort | [`SpaceSpec`](../spec.md#lofi.spec.SpaceSpec) | [Spec](../spec.md) |
| `readouts` | no, keyed by role | [`ReadoutSpec`](components.md#lofi.spec.components.model.ReadoutSpec) | [Model](model.md#readouts) |
| `span_decoders` | no, keyed by role | [`SpanDecoderSpec`](components.md#lofi.spec.components.model.SpanDecoderSpec) | [Model](model.md#span-decoders) |
| `actions` | no | [`ActionsSpec`](../spec.md#lofi.spec.ActionsSpec) | [Spec](../spec.md) |

### The `data` section

[`DataSection`](../spec.md#lofi.spec.DataSection) has one field, `datasets`, which maps each training dataset role
to a [`DatasetSection`](../spec.md#lofi.spec.DatasetSection) and must name at least one.

| Field of a dataset role | Required | Class | Rendered on |
| --- | --- | --- | --- |
| `dataset` | yes | [`DatasetSpec`](components.md#lofi.spec.components.data.DatasetSpec) | [Datasets](datasets.md#datasets) |
| `featurizer` | no | [`FeaturizerSpec`](components.md#lofi.spec.components.data.FeaturizerSpec) | [Datasets](datasets.md#featurizers) |
| `constraint_source` | no | [`ConstraintSourceSpec`](components.md#lofi.spec.components.data.ConstraintSourceSpec) | [Spec](../spec.md) |
| `constraint_policy` | no | [`ConstraintPolicySection`](../spec.md#lofi.spec.ConstraintPolicySection) | [Spec](../spec.md) |

A dataset that declares a grammar does so in its `grammar` field, which
[Grammar declarations](grammar.md) renders.

### The `loss` section

[`LossSection`](../spec.md#lofi.spec.LossSection) has one field, `terms`: a list of [`WeightedLossSpec`](../spec.md#lofi.spec.WeightedLossSpec) values, at
least one and no two under one name. Each term's objective is a [`LossSpec`](components.md#lofi.spec.components.loss.LossSpec)
variant, rendered on [Training](training.md#losses).

## Experiment document

[`ExperimentDocument`](../run.md#lofi.run.ExperimentDocument) validates an experiment document. Paths in it are relative
to the directory holding the document. [`lofi.run`](../run.md#lofi.run) exports the class and every
section class the table names.

| Section | Required | Class | Rendered on |
| --- | --- | --- | --- |
| `role` | yes | a nonblank name | |
| `freeze_id` | yes | a string of the form `freeze/<n>` | |
| `config` | yes | [`ExperimentConfigSpec`](../run.md#lofi.run.ExperimentConfigSpec) | [Run](../run.md) |
| `launch` | yes | [`LaunchSpec`](../run.md#lofi.run.LaunchSpec) | [Run](../run.md) |
| `failure` | yes | [`FailurePolicySpec`](../run.md#lofi.run.FailurePolicySpec) | [Run](../run.md) |
| `sweep` | yes | [`Sweep`](../run.md#lofi.run.Sweep) | [Run](../run.md) |
| `signature_twins` | no | [`SignatureTwinsSpec`](../run.md#lofi.run.SignatureTwinsSpec) | [Run](../run.md) |
| `seed_policy` | yes | [`ExperimentSeedSpec`](../run.md#lofi.run.ExperimentSeedSpec) | [Run](../run.md) |
| `rows` | yes | [`RowReference`](../run.md#lofi.run.RowReference) list | [Run](../run.md) |
| `reports` | no | [`ReportSpec`](../run.md#lofi.run.ReportSpec) list | [Run](../run.md) |
| `reducers` | no | [`ReducerSpec`](../spec.md#lofi.spec.ReducerSpec) list | [Spec](../spec.md) |
| `primary` | yes | `RowReference` | [Run](../run.md) |
| `preregistration` | yes | a path | |
| `factors` | no, keyed by factor | [`FactorSpan`](../run.md#lofi.run.FactorSpan) | [Run](../run.md) |
| `compute` | yes | [`ComputeSpec`](../run.md#lofi.run.ComputeSpec): [`UnpricedCompute`](../run.md#lofi.run.UnpricedCompute) or [`ExperimentCompute`](../run.md#lofi.run.ExperimentCompute) | [Run](../run.md) |
| `pricing` | no | [`PricingSpec`](../run.md#lofi.run.PricingSpec): [`StackBootstrapPricing`](../run.md#lofi.run.StackBootstrapPricing) or [`LengthBatchPricing`](../run.md#lofi.run.LengthBatchPricing) | [Run](../run.md) |
| `contraction` | yes | [`ExperimentContraction`](../run.md#lofi.run.ExperimentContraction), whose rungs are [`RungSpec`](../run.md#lofi.run.RungSpec) variants | [Run](../run.md) |
| `stop_rules` | yes | [`StopRuleSpec`](../run.md#lofi.run.StopRuleSpec) list, each `when` a [`PredicateSpec`](../run.md#lofi.run.PredicateSpec) | [Run](../run.md) |
| `decision_point` | yes | [`ExperimentDecision`](../run.md#lofi.run.ExperimentDecision), whose `signal` is a `PredicateSpec` | [Run](../run.md) |
| `evaluation` | yes | [`EvaluationSpec`](../run.md#lofi.run.EvaluationSpec), whose `correction` is a [`CorrectionSpec`](../run.md#lofi.run.CorrectionSpec) | [Run](../run.md) |

## Baseline fit document

[`BaselineFitSpec`](../run.md#lofi.run.BaselineFitSpec) has one field, `system`, a [`BaselineSystemSpec`](components.md#lofi.spec.components.baseline.BaselineSystemSpec) variant
rendered on [Evaluation](evaluation.md#baselines). The system must declare at
least one named metric and no metric twice.

## Judgment fit document

[`JudgmentFitSpec`](../judgment.md#lofi.judgment.JudgmentFitSpec) validates a judgment fit document. Every field is required;
a field typed `| None` is written as `null` when it is absent.

| Field | Class | Rendered on |
| --- | --- | --- |
| `dataset` | a nonblank dataset identifier | |
| `response` | [`ResponseDeclaration`](https://github.com/logical-form-induction/lofi/blob/main/lofi/judgment/_spec.py#L3080), written under one constructor key: `ResponseDeclaration_responsespec` ([`ResponseSpec`](../judgment.md#lofi.judgment.ResponseSpec)) or `ResponseDeclaration_boundedresponsespec` ([`BoundedResponseSpec`](../judgment.md#lofi.judgment.BoundedResponseSpec)) | [Judgment](../judgment.md) |
| `inference` | [`FitInference`](../judgment.md#lofi.judgment.FitInference): [`NutsInference`](../judgment.md#lofi.judgment.NutsInference), [`VariationalInference`](../judgment.md#lofi.judgment.VariationalInference) (with a required `learning_rate`) or [`MapInference`](../judgment.md#lofi.judgment.MapInference) | [Judgment](../judgment.md) |
| `elicitation` | [`Elicitation`](../judgment.md#lofi.judgment.Elicitation) | [Judgment](../judgment.md) |
| `calibration` | [`JudgmentFitCalibration`](../judgment.md#lofi.judgment.JudgmentFitCalibration) (with a required `guide_learning_rate`) or `null` | [Judgment](../judgment.md) |
| `comparison` | [`ComparisonSpec`](../judgment.md#lofi.judgment.ComparisonSpec) or `null` | [Judgment](../judgment.md) |
| `dynamic_range` | [`DynamicRangeSpec`](../judgment.md#lofi.judgment.DynamicRangeSpec) or `null` | [Judgment](../judgment.md) |
| `ablation` | a list of [`TableAblationSpec`](../judgment.md#lofi.judgment.TableAblationSpec) | [Judgment](../judgment.md) |

## Site profile

[`SiteProfile`](../run.md#lofi.run.SiteProfile) validates the site profile that the `LOFI_SITE_PROFILE`
environment variable names.

| Field | Class | Rendered on |
| --- | --- | --- |
| `submission` | [`SiteSubmission`](../run.md#lofi.run.SiteSubmission) | [Run](../run.md) |
| `placement` | [`SitePlacement`](../run.md#lofi.run.SitePlacement) | [Run](../run.md) |
| `storage` | [`SiteStorage`](../run.md#lofi.run.SiteStorage) | [Run](../run.md) |
| `environment` | [`SiteEnvironment`](../run.md#lofi.run.SiteEnvironment) | [Run](../run.md) |
| `interpreter` | [`SiteInterpreter`](../run.md#lofi.run.SiteInterpreter) | [Run](../run.md) |
| `transport` | [`SiteTransport`](../run.md#lofi.run.SiteTransport) | [Run](../run.md) |
| `allocation` | [`SiteAllocation`](../run.md#lofi.run.SiteAllocation) | [Run](../run.md) |
