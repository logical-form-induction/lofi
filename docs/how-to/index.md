# Guides

Choose a task below. The [tutorial](../tutorials/index.md) follows one run from
configuration to evaluation; [Reading the notation](../explanation/reading-the-notation.md)
introduces categories, types, inference rules, and probability expressions.

## Set up a run

- [Install LoFI](setup/installing.md) and select the dependencies needed for the task.
- [Author a configuration](setup/configuration.md) by combining reusable fragments and inspecting the final settings before execution.

## Declare a grammar and its meanings

- [Choose a formalism](grammars/choose-a-formalism.md) for phrase-structure productions, argument-taking categories, or the supported type-logical fragment.
- [Declare a signature](grammars/declare-a-signature.md) to specify primitive categories and the constructors that combine them.
- [Declare a lexicon](grammars/declare-a-lexicon.md) to assign categories to words or draw a controlled lexical inventory.
- [Choose combinatory rules](grammars/choose-combinatory-rules.md) to license combinations of those categories.
- [Choose matching, normal forms, and goals](grammars/choose-normal-form-matching-goal.md) to specify category comparisons, admitted derivations, complete analyses, and search bounds.
- [Declare a logical form language](grammars/declare-a-logical-form-language.md) for the expressions used as meanings.
- [Declare an interpretation](grammars/declare-an-interpretation.md) to build those meanings from grammatical derivations.
- [Map between signatures](grammars/map-between-signatures.md) when erasing features, assigning [semantic types](../explanation/logical-forms-as-types.md#semantic-typing), or aligning induced labels for evaluation.

## Prepare data

- [Generate a synthetic corpus](data/generate-a-synthetic-corpus.md) with a declared size, length distribution, and split policy.
- [Read a corpus](data/read-a-corpus.md) from an existing annotated release and inspect which records survive loading.
- [Featurize a dataset](data/featurize-a-dataset.md) to produce the batches the trainer expects.
- [Corrupt strings and labels](data/corrupt-strings.md) to vary grammatical evidence and annotation noise separately.
- [Generate ill-typed forms](data/generate-ill-typed-forms.md) to provide negative examples for semantic typing.
- [Generate inference pairs](data/generate-inference-pairs.md) with source meanings and declared relations.
- [Control semantic distributions](data/control-semantic-distributions.md) to vary meaning inventories, complexity, and bounded scope-reading counts.

## Configure the learner

- [Choose a token encoder](models/choose-a-token-encoder.md) for fixed, learned, or contextual word features.
- [Configure a chart parser](models/configure-a-chart-parser.md) to combine spans and choose symbolic parsing operations.
- [Configure a controller](models/configure-a-controller.md) to score the grammar's actions from child representations.
- [Configure term encoders and decoders](models/configure-term-encoders-and-decoders.md) to connect symbolic expressions with learned vectors.
- [Configure readouts](models/configure-readouts.md) for symbol probabilities, target-set mass, or explicit candidate expressions.

## Train and recover a point

- [Combine losses](training/combine-losses.md) to declare the evidence and regularization used for learning.
- [Configure a trainer](training/configure-a-trainer.md) to choose batches, update schedules, stages, and stopping criteria.
- [Choose a device policy](training/choose-a-device-policy.md) for the available hardware and numerical precision.
- [Checkpoint and resume](training/resume-from-a-checkpoint.md) to recover an interrupted point and select its released fitted state.
- [Supervise and constrain](training/supervise-and-constrain.md) with partial category, meaning, or constituent annotations.
- [Use theorem-prover feedback](training/reward-with-a-theorem-prover.md) to train from decided logical relations.
- [Use model checking](training/reward-with-model-checking.md) to score relations in declared finite structures.
- [Acquire labels actively](training/acquire-labels-actively.md) to select successive judgment batches from a candidate pool.

## Evaluate and compare

- [Evaluate a point](evaluation/evaluate-a-run.md) on retained data with explicit metrics and scoring conventions.
- [Decode derivations](evaluation/decode-derivations.md) to select complete analyses from a trained model's candidates.
- [Fit a baseline](evaluation/fit-a-baseline.md) under a declared comparison protocol.
- [Run and price a sweep](experiments/sweeps.md) to expand conditions across seeds and execute an experiment.
- [Choose a sweep design](experiments/choose-a-sweep-design.md) when a full factorial is too large or a planned fraction is appropriate.
- [Contract an experiment](experiments/contract-an-experiment.md) using its predeclared reductions and amendment record.
- [Run on a cluster](experiments/run-on-a-cluster.md) with the site's scheduler and storage settings.
- [Track runs](experiments/track-runs.md) in a metric service.
- [Reduce and report results](experiments/reduce-and-report-results.md) after the accepted rows have been collected.
- [Publish prediction records](experiments/publish-records.md) in a shared annotation format.

## Work with judgments

- [Construct items](judgments/construct-judgment-items.md) from templates, fillers, and constraints.
- [Lay out judgment corpora](judgments/judgment-corpora.md) with their design documents and predictor columns.
- [Fit a judgment model](judgments/fit-a-judgment-model.md) to relate item predictions and participant effects to responses.
- [Simulate a design](judgments/simulate-judgment-designs.md) to inspect recovery and calibration under known generating conditions.
- [Train on judgments](judgments/train-on-judgments.md) through a declared response instrument.

## Inspect or distribute a trained point

- [Run probes](probes/run-probes.md) to observe or intervene on published model values.
- [Run diagnostics](probes/run-diagnostics.md) to measure numerical, distributional, and representational behavior.
- [Instrument an external model](probes/external-model-site-maps.md) by declaring its accessible sites.
- [Export a model bundle](serving/export-to-the-hub.md) for loading outside the original run directory.
- [Serve through bead](serving/serve-through-bead.md) when the trained point publishes inference predictions.

## Extend LoFI

- [Package a plugin](extending/plugins.md) to install another declared component or capability.
- [Run law suites](extending/laws.md) to test the required behavior of an implementation.
- [Enable debug checks](extending/checks.md) when investigating internal invariant failures.
