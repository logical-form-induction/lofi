# Entry point groups

An entry point is package metadata announcing an installed extension. A group
names what can be extended; an entry in that group names an implementation.
For instance, `lofi.loss` registers the [loss declarations](configuration/training.md#losses)
a run can select. A separate package can add an implementation by registering
it in that group.

Run documents select installed variants by their tags. The
[plugin guide](../how-to/extending/plugins.md) shows how an author registers a
new variant, and [swappable assumptions](../explanation/swappable-assumptions.md)
explains the comparison this permits.

This page lists every entry point group LoFI's [`pyproject.toml`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml) declares, the
[declaration family](../explanation/swappable-assumptions.md#how-a-family-is-discovered) or runtime capability each group extends, and the names LoFI
packages in it.

A distribution extends LoFI by registering an object under one of these groups;
[Writing a plugin](../how-to/extending/plugins.md) shows how. The console
script `lofi` is the only `[project.scripts]` entry and names
[`lofi.run._entrypoint:main`](https://github.com/logical-form-induction/lofi/blob/main/lofi/run/_entrypoint.py) (see [Command line](cli.md)).

[`lofi.spec.load_plugins`](spec.md#lofi.spec.load_plugins) loads the groups in three passes before a document is
validated: the family roots in `lofi.families`, then every declaration family,
then the plugin objects of every component family. A component family's group
is named `lofi.<family>` after its entry in `lofi.families`. The remaining
groups are read by the component or procedure that owns them, when it is
built.

## Family roots

`lofi.families` registers one declaration root per component family. Every
variant of the family derives from that root, and the
[Component roots](configuration/components.md) page renders them.

The following catalog records the names declared in this checkout's
[`pyproject.toml`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml). Installed extensions can add further names
to these built-in registrations. Each family root names
the declaration class shared by its variants:

| Family | Root declaration |
| --- | --- |
| `active_collector` | [`lofi.spec.acquisition.collectors:ActiveCollectorSpec`](configuration/training.md#lofi.spec.acquisition.ActiveCollectorSpec) |
| `baseline_system` | [`lofi.spec.components.baseline:BaselineSystemSpec`](configuration/components.md#lofi.spec.components.baseline.BaselineSystemSpec) |
| `batch_sampler` | [`lofi.spec.components.trainer:BatchSamplerSpec`](configuration/components.md#lofi.spec.components.trainer.BatchSamplerSpec) |
| `constraint_source` | [`lofi.spec.components.data:ConstraintSourceSpec`](configuration/components.md#lofi.spec.components.data.ConstraintSourceSpec) |
| `controller` | [`lofi.spec.components.model:ControllerSpec`](configuration/components.md#lofi.spec.components.model.ControllerSpec) |
| `chart` | [`lofi.spec.components.model:ParserSpec`](configuration/components.md#lofi.spec.components.model.ParserSpec) |
| `dataset` | [`lofi.spec.components.data:DatasetSpec`](configuration/components.md#lofi.spec.components.data.DatasetSpec) |
| `evaluator` | [`lofi.spec.components.run:EvaluatorSpec`](configuration/components.md#lofi.spec.components.run.EvaluatorSpec) |
| `featurizer` | [`lofi.spec.components.data:FeaturizerSpec`](configuration/components.md#lofi.spec.components.data.FeaturizerSpec) |
| `gold_aggregation` | [`lofi.spec.components.data:GoldAggregationSpec`](configuration/components.md#lofi.spec.components.data.GoldAggregationSpec) |
| `grounding` | [`lofi.spec.components.model:GroundingSpec`](configuration/components.md#lofi.spec.components.model.GroundingSpec) |
| `loss` | [`lofi.spec.components.loss:LossSpec`](configuration/components.md#lofi.spec.components.loss.LossSpec) |
| `optimizer` | [`lofi.spec.components.trainer:OptimizerSpec`](configuration/components.md#lofi.spec.components.trainer.OptimizerSpec) |
| `pair_sampler` | [`lofi.spec.components.trainer:PairSamplerSpec`](configuration/components.md#lofi.spec.components.trainer.PairSamplerSpec) |
| `pair_scorer` | [`lofi.spec.components.model:PairScorerSpec`](configuration/components.md#lofi.spec.components.model.PairScorerSpec) |
| `probe` | [`lofi.spec.components.run:ProbeSpec`](configuration/components.md#lofi.spec.components.run.ProbeSpec) |
| `prover` | [`lofi.spec.provers:ProverSpec`](configuration/model.md#lofi.spec.provers.ProverSpec) |
| `readout` | [`lofi.spec.components.model:ReadoutSpec`](configuration/components.md#lofi.spec.components.model.ReadoutSpec) |
| `scheduler` | [`lofi.spec.components.trainer:SchedulerSpec`](configuration/components.md#lofi.spec.components.trainer.SchedulerSpec) |
| `sink` | [`lofi.spec.components.run:SinkSpec`](configuration/components.md#lofi.spec.components.run.SinkSpec) |
| `span_decoder` | [`lofi.spec.components.model:SpanDecoderSpec`](configuration/components.md#lofi.spec.components.model.SpanDecoderSpec) |
| `subword_pooler` | [`lofi.spec.components.model:SubwordPoolerSpec`](configuration/components.md#lofi.spec.components.model.SubwordPoolerSpec) |
| `term_decoder` | [`lofi.spec.components.model:TermDecoderSpec`](configuration/components.md#lofi.spec.components.model.TermDecoderSpec) |
| `term_encoder` | [`lofi.spec.components.model:TermEncoderSpec`](configuration/components.md#lofi.spec.components.model.TermEncoderSpec) |
| `token_encoder` | [`lofi.spec.components.model:TokenEncoderSpec`](configuration/components.md#lofi.spec.components.model.TokenEncoderSpec) |
| `tokenizer` | [`lofi.spec.components.data:TokenizerSpec`](configuration/components.md#lofi.spec.components.data.TokenizerSpec) |
| `trainer` | [`lofi.spec.components.trainer:TrainerSpec`](configuration/components.md#lofi.spec.components.trainer.TrainerSpec) |
| `type_sampler` | [`lofi.spec.components.trainer:TypeSamplerSpec`](configuration/components.md#lofi.spec.components.trainer.TypeSamplerSpec) |

## Registered groups and names

The group names identify the extension being selected, and the names in the
second column are its packaged alternatives. The full registration targets
remain in [the package metadata](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml).
Some groups supply procedures or logical roles rather than configurable model
components, as the loading rules above distinguish.

| Group | Registered names |
| --- | --- |
| [`lofi.resolver`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L119) | `lofi.path`, `lofi.pkg`, `lofi.seed` |
| [`lofi.laws`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L127) | `data`, `deduction`, `evaluation`, `interpretation`, `judgment`, `lexicon`, `neural-loss`, `probe`, `run`, `statistics`, `terms`, `training`, `typing` |
| [`lofi.config_dirs`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L144) | No built-in entries. |
| [`lofi.rule_family`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L147) | `pure_ccg`, `prefix_closed_vw_ccg` |
| [`lofi.rule_laws`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L150) | `composition`, `substitution`, `raising`, `pattern` |
| [`lofi.rule`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L158) | `absorption`, `application`, `clausal-adjunct`, `coercion`, `composition`, `coordination`, `pattern`, `substitution`, `type-raising` |
| [`lofi.erasure`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L168) | `signature-morphism` |
| [`lofi.matching`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L170) | `alpha`, `feature-erasure`, `identity`, `modal`, `product`, `subsumption` |
| [`lofi.normal_form`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L177) | `all-of`, `eisner`, `hockenmaier-bisk`, `none`, `undirected-eisner` |
| [`lofi.goal`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L183) | `cover`, `root` |
| [`lofi.interpretation`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L186) | `category`, `dependency`, `meaning`, `string` |
| [`lofi.parse_strategy`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L191) | `cky` |
| [`lofi.span_index`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L193) | `contiguous` |
| [`lofi.chart_store`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L195) | `native` |
| [`lofi.cell_pruner`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L197) | `top-k` |
| [`lofi.semiring`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L199) | `boolean`, `counting`, `expectation`, `forest`, `k-best`, `log-inside`, `viterbi` |
| [`lofi.astar_strategy`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L207) | `binary`, `linear` |
| [`lofi.astar_root_mode`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L210) | `free`, `goal` |
| [`lofi.astar_heuristic`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L213) | `supertag-outside`, `zero` |
| [`lofi.astar_proposer`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L216) | `supertag-beam` |
| [`lofi.mbr_utility`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L218) | `constituent` |
| [`lofi.intervention`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L220) | `mean`, `noise`, `resample`, `shuffle`, `zero` |
| [`lofi.acdc_threshold`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L226) | `absolute`, `relative` |
| [`lofi.artifact_source`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L229) | `hub`, `url` |
| [`lofi.diora_backend`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L232) | `released-mlp` |
| [`lofi.das_search`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L234) | `boundless`, `fixed-rank` |
| [`lofi.decode_method`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L237) | `mbr`, `viterbi` |
| [`lofi.formalism`](formalism.md#lofi.formalism) | `ab`, `cfg`, `tlg`, `undirected-ccg` |
| [`lofi.lf_language`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L245) | `neo-davidsonian`, `propositional` |
| [`lofi.lf_complexity`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L248) | `connective_count`, `meaning_depth`, `operator_nesting`, `predicate_count`, `quantifier_depth`, `variable_count` |
| [`lofi.lf_class`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L255) | `quantifier`, `connective`, `predicate`, `identity`, `constant` |
| [`lofi.pair_strategy`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L261) | `independent`, `overlap-matched` |
| [`lofi.corruption`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L264) | `swap`, `deletion`, `insertion`, `substitution` |
| [`lofi.ill_typing`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L269) | `argument-swap`, `over-application`, `sort-mismatch` |
| [`lofi.token_encoder`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L273) | `static`, `pretrained`, `learned`, `llm2vec` |
| [`lofi.subword_pooler`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L278) | `first`, `mean`, `attention` |
| [`lofi.term_encoder`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L282) | `hrr`, `linear-rnn`, `tree-rnn`, `transformer` |
| [`lofi.term_decoder`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L287) | `hrr`, `linear-rnn`, `tree-rnn`, `transformer`, `diffusion` |
| [`lofi.chart`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L293) | `inside-outside`, `tagged`, `flat`, `span-crf` |
| [`lofi.chart_composition`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L298) | `mean-field`, `gated-residual` |
| [`lofi.split_weigher`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L301) | `marginal`, `attend` |
| [`lofi.span_function`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L304) | `mlp`, `bilinear`, `cp-bilinear`, `gated-mlp`, `hrr`, `transformer` |
| [`lofi.controller`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L311) | `mlp`, `bilinear`, `transformer` |
| [`lofi.launcher`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L315) | `serial`, `array`, `remote` |
| [`lofi.correction`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L319) | `bonferroni`, `holm`, `benjamini-hochberg`, `benjamini-yekutieli` |
| [`lofi.report_format`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L324) | `markdown`, `latex` |
| [`lofi.experiment_config`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L327) | `run`, `judgment-fit`, `baseline-fit`, `active-learning` |
| [`lofi.pricing`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L332) | `stack-bootstrap`, `length-batch` |
| [`lofi.contraction_rung`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L335) | `unpreregistered-cells`, `resolution-iv`, `languages`, `seeds`, `corpus-scale`, `fallback` |
| [`lofi.sweep_design`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L342) | `grid`, `random`, `latin-hypercube`, `fractional-factorial` |
| [`lofi.activation`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L347) | `tanh`, `relu`, `gelu`, `silu` |
| [`lofi.recurrent_cell`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L352) | `gru`, `lstm`, `elman-tanh`, `elman-relu` |
| [`lofi.decoding_strategy`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L357) | `greedy`, `beam`, `exhaustive`, `sample` |
| [`lofi.pair_feature`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L362) | `left`, `right`, `difference`, `product` |
| [`lofi.sampling_restriction`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L367) | `full`, `top-k`, `nucleus` |
| [`lofi.similarity_kernel`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L371) | `cosine`, `rbf` |
| [`lofi.transformer_structure`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L374) | `positions`, `positions-with-depth`, `ancestor-mask`, `ancestor-and-sibling-bias`, `parent-recurrence` |
| [`lofi.vector_format`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L380) | `glove`, `word2vec` |
| [`lofi.description_cost`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L383) | `uniform`, `log-alphabet`, `per-constructor` |
| [`lofi.oov_initialization`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L387) | `zeros`, `normal` |
| [`lofi.pair_selection`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L390) | `forced-choice`, `relational`, `relational-scalar` |
| [`lofi.mask_replacement`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L394) | `mixed`, `mask` |
| [`lofi.adaptation`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L397) | `full`, `lora` |
| [`lofi.overflow_policy`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L400) | `refuse`, `report` |
| [`lofi.encoder_architecture`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L403) | `encoder-only`, `t5-encoder` |
| [`lofi.readout`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L406) | `terms` |
| [`lofi.meaning_weighting`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L408) | `distinct`, `multiplicity` |
| [`lofi.span_decoder`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L411) | `rnn-span-decoder` |
| [`lofi.loss`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L413) | `coherence`, `grammaticality`, `pinning`, `vector-anchor`, `simplicity`, `chart-scale`, `derivation-entropy`, `span-reconstruction`, `outside-reconstruction`, `inference`, `form-supervision`, `shaped-proof`, `form-typing` |
| [`lofi.inference_response`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L427) | `label`, `judgment` |
| [`lofi.inference_labeling`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L430) | `three-way`, `rte`, `graded` |
| [`lofi.grounding`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L434) | `pairwise`, `expected-embedding` |
| [`lofi.control_variate`](rewards.md#lofi.control_variate) | `moving` |
| [`lofi.lexicon_generator`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L439) | `entailment-graph`, `masked-lm` |
| [`lofi.entailment_graph_reader`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L442) | `edge-list` |
| [`lofi.acquisition`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L444) | `rc2.1` |
| [`lofi.active_collector`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L446) | `bead`, `drawn-effects`, `predictor-truth`, `registered-design` |
| [`lofi.judgment_source`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L451) | `predicted` |
| [`lofi.pair_scorer`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L453) | `prover`, `model-checking`, `guided-proof`, `neural-proof` |
| [`lofi.fuzzy_logic`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L459) | `product`, `goedel`, `lukasiewicz` |
| [`lofi.fuzzy_implication`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L464) | `residuum`, `material` |
| [`lofi.fuzzy_quantifier`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L468) | `tnorm`, `extreme` |
| [`lofi.fuzzy_equality`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L472) | `crisp`, `margin` |
| [`lofi.atomic_sampling`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L476) | `uniform` |
| [`lofi.denotation_streams`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L479) | `independent` |
| [`lofi.function_sampling`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L482) | `table` |
| [`lofi.truth_sampling`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L485) | `bernoulli` |
| [`lofi.structure_source`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L488) | `sampled`, `given`, `learned` |
| [`lofi.unmapped_constant`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L493) | `undecided` |
| [`lofi.knowledge_base`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L496) | `wikidata` |
| [`lofi.prover`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L499) | `truth-table`, `smt`, `tptp` |
| [`lofi.smt_engine`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L503) | `z3`, `cvc5` |
| [`lofi.tptp_engine`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L506) | `eprover`, `vampire`, `leo3` |
| [`lofi.logical_role`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L510) | `negation`, `conjunction`, `disjunction`, `conditional`, `biconditional`, `exclusive-or`, `equality`, `universal`, `existential`, `generalized-universal` |
| [`lofi.logical_constant`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L521) | `atom`, `choice`, `most`, `part-of`, `sum` |
| [`lofi.constraint_source`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L527) | `declared`, `gold-derivation`, `gold-lexicon`, `gold-logical-form` |
| [`lofi.batch_sampler`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L532) | `length-bucket`, `random`, `sequential` |
| [`lofi.unsatisfiable_policy`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L536) | `error`, `drop_clamp`, `skip_item` |
| [`lofi.type_sampler`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L540) | `recursive`, `exhaustive` |
| [`lofi.pair_sampler`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L543) | `rule` |
| [`lofi.dataset`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L545) | `ccgbank`, `ccgbank-reskin`, `chaosnli`, `expert-ratings`, `fracas`, `mnli`, `pmb-ccg`, `pmb-drs`, `snli`, `synthetic`, `synthetic-inference`, `unli` |
| [`lofi.baseline_system`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L558) | `bisk-hockenmaier`, `lm-prover` |
| [`lofi.featurizer`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L561) | `unlabeled`, `scalar`, `derivation`, `span-constraint`, `pair` |
| [`lofi.tokenizer`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L567) | `regex` |
| [`lofi.gold_aggregation`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L569) | `majority` |
| [`lofi.probe`](probe.md#lofi.probe) | `causal`, `circuit`, `classifier`, `diagnostics`, `external`, `intervention`, `sae` |
| [`lofi.diagnostic`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L579) | `action-margin`, `atomic-complex-differentiation`, `attention-patterns`, `constructor-embedding`, `controller-choice`, `controller-relations`, `cutpoint-entropy`, `depth-graded-encoding`, `feed-forward-specialization`, `finite-readings`, `gate-patterns`, `hidden-state-evolution`, `memory-cell-dynamics`, `recurrent-connection-analysis`, `recursion-spread`, `residual-contributions`, `split-posterior-kl`, `span-length-bias`, `superposition-interference` |
| [`lofi.optimizer`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L599) | `adamw`, `adam`, `sgd` |
| [`lofi.scheduler`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L603) | `cosine`, `linear`, `constant` |
| [`lofi.trainer`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L607) | `alternating`, `parser`, `distance`, `staged`, `terms` |
| [`lofi.mixing`](train.md#lofi.mixing) | `round-robin`, `proportional`, `ratio`, `temperature` |
| [`lofi.step`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L618) | `alternating`, `controller`, `grounded`, `judgment`, `parser`, `supervised-warm-start`, `terms`, `token-warm-up` |
| [`lofi.sink`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L627) | `wandb`, `trackio`, `mlflow` |
| [`lofi.evaluator`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L631) | `bracketing`, `category-side`, `coherence-square`, `coherence-diagnostics`, `constructor-diagnostics`, `decode`, `equivalence`, `expert-categories`, `grammaticality`, `held-out-likelihood`, `inference`, `lexical-diagnostics`, `lexical-types`, `logical-form`, `reachability`, `structural-diagnostics` |
| [`lofi.reducer`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L648) | `coherence-collapse`, `cross-lingual`, `crossed-recovery`, `encoder-ablation`, `node-vocabulary`, `residual-structure`, `restitutive-coverage`, `variance`, `rescoring`, `rank-churn`, `source-comparison`, `transfer` |
| [`lofi.prediction_format`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L661) | `lofi-json` |
| [`lofi.rescoring_metric`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L663) | `recall-homogeneity` |
| [`lofi.publisher`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L665) | `layers-records` |
| [`lofi.guide`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L667) | `mean_field`, `structured` |
| [`lofi.linking`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L670) | `linear`, `logit`, `morcela`, `probability` |
| [`lofi.splitter`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L675) | `kfold`, `leave-one-group-out` |
| [`lofi.judgment_design`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L678) | `known-random-effects`, `prior-predictive-reference`, `megaacceptability`, `megaacceptability-morcela`, `megaintensionality`, `meganegraising`, `megaveridicality` |
| [`lairs.codecs`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml#L687) | `ccgbank-auto`, `ccgbank-parg`, `chaosnli`, `fracas`, `layers-export`, `megaacceptability`, `megaintensionality`, `meganegraising`, `megaveridicality`, `mnli`, `pmb-ccg`, `pmb-drs`, `snli`, `unli` |
