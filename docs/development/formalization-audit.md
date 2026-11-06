# Documentation formalization audit

This report records the mathematical documentation changes in the current working tree. The review separates definitions, implementation behavior, and claims that require additional assumptions. Code examples, API names, configuration fields, and serialized grammar syntax remain code; mathematical expressions in prose use LaTeX.

Probability measures use $\mathbb{P}$, expectations use $\mathbb{E}$, PMFs use lowercase $p,q,r$, and PDFs use lowercase $f,g,h$. Numeric scalars are lowercase italic, vectors lowercase bold, and matrices and higher-order arrays uppercase bold. Grammar, logic, random variable, and information-theoretic notation follows the conventions of those fields.

## Foundations and architecture

| Page | Definitions or changes | Corrected claim or scope |
| --- | --- | --- |
| [Categories, semantic types, and logical forms](../explanation/logical-forms-as-types.md) | Sorted signatures, binding, semantic typing, category-to-type erasure, substitution and normalization. | Categories, semantic types, and logical forms are distinct; a common term representation does not identify them. |
| [Grammars as deductive systems](../explanation/derivations-as-proofs.md) | CFG and CCG deduction, rule relations, normal forms, goals, and capacity assumptions. | Normal form preservation and capacity theorems require their stated fragments; bounded search is separate from derivability. |
| [Charts and approximation](../explanation/charts-and-approximation.md) | Semirings, inside–outside quantities, decoding decisions, pruning, and residual mass. | Unnormalized weights, normalized probabilities, and approximate vector composition are distinguished. |
| [Term layout capability is not an arity ceiling](../explanation/term-layout-capabilities.md) | Term domains, codec round trips, preorder and complete tree bounds, HRR composition, supported ranks. | Round trips concern valid encoded terms; learned encoders need not be injective and HRR scaling does not guarantee fixed norms. |
| [Documents are the declaration language](../explanation/documents-are-the-dsl.md) | Ordered partial composition and scientific configuration identity. | Merge is not commutative; truncated hashes are not injective identifiers. |
| [The layer order](../explanation/layering.md) | Directed import graph and strict layer inequalities. | Acyclic imports do not imply statistical independence. |
| [Boundaries](../explanation/boundaries.md) | Input and output invariants as predicates defining admitted domains. | Validation and finite tests do not prove universal contract preservation. |

## Objectives, evidence, and experiments

| Page | Definitions or changes | Corrected claim or scope |
| --- | --- | --- |
| [Induction objectives](../explanation/induction-objectives.md) | Equations for all packaged loss families, masks, denominators, schedules, and gradient conventions. | Weighted losses need not form a joint likelihood; constructor costs are not automatically a prior or code length. |
| [Inference as evidence for logical forms](../explanation/inference-as-evidence.md) | Proof/refutation cells, model semantics, fuzzy operations, pair and label normalization, reward estimators, and moving average control variates. | Abstention is not refutation; finite model checking and sampled gradient estimators have explicit scope. |
| [Judgment models](../explanation/judgment-models.md) | Ordinal and choice PMFs, bounded PDFs, linking functions, identification, marginalization, ELBO, and calibration. | Scale invariances depend on free parameters; LoFI’s length exponent extends MORCELA; finite marginal domains remain explicit. |
| [Probing through published sites](../explanation/probing-through-sites.md) | Site tensors, intervention effects, selectivity, subspace interchange, circuit recovery, and the sparse autoencoder objective and decoder normalization. | Effects depend on populations and replacements; normalized recovery need not lie between zero and one. |
| [Swappable grammatical assumptions](../explanation/swappable-assumptions.md) | Nested model families, optimized training fit, and paired-seed contrasts. | Nesting does not guarantee superior held-out fit or make parameter count a capacity theorem. |
| [Source and learner declarations](../explanation/synthetic-designs.md) | Source versus learner laws, shifted negative binomial sampling, rejection conditioning, label noise, and source scoring. | Clipping differs from conditioning, and requested factors differ from realized sample measurements. |
| [Experiments and commitments](../explanation/experiments-and-commitments.md) | Paired randomization, multiplicity corrections, FWER/FDR, equivalence hypotheses, and Monte Carlo test values. | Nonsignificance does not establish equivalence; correction does not repair invalid individual tests or repeated-look assumptions. |

## Grammar and language guides

| Page | Definitions or changes | Corrected claim or scope |
| --- | --- | --- |
| [Choose a formalism](../how-to/grammars/choose-a-formalism.md) | CFG tuple and deduction, AB/CCG category formation, undirected application, and mode lattices. | Plugin names do not imply every rule or theorem of a broader calculus. |
| [Choose combinatory rules](../how-to/grammars/choose-combinatory-rules.md) | Rule relations and instances under shared substitutions. | Repeated variables constrain equality under the declared matching relation; unbounded rule declarations still meet computational bounds. |
| [Declare a category signature](../how-to/grammars/declare-a-signature.md) | Constructor profiles and recursive category term formation. | Removed the incorrect category-theoretic use of “endofunctor.” |
| [Declare a lexicon](../how-to/grammars/declare-a-lexicon.md) | Log mass normalization, rank PMFs, bounded ambiguity mixtures, joint mass targets, and coupling. | The mean ambiguity lies between one and its maximum; finite apportionment and marginal fitting need not realize targets exactly. |
| [Declare an interpretation](../how-to/grammars/declare-an-interpretation.md) | Lexical and action typing, capture-avoiding substitution, eager/lazy policies, and resource bounds. | Search uniqueness is relative to returned inhabitants; lazy interpretation retains redexes. |
| [Map between signatures](../how-to/grammars/map-between-signatures.md) | Feature erasure, recursive constructor images, and label assignment objectives. | A well-formed projection need not preserve product semantics; fitted label alignment is not a signature morphism. |
| [Choose a normal form, matching relation and goal](../how-to/grammars/choose-normal-form-matching-goal.md) | Filter conjunction, relation intersection, root weights, cover penalties, and reached sets. | Finite unsaturated closure does not prove unreachability; root weights are not necessarily a PMF. |
| [Declare a logical form language](../how-to/grammars/declare-a-logical-form-language.md) | Choice functions, relational dynamic semantics, plural sums, events, generalized quantifiers, and links to the central typed syntax. | These are mathematical interpretations of the declared fragments; registration alone does not equip every executor with those semantics. |

## Synthetic data guides

| Page | Definitions or changes | Corrected claim or scope |
| --- | --- | --- |
| [Control semantic distributions](../how-to/data/control-semantic-distributions.md) | Class size coupling, conditional assignments, polysemy sampling, empirical mutual information, complexity bands, and scope equivalence. | Equal symbol counts do not rule out inventory effects; normalized mutual information is undefined when either entropy is zero; bounded scope counts are not exact quotients. |
| [Generate a synthetic corpus](../how-to/data/generate-a-synthetic-corpus.md) | Length PMFs, split allocation, derivation weights, string probabilities, and acceptance bands. | Split counts round down; uniform rule weights do not imply uniform strings; accepted samples condition on grammar support and targets. |
| [Corrupt strings and their labels](../how-to/data/corrupt-strings.md) | Exact/Bernoulli counts, operator mixtures, negative certificates, and binary label transition matrices. | Nonderivability certificates are relative to the configured search; accepted operator frequencies differ from proposal weights. |
| [Generate inference pairs](../how-to/data/generate-inference-pairs.md) | Oracle questions and short circuit, coarse labels, target proportions, symmetric difference complexity, and label noise. | Unproved is not refuted; an entailment label does not independently exclude contradiction; zero routing rows retain labels. |
| [Generate ill-typed forms](../how-to/data/generate-ill-typed-forms.md) | Application edit schemata, typing failure acceptance, injection counts, and observed label laws. | Edits alone do not prove ill-typing; injected negatives augment the gold examples. |

## Neural model guides

| Page | Definitions or changes | Corrected claim or scope |
| --- | --- | --- |
| [Choose a token encoder](../how-to/models/choose-a-token-encoder.md) | Embedding tensors, word/subword alignment and pooling, masked next-token adaptation, and LoRA. | Pooling is local to aligned words; optional adaptation is not the complete LLM2Vec training pipeline. |
| [Configure a controller](../how-to/models/configure-a-controller.md) | Feature vectors, MLP/bilinear logits, attention, and normalized action distributions. | Action probabilities are distinct from symbolic applicability. |
| [Configure a chart parser](../how-to/models/configure-a-chart-parser.md) | Span-CRF distributions, gated composition, split mixtures, bilinear/CP functions, and HRR operations. | Inside values are log weights; mean-field vector composition is not exact tree expectation; lost mass need not be normalized. |
| [Configure term encoders and decoders](../how-to/models/configure-term-encoders-and-decoders.md) | Encoder/decoder domains, autoregressive factors, masked denoising, and pushforward consistency. | Layout validity, decoder probability, vector mappings, and preservation of symbolic typing are separate properties. |
| [Configure readouts](../how-to/models/configure-readouts.md) | Coupled and factorized target scores, constituent factors, masks, and clipping. | Products of marginal slot rows need not equal joint term probabilities; clipping does not establish disjoint target events. |

## Training and judgment guides

| Page | Definitions or changes | Corrected claim or scope |
| --- | --- | --- |
| [Combine losses](../how-to/training/combine-losses.md) | Links every loss family to the central equations and denominator/gradient definitions. | Renamed the generic simplicity contribution a regularizer rather than assuming a probabilistic prior. |
| [Configure a trainer](../how-to/training/configure-a-trainer.md) | Gradient accumulation/clipping, distance regression, Adam/AdamW, learning rate schedules, and exhaustive, recursive, memorized, and rule instance sampling laws. | Accumulation averages microbatch objectives; schedules count closed windows even when mixed precision scaling skips an update. |
| [Reward with a theorem prover](../how-to/training/reward-with-a-theorem-prover.md) | Affine payoff and decay interval in LaTeX, with named endpoints. | Code fields remain literal identifiers; the reward equation is mathematical notation. |
| [Acquire labels actively](../how-to/training/acquire-labels-actively.md) | Predictive means, extremity scores, tied-rank correlation, quotas, agreement, intervals, stopping, and simulated cutpoints. | The score is not predictive uncertainty; the implemented squared distance agreement statistic is not standard ordinal alpha; pointwise intervals do not establish sequential coverage. |
| [Train on judgments](../how-to/judgments/train-on-judgments.md) | Response instrument as a stochastic matrix, induced PMF, and observed response log loss. | Identifiability is limited by the instrument; probability reward and negative log likelihood are different objectives. |
| [Simulate a judgment design](../how-to/judgments/simulate-judgment-designs.md) | Explicit interior cutpoint equation and parameter correspondence. | Mathematical scalars replace formula-like code formatting. |

## Evaluation and experimental design guides

| Page | Definitions or changes | Corrected claim or scope |
| --- | --- | --- |
| [Evaluate a trained point](../how-to/evaluation/evaluate-a-run.md) | Sentence/corpus metric denominators and links to calibration, proof, and interval definitions. | Coverage and resampling units determine what an aggregate or interval estimates. |
| [Fit an external baseline](../how-to/evaluation/fit-a-baseline.md) | Categorical derivation factors, full forest and top-$k$ expected counts, floor-constrained updates, parameter convergence, and raising/coordination/coercion rules. | Lexical outcomes are POS tags; the convergence test is not a likelihood test, and the mixture/floor update carries no unrestricted claim of likelihood monotonicity. |
| [Choose a sweep design](../how-to/experiments/choose-a-sweep-design.md) | Cartesian products and yoking, random inclusion, discrete Latin hypercube indices, factorial generators, and error criteria. | Discrete rows can duplicate; exclusions can break design balance; declared dependence is an assumption, not an empirical test. |
| [Publish records](../how-to/experiments/publish-records.md) | Category sequences in mathematical prose use LaTeX. | Serialized examples remain literal code. |
| [Run on a cluster](../how-to/experiments/run-on-a-cluster.md) | Task index interval and allocation share domain. | The final chunk is bounded by the number of pending points. |

## Diagnostic guide

| Page | Definitions or changes | Corrected claim or scope |
| --- | --- | --- |
| [Run the packaged diagnostics](../how-to/probes/run-diagnostics.md) | Probability normalization, entropy, raw/probability margins, span aggregation, split posterior KL, controller relations, angular representation scores, recurrent changes, attention, group effects, and residual norm shares. | Masks and zero cases are explicit; scores describe observed encodings and supplied relations, not causal or linguistic identification; residual norm shares are not output energy fractions. |

## Tutorials and reference entry points

| Page | Definitions or changes | Corrected claim or scope |
| --- | --- | --- |
| [Your first run](../tutorials/01-first-run.md) | Application inference rules rendered as LaTeX. | Executable configuration and command examples retained. |
| [Declaring a grammar](../tutorials/02-declare-a-grammar.md) | Application inference rules rendered as LaTeX. | Surface notation examples remain literal grammar syntax. |
| [Declaring logical forms](../tutorials/04-declare-logical-forms.md) | Category erasure rendered with semantic type arrows. | Distinguished the mathematical type from its serialized spelling. |
| [Eval](../reference/eval.md) | Classification, macro-F1, Brier score, ECE, proof reproduction, and relaxation gap. | Undefined denominators, binning conventions, abstentions, degenerate proofs, and coverage are explicit. |
| [Stats](../reference/stats.md) | Percentile/BCa bootstrap equations and links to testing definitions. | Fallback cases and the sampling unit assumptions are explicit. |
| [Judgment](../reference/judgment.md) | Links the generated API to likelihood, linking, identification, and posterior definitions. | Avoided duplicating the complete mathematical chapter in the API listing. |
| [Neural components](../reference/nn.md) | Links the generated API to formal controller, chart, encoder, decoder, readout, and loss definitions. | Mathematical prose in generated docstrings is rendered as math. |

## Citation system and navigation

An actual BibTeX bibliography is rendered by Pandoc citeproc through the MkDocs/Markdown integration. The author–year style supports parenthetical `(Name Year)`, attributive `Name (Year)`, possessive `Name’s (Year)`, and paper reference `Name Year` forms. Full entries appear in the citing page’s References section. Literal code is excluded from citation processing.

| File or page | Change |
| --- | --- |
| [Documentation style](https://github.com/logical-form-induction/lofi/blob/main/docs/STYLE.md) | Mathematical notation and citation syntax conventions. |
| [Documentation development](documentation.md) | Citation authoring, build dependencies, and verification workflow. |
| [Home](../index.md) | Replaced manually written literature citations with BibTeX citations. |
| [Explanation index](../explanation/index.md) | Names the distinction between categories, semantic types, and logical forms. |
| Bibliography, CSL style, citation hook, and citation tests | Central bibliography; author–year rendering; missing-key/reference failures; tests for the supported forms and code protection. |
| MathJax configuration | Explicit support for bold Greek letters and serialized typesetting on initial load and instant navigation. The hook avoids processing already rendered mathematics. |

## Generated API documentation

The following 32 Python modules received mathematical changes confined to docstrings. Source links identify their repository locations; the revised wording is in the current working tree. Raw string literals protect LaTeX backslashes. Runtime behavior is unchanged.

| Area | Modules | Changes |
| --- | --- | --- |
| Judgment models | [lofi/judgment/mass.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/judgment/mass.py), [lofi/judgment/morcela.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/judgment/morcela.py), [lofi/judgment/bounded.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/judgment/bounded.py), [lofi/judgment/crossed.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/judgment/crossed.py), [lofi/judgment/_active.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/judgment/_active.py) | Linking equations, bounded location, parameter counts, and seeds for each round. |
| Neural representations | [lofi/nn/arity.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/nn/arity.py), [lofi/nn/parsers/_stacked.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/nn/parsers/_stacked.py), [lofi/nn/tokens/learned.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/nn/tokens/learned.py), [lofi/nn/ports.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/nn/ports.py), [lofi/nn/spacemap.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/nn/spacemap.py) | Role–filler binding, concatenation, initializer moments, and parameter domains; fixed overstrong norm claims. |
| Grammar and typing | [lofi/judge/_normal.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/judge/_normal.py), [lofi/spec/_rule.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/spec/_rule.py), [lofi/spec/grammar/_lexicon.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/spec/grammar/_lexicon.py), [lofi/typecheck/_templates.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/typecheck/_templates.py) | Normal form inequalities, type raising/coercion, coupling exponent, and semantic type arrows. |
| Probes and evaluation | [lofi/probe/_attribution.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/probe/_attribution.py), [lofi/probe/_scores.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/probe/_scores.py), [lofi/eval/grammaticality.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/eval/grammaticality.py), [lofi/eval/_seeds.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/eval/_seeds.py) | Integration counts, Jensen–Shannon bounds, noise-adjusted AUC, and interval domains. |
| Acquisition and scheduling | [lofi/spec/acquisition/collectors.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/spec/acquisition/collectors.py), [lofi/spec/_interpolation.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/spec/_interpolation.py), [lofi/run/_array.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/run/_array.py), [lofi/run/_boundary/site.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/run/_boundary/site.py) | Cutpoints, seed range, task chunks, and resource share domains. |
| Optimization and validation | [lofi/_checks.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/_checks.py), [lofi/data/_sampling.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/data/_sampling.py), [lofi/spec/layers.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/spec/layers.py), [lofi/spec/optimizers/_checks.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/spec/optimizers/_checks.py), [lofi/spec/optimizers/adam.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/spec/optimizers/adam.py), [lofi/spec/optimizers/adamw.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/spec/optimizers/adamw.py), [lofi/spec/schedulers/_checks.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/spec/schedulers/_checks.py), [lofi/spec/schedulers/constant.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/spec/schedulers/constant.py), [lofi/spec/schedulers/cosine.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/spec/schedulers/cosine.py), [lofi/spec/schedulers/linear.py](https://github.com/logical-form-induction/lofi/blob/main/lofi/spec/schedulers/linear.py) | Probability, dropout, optimizer, and scheduler parameter intervals. |

## Pages retained as procedural guides

The remaining operational pages need no independent mathematical model merely because they describe software. The following groups retain their command, schema, and workflow focus. Mathematical procedures shared with revised pages should be read with the linked central definitions. This is a scope decision, not a claim that every possible clarification or literature addition has been exhausted.

| Group | Retained focus and location of mathematical definitions |
| --- | --- |
| Installation, configuration, device policies, checkpoints | Environment setup, declaration composition, execution placement, and recovery. Composition and invariants are defined in the architecture explanations. |
| Corpus readers, featurization, judgment resources and item construction | Data formats, resource schemas, retained columns, and item construction. Sampling, response models, and token representations are defined in the revised data/model pages. |
| Sweep execution, contraction, tracking, publishing and report rendering | Orchestration, ledgers, sinks, serialization, and table formats. Design spaces and statistical commitments are defined in the sweep and experiment pages. |
| Plugins, conformance laws, debug checks, serving and export | Interfaces, registration, executable checks, bundles, and endpoints. These interfaces do not themselves imply mathematical guarantees. |
| Derivation decoding, model checking rewards, constraints and supervision | Configuration recipes for the decoding, fuzzy semantics, pinning, and anchoring definitions in Charts and approximation, Inference as evidence, and Induction objectives. |
| Probe configuration and external site maps | Selecting probes, observations, patches, and tensor sites. Probing through sites contains the intervention and recovery definitions; Run the packaged diagnostics supplies measurement equations. |
| Judgment fitting and later tutorials | Fit/run recipes and integrated examples. Judgment models and the model, training, and experiment guides hold the underlying equations. |
| Other generated reference pages | API declarations and signatures remain generated from source; only mathematical docstring expressions are converted, while shape tuples and executable syntax remain code. |

## Validation

- **Completed:** all 32 edited Python modules parse under Python 3.14; stripping docstrings yields executable ASTs identical to HEAD. Math literals containing backslashes use raw string prefixes.
- **Completed:** the grammar, data, sweep, acquisition, baseline, and diagnostic guide batches preserve their fenced code/output examples. Their citation keys resolve and display math delimiters balance. Whitespace checks pass.
- **Completed:** all four citation system tests pass, including the four requested citation forms, code protection, missing-key failures, and disambiguating year suffixes. Targeted Ruff lint and formatting checks pass for 35 Python files. The strict MkDocs build passes.
- **Completed:** browser checks on 20 representative pages found no MathJax errors, unrendered commands, or duplicate nested equations. The checked pages cover logical forms, derivations, charts, objectives, judgments, inference, probing, synthetic designs, experiments, specialized logics, trainer samplers, chart/term models, acquisition, baseline fitting, diagnostics, and the Eval, Stats, Judgment, and Neural reference pages. Visual inspection confirmed displayed equations, bold Greek notation, author–year citations, and page end hanging indent references. An internal-link navigation check also passed after the MathJax lifecycle fix.

## Verification limits

Equations were checked against the relevant implementation rather than inferred solely from component names. A documentation audit does not prove soundness, completeness, estimator unbiasedness, numerical accuracy, or validity outside the stated domains. Literature sources provide background and particular results; citing a related method does not establish architectural identity with LoFI.

The bibliography contains 91 distinct cited entries. Publisher, official proceedings, or journal records were consulted for their metadata and attribution. Access restrictions and conflicting records prevent an unconditional claim that every field was verified on its original publisher landing page. The following qualifications remain; entry level comments retain the verification trail.

| Source | Verification and limitation |
| --- | --- |
| \citealt{howard1980} | The original Academic Press chapter landing page was not located. Metadata was corroborated through an Elsevier-hosted bibliography; the text was checked against a nonpublisher transcription. |
| \citealt{plate1995} | The original IEEE page was inaccessible. PubMed supplied metadata and the abstract. The documentation distinguishes LoFI's Hadamard product binding from the paper's circular convolution. |
| \citealt{kuhlmann-etal2015} | The official ACL record and PDF give pages 187–219, retained here. An institutional copy gives 215–247; the original MIT landing page remained inaccessible. |
| \citealt{parsons1990} | The conventional 1990 date is corroborated in Cambridge and Wiley bibliographies. MIT's current retail listing gives January 28, 1991. This date discrepancy remains unresolved. |
| \citealt{huben-etal2024} | The author list follows the current ICLR proceedings record. Its linked PDF retains an older list/order. The citation supplies methodological background rather than an assertion that LoFI reproduces that architecture. |
| \citealt{bisk-hockenmaier2012} | The original AAAI issue and paper establish 2012. The migrated landing page's generated 2021 publication field is not used. |
| \citealt{lin1991} | Indexed IEEE metadata was checked, but direct access returned a robot challenge. The author's name and the paper's content were checked against an original article PDF hosted by the University of Florida. |
| Other challenged publisher pages | Indexed publisher records or publisher journal archives supported checks for Efron's bootstrap paper, McKay and colleagues' sampling paper, Gers and colleagues' LSTM paper, and the Kullback–Leibler and Benjamini–Yekutieli papers. These checks are distinct from unrestricted access to every live landing page. |

MathJax configuration follows its documentation for [bold symbols](https://docs.mathjax.org/en/v3.2/input/tex/extensions/boldsymbol.html) and [asynchronous typesetting](https://docs.mathjax.org/en/v3.2/web/typeset.html).

The final inventory covers 55 revised existing Markdown files, this report, and 32 Python modules with mathematical docstring changes. The page tables include every revised substantive page; procedural pages retain their operational focus as described above. The scan for backticks in prose found only literal API/configuration syntax and authoring examples after conversion. No further substantive formalization gap was identified in this pass. This is a source-grounded documentation audit, not an exhaustive literature review.

## References

\bibliography
