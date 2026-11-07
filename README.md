# LoFI

LoFI induces categorial grammars and typed logical forms from inferential
annotation. It treats categories as types, derivations as proofs, and logical
forms as proof terms so that grammatical assumptions can be compared within a
single implementation.

The base package contains the symbolic induction stack and does not require a
tensor library. Optional capabilities add neural models, corpus readers,
judgment models, reporting, publishing, and experiment tracking.

## Installation

LoFI requires Python 3.14 or newer. From a source checkout, install the base
distribution into the active environment with:

```bash
uv pip install "lofi @ file://$PWD"
```

The optional extras and their dependency boundaries are described in the
[installation guide](docs/how-to/setup/installing.md).

## Documentation

The [documentation index](docs/index.md) links the tutorial, task-oriented
guides, design explanations, and API reference. Contributors should begin with
[CONTRIBUTING.md](CONTRIBUTING.md).

## Component combinations

Every registered term encoder is paired with every registered term decoder
through the wiring boundary by `tests/composability/test_combinations.py`,
which generates the count below and checks that this section quotes it.

10 of 12 combinations construct; 2 declared incompatible

A pair that cannot work is declared, with the mechanism that excludes it, and
the test asserts that the wiring boundary refuses the pair with that reason:

| Term encoder | Term decoder | Reason |
|---|---|---|
| `tree-rnn` | `hrr-term-decoder` | selects publication 'codebook' with sharing the space producer does not publish |
| `transformer` | `hrr-term-decoder` | selects publication 'codebook' with sharing the space producer does not publish |

## Citation

Use [CITATION.cff](CITATION.cff) to cite the software and record the version
used in your analysis.

## License

LoFI is licensed under the [MIT License](LICENSE).
