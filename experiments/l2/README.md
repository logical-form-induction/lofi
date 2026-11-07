# L2 source-logic experiments

This experiment asks whether a learner recovers the logical fragment that generated
its training pairs. The source declaration owns string generation and inference
labels, while `model.language` independently owns the candidate grammar and
semantic inventory. Thus every source fragment can be crossed with every
candidate logic without changing the corpus that a row compares.

`source-factors.yaml` is both the production comparison and its
scientifically minimal reduced freeze. It crosses quantifier-free,
universal-only, existential-only, and second-order source fragments with the
same four candidate fragments. Every candidate is fit at a declared
parameter-count target. The 16 primary cells each receive three
signature-twin augmentations, while the four reversed-ground-truth and four
lexicon-scrambled controls retain only the observed assignment. Thus the one
canonical comparison contains 72 points: 16 observed primaries, 48 signature
twins, and eight validity controls. A smaller grid cannot emit a verdict for
every source while satisfying the reducer's candidate, twin, and control
requirements.

The comparison reports held-out string log likelihood, description length,
and three-way inference accuracy. Its reducer reads source identity from
`comparison.source`, candidate identity from the learner grammar, and capacity
from the measured point. It produces one verdict and one source-by-winner row
per source fragment after checking candidate, signature-twin, and validity
controls.

`vocabulary-size.yaml` varies the source constant inventory, and
`decompositions.yaml` contrasts the flat connective inventory with a
CAUSE/BECOME decomposition. These smaller documents use the same run boundary
as the source comparison.

The logical declarations use Vampire's THF interface so that the second-order
condition is decided by the same external prover used during data generation.
Vampire is the sole external execution prerequisite and must be on `PATH`
when the comparison is expanded, executed, or ingested. After freeze 1 has
been minted, the tagged checkout is launched with:

```bash
uv run lofi launch \
  --experiment experiments/l2/source-factors.yaml \
  --freeze freezes/1/FREEZE.json --out launch.json
```

Its manifest is `freezes/1/FREEZE.json`, and its point results are written
below `results/1/source-factors/`. The completed results are ingested with:

```bash
uv run lofi ingest --experiment experiments/l2/source-factors.yaml \
  --freeze freezes/1/FREEZE.json --root .
```

Ingestion must yield decided-label rows, one `source-comparison-verdict` per
fragment, and the four-row `source-by-winner` matrix. Ordinary document and
proof tests remain executable without Vampire; execution tests skip when it is
absent.
