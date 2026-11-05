# Judgment corpora

A judgment fit names its dataset by a design tag. Every installed tag
(`megaacceptability`, `megaacceptability-morcela`, `megaveridicality`,
`meganegraising` and `megaintensionality`) is read by one reader, which takes
every input from the data root `LOFI_DATA_ROOT` names. Nothing about a release,
its factors or a language model checkpoint is fixed in code: each tag reads the
design document `designs/<tag>.json`, and that document names the files it
reads and the attributes its factors are read from.

## Layout

```text
$LOFI_DATA_ROOT/
  designs/
    megaveridicality.json
    megaacceptability-morcela.json
    ...
  releases/
    mega-veridicality-v2.1/
      mega-veridicality-v2.tsv
      release.json
    mega-acceptability-v2/
      mega-acceptability-v2.tsv
      release.json
    morcela/
      unigrams.tsv
```

Only `designs/<tag>.json` has a fixed location. Every other path is whatever the
design document declares, relative to the data root; the `releases/` directory
above is a convention, not a requirement. The released response tables are used
exactly as their distributions ship them, so a release directory can be unpacked
in place. A design records the SHA-256 digest of its document and of every file
it reads in its provenance.

## Design documents

A design document declares where its judgments come from under `source`:

- `fragment`: a release's Layers records. `fragment` names the records'
  source, either `{"kind": "codec", "codec", "declaration", "responses"}`,
  which decodes the released `responses` file through the codec registered
  under `codec` in the `lairs.codecs` group, configured by the release
  `declaration` document, or `{"kind": "layers", "path"}`, a Layers export.
  `responses` names the responses read: `{"kind": "ordinal"}` ratings on the
  scale the release declares, `{"kind": "relation", "inventory"}` relations
  in response order, or `{"kind": "bounded"}` continuous responses on a
  finite scale. `readouts` is `null` or the readout columns joined to every
  row of an ordinal design (below), and `keep` is `null` or a bead
  expression over each judgment's own features, bound to `judgment`, that
  keeps the judgments it holds of, so
  `"judgment['nativeenglish'] == 'True'"` reads only the self-reported native
  speakers of a MegaAcceptability release.
- `bead`: constructed bead items and bead annotation records. `items` names a
  [judgment-item construction](construct-judgment-items.md) document,
  `judgments` the annotation records, one JSON object per line, and `encoding`
  the bead response encoding they answer.

Either way `factors` maps each factor name, in column order, to a
[bead expression](construct-judgment-items.md) over the judged item records. A
codec lowers what a release states of each item into such a record: the
release's own feature names are the record's attributes, so a MegaVeridicality
item carries `item['verb']`, `item['frame']` and `item['polarity']`. A
relation's record is that of its related-item key, the item identities in the
order the relation runs. A design records exactly the declared factors, and a
factor an item record cannot answer is refused before any row is read.

For instance, a MegaVeridicality v2.1 design reads:

```json
{
  "source": "fragment",
  "fragment": {
    "kind": "codec",
    "codec": "megaveridicality",
    "declaration": "releases/mega-veridicality-v2.1/release.json",
    "responses": "releases/mega-veridicality-v2.1/mega-veridicality-v2.tsv"
  },
  "responses": {"kind": "relation", "inventory": ["contradicts", "neutral", "entails"]},
  "factors": {
    "verb": "item['verb']",
    "frame": "item['frame']",
    "polarity": "item['polarity']",
    "conditional": "item['conditional']"
  },
  "readouts": null,
  "keep": null
}
```

MegaVeridicality responses `no`, `maybe` and `yes` are read as the relations
`contradicts`, `neutral` and `entails`; the inventory orders them as response
levels. The consequent of each item is generated from its antecedent, and each
row keeps its participant, so the design carries participant indices. A row whose
response field is empty contributes no design row.

## Readout columns

An ordinal design may join one of two readout sources to every row:

| `kind` | Fields |
| --- | --- |
| `root-mass` | `path`, a published root-mass artifact; `mass` and `complement`, the readout columns its pair fills |
| `causal-lm` | `element`, the element each item record renders its corpus text under, which the model scores; `unigrams`, the token-count table; `checkpoint`, a hub source `{kind: hub, repository, revision}` with a full 40-character `revision`; `artifacts`, the cache (`root`, `offline`) the model is fetched into; `columns`, the readout columns `log_probability`, `unigram_log_probability` and `length` fill |

MORCELA's unigram table is a UTF-8 tab-separated file with the header
`token	count`, one token per line. The checkpoint is fetched through the one
artifact fetcher into the declared cache and read from the fetched directory;
with `offline: true` a release absent from the cache is refused rather than
downloaded.

## Release metadata

A MegaAttitude release document (MegaVeridicality, MegaNegRaising,
MegaIntensionality) declares `source_id`, `release_id`, `language`,
`source_method`, `spdx` and, optionally, `eprint_refs`:

```json
{
  "source_id": "megaveridicality",
  "release_id": "v2.1",
  "language": "en",
  "source_method": "elicited",
  "spdx": "CC-BY-SA-4.0"
}
```

A MegaAcceptability release document declares `source_id`, `release_id`,
`spdx`, the response scale as `scale_min` and `scale_max`, and two reference
maps, `frame_refs` and `verb_refs`, which give a Layers record reference for
every frame and every verb in the table. Version 1 tables spell a missing
response `NA`; version 2 tables leave the field empty. A participant may rate
the same stimulus more than once, and each rating is kept.

## Checking a local release

The integration test `tests/integration/data/test_megaveridicality_release.py`
reads a local MegaVeridicality v2.1 distribution through the registered design:

```bash
LOFI_MEGAVERIDICALITY=~/Downloads/mega-veridicality-v2.1 \
  uv run pytest --run-integration tests/integration/data/test_megaveridicality_release.py
```

## Agreement ceiling

`scripts/judgment/agreement_ceiling.py` computes the split-half annotator
agreement ceiling of a registered ordinal design, grouping its responses by the
levels of the factors `--group` names, each repeated `--group` one more factor
of the design's own document:

```bash
uv run python scripts/judgment/agreement_ceiling.py megaacceptability \
  --group verb --group frame --splits 10 --level 0.95 --out ceiling.json
```

The record names the dataset, the grouping factors and the digest of every file
the design read, so the MORCELA pilot can check the ceiling against its
candidate's design. A participant filter is the design document's `keep`.
