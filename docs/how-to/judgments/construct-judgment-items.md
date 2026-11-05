# Construct judgment items

A judgment-item construction document declares a set of
[bead](https://github.com/FACTSlab/bead) items: the templates filled, the
lexicons their slots are filled from, the bead strategy each template is filled
by, the bead item template naming what each item renders and the constraints
every item satisfies, and the attributes each item records. Every resource is
written in bead's own serialized shape, so a document holds no LoFI notion of
what an item is about: a template's slots, a lexical item's features and an
item's attributes are whatever the document names. The active-learning pool,
a [bead design](judgment-corpora.md) and a
[predicted judgment source](train-on-judgments.md) are all built from such a
document by `lofi.judgment.construct_items`.

## Resources

| Field | What it declares |
| --- | --- |
| `namespace` | The UUID namespace every constructed identity derives from. |
| `templates` | Bead templates: `{kind: jsonl, path}`, one template per line; `{kind: collection, path}`, a bead `TemplateCollection` file; or `{kind: inline, template}`. |
| `lexicons` | Bead lexicons: `{kind: jsonl, path, name}`, one lexical item per line; `{kind: table, path, name, format, language_code, column_mapping, feature_columns, pos}`, a CSV or TSV table read by bead's `from_csv` or `from_tsv`; `{kind: glazing, name, resource, query, language_code, options}` or `{kind: unimorph, name, query, language_code, options}`, fetched through bead's resource adapters; or `{kind: inline, lexicon}`. |
| `generators` | Generators from the `lofi.lexicon_generator` entry-point group, each proposing one more lexicon. |
| `filling` | The bead strategy a template is filled by unless `template_fillings` names it. |
| `template_fillings` | The strategy each named template is filled by instead. |
| `item_template` | The bead item template, `{kind: json, path}` or `{kind: inline, item_template}`. |
| `constraints` | The bead constraints the item template names, `{kind: jsonl, path}` or `{kind: inline, constraint}`; exactly the constraints it names are declared. |
| `admit` | `null`, or a bead expression over an item's fillings that admits the combination as an item. |
| `attributes` | Each item attribute and the bead expression it is read by. |

Every declared resource carries its bead `id`, and one without is refused,
since bead would draw it at random. A lexicon bead loads from a table or a
resource adapter draws its items' identities at random, so each is derived
instead from the namespace, the lexicon's name and the item's content.

## Filling strategies

Each strategy is bead's:

| `strategy` | Fields | Bead strategy |
| --- | --- | --- |
| `exhaustive` | none | `ExhaustiveStrategy`: every combination. |
| `random` | `samples`, `seed` | `RandomStrategy`: a seeded sample. |
| `stratified` | `samples`, `grouping_property`, `seed` | `StratifiedStrategy`: a sample balanced over a filler property such as `pos` or `features.number`. |
| `csp` | `count`, `max_attempts` | `CSPFiller`: backtracking search honoring a template's multi-slot constraints. |
| `mlm` | `checkpoint`, `artifacts`, `device`, `beam_size`, `fill_direction`, `custom_order`, `top_k`, `budget`, `per_slot_max_fills`, `per_slot_enforce_unique` | `MLMFillingStrategy` through bead's `HuggingFaceMLMAdapter`: beam search ranking the lexicon's admissible items by the model. |
| `mixed` | `slots`, `masked_lm_slots`, `masked_lm`, `default` | `MixedFillingStrategy`: each named slot by its own enumerating strategy, the masked-LM slots by the one masked-LM filling, in the context of each first-pass filling. |
| `streaming` | `max_combinations` | `StreamingFiller`: lazy enumeration, bounded or not. |

A `seed` of `null` is derived from the namespace and the template's identity,
so a document always constructs the same items. Only `csp` reads a
template's multi-slot constraints, so a template declaring one is refused under
any other strategy.

A masked-LM *filling* and a masked-LM *generator* are different things. bead's
masked-LM strategy fills a template's slots with items the lexicons already
hold, ranked by the model; it proposes no lexical item. The `masked-lm`
generator proposes new lexical items: the `count` highest-scoring distinct
single-token completions of its prompts under its model, each recording
`masked-lm:<repository>@<revision>` as its source. Both name their model as a
`checkpoint`, a hub source `{kind: hub, repository, revision}` pinned to a full
commit, and the `artifacts` cache (`root`, `offline`) it is fetched into; the
model is read from the directory the one artifact fetcher materializes, so
neither the generator nor bead's adapter downloads anything itself.

## Items

Items are built by bead's `ItemConstructor`, which renders each element and
evaluates the item template's constraints, including bead's model constraints
(`lm_prob`, `nli`, `similarity`) through bead's model registry; an item a
constraint rejects is not constructed. An item template whose
`filled_template_ref` elements name no template renders each filling of every
template as one item. One whose elements each name a declared template, by
`filled_template_ref_id`, renders every combination of those templates'
fillings as one item, each element showing its template's filling, and every
declared template is rendered by some element.

## Attributes

An attribute, and the `admit` expression, is a bead expression over one item's
fillings:

| Variable | Bound to |
| --- | --- |
| `slots` | For an item of one template, each slot's lexical item: `slots.verb.lemma`, `slots.verb.features['number']`. |
| `template` | For an item of one template, the bead template: `template.name`. |
| `templates` | For an item combining several templates, each template's name mapped to its `template` and `slots`: `templates.first.slots.node.lemma`. |
| `item_template` | The bead item template: `item_template.template_metadata['study']`. |
| `elements` | Each element's declaration by name: `elements.stimulus.element_metadata['shown']`. |

No slot name is reserved. No attribute is recorded that the document does not
declare, and every rule that reads an item's attributes later reads them
through `item`, so `item['verb']` reads the attribute `verb`.

An item's identity is a UUID in `namespace` over, for each template it renders,
the template's id and the id of the lexical item filling each slot, in
slot-name order, so bead returns judgments under the same identities on every
read.

## Graph-seeded pairs

The `entailment-graph` generator proposes a local entailment graph's nodes as
one lexicon. Each node's features record, under the declared `feature`, every
node it is paired with and the pair's declared label: `linked` for each node a
graph link reaches, and `sampled` for each node of a seeded draw of
`negative_ratio` outside pairs per link. A pair item combines two one-slot
templates, admitted only where the second node is paired with the first:

```json
{
  "generators": [
    {
      "kind": "entailment-graph",
      "graph": {"kind": "edge-list", "source": "graph.tsv", "language": "en"},
      "lexicon": "nodes",
      "feature": "paired",
      "linked": "plausible",
      "sampled": "presumed-non-inference",
      "negative_ratio": 1,
      "seed": null
    }
  ],
  "admit": "templates.second.slots.node.lemma in templates.first.slots.node.features['paired']",
  "attributes": {
    "left": "templates.first.slots.node.lemma",
    "right": "templates.second.slots.node.lemma",
    "seed": "templates.first.slots.node.features['paired'][templates.second.slots.node.lemma]"
  }
}
```
