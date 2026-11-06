# Probe role inventory

Diagnostics declare the roles they require. Each immutable `SiteMap` names one
component and one `ComponentKind`; every site in the map belongs to that component.
`ensure_publishes` reads this metadata and holds a real component's map to the axes,
numeric classes, and inventory for its kind. `SiteMapBuilder.build` performs the same
check. `INVENTORY`, `ROLE_AXES`, and `ROLE_DTYPE` in `_roles.py` define these
constraints.

External models, parameter-only views, and components with no shared role inventory
use `SiteMap.custom(component, sites)` or `SiteMapBuilder.custom(component)`. This
explicit path still enforces single-component ownership and immutability, but it does
not apply a shared role inventory.

## Component requirements

Required roles must appear for every component of the given kind. Roles in a named
structure are optional as a group: a site map may omit the structure or publish every
role in it. Roles in a named alternative group are mutually exclusive: a site map
must publish exactly one of them. Roles in a capability family may all be absent;
otherwise, the roles present must equal one of the declared capability sets.

| kind | required | structures | alternatives | capabilities |
| --- | --- | --- | --- | --- |
| `token-encoder` | `TOKEN_EMBEDDING` | none | none | none |
| `subword-pooler` | `WORD_VECTOR` | none | none | none |
| `term-encoder` | `TERM_ENCODING`, `NODE_ENCODING` | `attention`: `ATTENTION_LOGITS`, `ATTENTION_WEIGHTS`; `residual`: `RESIDUAL_STREAM`, `FFN_PREACTIVATION` | none | `recurrence`: (`RECURRENT_HIDDEN`) / (`RECURRENT_HIDDEN`, `GATE`) / (`RECURRENT_HIDDEN`, `RECURRENT_CELL`, `GATE`) |
| `term-decoder` | `SYMBOL_DISTRIBUTION` | `attention`: `ATTENTION_LOGITS`, `ATTENTION_WEIGHTS` | `state`: `DECODER_RECURRENT_STATE`, `UNBOUND_VECTOR`, `DECODER_RESIDUAL_STREAM` | none |
| `controller` | `COMBINATOR_LOGITS` | none | none | none |
| `parser` | `CHART_CELL` | `attention`: `ATTENTION_LOGITS`, `ATTENTION_WEIGHTS`; `deep`: `INSIDE_SCORE`, `OUTSIDE_SCORE`, `SPLIT_DISTRIBUTION` | none | `span-context`: (`OUTSIDE_CELL`, `SPAN_REPRESENTATION`) |

The three recurrence capability sets describe a vanilla recurrence, a GRU, and an
LSTM, respectively. A term encoder that is not recurrent publishes none of them.

## Role conventions

Diagnostics address reading axes by name. Site specifications must use the axes and
numeric class listed here, and `ensure_publishes` holds a map to both.

| role | axes, outermost first | numeric class |
| --- | --- | --- |
| `TOKEN_EMBEDDING` | `item`, `row`, `width` | `real` |
| `WORD_VECTOR` | `item`, `word`, `width` | `real` |
| `TERM_ENCODING` | `term`, `width` | `real` |
| `NODE_ENCODING` | `term`, `slot`, `width` | `real` |
| `SYMBOL_DISTRIBUTION` | `term`, `slot`, `symbol` | `log-probability` |
| `DECODER_RECURRENT_STATE` | `term`, `slot`, `width` | `real` |
| `UNBOUND_VECTOR` | `term`, `slot`, `width` | `real` |
| `DECODER_RESIDUAL_STREAM` | `term`, `slot`, `width` | `real` |
| `RESIDUAL_STREAM` | `batch`, `position`, `feature` | `real` |
| `FFN_PREACTIVATION` | `batch`, `position`, `feature` | `real` |
| `CHART_CELL` | `item`, `span`, `width` | `real` |
| `OUTSIDE_CELL` | `item`, `span`, `width` | `real` |
| `SPAN_REPRESENTATION` | `item`, `span`, `span_width` | `real` |
| `RECEIVER_EDGE` | `item`, `edge`, `width` | `real` |
| `INSIDE_SCORE` | `item`, `span` | `real` |
| `OUTSIDE_SCORE` | `item`, `span` | `real` |
| `SPLIT_DISTRIBUTION` | `item`, `span`, `split` | `log-probability` |
| `CONSTITUENT_MARGINAL` | `item`, `span` | `probability` |
| `LOG_PARTITION` | `item` | `real` |
| `IDENTITY_SCORE` | `item`, `span`, `split` | `real` |
| `COMBINATOR_LOGITS` | `cell`, `action` | `log-probability` |

`PARAMETER` has no fixed axes or numeric class. Its site specification describes the
component-specific parameter shape. `RECURRENT_HIDDEN`, `RECURRENT_CELL`, `GATE`,
`ATTENTION_LOGITS` and `ATTENTION_WEIGHTS` have no entry either. A point of one of
these roles is published with `SiteMapBuilder.publish_at`, and its axes and numeric
class go unchecked.

## Build and validate a site map

`SiteMapBuilder.publish` adds a role with its standard axes and numeric class.
`SiteMapBuilder.build` validates the result against a component kind.

```python
published = (
    SiteMapBuilder(ComponentId("cky-parser"), ComponentKind.PARSER)
    .publish(Role.CHART_CELL)
    .publish(Role.INSIDE_SCORE)
    .publish(Role.OUTSIDE_SCORE)
    .publish(Role.SPLIT_DISTRIBUTION)
    .build()
)
```

Call `ensure_publishes(published)` directly to validate a tagged site map that was
built elsewhere.
