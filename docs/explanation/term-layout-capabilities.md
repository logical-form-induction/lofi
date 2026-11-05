# Term-layout capability is not an arity ceiling

This page explains why the arity of the terms LoFI can represent and the arity
a particular neural family can parameterize are two separate questions. A
**layout capability** determines how a term is laid out in slots. A
**rank-family capability** determines which operator arities a neural family
has declared parameters for. N-ary support in the term algebra does not answer
the second question, and a rank-family limit does not make the term algebra
binary.

## Supported paths

The package exercises these paths with a signature containing a ternary
constructor:

| Layout | Exact neural pairing | Why the pairing works |
| --- | --- | --- |
| Preorder with arity | Holographic term encoder and holographic term decoder | The codec records the constructor symbols and parent channels directly. Holographic components bind and unbind slot roles, so they do not require term-independent tree paths. |
| Complete tree | Holographic term encoder and recursive term decoder | The layout computes its branching factor from the operators the grammar reaches. A reached ternary constructor therefore supplies three child positions to the decoder, and the holographic encoder supplies the decoder's input representation and width. |

Both paths use the ordinary layout declarations and component factories. Each
encodes, decodes and assigns a finite score to a term whose root has three
children.

## Where a rank limit applies

Some neural families allocate separate parameters by operator rank. These
families declare an `ArityBound`. Construction compares the arities the grammar
reaches with that declaration. If a declaration covers ranks one and two but
the grammar reaches the ternary constructor `triple`, construction raises
`ArityUnsupportedError` and names both `arity 3` and `triple`.

This refusal is local to the rank-indexed declaration. It does not invalidate
the ternary term, the preorder encoding, the complete-tree encoding or either
supported decoder path above. Widening such a family requires it to declare
and provide parameters for the additional rank; the layouts need no new arity
mechanism.

## Why the distinction matters

A comparison between grammars that reach different operator arities is a
comparison only if every arm runs under a component that serves its arities.
Because the rank limit is declared and checked at construction, an arm that
exceeds it is refused rather than run under a degraded representation, and the
refusal names the component responsible. The same principle governs rule
arities in the chart, through deduction profiles (see [Grammars as deductive
systems](derivations-as-proofs.md) and [Swappable grammatical
assumptions](swappable-assumptions.md)). To choose a layout and a
component pairing, see [Configure term encoders and
decoders](../how-to/models/configure-term-encoders-and-decoders.md).
