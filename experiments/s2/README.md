# Source-grammar comparison campaign

This campaign asks whether held-out fit, charged for description length,
identifies the categorial grammar that generated a corpus.
`source-complexity.yaml` declares eight grammars that vary along four
dimensions: the functor notation (directed slashes or an undirected arrow), the
rule set (application alone, first-order composition, composition with type
raising to `S`, or composition of unbounded order), the size of the primitive
inventory (2, 4, 8, or 16), and the maximum depth of a lexical category (1, 2,
or 3). Each grammar's name spells these out, so `slashes-raised-k8-depth3` is
the directed grammar with type raising over eight primitives and lexical
categories up to depth three.

Every grammar serves once as the source. A corpus is sampled from the source,
and each of the eight grammars is then fitted to it as a candidate. The
candidate supplies the learner's language and its combinator decoders, while
one capacity target keeps the model at a fixed parameter count across
candidates. Two validity controls are fitted with the source grammar as the
candidate: a reversed ground truth, which reverses the order of the source
signature's primitive inventory, and a scrambled lexicon, which redraws the
source lexicon under a different seed. Each primary fit is also paired with
three signature twins: fits of the same candidate under distinct non-identity
permutations of its primitives, constructors, and combinators, which leave its
capacity unchanged. Thus the design crosses eight sources with eight candidates
and two controls at training sizes of 16, 64, and 256 sentences, under five
seeds.

Every point reports the per-sentence string log-likelihood and the description
length of its fitted grammar on the held-out split. The source-comparison
reducer charges each fit its description length per held-out unit and, within
each source and data size, tests the source against every other candidate and
both controls with paired permutation tests and equivalence tests at the
preregistered margin. Those contrasts are corrected together by the
experiment's declared correction. The verdict is positive only when the source
ranks first, beats every candidate and its signature twins, and both controls
pass. Otherwise the reducer classifies the winner as simpler, richer, or
differently directed than the source. The source-by-winner matrix and the first
data size at which the verdict turns positive are reported beside it.
`source-complexity.json` preregisters all twenty-eight pairwise grammar
contrasts in one family.

`reduced-freeze.yaml` is the executable local contraction. It samples one
corpus of eight sentences from `slashes-application-k2-depth1`, fits that
grammar and `arrow-composition-k4-depth2` as candidates with their signature
twins, and fits both validity controls, for ten points under one seed. Its test
trains every point through the production executor and reduces the real
evaluator rows to one verdict and one source-by-winner cell.
