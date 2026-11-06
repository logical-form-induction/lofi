# Grammars as deductive systems

This page explains how a grammar declaration becomes one deductive system: what
a signature, a rule inventory, a matching relation, a normal form and a goal
each contribute, what the compiler derives from them rather than reading from
the declaration, and why every formalism LoFI compares passes through the same
compiled value.

## One compiled value per grammar

A grammar declaration specifies grammatical components and computational
bounds; the consumer supplies the deduction strategy. Its complete form,
[`GrammarDecl`](../reference/configuration/grammar.md)
(`formalism: declared`), holds a signature, a rule set, a matching relation, a
normal form, a goal, a lexicon, an approximation budget and, optionally, the
hypotheses of a capacity theorem. A concise declaration such as `formalism:
ab`, `cfg`, `tlg` or `undirected-ccg` is a plugin that expands into a
`GrammarDecl`; after that expansion every formalism is the same kind of value.

Compilation produces a [`Formalism`](../reference/formalism.md), and
`Formalism.compile` returns the run-scoped `Grammar` that every downstream
consumer reads. The pieces correspond to the parts of a deductive system:

| Part | Role in the deductive system |
| --- | --- |
| Signature | The formulas. Sorts and operators over which categories are terms in one store. |
| Lexicon | The axioms. Each entry pairs a form with a category and, optionally, a meaning. |
| Rule set | The inference rules. Each rule relates premise categories to a conclusion. |
| Matching | When two categories count as the same premise. |
| Normal form | Which derivation steps are admitted; preservation and uniqueness require separate justification. |
| Goal | Which conclusions over which input regions count as a proof of the sentence. |
| Budget | How far an approximate consumer may go: term depth, beam, mass target, unary rounds, beta-reduction fuel. |

[Categories, semantic types, and logical forms](logical-forms-as-types.md)
distinguishes categories, semantic types, and proof terms. A common deduction
engine makes their computational interface uniform; it does not make different
grammars the same logic. The separation of inference rules from a parsing
strategy follows the deductive approach to parsing
\citep{pereira-warren1983,shieber-etal1995}.

## Categories, productions, and lexical axioms

A context-free grammar (CFG) is conventionally written $G=(N,\Sigma,P,S)$:
$N$ is a finite nonterminal alphabet, $\Sigma$ a disjoint terminal alphabet,
$P\subseteq N\times(N\cup\Sigma)^*$ a finite production set, and $S\in N$
the start symbol. A production is written $A\to\alpha$. These capital
letters name grammatical objects, not numerical scalars. LoFI's concise
CFG declaration separates productions over nonterminals from lexical
assignments of terminal forms to nonterminals. This separation supplies the
lexical axioms and structural rules of a deductive parser
\citep{shieber-etal1995}.

For an input $w_1\cdots w_n$, write $[A,i,j]$ for the assertion that category
$A$ derives the substring $w_{i+1}\cdots w_j$, with $0\leq i<j\leq n$.
For one-token lexical entries and a binary production, the corresponding
inference schemata are:

$$
\frac{A\in\operatorname{Lex}(w_{i+1})}{[A,i,i+1]}
\qquad
\frac{[B,i,k]\quad[C,k,j]\quad(A\to BC)\in P}{[A,i,j]}.
$$

Here $\operatorname{Lex}(w)$ is the set of categories assigned to $w$.
Productions of other positive arities use correspondingly many adjacent
premise spans; unary productions keep the same span. These are inference
schemata, not claims that every configured consumer supports every arity
or empty yield. The span consumer must support the grammar's profile.

For the ordinary directional fragment of combinatory categorial grammar
(CCG), categories are generated from atomic categories $a\in\mathcal A$ by
$A,B ::= a\mid A/B\mid A\backslash B$. In this CCG convention, both slash
categories return $A$, with $B$ sought on the right or left, respectively
\citep{steedman2000}. Forward and backward application are:

$$
\frac{[A/B,i,k]\quad[B,k,j]}{[A,i,j]}\;(>)
\qquad
\frac{[B,i,k]\quad[A\backslash B,k,j]}{[A,i,j]}\;(<).
$$

The AB fragment uses these application rules. CCG may additionally admit
composition and type raising. Familiar instances are forward harmonic
composition and forward type raising:

$$
\frac{[A/B,i,k]\quad[B/C,k,j]}{[A/C,i,j]}\;(>B)
\qquad
\frac{[A,i,j]}{[T/(T\backslash A),i,j]}\;(>T).
$$

For meanings $u$ and $v$, the corresponding forward application,
composition, and raising actions are $u\,v$, $\lambda x.u(vx)$, and
$\lambda f.fu$, with fresh bound variables and appropriate semantic types
\citep{steedman2000}. Their availability, composition degree, directions,
and permitted target categories are declaration choices. An unrestricted
metavariable $T$ in the raising schema does not imply that a finite forward
enumerator can list every instance.

Type-logical grammar instead states derivability by sequents such as
$A_1,\ldots,A_k\vdash B$. In the original associative Lambek calculus,
antecedents are nonempty ($k\geq1$) and ordered; exchange, weakening, and contraction are not generally
available \citep{lambek1958}. Its residual notation writes the left-taking
category as $B\backslash A$, reversing the CCG argument order above.
The packaged `tlg` declaration supplies particular rules and modalities;
sharing the deduction interface does not establish completeness for every
type-logical calculus.

## Rules are relations over terms

A rule is declared by its family (`application`, `composition`,
`substitution`, `type-raising`, `coordination`, `pattern` and others, each
registered in the `lofi.rule` entry-point group) and compiled to a value that
implements the [`Rule`](../reference/judge.md) protocol. The deduction engine
in `lofi.deduce` depends only on that protocol. It imports no formalism and no
grammar pack, and an import contract enforces this (see [The layer
order](layering.md)). One engine thus serves CFG productions, AB
application, CCG combinators and type-logical patterns alike, and a difference
between two grammars need not require a different deduction implementation.

For a category set $\mathcal C$, a rule of arity $k$ is a relation
$R\subseteq\mathcal C^k\times\mathcal C$. An instantiated rule licenses:

$$
\frac{[A_1,i_0,i_1]\quad\cdots\quad[A_k,i_{k-1},i_k]}
     {[A,i_0,i_k]}
\quad\text{if }(A_1,\ldots,A_k,A)\in R.
$$

The boundaries satisfy $i_0<\cdots<i_k$ for this nonempty contiguous-span
schema. The runtime also retains rule and normal-form state; the notation
suppresses that state to display the grammatical relation.

The matching relation decides when a category fills a premise slot. Identity
matching compares terms structurally; feature erasure, subsumption, alpha
equivalence, product and modal matchings specify other declared relations, each named and
registered in `lofi.matching`. Because the matching is a declared value rather
than a property of term equality, a grammar that ignores morphosyntactic
features and one that respects them differ in one field and are otherwise
compiled identically. The compiler refuses a goal whose target set was built
under a different matching from the rule set's, compared by identity.

## What the compiler derives

Compilation derives three reports from the grammar's components:

- **Ranks and unary closure.** `deduction_requirements` collects the arity of
  every rule. A rule set containing a unary rule requires unary-closure rounds
  in each chart cell, and a declaration whose budget allows zero such rounds is
  refused.
- **The reachable category set.** A closure report classifies the categories
  the rules can reach as `finite`, `infinite` or `unknown`, and lists the
  reached and the escaping categories. Every goal
  category must be reachable, or the grammar is refused.
- **A capacity report.** When a declaration supplies CCG-family evidence, the
  judge reports the weak generative capacity the theorem gives for that family
  together with its citation: bounded-degree pure CCG is a proper superclass of
  the context-free languages \citep{kuhlmann-etal2010}, and
  prefix-closed VW-CCG with target restrictions on application is
  TAG-equivalent \citep{kuhlmann-etal2015}. When the evidence is
  absent or a hypothesis fails (an unbounded composition degree, an inventory
  that is not exclusively generalized composition, a lexical-anchor
  restriction), the report is `not_characterized` and names the unmet
  hypothesis. An applicable theorem with no citation is refused. The report
  characterizes the rule family, not one concrete grammar.

For a grammar family $\mathcal G$, weak generative capacity concerns
$\{L(G):G\in\mathcal G\}$, where $L(G)$ is the set of strings derived by
$G$ under its grammatical start condition. The pure-CCG result ranges over
finite lexica and bounded but arbitrary composition degree, with application
as degree zero; it does not say that every degree bound exceeds context-free
power \citep{kuhlmann-etal2010}. The VW-CCG results concern a specified
generalized-composition fragment \citep{vijay-shanker-weir1994,kuhlmann-etal2015}.
Type raising, substitution, or coordination cannot be added while retaining
the theorem merely by retaining its name. Moreover, the TAG-to-CCG
construction in \citealt{kuhlmann-etal2015} permits empty-string lexical
entries. A declaration's evidence and the report's family label do not
establish that a particular terminal-anchored consumer realizes that full
family. Computational bounds and a fragment-covering goal are additional
choices, not hypotheses that can be omitted from a comparison.

The rule arities, unary closure, category-set class, fan-out and
lexicalization together form the [`DeductionProfile`](../reference/formalism.md).
A consumer that serves only some profiles, such as a neural chart parser built
for binary rules, declares a `ProfileBound`. `Formalism.compile` checks each
supplied bound and raises `ProfileUnsupportedError` naming the consumer, the
arities it serves and the arities the grammar uses. A grammar outside a
consumer's reach is thus refused at compilation rather than parsed
incorrectly.

## Normal forms

Composition and type raising may give a CCG several derivations with the
same interpretation, a form of spurious ambiguity \citep{eisner1996}.
For a fixed lexical leaf sequence over $w$, let $\mathcal D_G(w)$ be its
derivations, and let $d\sim d'$ mean equivalence of their semantic recipes
in the normal-form result. This excludes accidental equality of denotations
in a particular model and ambiguity between different lexical assignments.
A filter selects $\mathcal D_G^{\mathrm{NF}}(w)\subseteq\mathcal D_G(w)$.
Two distinct properties matter:

$$
\begin{aligned}
\text{Safety: }&
\forall d\in\mathcal D_G(w)\;\exists d'\in\mathcal D_G^{\mathrm{NF}}(w)
\quad d\sim d',\\
\text{Uniqueness: }&
\forall d_1,d_2\in\mathcal D_G^{\mathrm{NF}}(w),
\quad d_1\sim d_2\Longrightarrow d_1=d_2.
\end{aligned}
$$

Together they give exactly one representative per equivalence class.
\citet{eisner1996} calls the second property *completeness* and proves both
for the paper's pure-CCG setting. Removing rules can destroy safety: the
normal-form representative may require a rule that the restricted grammar
does not admit. Thus a filter's name is not a guarantee for an arbitrary
configured inventory.

LoFI registers `none`, `eisner`, `undirected-eisner`, `hockenmaier-bisk`,
and `all-of`, which conjoins filters. \citet{hockenmaier-bisk2010} addresses
bounded composition and grammatical type raising, but explicitly leaves a
general proof of exhaustive ambiguity coverage open. The paper supplies
local arguments and empirical evaluation; it also identifies cases in which
its normal forms require unbounded type-raising degree.
LoFI's compiler does not prove safety or uniqueness for each filter–inventory
pairing. The normal form is nevertheless part of the rule set and grammar
fingerprint, because changing the admitted derivations changes counting
and weighted inference.

## Goals

A goal says which chart items count as a parse. Let $\mathcal T$ be the
declared target categories and $\simeq$ their matching relation. The `root`
goal accepts exactly the items satisfying:

$$
\operatorname{accept}_{\mathrm{root}}([A,i,j])
\iff i=0\land j=n\land
\exists B\in\mathcal T\;(A\simeq B).
$$

Each target matching class may carry a scalar log weight $\omega_B$.
These weights need not be normalized probabilities.

For $n>0$, a cover consists of boundaries $0=i_0<\cdots<i_m=n$ such that
each span $[i_{r-1},i_r)$ contains a chart item and $m\leq m_{\max}$.
The runtime selects one tiling: `fewest-fragments` minimizes $m$ and prefers
the longest first fragment at a tie, recursively; `lexical` selects the
$n$ one-token spans. It does **not** marginalize over every possible tiling.
All items within each selected span form that fragment's alternative group.
For a nonnegative penalty $\lambda$, the added log score is:

$$
\operatorname{penalty}(m)=
\begin{cases}
-\lambda(m-1),&\texttt{after-first},\\
-\lambda m,&\texttt{each-fragment}.
\end{cases}
$$

A cover goal may score a string without a full-span derivation at a target
category. It thus answers a different question from grammatical acceptance
by the root goal. Empty inputs produce no cover in this implementation.

## Identity

A compiled grammar has a fingerprint: the SHA-256 of a canonical JSON record
of its name, signature digest, every rule's specification, the matching and
normal form, the closure report, the profile, the budget, the capacity report,
the clamps, the targets, the goal settings, the interpretations and the digest
of its checked theory. Two declarations that compile to the same parts have
the same fingerprint however they were spelled; a change to any part, the
budget included, changes it. [Documents are the declaration
language](documents-are-the-dsl.md) describes how that identity enters a
point's identity.

## What this licenses

Because every formalism compiles to one value read by one engine, a comparison
between two grammars varies only what their declarations vary. It does not
show that a difference in results reflects the grammars' generative capacity:
the budget, the normal form and the goal are part of each grammar and can
account for a difference on their own. A comparison that means to isolate the
rule inventory holds those three fixed. [Swappable grammatical
assumptions](swappable-assumptions.md) states this condition in general.

To declare each part, see [Declare a signature](../how-to/grammars/declare-a-signature.md),
[Choose combinatory rules](../how-to/grammars/choose-combinatory-rules.md) and
[Choose a normal form, matching and goal](../how-to/grammars/choose-normal-form-matching-goal.md).

## References

\bibliography
