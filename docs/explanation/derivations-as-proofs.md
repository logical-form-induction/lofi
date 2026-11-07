# Grammars as deductive systems

## From words to a derivation

Take *John sleeps*, with *John* assigned $\mathrm{NP}$ and *sleeps*
assigned $\mathrm{S}\backslash\mathrm{NP}$. The latter category says that
the expression needs a noun phrase on its left to form a sentence. A
derivation records the two lexical assignments and the application rule
that combines them. This proves grammatical acceptance under the declared
rules. Whether John sleeps is a question for the sentence's interpretation.

Number the word boundaries as $0\;\text{John}\;1\;\text{sleeps}\;2$.
Then $[\mathrm{NP},0,1]$ says that the first word can serve as an NP,
and $[\mathrm{S}\backslash\mathrm{NP},1,2]$ says what the second word
can combine with. Read the rule as “if both assertions above the line hold,
the assertion below follows”:

$$
\frac{[\mathrm{NP},0,1]\qquad[\mathrm{S}\backslash\mathrm{NP},1,2]}
     {[\mathrm{S},0,2]}.
$$

A context-free grammar can describe the same string with lexical categories
$\mathrm{NP}$ and $\mathrm{VP}$ and the production
$\mathrm{S}\to\mathrm{NP}\;\mathrm{VP}$. Here the production states the
combination that the slash category expressed in the first analysis. The
analyses reach the same successful result through these different grammatical
representations.

## One compiled value per grammar

A grammar declaration specifies grammatical components and computational
bounds; the consumer supplies the deduction strategy. Its complete form,
[`GrammarDecl`](../reference/configuration/grammar.md)
([`formalism: declared`](../how-to/grammars/choose-a-formalism.md)), holds a signature, a rule set, a matching relation, a
normal form, a goal, a lexicon, an approximation budget and, optionally, the
hypotheses of a capacity theorem. A concise declaration such as `formalism:
ab`, `cfg`, `tlg` or `undirected-ccg` is a plugin that expands into a
`GrammarDecl`; after that expansion every formalism is the same kind of value.

Compilation produces a [`Formalism`](../reference/formalism.md#lofi.formalism.Formalism), and
[`Formalism.compile`](../reference/formalism.md#lofi.formalism.Formalism.compile) returns the run-scoped [`Grammar`](../reference/formalism.md#lofi.formalism.Grammar) that every downstream
consumer reads. The pieces correspond to the parts of a deductive system:

| Part | Role in the deductive system |
| --- | --- |
| Signature | The formulas. Sorts and operators over which categories are terms in one store. |
| Lexicon | The axioms. Each entry pairs a form with a category and, optionally, a meaning. |
| Rule set | The inference rules. Each rule relates premise categories to a conclusion. |
| Matching | When two categories count as the same premise. |
| Normal form | Which derivation steps are admitted; preservation and uniqueness require separate justification. |
| Goal | Which conclusions over which input regions count as a proof of the sentence. |
| Budget | How far an approximate consumer may go: term depth, beam, mass target, unary rounds, beta reduction fuel. |

[Categories, semantic types, and logical forms](logical-forms-as-types.md)
distinguishes categories, [semantic types](logical-forms-as-types.md#semantic-typing), and proof terms. A common deduction
engine supplies a uniform computational interface to grammars whose logical
rules can differ. The separation of inference rules from a parsing
strategy follows the deductive approach to parsing
\citep{pereira-warren1983,shieber-etal1995}.

## Categories, productions, and lexical axioms

A context-free grammar (CFG) is conventionally written $G=(N,\Sigma,P,S)$:
$N$ is a finite nonterminal alphabet, $\Sigma$ a disjoint terminal alphabet,
$P\subseteq N\times(N\cup\Sigma)^*$ a finite production set, and $S\in N$
the start symbol. A production is written $A\to\alpha$. These capital
letters name the grammatical objects just defined. LoFI's concise
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
premise spans; unary productions keep the same span. Executing these inference
schemata requires a consumer that supports the grammar's arities and yield
conditions, as declared in its profile.

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
composition and [type raising](../how-to/grammars/choose-combinatory-rules.md#the-rule-families). Familiar instances are forward harmonic
composition and forward type raising:

$$
\frac{[A/B,i,k]\quad[B/C,k,j]}{[A/C,i,j]}\;(>B)
\qquad
\frac{[A,i,j]}{[T/(T\backslash A),i,j]}\;(>T).
$$

For meanings $u$ and $v$, the corresponding forward application,
composition, and raising actions are $u\,v$, $\lambda x.u(vx)$, and
$\lambda f.fu$, with fresh bound variables and appropriate [semantic types](logical-forms-as-types.md#semantic-typing)
\citep{steedman2000}. Their availability, composition degree, directions,
and permitted target categories are declaration choices. An unrestricted
metavariable $T$ ranges over target categories, so unrestricted raising can
have more instances than a finite forward enumerator can list.

Type-logical grammar instead states derivability by sequents such as
$A_1,\ldots,A_k\vdash B$. In the original associative Lambek calculus,
antecedents are nonempty ($k\geq1$) and ordered; exchange, weakening, and contraction are not generally
available \citep{lambek1958}. Its residual notation writes the left-taking
category as $B\backslash A$, reversing the CCG argument order above.
The packaged `tlg` declaration supplies particular rules and modalities;
completeness must be established for the particular calculus those declarations
implement.

## Rules are relations over terms

A rule is declared by its family (`application`, `composition`,
`substitution`, `type-raising`, `coordination`, `pattern` and others, each
registered in the `lofi.rule` [entry point](swappable-assumptions.md#how-a-family-is-discovered) group) and compiled to a value that
implements the [`Rule`](../reference/judge.md#lofi.judge.Rule) protocol. The deduction engine
in [`lofi.deduce`](../reference/deduce.md#lofi.deduce) depends only on that protocol. It imports no formalism and no
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
schema. The runtime also retains rule and normal form state; the notation
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

- **Ranks and unary closure.** [`deduction_requirements`](../reference/judge.md#lofi.judge.deduction_requirements) collects the arity of
  every rule. A rule set containing a unary rule requires unary closure rounds
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
  that is not exclusively generalized composition, a lexical anchor
  restriction), the report is `not_characterized` and names the unmet
  hypothesis. An applicable theorem with no citation is refused. The report
  characterizes the rule family over its admissible grammars.

For a grammar family $\mathcal G$, weak generative capacity concerns
$\{L(G):G\in\mathcal G\}$, where $L(G)$ is the set of strings derived by
$G$ under its grammatical start condition. The pure-CCG result ranges over
finite lexica and bounded but arbitrary composition degree, with application
as degree zero. Establishing greater-than-context-free power for a particular
degree bound requires a result for that bound \citep{kuhlmann-etal2010}. The VW-CCG results concern a specified
generalized composition fragment \citep{vijay-shanker-weir1994,kuhlmann-etal2015}.
[Type raising](../how-to/grammars/choose-combinatory-rules.md#the-rule-families), substitution, or coordination cannot be added while retaining
the theorem merely by retaining its name. Moreover, the TAG-to-CCG
construction in \citealt{kuhlmann-etal2015} permits empty-string lexical
entries. Applying that family result to a terminal-anchored consumer requires
showing that the consumer realizes the same family. A comparison must also
account for its computational bounds and any fragment-covering goal.

The rule arities, unary closure, category set class, fan-out and
lexicalization together form the [`DeductionProfile`](../reference/formalism.md#lofi.formalism.DeductionProfile).
A consumer that serves only some profiles, such as a neural [chart parser](../how-to/models/configure-a-chart-parser.md) built
for binary rules, declares a [`ProfileBound`](../reference/formalism.md#lofi.formalism.ProfileBound). `Formalism.compile` checks each
supplied bound and raises [`ProfileUnsupportedError`](../reference/formalism.md#lofi.formalism.ProfileUnsupportedError) naming the consumer, the
arities it serves and the arities the grammar uses. A grammar outside a
consumer's reach is thus refused at compilation.

## Normal forms

Suppose three expressions have categories $A/B$, $B/C$, and $C$, and the
grammar admits both application and [forward composition](../how-to/grammars/choose-combinatory-rules.md#the-rule-families). One derivation
combines the last two expressions to obtain $B$, then applies the first to
obtain $A$. Another first composes the two functions to obtain $A/C$, then
applies that result to $C$. With meanings $f$, $g$, and $x$, both compute
$f(g(x))$. They are different derivations of the same semantic recipe.

A normal form can remove such redundant routes. But removing a route is
safe only if an equivalent permitted route remains. The two properties
below separate preserving at least one route from leaving at most one.

Composition and [type raising](../how-to/grammars/choose-combinatory-rules.md#the-rule-families) may give a CCG several derivations with the
same interpretation, a form of spurious ambiguity \citep{eisner1996}.
For a fixed lexical leaf sequence over $w$, let $\mathcal D_G(w)$ be its
derivations, and let $d\sim d'$ mean equivalence of their semantic recipes
in the normal form result. This excludes accidental equality of denotations
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
normal form representative may require a rule that the restricted grammar
does not admit. Safety thus depends on the configured inventory as
well as the selected filter.

LoFI registers `none`, [`eisner`](../how-to/grammars/choose-normal-form-matching-goal.md#declare-a-normal-form), `undirected-eisner`, `hockenmaier-bisk`,
and `all-of`, which conjoins filters. \citet{hockenmaier-bisk2010} addresses
bounded composition and grammatical type raising, but explicitly leaves a
general proof of exhaustive ambiguity coverage open. The paper supplies
local arguments and empirical evaluation; it also identifies cases in which
its normal forms require unbounded type raising degree.
LoFI's compiler does not prove safety or uniqueness for each filter–inventory
pairing. The normal form is nevertheless part of the rule set and grammar
fingerprint, because changing the admitted derivations changes counting
and weighted inference.

## Goals

Parsing every word or short phrase does not necessarily yield an analysis
of the whole sentence. A root goal requires the complete input to have a
declared target category, often $\mathrm{S}$. A cover goal can instead score several parsed
fragments that collectively cover the input. For a string whose two halves
parse but cannot combine, a cover can provide a score even though the root
goal rejects it. The choice thus changes what a reported score means.

A goal says which chart items count as a parse. Let $\mathcal T$ be the
declared target categories and $\simeq$ their matching relation. The [`root`](../how-to/grammars/choose-normal-form-matching-goal.md#declare-a-goal)
goal accepts exactly the items satisfying:

$$
\operatorname{accept}_{\mathrm{root}}([A,i,j])
\iff i=0\land j=n\land
\exists B\in\mathcal T\;(A\simeq B).
$$

Each target matching class may carry a scalar log weight $\omega_B$.
The target categories carry log weights, with no requirement that their
exponentials sum to one.

For $n>0$, a cover consists of boundaries $0=i_0<\cdots<i_m=n$ such that
each span $[i_{r-1},i_r)$ contains a chart item and $m\leq m_{\max}$.
The runtime selects one tiling: [`fewest-fragments`](../how-to/grammars/choose-normal-form-matching-goal.md#declare-a-goal) minimizes $m$ and prefers
the longest first fragment at a tie, recursively; `lexical` selects the
$n$ one-token spans. Scoring is conditional on this selected tiling, with all
items within each selected span forming that fragment's alternative group.
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
between two grammars varies what their declarations vary. Results can vary
with the budget, normal form, and goal. A comparison that means to isolate the
[rule inventory](../how-to/grammars/choose-combinatory-rules.md#expand-a-concise-inventory) holds those three fixed. [Swappable grammatical
assumptions](swappable-assumptions.md) states this condition in general.

To declare each part, see [Declare a signature](../how-to/grammars/declare-a-signature.md),
[Choose combinatory rules](../how-to/grammars/choose-combinatory-rules.md) and
[Choose a normal form, matching and goal](../how-to/grammars/choose-normal-form-matching-goal.md).

## References

\bibliography
