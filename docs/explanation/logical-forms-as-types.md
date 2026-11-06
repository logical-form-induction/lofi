# Categories, semantic types, and logical forms

A syntactic category specifies how an expression combines; a semantic type
specifies the kind of value its meaning denotes; and a logical form is a term of
that semantic type. LoFI represents all three with its
[term algebra](../reference/terms.md#lofi.terms.Term). This page defines term formation and
typing, distinguishes equality from matching, and explains how an
interpretation relates a derivation to its meaning.

## Start with a sentence

Consider *John sleeps*. There are three questions to keep separate: how its
words fit together, what kinds of meanings they have, and which particular
meaning each word contributes.

The syntactic derivation is

$$
\frac{\text{John}:\mathrm{NP}\qquad
      \text{sleeps}:\mathrm{S}\backslash\mathrm{NP}}
     {\text{John sleeps}:\mathrm{S}}\;(<).
$$

The two expressions above the line supply a noun phrase and an expression
that needs a noun phrase on its left. Backward application, labeled $<$,
combines them into the sentence below the line. Their meanings combine by
function application:

$$
\frac{\operatorname{john}:e\qquad
      \operatorname{sleep}:e\to t}
     {\operatorname{sleep}(\operatorname{john}):t}.
$$

Here the verb contributes a function and the noun phrase supplies its
argument. The [semantic typing rules](#semantic-typing) check that the
argument is an individual, as the function requires.

Here $e$ names the type of individuals and $t$ the type of truth values.
An arrow in a semantic type describes a function's input and output; the
backward slash in the syntactic category specifies word order. Type $t$
classifies the possible value. An interpretation of the predicates and
individuals determines which truth value the expression has.

This example uses *category* for a grammatical classification such as noun
phrase. Category theory studies a separate mathematical structure. Composition
with compatible inputs and outputs helps explain the grammatical and typing
rules below, which specify the combinations licensed here. The
[notation guide](reading-the-notation.md) explains how to read those rules.

## The correspondence

In type-logical grammar, categories are formulas and grammatical derivations
are proofs [@lambek1958]. Under the Curry–Howard correspondence, a proof has
an associated term [@howard1980]. These statements concern different levels:

| Object | Notation | LoFI representation |
| --- | --- | --- |
| Syntactic category | $A$, such as $\mathrm{NP}$ or $\mathrm{S}/\mathrm{NP}$ | A term of the category sort, usually `Cat`. |
| Semantic type | $\tau$, such as $e$ or $e\to t$ | A term of the type sort, usually `Ty`. |
| Logical form | $u$, such as $\lambda x_e.\operatorname{sleep}(x)$ | A term of the expression sort, usually `Tm`. |

The sort `Tm` identifies an expression's position in the term algebra. A
typing judgment assigns it a semantic type such as $e$, $t$, or $e\to t$.
The shared representation supports these distinct grammatical and semantic
classifications.

## Signatures and terms

A [term](../reference/terms.md#lofi.terms.Term) is a structured expression made from declared symbols. For
categories, $\mathrm{NP}$ can be a term with no smaller parts, while
$\mathrm{S}/\mathrm{NP}$ is built from two category terms and a slash
constructor. A [signature](../how-to/grammars/declare-a-signature.md) lists the available symbols and how many
arguments of each sort their constructors take. It plays the role of a
grammar for these expressions, before the grammar for sentences is applied.

The formation rules below read from top to bottom. If each argument is a
term of the required sort, and the constructor is declared to accept those
sorts, the constructed expression is a term of its result sort. The line
separates the required premises from the conclusion they license.

A many-sorted signature $\Sigma$ specifies sorts $S$ and operators with
declarations $c:s_1\times\cdots\times s_n\to s$. For a variable context
$\Gamma$, the nonbinding formation rules are the sorted version of term
formation in universal algebra [@meinke-tucker1992]:

$$
\frac{x:s\in\Gamma}{\Gamma\vdash_\Sigma x:s}
\qquad
\frac{\Gamma\vdash_\Sigma u_1:s_1\quad\cdots\quad
      \Gamma\vdash_\Sigma u_n:s_n
      \qquad(c:s_1\times\cdots\times s_n\to s)\in\Sigma}
     {\Gamma\vdash_\Sigma c(u_1,\ldots,u_n):s}.
$$

A nullary operator is a constant. LoFI additionally derives payload-bearing
operators for vocabulary constants, free and bound variables, and, when
enabled, holes. A hole marks a position that must be filled to complete a
term. The notation is the usual inductive
presentation of syntax; binding and semantic typing add conditions to this
nonbinding fragment [@pierce2002].

A [`BindingSite`](../reference/terms.md#lofi.terms.BindingSite) specifies a bound variable's sort, the argument positions
in its scope, and an optional annotation position. In a typed lambda
abstraction, the annotation is outside the newly introduced variable's
scope and the body is inside it. The runtime represents bound variables
with de Bruijn indices [@debruijn1972], which determine binding independently
of the names used to print the term.

Write $T_\Sigma(\Gamma,s)$ for the terms of sort $s$ in context $\Gamma$.
The syntactic and logical form signatures have separate stores. Within a
store, terms are interned, so the same structural term has one
representative. Terms contain no neural vectors or probabilities; those
belong to the computations that read them.

## Semantic typing

The expression $\operatorname{john}$ and the function
$\lambda x_e.\operatorname{sleep}(x)$ are both terms of the expression sort
`Tm`, but they have different semantic types. The first supplies an individual;
the second still needs one. The lambda expression means “given an individual
$x$, return whether $x$ sleeps.” Applying it to John fills that argument.

A typing context records the type assignments currently available. Thus
$\Delta;\Gamma\vdash u:\tau$ can be read aloud as “using these declarations
for constants and variables, expression $u$ has type $\tau$.” The application
rule below checks that the supplied argument has the type the function needs.
The abstraction rule checks the body under an additional assumption about
its input, then records that assumption in the resulting function type.

For the simply typed lambda calculus fragment [@church1940], let
$\mathcal{B}$ be the atomic semantic types. Types are generated by:

$$
\tau,\sigma ::= b\ (b\in\mathcal{B})\mid\tau\to\sigma.
$$

The packaged languages include $e$ for individuals, $v$ for events, and $t$
for truth values. An extended language may declare further type
constructors. The following rules describe the function type fragment,
using the standard typing judgments for variables, constants, application,
and abstraction [@howard1980; @pierce2002].

Let $\Gamma$ associate variables with semantic types and $\Delta$ associate
constants with declared types. The judgment $\Delta;\Gamma\vdash u:\tau$
says that $u$ has semantic type $\tau$. Contexts assign at most one type
to each variable; rename a bound variable when necessary so that the
variable introduced by abstraction is fresh for $\Gamma$:

$$
\frac{x:\tau\in\Gamma}{\Delta;\Gamma\vdash x:\tau}
\qquad
\frac{c:\tau\in\Delta}{\Delta;\Gamma\vdash c:\tau}
$$

$$
\frac{\Delta;\Gamma\vdash u:\sigma\to\tau
      \qquad\Delta;\Gamma\vdash w:\sigma}
     {\Delta;\Gamma\vdash u\,w:\tau}
\qquad
\frac{\Delta;\Gamma,x:\sigma\vdash u:\tau}
     {\Delta;\Gamma\vdash\lambda x_\sigma.u:\sigma\to\tau}.
$$

These semantic type contexts are distinct from the sort contexts in the
formation judgment. A polymorphic constant is instantiated before applying
the constant rule: the equality scheme
$\forall\alpha.\alpha\to\alpha\to t$, for instance, allows individual or
event equality with a consistent choice of $\alpha$ within each instance.
Type schemes and their instances are distinct from object language
quantification [@hindley1969; @milner1978; @pierce2002]. The [type checking
API](../reference/typecheck.md) supplies the language's typing rules and
unification machinery.

## Category-to-type interpretation

An [erasure](../how-to/grammars/map-between-signatures.md) assigns semantic
types to categories. The distinction between syntactic composition and its
semantic interpretation follows \citeauthor{montague1970}'s
(\citeyear{montague1970}) compositional account. Write the type map as $E$. For the usual CCG slash
constructors, where both $A/B$ and $A\backslash B$ return $A$, one
interpretation is:

$$
E(\mathrm{NP})=e,\qquad E(\mathrm{S})=t,\qquad
E(A/B)=E(A\backslash B)=E(B)\to E(A).
$$

These clauses retain argument types while erasing slash direction. Other
signature morphisms permitted by LoFI can specify different interpretations. In Lambek's
residual notation, the corresponding left-taking category is
$B\backslash A$ [@lambek1958]; translate that convention when comparing the
two presentations. The CCG convention is used in this page's examples
[@steedman2000].

For an entry with category $A$ and meaning $u$, the interpretation requires
$\Delta;\varnothing\vdash u:E(A)$. An action for a rule with premises
$A_1,\ldots,A_n$ and conclusion $A$ must combine meanings of types
$E(A_1),\ldots,E(A_n)$ into a meaning of type $E(A)$. Subject to these
conditions, typing is preserved by induction on the derivation tree, using
the application and substitution properties of the typed calculus
[@pierce2002]. The interpretation must establish these typing conditions for
its entries and actions.

Take *John* at $\mathrm{NP}$ with meaning $\operatorname{john}:e$, and
*sleeps* at $\mathrm{S}\backslash\mathrm{NP}$ with meaning
$\lambda x_e.\operatorname{sleep}(x)$, where $\operatorname{sleep}:e\to t$.
Backward application yields:

$$
(\lambda x_e.\operatorname{sleep}(x))\,\operatorname{john}
\longrightarrow_\beta\operatorname{sleep}(\operatorname{john}):t.
$$

Both the application and its reduct have type $E(\mathrm{S})=t$.
[Declare an interpretation](../how-to/grammars/declare-an-interpretation.md)
connects these objects to configuration fields and semantic actions.

## Substitution, equality, and matching

Write $u[x:=w]$ for capture-avoiding substitution. Beta reduction is:

$$
(\lambda x_\sigma.u)\,w\longrightarrow_\beta u[x:=w].
$$

Capture avoidance matters: substituting the free variable $y$ for $x$ in
$\lambda y.x$ gives a term alpha-equivalent to $\lambda z.y$, not
$\lambda y.y$. De Bruijn's representation addresses this problem by
representing binding through indices rather than printed names
[@debruijn1972; @pierce2002].

In this example, the original function ignores its own input and returns
whatever $x$ denotes. Replacing $x$ with the free $y$ should leave it ignoring
its input. Renaming the function's bound variable to $z$ makes that intention
visible. Keeping the name $y$ for both roles would instead turn the function
into one that returns its input, changing the meaning.

Structural equality, alpha-equivalence, and beta-convertibility are
separate notions. The [`alpha` matching](../how-to/grammars/choose-normal-form-matching-goal.md#declare-a-matching-relation) uses the same representative
identity as `identity` for LoFI's interned binding representation. Beta
reduction is a separate operation with a [fuel budget](../reference/formalism.md#lofi.formalism.Budget.beta_fuel). If a redex remains
after that budget is spent, the runtime raises [`NonTerminatingError`](../reference/terms.md#lofi.terms.NonTerminatingError);
the exception records budget exhaustion, leaving divergence undecided.
Equivalence in an
object logic is a further relation, relative to its theory and semantics.

A [`Matching`](../reference/terms.md#lofi.terms.Matching) specifies the equivalence relation a
rule uses. Where it provides a representative key $k$, write:

$$
u\sim w\quad\Longleftrightarrow\quad k(u)=k(w).
$$

Feature erasure uses feature-erased cores as representatives. The
`subsumption` matching requires free variable subsumption in **both**
directions: if $u\preceq w$ means that a substitution takes $u$ to
$w$, its equivalence test is $u\preceq w\land w\preceq u$. Product matching
intersects its component relations. These are the runtime's matching
choices; none changes the underlying structural identity.

## Rules and computational directions

Application can be queried in several ways. Given $\mathrm{S}/\mathrm{NP}$
and $\mathrm{NP}$, ask what result they license: $\mathrm{S}$. Given all
three, ask whether the combination is licensed. Or, given
$\mathrm{S}/\mathrm{NP}$ and a desired $\mathrm{S}$, ask which argument is
missing. These are different computational tasks about the same rule.
Representing the rule as a relation allows those questions to be stated
without assuming that every question has one answer or a finite list of answers.

For categories in $\mathcal{C}$, a rule of arity $n$ is a relation
$R\subseteq\mathcal{C}^n\times\mathcal{C}$. For fixed premises, its
conclusions form the set:

$$
R(A_1,\ldots,A_n)
=\{A\in\mathcal{C}:(A_1,\ldots,A_n,A)\in R\}.
$$

This set need not be a singleton or finite. [Type raising](../how-to/grammars/choose-combinatory-rules.md#the-rule-families) with unrestricted
target categories is one instance [@steedman2000]. A
[rule](../reference/judge.md#lofi.judge.Rule) declares which directions it computes:
forward enumeration, validation, or residuation of a missing premise.
[`forward_is_complete`](../reference/judge.md#lofi.judge.Rule.forward_is_complete) states whether its forward enumeration exhausts the
relation. Consumers check that capability instead of inferring it from the
existence of a rule.

## Implementation boundaries

Categories and logical forms use the same term operations but separate
signatures and stores. The category induction packages do not import the
logical form typing packages, as enforced by the [layer
contracts](layering.md). Category induction can thus run without declaring
a logical form language at that API boundary; complete [run document](documents-are-the-dsl.md#the-document-is-the-program)
workflows may impose further language requirements.

This separates three questions: (i) whether an object is a term of a sort;
(ii) whether a logical form has a semantic type; and (iii) whether a
meaning has the type assigned to its derivation's category. The term
algebra answers the first; typing and interpretation answer the others.

## References

\bibliography
