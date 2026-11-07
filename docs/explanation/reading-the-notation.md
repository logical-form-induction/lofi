# Reading the notation

## Words, categories, types, and meanings

Consider *John sleeps*. A grammar assigns *John* a [syntactic category](logical-forms-as-types.md#start-with-a-sentence) such as
$\mathrm{NP}$ (noun phrase). It assigns *sleeps* a category describing what
the word needs to combine with: a noun phrase on its left to form a sentence.
In the CCG convention used here, that category is
$\mathrm{S}\backslash\mathrm{NP}$.

A [syntactic category](logical-forms-as-types.md#start-with-a-sentence) concerns combination. A [semantic type](logical-forms-as-types.md#semantic-typing) concerns
the kind of value an expression denotes. For this example, $e$ is the type
of individuals and $t$ the type of truth values. The meaning of *John* has
type $e$; the meaning of *sleeps* has type $e\to t$, taking an individual
and returning a truth value.

A [logical form](../how-to/grammars/declare-a-logical-form-language.md) is an explicit expression for the meaning, such as
$\operatorname{sleep}(\operatorname{john})$. A [structure](inference-as-evidence.md#syntax-consequence-and-models) assigns this expression
a truth value by interpreting the individual and predicate in the world under
discussion. Grammatical
acceptance, semantic typing, and truth are three separate questions.

Grammatical categories, categories in category theory, semantic types, and
Python classes belong to different formalisms. Connections between them require
an explicit correspondence, supplied wherever the documentation uses one.

## Sets and membership

A set collects the objects under discussion. For a small set of individuals
$D=\{j,m\}$, the statement $j\in D$ says that $j$ is a member. Braces
enclose a set's members; their order and repetition do not change the set.
A sequence, in contrast, retains order, as the words of a sentence must.

| Notation | Read it as | Example |
| --- | --- | --- |
| $x\in D$ | $x$ belongs to $D$. | $j\in\{j,m\}$. |
| $A\subseteq B$ | Every member of $A$ belongs to $B$. | $\{j\}\subseteq\{j,m\}$. |
| $A\cap B$ | Members shared by $A$ and $B$. | $\{j,m\}\cap\{m\}=\{m\}$. |
| $A\cup B$ | Members in either set. | $\{j\}\cup\{m\}=\{j,m\}$. |
| $A\times B$ | Ordered pairs with a first member in $A$ and a second in $B$. | $\{j\}\times\{0,1\}=\{(j,0),(j,1)\}$. |
| $\varnothing$ | The empty set. | No admissible candidates remain. |
| $\lvert A\rvert$ | The number of members of a finite set $A$. | $\lvert\{j,m\}\rvert=2$. |

For strings and trees, the same vertical bars may denote length or node
count instead; the surrounding definition specifies which. A set written
$\{x\in D:P(x)\}$ contains exactly those members of $D$ satisfying
condition $P$. This notation describes a collection without prescribing an
algorithm for finding its members.

## Functions and relations

Writing $f:A\to B$ states that $f$ takes each input in $A$ to one output
in $B$. The sets $A$ and $B$ are its **domain** and **codomain**. A codomain
can contain values the function never returns. The notation $x\mapsto f(x)$
describes what happens to a particular input.

For instance, let $f$ count the words in a sentence and let $g$ test whether
a count is at most three. Then $f(\text{John sees Mary})=3$ and $g(3)$ is
true. Their composition answers whether a sentence has at most three words.
The output of the first function is the kind of input the second expects.

In general, if $f:A\to B$ and $g:B\to C$, composition feeds the result of $f$ into
$g$:

$$
 A\xrightarrow{\ f\ }B\xrightarrow{\ g\ }C,
 \qquad (g\circ f)(x)=g(f(x)).
$$

Read $g\circ f$ as “$g$ after $f$.” The intermediate types match, so the
composite accepts an $A$ and produces a $C$. Composing LoFI components
similarly requires compatible declared interfaces and capabilities.

A [relation](logical-forms-as-types.md#rules-and-computational-directions) specifies allowed combinations without requiring one output
per input. For categories in $\mathcal C$, a binary grammar rule can be
written $R\subseteq\mathcal C\times\mathcal C\times\mathcal C$.
The tuple $(\mathrm{S}/\mathrm{NP},\mathrm{NP},\mathrm{S})$ belongs to
the forward-application relation. The tuple
$(\mathrm{S}/\mathrm{NP},\mathrm{NP},\mathrm{NP})$ does not. Checking
whether a tuple belongs and enumerating every possible result are different
tasks. Complete enumeration requires an algorithm that reaches every result
licensed by the relation.

## Reading an inference rule

An inference rule places its premises above a line and its conclusion below.
If the premises hold and the rule's side conditions are met, the conclusion
is licensed. For forward application:

$$
\frac{A/B\qquad B}{A}.
$$

An expression of category $A/B$ needs a $B$ on its right. Supplying that
argument yields an expression of category $A$. With $A=\mathrm{NP}$ and
$B=\mathrm{N}$, a determiner of category $\mathrm{NP}/\mathrm{N}$ can
combine with a noun of category $\mathrm{N}$ to form a noun phrase.
The letters stand for categories consistently throughout the rule: the two
occurrences of $B$ cannot be filled independently.

In the general rule, $A$ and $B$ are **metavariables**, placeholders used to
state a rule about many possible expressions. They differ from a variable
inside a logical form. The formation rules on the
[types and logical forms page](logical-forms-as-types.md#signatures-and-terms)
also use contexts, written $\Gamma$ or $\Delta$, to record declarations.
The judgment $\Gamma\vdash u:\tau$ reads “under the assignments in
$\Gamma$, expression $u$ has type $\tau$.” A colon separates an expression
from its type; the turnstile separates assumptions from what follows under them.

The symbol $\models$ concerns [semantic satisfaction or consequence](inference-as-evidence.md#syntax-consequence-and-models) instead.
$\mathcal M\models\varphi$ says that formula $\varphi$ is true in
structure $\mathcal M$. Entailment asks about every structure compatible
with the stated premises and theory. A derivation under a proof system and
truth in a structure are related by the system's soundness and completeness
properties, when those properties hold; the symbols are not interchangeable.

## Lambda expressions and substitution

The expression $\lambda x_e.\operatorname{sleep}(x)$ describes a function:
take an individual $x$ and return whether $x$ sleeps. The dot introduces the
body of that function. Applying it to John gives:

$$
(\lambda x_e.\operatorname{sleep}(x))\,\operatorname{john}
\longrightarrow_\beta\operatorname{sleep}(\operatorname{john}).
$$

Beta reduction substitutes the supplied argument into the body. The notation
$u[x:=w]$ means substituting $w$ for the free occurrences of $x$ in $u$,
while respecting variable binding. It is not an instruction to replace
every printed letter `x`: a variable introduced by a different binder has
a different role.

A **bound variable** gets its interpretation from its enclosing binder;
a **free variable** has no such binder in the expression under discussion.
Renaming a bound variable consistently preserves its role. Moving a free
variable under a binder can change the meaning, which is why substitution
must avoid variable capture. The [substitution discussion](logical-forms-as-types.md)
works through a case where careless replacement changes the meaning; the
[`lofi.terms` reference](../reference/terms.md) documents the representation.

## Sums, products, and probabilities

The symbols $\sum$ and $\prod$ describe repeated addition and
multiplication. For three numbers $a_1,a_2,a_3$,
$\sum_{i=1}^3 a_i=a_1+a_2+a_3$ and
$\prod_{i=1}^3 a_i=a_1a_2a_3$. A subscript such as $d\in\mathcal D$
says to use one contribution for every member of the declared set.

Weights need not be probabilities. Suppose two candidate derivations have
chosen weights 2 and 3. Their total weight, often called a **partition
function**, is $z=2+3=5$. Normalizing gives probabilities $2/5$ and $3/5$.
The operation requires a finite, positive total, and the probabilities are
relative to the candidates included in that total. [Inside and outside
weights](charts-and-approximation.md#inside-outside-and-marginals) compute
such totals and constituent probabilities over a parsing chart.

LoFI often stores logarithms of weights. Natural logarithms turn products
into sums: $\log(ab)=\log a+\log b$ for positive $a,b$. The exponential
function reverses this operation: $\exp(\log a)=a$. Thus ordinary addition of
log weights represents multiplication of weights. Log-sum-exp represents
addition of weights while keeping their logarithmic representation;
probabilities require the normalization described above.

The notation $\mathbb P(E)$ gives the probability of an event $E$;
$p(x)$ commonly gives probability mass at a discrete outcome $x$.
$\mathbb E[X]$ denotes an expectation, a probability-weighted average.
A conditional probability $p(x\mid y)$ describes outcomes after specifying
the condition $y$. If a parser discards candidates and renormalizes the rest,
its probabilities are conditional on what survived, which matters when
comparing [approximations](charts-and-approximation.md).

For continuous quantities, integrating a probability density $f(x)$ over a
region gives that region's probability. The value at a single point $x$
describes density there; that point has zero probability under the density. This distinction matters for the
[judgment models](judgment-models.md), which include both discrete ratings
and continuous responses.

## Quantifiers and assumptions

$\forall x\in A$ means “for every $x$ in $A$”; $\exists x\in A$ means
“there is at least one $x$ in $A$.” Their order matters. The statement
“every derivation has an equivalent retained derivation” allows a different
retained representative for each original. Requiring one retained derivation
to represent every original would reverse the quantifier order.

An implication $P\Rightarrow Q$ says that $P$ is sufficient for $Q$;
an equivalence $P\Leftrightarrow Q$ gives both directions. To use a result,
check its assumptions as well as its conclusion. Applying a grammar-family
theorem to a bounded search requires a result about that algorithm. Tests of
only some admitted inputs leave the remaining cases unexamined. Identifying a grammar from observed
data requires showing which alternatives the observations distinguish.

For another introduction to composing functions with compatible types, see
Bartosz Milewski's [discussion of composition](https://bartoszmilewski.com/2014/11/04/category-the-essence-of-composition/).
It provides background on category theory; the grammatical conventions used
here are specified on [Categories, semantic types, and logical forms](logical-forms-as-types.md).
