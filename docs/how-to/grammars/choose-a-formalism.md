# Choose a formalism

A grammar declaration's `formalism` field selects how the rest of the
declaration is read. `declared` reads a complete grammar by its parts: a
signature, a rule set, a lexicon, a goal, a matching relation, a normal form
and a budget. Every other value names an installed formalism plugin, which
reads a shorter declaration and expands it to a `declared` grammar before
anything is compiled. This guide lists the packaged formalisms, shows what
each one fixes, and shows how a declaration outside a formalism's fragment is
refused.

## The packaged formalisms

The `lofi.formalism` entry point group registers four concise formalisms,
each with a packaged example under
`lofi/configs/data/datasets/train/dataset/grammar/`:

| `formalism` | Declaration class | Packaged example | What the formalism fixes |
|---|---|---|---|
| `declared` | [`GrammarDecl`](../../reference/configuration/grammar.md) | `directed.yaml` | Nothing. Every part is declared. |
| `ab` | `ABFormalismSpec` | `ab.yaml` | Application is the only rule family. |
| `cfg` | `CFGFormalismSpec` | `cfg.yaml`, `cfg-collapse.yaml` | Categories are the declared nonterminals; each production becomes one pattern rule. |
| `undirected-ccg` | `UndirectedCCGFormalismSpec` | `undirected.yaml` | The signature has exactly one self-opposite functor view. |
| `tlg` | `TLGFormalismSpec` | `nl-diamond.yaml` | The rules are the elimination, product and modal rules of the NL-diamond fragment, one set per declared mode. |

A run document selects a packaged grammar by naming its fragment in the
`defaults` list, as
[`docs/examples/synthetic-parser.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/synthetic-parser.yaml)
does with `directed`:

```yaml
defaults:
  - data.datasets.train.dataset.grammar: directed
```

or writes the declaration under `data.datasets.train.dataset.grammar` with its
`formalism` field set. Fields the document writes beside a selected fragment
are merged over the fragment's own.

## Relate the declaration to the mathematical grammar

For a context-free grammar $G=(N,\Sigma,P,S)$, $N$ is the nonterminal
inventory, $\Sigma$ the terminal inventory, $P$ the productions, and $S\in N$
the start symbol. Here `nonterminals` supplies $N$, `goal` supplies $S$,
`productions` supplies nonlexical productions
$A\to B_1\cdots B_k$, and lexical entries supply $A\to w$. The compiled
rule runs from the children to the parent:

$$
\frac{[B_1,i_0,i_1]\quad\cdots\quad[B_k,i_{k-1},i_k]}
     {[A,i_0,i_k]}.
$$

This is the chart deduction presentation of the production, as developed for
parsing in \citealt{shieber-etal1995}. The declared plugin's accepted
production shapes and lexical inventory delimit its supported CFG fragment.

For AB grammars, categories are generated recursively from the atomic
inventory $\mathcal A$ by
$A,B::=a\mid A/B\mid A\backslash B$, with $a\in\mathcal A$; derivations use
forward and backward application. CCG adds declared combinators to this
category algebra \citep{steedman2000}. The undirected plugin instead uses a
single binary constructor and licenses its application with the argument on
either side. This identification changes the declaration's directions; it
does not itself establish equivalence to a directed grammar.

The `tlg` plugin expands the listed elimination, product, and modal rules of
its declared nonassociative fragment. It does not expose every inference rule
of the Lambek calculus merely by selecting the plugin. See
[Categories, productions, and lexical axioms](../../explanation/derivations-as-proofs.md#categories-productions-and-lexical-axioms)
for the sequent notation and the distinction between Lambek and CCG slash
conventions \citep{lambek1958,steedman2000}. A mode declaration is a finite
partially ordered set $(M,\leq)$ in which every pair has a unique meet and
join; these are the lattice conditions the validator checks.

## Compile each packaged formalism

Compose each packaged example against `GrammarSpec`, expand it with
`declare_grammar`, and compile it against a registered logical form language:

```python
from pathlib import Path

from lofi.spec import SpecLoader, load_plugins
from lofi.spec.grammar import GrammarIso, GrammarSpec, declare_grammar
from lofi.spec.language import registered_language

load_plugins()
loader = SpecLoader()
packaged = Path("lofi/configs/data/datasets/train/dataset/grammar")
language = registered_language("propositional")

for name in ("ab", "cfg", "undirected", "nl-diamond", "directed"):
    spec = loader.compose_traced(packaged / f"{name}.yaml", schema=GrammarSpec).spec
    assembly = GrammarIso().forward(declare_grammar(spec), language)
    rules = [rule.name for rule in assembly.grammar.rules]
    print(f"{name:11} {type(spec).__name__:28} {rules}")
```

```text
ab          ABFormalismSpec              ['application:/', 'application:\\']
cfg         CFGFormalismSpec             ['sentence', 'verb-phrase']
undirected  UndirectedCCGFormalismSpec   ['application-right', 'application-left']
nl-diamond  TLGFormalismSpec             ['/E_linear', '\\E_linear', 'dotI_linear', 'diamondI_linear', 'box-downE_linear']
directed    GrammarDecl                  ['application:/', 'application:\\']
```

`GrammarSpec` is a tagged union over `formalism`, so composition returns the
declaration class the field selects. The CFG's rules carry its production
names, and the type-logical grammar's rules carry the mode they belong to.

Compose a grammar document rather than validating its parsed YAML. A fragment
may select other fragments in its own `defaults` list, and only composition
reads them. The packaged `ccgbank.yaml` takes its slash constructors and
functor views from `pmb/slash-signature`:

```python
ccgbank = declare_grammar(
    loader.compose_traced(packaged / "ccgbank.yaml", schema=GrammarSpec).spec
)

print(ccgbank.signature.name, [functor.name for functor in ccgbank.signature.functors])
```

```text
ccgbank ['/', '\\']
```

## What an expansion produces

A concise declaration is expanded once, by its plugin, into the generic
declaration every formalism is compiled through:

```python
cfg = loader.compose_traced(packaged / "cfg.yaml", schema=GrammarSpec).spec
expanded = declare_grammar(cfg)

print(expanded.formalism, expanded.signature.syntax, expanded.signature.primitives)
print(expanded.lexicon.entries[0].category)
print(expanded.goal.categories)
```

```text
declared s-expression ('S', 'NP', 'VP', 'V')
("#const[Cat]" "NP")
('("#const[Cat]" "S")',)
```

The CFG's nonterminals become the primitives of an s-expression signature
with no constructors, each lexical category and the goal become constants of
the category sort, and each production becomes a pattern rule from its
children to its parent. The type-logical expansion, likewise, declares five
prefix constructors per mode (`/`, `\`, `dot`, `diamond` and `box-down`
suffixed with the mode's name) and a modal matching relation over the
declared mode order.

What each concise declaration accepts:

| Formalism | Fields |
|---|---|
| `ab` | `name`, `signature`, `rules` (application rules only), `lexicon`, and optionally `goal`, `matching`, `normal_form`, `budget`, `max_hypotheses`, `capacity_evidence`. |
| `undirected-ccg` | The same fields as `ab`; any rule family is admitted, but the signature declares one functor view with `argument_side: either` that is its own opposite. |
| `cfg` | `name`, `nonterminals`, `productions` (each a `name`, a `parent` and its `children`), an inline `lexicon` over the nonterminals, a `goal` nonterminal, and optionally `matching`, `normal_form`, `budget`. |
| `tlg` | `name`, `primitives`, `modes` (a list of `modes` and an `order` of `lower`/`upper` pairs that must form a lattice), `lexicon`, a `goal` primitive, and optionally `max_hypotheses`, `normal_form`, `budget` (three unary rounds by default). |

## How a declaration outside a fragment is refused

Each formalism checks its own fragment when the composed declaration is
validated. Four refusals, each an override on a packaged example:

```python
def refuse(name, *overrides):
    try:
        loader.compose_traced(
            packaged / f"{name}.yaml", schema=GrammarSpec, overrides=overrides
        ).spec
    except Exception as refused:
        print(refused)


composition = (
    '{"rule": "composition", "direction": "unrestricted", "order": 1, "view": "/"}'
)
refuse("ab", f"rules.rules+=[{composition}]")
refuse("cfg", "goal=VP2")
refuse("nl-diamond", "goal=N")
refuse("ab", "formalism=lambek")
```

```text
lofi/configs/data/datasets/train/dataset/grammar/ab.yaml cannot be composed: an AB grammar declares non-application rules ('composition',)
lofi/configs/data/datasets/train/dataset/grammar/cfg.yaml cannot be composed: CFG goal 'VP2' is not a declared nonterminal
lofi/configs/data/datasets/train/dataset/grammar/nl-diamond.yaml cannot be composed: type-logical goal 'N' is not a declared primitive
formalism was refused: Unknown variant 'formalism' = 'lambek' (set by override:formalism="lambek"); GrammarSpec registers: ['ab', 'cfg', 'declared', 'tlg', 'undirected-ccg']
```

A refusal the declaration raises as a whole names the document; one at a leaf,
such as an unregistered `formalism`, names the leaf.

A CFG also refuses a repeated nonterminal, production name or production, a
production or lexical entry over an undeclared nonterminal, and a lexicon that
is not inline. A mode order that relates two modes both ways, or leaves two
modes without a unique meet or join, is refused before the TLG is expanded.

## Add a formalism

A plugin adds a formalism by registering a `FormalismPlugin` under the
`lofi.formalism` entry point group. The plugin pairs a declaration class, a
`GrammarSpec` variant whose `formalism` literal is the entry point name, with
a function that expands a validated declaration to a `GrammarDecl`. The
[plugin guide](../extending/plugins.md) covers registration, and the
[entry point reference](../../reference/entry-points.md) lists the group.

## Related

- [Declare a category signature](declare-a-signature.md) for the signature a
  `declared` or `ab` grammar carries.
- [Choose combinatory rules](choose-combinatory-rules.md) for the rule set.
- [Swappable assumptions](../../explanation/swappable-assumptions.md) for why
  each formalism lowers to one generic declaration.

## References

\bibliography
