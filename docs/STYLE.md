# Writing the LoFI documentation

This guide fixes the voice and structure of every page under `docs/`. It is
distilled from the Quivers documentation, whose prose is the model. It is
a contributor guide and is excluded from the built site.

## The voice in one paragraph

A page says what a thing is in its first sentence, then shows it, then says
what it implies. Claims are exact and scoped. Where behavior depends on an
assumption the code does not check, the page names the assumption. Prose is
plain and declarative and is never promotional. Code and prose alternate in
short runs, and every displayed output is real.

> Each Quivers program is a typed value with an executable PyTorch
> implementation and a precise mathematical interpretation.

> Different algebras satisfy different laws; the documentation states those
> assumptions instead of treating them as properties of every runtime
> `Algebra`.

## Opening a page

The first paragraph names the subject and gives the page's route through it as
a sequence of verbs. No preamble, history or motivation comes before the
definition.

- **Tutorial chapter:** what the chapter builds, then the stages in order.
  "This chapter develops a Bayesian linear regression in QVR: it defines the
  model, generates synthetic data, fits the model with variational inference,
  and inspects the posterior."
- **How-to guide:** the declaration or command that does the task, defined in
  one sentence. "QVR's `deduction` block declares a weighted deductive system:
  a hypergraph of inference rules over an item algebra, paired with a semiring
  for weight aggregation."
- **Explanation:** the question the page answers and what it distinguishes.
  "This section gives a mathematical interpretation of the QVR DSL and records
  how that interpretation relates to the current implementation."
- **Reference:** one sentence naming what the page renders, then the `:::`
  directives.

A section index (`index.md`) is an enumerated list with one bold or linked
title and one sentence per page, as in the Quivers semantics index.

## Person and tense

- Present tense throughout. A command "writes" a directory and a declaration
  "is refused".
- Tutorials use **we** for the shared work and **you** for the reader's
  situation: "We model `n` real-valued observations…" and "If that prints
  `Program` without an error, you're set."
- How-to guides use the imperative for steps ("Declare…", "Run…") and the
  third person for behavior.
- Explanation and reference pages use the third person. Subjects are the
  package's own objects: "the chart", "a run document", "the trainer".

## Code and prose

- Introduce each block with one sentence ending in a colon that says what the
  block does: "Create finite sets:". Follow it with at most a paragraph that
  says what the output shows or what changed.
- Keep blocks short. A long declaration is shown once in full and then quoted
  in excerpts, with a line-by-line table ("Reading the QVR line by line:" with
  columns `Line | What it says`).
- Show output in a separate `text` block under the code that produced it, or
  as a trailing comment for one-line results (`print(f.tensor.shape)  #
  torch.Size([3, 4])`).
- **Every example in a tutorial or how-to guide runs against the current
  package, and its output is pasted from a run.** Point at the checked-in
  documents under `docs/examples/` and `lofi/configs/` instead of retyping
  them, so a document and its page cannot drift apart.
- Begin a tutorial with a smoke check of a few lines that confirms the
  install.
- Write shell commands as `uv run lofi …` from a source checkout.

## Stating claims

- State what the code does, at the boundary where it does it: "An unknown path
  or a value of the wrong type is refused before any model or data loader is
  constructed."
- Prefer a refusal with its reason to a vague guarantee: "Each transpiler
  either emits the reachable computation graph or reports which indexed-core
  capability it cannot represent."
- Never transcribe a measured number. Show the command that measures it.
- Mark unverified laws and approximations as such ("for idempotent
  semirings…"). Do not write "clearly", "simply", "just" or "obviously".
- Explanations may say why a design is the way it is. How-to guides do not.

## Linking

- Use relative Markdown links to `.md` files so strict mode checks them.
- Link a symbol's first mention on a page to its reference entry, e.g.
  ``[`RunSpecCodec`](../reference/spec.md)``.
- A how-to guide links to the explanation of why, and an explanation links to
  the how-to guide for doing it. Neither repeats the other.
- End a tutorial chapter with `## Try this` (two or three concrete variations)
  and `## Next` (one sentence and a link to the next chapter).
- Cite literature with BibTeX keys from `docs/references/references.bib`:
  `\citep{lambek1958}` → (Lambek 1958), `\citet{lambek1958}` → Lambek (1958),
  `\citeauthor{lambek1958}'s (\citeyear{lambek1958})` → Lambek's (1958),
  and `\citealt{lambek1958}` → Lambek 1958. Use the first for parenthetical
  citations, the second for attribution, the third for possessives, and the
  fourth for references to papers themselves. Parenthetical citations have
  no comma between author and year and no numbered markers. Group works with
  `\citep{lambek1958,howard1980}`. The parenthetical aliases `[@lambek1958]`
  and `[@lambek1958; @howard1980]` are accepted; bare `@key` remains literal.
  End cited pages with `## References` followed by `\bibliography` on its
  own line. MkDocs generates the bibliography; do not type reference entries
  or author–year citations by hand.

## Admonitions

Use them sparingly. Quivers uses about ten across the whole site.

- `!!! note` for a scope statement that would otherwise interrupt the text,
  such as a page generated from measurements or a module that is not public.
- `!!! tip` to redirect a reader who has landed on the wrong quadrant.
- `!!! warning` only for an action that destroys or overwrites data.

Never use an admonition for emphasis or a summary.

## What each quadrant contains

| Quadrant | Contains | Excludes |
| --- | --- | --- |
| Tutorials (`tutorials/`) | One continuous path in numbered chapters. Each chapter changes one part of a run document, runs it, and reads the result. | Option catalogs, alternatives, design rationale. |
| How-to guides (`how-to/`) | One task per page: the declaration or command, its fields, the variants worth knowing and how each is refused. | Teaching from zero and history. |
| Explanation (`explanation/`) | Why the design is so, the assumptions behind each family, and what a comparison does and does not license. | Step-by-step instructions and field lists. |
| Reference (`reference/`) | mkdocstrings pages for public packages, the configuration-document classes, the command line and the entry-point groups. Only what exists. | Prose beyond a one-paragraph lede and its sections. |

## Mathematical notation and definitions

Use established terminology and the conventions of the relevant field. Do
not coin a replacement for an existing term or abbreviation.

- Probability measures are $\mathbb{P}$ and expectation is $\mathbb{E}$.
  Probability mass functions are lowercase $p,q,r$; probability density
  functions are lowercase $f,g,h$. State events, supports, and reference
  measures where needed.
- Numeric scalars are lowercase italic, vectors lowercase bold, and matrices
  and higher-order arrays uppercase bold. Use `\boldsymbol` for Greek vectors
  and `\mathbf` for Latin vectors and arrays. Distinguish random variables
  from realizations: $X$ and $x$, or $\mathbf{X}$ and $\mathbf{x}$.
- Retain standard notation for nonnumeric objects: category metavariables
  $A,B$, contexts $\Gamma,\Delta$, signatures $\Sigma$, structures
  $\mathcal{M}$, and typing and consequence judgments. Uppercase category
  and random-variable symbols do not denote numeric scalars or matrices.
- State the slash convention. CCG's $A\backslash B$ takes a $B$ on the left
  and returns $A$; Lambek's corresponding left residual is $B\backslash A$.
- Define every symbol before use. Distinguish sorts from semantic types,
  scores from probabilities, probabilities from log probabilities, and
  omitted-mass bounds from error bounds on objectives.
- Give mathematical components their domains, codomains, equations or rules,
  assumptions, and configuration/API correspondences. State objective
  denominators, masks, empty cases, and gradient detachment. Include worked
  examples where they clarify the operation.
- Cite the relevant primary literature for definitions, algorithms, and
  substantive claims. Verify bibliographic entries against publisher pages;
  distinguish a verified entry from a verified theorem attribution. Record
  any verification limitation instead of guessing. Do not claim exhaustive
  coverage without a systematic literature search.
- Distinguish implemented checks, tested properties, and proved results.
  State the hypotheses of each theorem or identity.
- Use `$...$` and `$$...$$` for inline and display math. Use
  `\operatorname` for named operators. Check the browser rendering as well
  as the strict documentation build.

## Mechanics

- One `#` title per page, in sentence case. Tutorial chapters are numbered in
  the nav, not in the title.
- Sections are `##`. Use `###` sparingly.
- Use American spelling, the serial comma, and `e.g.`/`i.e.` inside
  parentheses.
- Use an em dash only to define or exemplify inside a sentence.
- Write "LoFI", "run document", "point", "sweep" and "experiment" with the
  meanings the [configuration guide](how-to/setup/configuration.md) gives them.
- Build with `--strict` before committing.
