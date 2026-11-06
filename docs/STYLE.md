# Writing the LoFI documentation

This guide describes the prose, notation, and links used under `docs/`.
The documentation serves readers learning the software and formalism as
well as specialists checking the implementation. This contributor guide is
excluded from the built site.

## The voice in one paragraph

A page says what a thing is in its first sentence, then shows it, then says
what it implies. Claims are exact and scoped. Where behavior depends on an
assumption the code does not check, the page names the assumption. Prose is
plain and declarative and is never promotional. Code and prose alternate in
short runs, and every displayed output is real.

Explain unfamiliar ideas through a concrete case, then give the general
definition and its assumptions. Keep the full definition: accessibility
comes from explaining the notation, reasoning, and qualifications together.
Use established terminology, and distinguish a mathematical illustration
from output measured by running the package.

## Opening a page

The first paragraph gives the reader a subject or problem they can identify.
Avoid opening with a list of what the page will explain. A small example may
precede a formal definition when it makes the definition easier to understand.

- **Tutorial chapter:** what the reader builds and what its result means.
- **How-to guide:** the task, necessary prerequisites, and the declaration or
  command that performs it. Explain the choices needed to carry it out.
- **Explanation:** a concrete instance of the problem, followed by its
  representation, general definition, assumptions, and consequences.
- **Reference:** the objects documented, how to read their signatures or
  fields, and links to the relevant explanation and worked use.

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

- Do not hyphenate ordinary noun phrases used as modifiers: write "logical
  form packages", "entry point groups", and "active learning runs". Retain
  conventional hyphens in established terms such as "first-order logic",
  "context-free grammar", and "well-typed terms". Rephrase long compounds
  when a short phrase states the relationship more clearly. Preserve literal
  identifiers, filenames, URLs, and published titles.
- Introduce each block by saying what it does. Explain the nonobvious parts
  and what the output establishes. Do not impose an arbitrary paragraph limit
  when interpreting the result requires more explanation.
- Keep blocks short. A long declaration is shown once in full and then quoted
  in excerpts. Explain important fields beside the excerpt, with links to
  their definitions.
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

- Build each paragraph around the behavior or mathematical claim the reader
  needs. Give its assumptions and scope as part of that explanation.
- Avoid following a positive statement with a stock denial of a stronger
  claim: "X does Y; this does not establish Z." State the relevant condition
  directly. For a learned reconstruction decoder, explain how training
  rewards probability assigned to the original expression and how decoding
  uses that probability. A warning about the word `identity` adds little to
  that account.
- Keep negative statements when they specify an error, exclusion, failed
  result, or mathematical restriction that the reader needs. Review their
  role in the argument rather than mechanically removing negation.
- State what the code does, at the boundary where it does it: "An unknown path
  or a value of the wrong type is refused before any model or data loader is
  constructed."
- Prefer a refusal with its reason to a vague guarantee: "Each transpiler
  either emits the reachable computation graph or reports which indexed core
  capability it cannot represent."
- Never transcribe a measured number. Show the command that measures it.
- Mark unverified laws and approximations as such ("for idempotent
  semirings…"). Do not write "clearly", "simply", "just" or "obviously".
- Explain why a choice matters where the reader makes that choice. Link to
  the fuller explanation rather than requiring an unexplained prerequisite.

## Linking

- Use relative Markdown links to `.md` files so strict mode checks them.
- Link a symbol's first substantive mention in each section to its reference entry, e.g.
  ``[`RunSpecCodec`](../reference/spec.md#lofi.spec.RunSpecCodec)``. Verify that
  the anchor is generated. Link to the implementation when the reader needs
  to inspect the code, using the actual file in the public repository.
- Link unfamiliar concepts to the section that defines them, at the point
  of use. A generic link to an entire reference package is insufficient when
  a specific concept or API anchor exists. Explain enough locally that the
  reader need not follow every link to understand the paragraph.
- Apply this throughout a page, including existing prose and configuration
  tables. An introductory link or a list of related pages does not replace
  a link where a later section uses the concept.
- Check what a destination describes as well as whether it exists. Fields
  such as `width` and `depth` belong to different declarations; link to the
  declaration that owns the field being discussed.
- A how-to guide links to the explanation of why, and an explanation links to
  the how-to guide for doing it. A brief local explanation can be repeated
  when needed to make the task understandable.
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

Use them sparingly.

- `!!! note` for a scope statement that would otherwise interrupt the text,
  such as a page generated from measurements or a module that is not public.
- `!!! tip` to redirect a reader who has landed on the wrong quadrant.
- `!!! warning` only for an action that destroys or overwrites data.

Never use an admonition for emphasis or a summary.

## What each quadrant contains

| Quadrant | Contains | Excludes |
| --- | --- | --- |
| Tutorials (`tutorials/`) | One continuous path in numbered chapters. Each chapter changes one part of a run document, runs it, and reads the result. | Option catalogs, alternatives, design rationale. |
| How-to guides (`how-to/`) | One task per page, the concepts needed to perform it, its declaration or command, and meaningful alternatives. | Unrelated background and history. |
| Explanation (`explanation/`) | Why the design is so, the assumptions behind each family, and what a comparison does and does not license. | Step-by-step instructions and field lists. |
| Reference (`reference/`) | Public APIs, configuration classes, commands, and entry point groups, with explanations and links needed to interpret them. | Undocumented promises about features that do not exist. |

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
  and random variable symbols do not denote numeric scalars or matrices.
- State the slash convention. CCG's $A\backslash B$ takes a $B$ on the left
  and returns $A$; Lambek's corresponding left residual is $B\backslash A$.
- Define every symbol before use. Distinguish sorts from semantic types,
  scores from probabilities, probabilities from log probabilities, and
  bounds on omitted mass from error bounds on objectives.
- Use the standard representation for the object: parsing charts for chart
  states, trees for hierarchical structure, inference rules for deductions,
  and equations for calculations. Explain how to read that representation
  and why the operation works. Do not replace a tree or derivation with a
  table listing its parts. Reserve tables for genuine comparisons and lookups.
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
