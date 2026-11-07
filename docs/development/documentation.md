# Documentation

Most pages are Markdown files under [`docs/`](https://github.com/logical-form-induction/lofi/tree/main/docs): text files with marks for
headings, links, code, and mathematics. Edit those files to change tutorials,
guides, or explanations. Much of the API reference instead comes from
[**docstrings**](../reference/index.md#reading-a-python-api-entry), the documentation attached to Python modules, classes, and
functions in the source. Edit the relevant docstring to change that text.

[MkDocs](https://www.mkdocs.org/) builds the website from these sources;
[mkdocstrings](https://mkdocstrings.github.io/) renders the NumPy-style
docstrings into reference pages. After the environment setup in
[CONTRIBUTING.md](https://github.com/logical-form-induction/lofi/blob/main/CONTRIBUTING.md),
run the preview command from the repository root:

```bash
uv run --with-requirements docs/requirements.txt mkdocs serve
```

Open `http://127.0.0.1:8000` in a browser. The command stays running and updates
the preview as you save edits. Inspect the changed page, including its links,
math, and narrow-screen layout. For mathematics, also follow links between
pages and use the browser's Back button to exercise the site's instant
navigation. Check that letters, operators, fractions, and subscripts remain
visible after returning to a page. This browser check complements the build
by testing the rendered mathematics.
Stop the preview with Ctrl-C when finished.
Then run the strict build used by the automated checks:

```bash
uv run --with-requirements docs/requirements.txt mkdocs build --strict
```

Both commands build locally; [publication](#publication) occurs on a push to
`main`. For a paragraph correction, preview the page and run the strict build. If
the correction changes an executable example, run that example as well.

Documentation dependencies are listed in [`docs/requirements.txt`](https://github.com/logical-form-induction/lofi/blob/main/docs/requirements.txt).
`--with-requirements` makes them available in a separate environment for the
command being run.

## Citations

Each work has a **BibTeX key**, a stable label such as `lambek1958` matching
an entry in [`docs/references/references.bib`](https://github.com/logical-form-induction/lofi/blob/main/docs/references/references.bib). Cite that key in a page rather
than maintaining an author's name and date separately. The entry supplies
the bibliographic facts; a Citation Style Language (CSL) file specifies how
those facts are formatted.

The MkDocs hook in [`scripts/docs/citations.py`](https://github.com/logical-form-induction/lofi/blob/main/scripts/docs/citations.py) sends citations to
[Pandoc's citation processor](https://pandoc.org/MANUAL.html#citations),
using `docs/references/references.bib` and the checked-in CSL style. The
pinned `pypandoc-binary` dependency supplies Pandoc locally and in CI.
Only citations pass through Pandoc; MkDocs renders the rest of each page.

Use these BibTeX key commands to generate the four citation forms:

| Context | Source | Result |
|---|---|---|
| Parenthetical | `\citep{lambek1958}` | (Lambek 1958) |
| Attribution | `\citet{lambek1958}` | Lambek (1958) |
| Possessive | `\citeauthor{lambek1958}'s (\citeyear{lambek1958})` | Lambek's (1958) |
| Reference to the paper | `\citealt{lambek1958}` | Lambek 1958 |

The parenthetical form has no comma between author and year. Group works
with `\citep{lambek1958,howard1980}`. Pandoc-style `[@lambek1958]` and
`[@lambek1958; @howard1980]` are also accepted for parenthetical citations.
Other commands take one key. Bare `@key` is ordinary text, so decorators
and grammar notation are unaffected. Citation syntax inside code remains literal.

Close a cited page with a `## References` heading and `\bibliography` alone
on the final line. Only works cited on that page appear in the generated,
alphabetized reference list. In-text citations link to that list without
numbered markers. Unknown keys and missing reference lists fail the build.
Names, dates, and same-author/year suffixes come from the citation processor;
do not maintain them separately in prose.

Run the rendering checks with:

```bash
uv run --with-requirements docs/requirements.txt python -m unittest discover -s scripts/docs -p 'test_*.py'
```

Before adding an entry, verify its author, title, publication year, venue,
and pages against the publisher's record. Put the DOI and publisher URL in
the entry, and record the verification source and any limitation in an
adjacent BibTeX comment. Read the relevant result before attributing a
theorem to it, checking both the theorem's hypotheses and its conclusion.

The CSL style adapts the [ACL Conference Proceedings style](https://github.com/citation-style-language/styles/blob/master/association-for-computational-linguistics.csl)
by removing the comma between author and year. Its provenance, modification
notice, and CC BY-SA 3.0 license are retained in [`docs/references/acl.csl`](https://github.com/logical-form-induction/lofi/blob/main/docs/references/acl.csl).

## Strict mode

`--strict` turns every MkDocs warning into a failure. It detects:

- links to missing pages or anchors
- pages missing from `nav` and `nav` entries that name missing pages
- identifiers on `:::` lines that mkdocstrings cannot resolve

Ruff's pydocstyle rules require docstrings for every
public module, class, and function. Check [array ranks, axes, and dtype classes](../reference/index.md#reading-a-python-api-entry)
against the implementation during review.

An **anchor** is a link target within a page, commonly a heading. `nav` is the
site's list of pages in [`mkdocs.yml`](https://github.com/logical-form-induction/lofi/blob/main/mkdocs.yml). A line beginning `:::` tells mkdocstrings
to render documentation for the named Python object. The build checks that
the site can resolve its structure. Validate executable examples by running
them, and review mathematical claims against their definitions and sources.

## Writing a page

The site follows [Diataxis](https://diataxis.fr/): tutorials teach by building,
how-to guides solve a task, explanation pages explain designs, and reference
pages describe the [public API](../project/stability.md#supported-api). [`docs/STYLE.md`](https://github.com/logical-form-induction/lofi/blob/main/docs/STYLE.md) fixes the voice and structure of
each quadrant for contributors. It is excluded from the built site.

For a grammar rule, show the expressions it combines before giving its
schematic notation. For a command, explain its inputs and the files it
writes. Link the first useful mention of a concept to its explanation and
link API names to their reference entries. The
[notation guide](../explanation/reading-the-notation.md) helps readers interpret
symbols, but each page still needs to define the quantities its equations use.

**Document implemented behavior only.** Reference pages describe the public
interfaces available in the current package.

**Run examples before committing them.** Keep displayed output consistent with
the implementation.

## Adding a page

Write the file under [`docs/`](https://github.com/logical-form-induction/lofi/tree/main/docs), add it to `nav` in [`mkdocs.yml`](https://github.com/logical-form-induction/lofi/blob/main/mkdocs.yml), and run the
strict build. Follow a neighboring entry in the appropriate section: each
navigation entry pairs the visible page title with a path relative to
`docs/`. Internal Markdown links use paths relative to the page containing
the link. The strict build catches mismatches, so a moved file needs its
navigation entry and links updated together.

## Publication

Pushing to `main` builds the site and deploys it to GitHub Pages at
<https://logical-form-induction.github.io/lofi/>. Changes to [`docs/`](https://github.com/logical-form-induction/lofi/tree/main/docs),
[`mkdocs.yml`](https://github.com/logical-form-induction/lofi/blob/main/mkdocs.yml), [`pyproject.toml`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml), or `lofi/` trigger the [workflow](https://github.com/logical-form-induction/lofi/blob/main/.github/workflows/docs.yml).

Here `main` is the repository's main branch, and pushing sends committed
changes to GitHub. Its automated workflow builds and publishes the accepted
source; local edits remain in the checkout until committed and pushed.
