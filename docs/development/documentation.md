# Documentation

These pages are built with [MkDocs](https://www.mkdocs.org/), and the API
reference is rendered from the NumPy-style docstrings in the source by
[mkdocstrings](https://mkdocstrings.github.io/).

```bash
uv run --with-requirements docs/requirements.txt mkdocs serve
uv run --with-requirements docs/requirements.txt mkdocs build --strict
```

The first command serves a preview at `http://127.0.0.1:8000`. The second runs
the strict build used in continuous integration.

Documentation dependencies are listed in `docs/requirements.txt` and are not
installed into the project environment. `--with-requirements` makes them
available only for the command being run.

## Citations

The MkDocs hook in `scripts/docs/citations.py` sends citations to
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
theorem to it; matching bibliographic metadata does not verify a claim.

The CSL style adapts the [ACL Conference Proceedings style](https://github.com/citation-style-language/styles/blob/master/association-for-computational-linguistics.csl)
by removing the comma between author and year. Its provenance, modification
notice, and CC BY-SA 3.0 license are retained in `docs/references/acl.csl`.

## Strict mode

`--strict` turns every MkDocs warning into a failure. It detects:

- links to missing pages or anchors
- pages missing from `nav` and `nav` entries that name missing pages
- identifiers on `:::` lines that mkdocstrings cannot resolve

Strict mode does not require docstrings. Ruff's pydocstyle rules cover every
public module, class, and function. Check array ranks, axes, and dtype classes
against the implementation during review.

## Writing a page

The site follows [Diataxis](https://diataxis.fr/): tutorials teach by building,
how-to guides solve a task, explanation pages explain designs, and reference
pages describe the public API. `docs/STYLE.md` fixes the voice and structure of
each quadrant for contributors. It is excluded from the built site.

**Document implemented behavior only.** Reference pages describe the public
interfaces available in the current package.

**Run examples before committing them.** Keep displayed output consistent with
the implementation.

## Adding a page

Write the file under `docs/`, add it to `nav` in `mkdocs.yml`, and run the
strict build.

## Publication

Pushing to `main` builds the site and deploys it to GitHub Pages at
<https://logical-form-induction.github.io/lofi/>. Changes to `docs/`,
`mkdocs.yml`, `pyproject.toml`, or `lofi/` trigger the workflow.
