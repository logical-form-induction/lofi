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
