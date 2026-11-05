# Checking public-release preconditions

Making a repository public exposes every commit reachable from every advertised
branch and tag. Run the release preflight from the exact private checkout whose
GitHub `origin` would become public:

```bash
uv run python scripts/preflight/public_release.py
```

The command checks six conditions and reports each one separately:

1. target `main` contains the canonical LoFI MIT license as a regular root-level
   `LICENSE` file.
2. gitleaks 8.30.1, using its default rules with no repository or environment
   allowlists, finds no secret in the history reachable from any advertised ref.
3. no advertised history contains a proposal- or grant-named artifact, a
   distinctive funded-proposal heading, or an opaque document format capable of
   hiding proposal text. Heading matching tolerates Markdown and LaTeX markup and
   differences in capitalization. PDF, word-processing, presentation, and
   spreadsheet artifacts are rejected categorically, including LFS pointers,
   because the preflight cannot prove their contents safe from Git history alone.
4. no advertised history contains a path under the root `planning/` directory.
5. the documentation builds with `mkdocs build --strict` and `uv --locked` from
   a detached checkout of target `main`.
6. local `HEAD`, mirrored `main`, GitHub `main`, and the successful `ci.yml` run
   all name the same commit.

The command first makes an isolated mirror of `origin`. Thus the history checks
include remote-only branches and tags as well as deleted files. Removing a
sensitive path in a later commit, or omitting a remote branch from the local
checkout, consequently does not make the preflight pass; the public repository
must instead have clean advertised history.

Every failed condition is printed with `FAIL` and the command exits nonzero.
Missing tools or GitHub authentication are failures rather than skipped checks.
The required command-line tools are `git`, gitleaks 8.30.1, `uv`, and `gh`.

The mirror, detached documentation checkout, and generated site live in
temporary directories. The command does not build documentation from dirty
working-tree files, update the source checkout or its lockfile, or call GitHub's
repository-visibility API. A reviewer must confirm that a clean preflight is
for the same commit immediately before the separate manual visibility change.
