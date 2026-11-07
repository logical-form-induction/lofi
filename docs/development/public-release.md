# Checking public release preconditions

This check is for maintainers preparing to change a private repository's
visibility. LoFI's published repository is already public; ordinary code or
documentation contributions do not require another visibility change.

Making a repository public exposes every commit reachable from every advertised
branch and tag. Run the release preflight from the exact private checkout whose
GitHub `origin` would become public:

```bash
uv run python scripts/preflight/public_release.py
```

Here `origin` is the configured remote repository. Branches and tags name
commits; an **advertised** branch or tag is available from that remote.
Earlier commits reachable through those names remain part of its exposed
history. A file added in one commit and deleted in the next can still be read
from the earlier commit. Deleting it from the current file list is thus
insufficient to remove it from history.

The [preflight implementation](https://github.com/logical-form-induction/lofi/blob/main/scripts/preflight/public_release.py)
copies remote history into temporary locations and checks a specific revision.
It does not change repository visibility. The six conditions below include
both content checks and confirmation that CI checked that same revision.

The command checks six conditions and reports each one separately:

1. target `main` contains the canonical LoFI MIT license as a regular root level
   [`LICENSE`](https://github.com/logical-form-induction/lofi/blob/main/LICENSE) file.
2. gitleaks 8.30.1, using its default rules with no repository or environment
   allowlists, finds no secret in the history reachable from any advertised ref.
3. no advertised history contains a proposal- or grant-named artifact, a
   distinctive funded proposal heading, or an opaque document format capable of
   hiding proposal text. Heading matching tolerates Markdown and LaTeX markup and
   differences in capitalization. PDF, word processing, presentation, and
   spreadsheet artifacts are rejected categorically, including LFS pointers,
   because the preflight cannot prove their contents safe from Git history alone.
4. no advertised history contains a path under the root `planning/` directory.
5. the documentation builds with [`mkdocs build --strict`](documentation.md#strict-mode) and `uv --locked` from
   a detached checkout of target `main`.
6. local `HEAD`, mirrored `main`, GitHub `main`, and the successful [`ci.yml`](https://github.com/logical-form-induction/lofi/blob/main/.github/workflows/ci.yml) run
   all name the same commit.

The command first makes an isolated mirror of `origin`. Thus the history checks
include remote-only branches and tags as well as deleted files. Removing a
sensitive path in a later commit, or omitting a remote branch from the local
checkout, consequently does not make the preflight pass; the public repository
must instead have clean advertised history.

A **mirror** copies the remote's branches and tags for inspection, including
ones absent from the working checkout. The documentation check uses a
**detached checkout**, a copy of the files at the target commit without moving
the contributor's active branch. These copies keep the evidence tied to the
remote revision rather than to uncommitted local edits.

Every failed condition is printed with `FAIL` and the command exits nonzero.
Missing tools or GitHub authentication are failures rather than skipped checks.
The required command line tools are `git`, gitleaks 8.30.1, `uv`, and `gh`.

`git` reads repository history, gitleaks scans it for secrets, `uv` runs the
documentation environment, and `gh` reads GitHub's repository and CI state.
A [nonzero exit status](../reference/cli.md#exit-status) tells a shell or automated workflow that a check failed;
the printed `FAIL` lines identify the conditions to investigate.

The mirror, detached documentation checkout, and generated site live in
temporary directories. The command does not build documentation from dirty
working tree files, update the source checkout or its lockfile, or call GitHub's
repository visibility API. A reviewer must confirm that a clean preflight is
for the same commit immediately before the separate manual visibility change.
A successful report for one commit does not cover changes made afterward.
