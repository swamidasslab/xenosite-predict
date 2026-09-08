# News fragments

Drop one Markdown file here per **user-facing** change. [towncrier](https://towncrier.readthedocs.io/) compiles them into `CHANGELOG.md` at release.

## Filename

```
<issue>.<type>.md
```

- **issue:** GitHub issue/PR number (`123.fixed.md` → links to `#123`), or `+slug` when there is no ticket (`+http-msgpack.changed.md`).
- **type:** `added` | `changed` | `fixed` | `removed` | `deprecated` | `security`

## Content

One or two sentences a user would care about. Not the git subject.

```
uv run towncrier create --no-edit -c "Keep RDKit mols when rdkit=True." +rdkit-mols.added.md
```

Or: `make changelog-create TYPE=added NAME=rdkit-mols MSG="Keep RDKit mols when rdkit=True."`

Internal-only work (tests, tooling, refactors with no API/score change) does not need a fragment.

## Preview and release

```
make changelog VERSION=0.3.3   # draft against a version; does not write files
```

Pushing a tag `vX.Y.Z` compiles fragments into `CHANGELOG.md` (GitHub Action), commits that to the default branch when the tag can fast-forward it, and opens a GitHub Release from the new section. Bump `project.version` before tagging. `make changelog-release` is optional if you want to compile locally instead.
