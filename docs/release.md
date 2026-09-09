# Release

Package version comes from git tags ([hatch-vcs](https://github.com/ofek/hatch-vcs)), not `pyproject.toml`. You do **not** bump a version field, and you do **not** run `towncrier build` locally. Fragments stay in `changelog.d/` until the tag job compiles them.

## During development

User-facing API, scoring, defaults, errors, or install/publish changes need a towncrier fragment in the same unit of work. See [`changelog.d/README.md`](../changelog.d/README.md).

```
make changelog-create TYPE=added NAME=slug MSG="Keep RDKit mols when rdkit=True."
make changelog VERSION=0.3.3   # optional draft; does not write files
```

Skip fragments for internal-only tests, refactors, and tooling. Do not run `uv run towncrier build` (without `--draft`): hatch-vcs will invent a `.devN+g…` version and write that into `CHANGELOG.md`.

## Cut a release

On `main`, with fragments committed:

```
git tag v0.3.3
git push origin v0.3.3
```

Pushing `v*` runs `.github/workflows/publish.yml`:

1. `uv build` on the **tagged** commit (hatch-vcs → sdist/wheel version `X.Y.Z`)
2. Trusted publish to PyPI
3. `towncrier build --version X.Y.Z` → `CHANGELOG.md`, consume fragments, **new** commit on the default branch (not an amend; the tag is not moved)
4. GitHub Release from that section

Do not force-push the tag (that re-runs publish). If `CHANGELOG.md` already has `## [X.Y.Z]`, towncrier is skipped.

`make changelog-release VERSION=X.Y.Z` is only for a local dry-run you intend to discard, or for recovering if CI could not fast-forward `main`.

## One-time PyPI trusted publishing

No long-lived PyPI tokens. OIDC via `publish.yml`:

1. [pypi.org/manage/account/publishing](https://pypi.org/manage/account/publishing/): pending publisher for `xenosite-predict`, owner `swamidasslab`, repo `xenosite-predict`, workflow `publish.yml`, environment `pypi`
2. GitHub → Settings → Environments → create `pypi` (optional required reviewers)
3. After the first successful tag publish, later `v*` tags reuse the same publisher

Keep `XENOSITE_ONNX_URL`, API keys, and weight hostnames out of the repo.
