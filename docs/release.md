# Release

Package version comes from git tags ([hatch-vcs](https://github.com/ofek/hatch-vcs)), not `pyproject.toml`. Compile the changelog onto the commit you tag so `git checkout vX.Y.Z` has that version’s notes.

## During development

User-facing API, scoring, defaults, errors, or install/publish changes need a towncrier fragment in the same unit of work. See [`changelog.d/README.md`](../changelog.d/README.md).

```
make changelog-create TYPE=added NAME=slug MSG="Keep RDKit mols when rdkit=True."
```

Skip fragments for internal-only tests, refactors, and tooling. Preview with `make changelog VERSION=X.Y.Z` (does not write files).

## Cut a release

On `main`, with a clean tree:

```
make changelog VERSION=0.3.3          # preview
make changelog-release VERSION=0.3.3  # writes CHANGELOG.md, git-rms fragments
git add CHANGELOG.md changelog.d
git commit -m "Update CHANGELOG.md for 0.3.3."
git tag v0.3.3
git push origin main v0.3.3
```

Do not bump a version field. Do not amend the tagged commit. Do not force-push the tag (that re-runs publish).

Pushing `v*` runs `.github/workflows/publish.yml`:

1. `uv build` on the **tagged** commit (hatch-vcs → sdist/wheel version `X.Y.Z`)
2. Trusted publish to PyPI
3. GitHub Release from that version’s `CHANGELOG.md` section

If `CHANGELOG.md` already has `## [X.Y.Z]`, the job does not compile again. If you tagged without compiling, it compiles fragments as a **new** commit on the default branch (the tag itself stays unchanged). Prefer compiling before the tag so the notes live on `vX.Y.Z`.

## One-time PyPI trusted publishing

No long-lived PyPI tokens. OIDC via `publish.yml`:

1. [pypi.org/manage/account/publishing](https://pypi.org/manage/account/publishing/): pending publisher for `xenosite-predict`, owner `swamidasslab`, repo `xenosite-predict`, workflow `publish.yml`, environment `pypi`
2. GitHub → Settings → Environments → create `pypi` (optional required reviewers)
3. After the first successful tag publish, later `v*` tags reuse the same publisher

Keep `XENOSITE_ONNX_URL`, API keys, and weight hostnames out of the repo.
