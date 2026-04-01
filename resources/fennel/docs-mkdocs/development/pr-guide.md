# Pull Request Practices

This guide outlines how to prepare and submit PRs for Fennel.

## Branching
- Use feature branches off `master`; prefer descriptive names (e.g., `feat/v2-docs`, `fix/ci-artifact`).
- Avoid committing directly to `master`.

## Commits
- Keep commits focused and descriptive (imperative mood).
- Run pre-commit locally: `pre-commit run --all-files`.
- Clean notebooks: outputs are stripped automatically via `nbstripout`.

## Tests
- Run the test suite or a meaningful subset:
  - Standard: `pytest -m "not jax and not slow"`
  - JAX (optional): `pytest -m jax`
  - Physics regression (as needed): `pytest tests/test_physics_regression.py -m physics`
- Ensure Python 3.8+ compatibility (NumPy 1.x/2.x). Use `integrate_trapezoid`/`trapezoid_compat` for trapezoidal integration.

## CI Expectations
- CI runs on all branches and PRs across Python 3.8–3.11 and Ubuntu/macOS/Windows.
- Artifact upload uses `actions/upload-artifact@v4`.

## Documentation
- Update docs and README when changing user-facing behavior or workflows.
- Add links to new docs in the README or relevant nav pages.

## Changelog
- For user-visible changes, update `CHANGELOG.md` under the appropriate version section.

## PR Checklist
- [ ] Branch created from `master`
- [ ] Pre-commit hooks passed locally
- [ ] Tests run (standard or targeted) and results noted
- [ ] Docs/README updated if behavior or workflows changed
- [ ] Changelog updated if user-facing change
- [ ] CI green (or known flake documented)

## Review Tips
- Provide a brief summary of what changed and why.
- Note any follow-up work or known gaps.
- If touching physics outputs, mention whether reference values were regenerated.
