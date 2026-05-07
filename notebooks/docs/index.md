# CATIA documentation

Developer-facing documentation for the CATIA catastrophe-risk toolkit. Sources live in **`notebooks/docs/`** alongside the tutorial notebook.

## Contents

- [Regions](regions.md) — supported geographic region identifiers and how they are used across the codebase.
- [Perils](perils.md) — modeled peril slugs (`PERIL_CONFIG`), defaults, and extension checklist.

## Build locally

From the repository root (optional `[docs]` extras). MkDocs reads `notebooks/docs/` via `mkdocs.yml`.

```bash
pip install -e ".[docs]"
mkdocs serve
```

Then open the URL printed in the terminal (typically `http://127.0.0.1:8000`).
