# Updating CATIA

Use this page when you **pull new commits**, **switch branches**, or **upgrade from PyPI**. The goals are the same: refresh the environment, pick up new dependencies and console entry points, and sanity-check that the install still matches the tree you think you are running.

---

## Check what you are running

From your activated virtual environment:

```bash
python -c "import importlib.metadata as m; print(m.version('catia'))"
```

If you work from a **git checkout**, also note the commit:

```bash
git rev-parse --short HEAD
```

`pip` and `git` can disagree (e.g. editable install from an old branch while the folder moved on). When debugging “it worked yesterday,” confirm **both** the **package version** and the **commit** you expect.

---

## After you update the repository (git)

Typical flow from the repo root:

1. **Fetch and integrate** upstream changes (`git pull`, merge, or rebase—whatever your team uses).
2. **Re-install in editable mode** so dependency changes in `pyproject.toml` and console scripts apply:

   ```bash
   pip install -e ".[dev]"
   ```

   Use `".[agent]"` instead of `".[dev]"` if you only need CLI/agent extras; use `".[docs]"` when building MkDocs.
3. If the repo ships a **`requirements.txt`** and it changed on main, refresh constraints:

   ```bash
   pip install -r requirements.txt
   ```
4. **Run tests** after non-trivial pulls:

   ```bash
   pytest tests/ -q --tb=short
   ```

### Optional cleanup

- **Caches:** Climate and connector caches may live under `data/cache/` (or paths in your config). If results look stale after a logic change, clear the relevant cache directory rather than guessing at the model.
- **Local model registry:** `models/registry.json` may be regenerated or customized locally. Treat it like data: before discarding changes, confirm you do not need them for your experiments.

### When a pull still behaves oddly

- **Recreate the venv** if dependency resolution failed mid-upgrade or you mixed system and venv `pip`.
- **Entry points missing** (`catia`, `catia-agent` not found): activate the venv, then `pip install -e ".[dev]"` again. See the README section *Common issues* for Windows `PATH` and `python -m catia.agent_repl` fallbacks.

---

## Upgrading from PyPI

If you installed **`catia`** from PyPI (not editable from git):

```bash
pip install --upgrade catia
```

Extras are not upgraded automatically with a bare `catia` bump. Install or upgrade the extra you use, for example:

```bash
pip install --upgrade "catia[dev]"
```

Then re-check the version string as above. For production deployments, pin versions in your own `requirements.txt` or lockfile and upgrade intentionally.

---

## Documentation and notebooks

After an update, **rebuild or refresh local docs** if you rely on them:

```bash
pip install -e ".[docs]"
mkdocs serve
```

Open the tutorial notebook from `notebooks/tutorial.ipynb` and **restart the kernel** so imports pick up the newly installed package.

---

## Where to look for breaking changes

This repo does not guarantee a separate changelog file. For substantive upgrades:

1. **README** at the repo root (installation, CLI, agent, API).
2. **MkDocs** pages under [Home](index.md), [Transparency](transparency.md), and this page.
3. **Recent commits** or release tags on the host (e.g. GitHub), if you need line-level detail.

If you maintain a fork, document your own release notes for downstream teams so “update the repo” steps stay explicit.
