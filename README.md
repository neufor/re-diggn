# ML Project Template

A Claude Code workspace template for ML projects in **fintech/trading** and **cheminformatics**.
Pre-configured with MCP servers, sub-agents, permissions, and Serena LSP.

## What's included

| Component | Purpose |
|-----------|---------|
| `.mcp.json` | 6 MCP servers: filesystem, memory, context7, brave-search, jupyter, serena |
| `.claude/agents/` | 5 sub-agents: data-analyst, ml-trainer, backtesting-quant, molecule-analyst, docs |
| `.claude/settings.json` | Permissions allowlist — no prompts for routine commands |
| `.serena/project.yml` | Serena LSP config: Python + YAML + TOML language servers |
| `.vscode/tasks.json` | Auto-starts Jupyter Lab on folder open |
| `CLAUDE.md` | Domain context and agent routing guide for Claude |

---

## One-time: mark as a GitHub template

GitHub → this repo → **Settings** → check **"Template repository"**.

---

## Starting a new project

### 1. Create the new repo from the template

```bash
gh repo create my-new-project --template your-username/ml-template --private --clone
cd my-new-project
```

Or via GitHub UI: green **"Use this template"** → "Create a new repository" → clone it.

### 2. Update the two project-specific files

**`CLAUDE.md`** — keep only the relevant domain line:
```markdown
- **Fintech / Trading**: ML-driven trading signals, risk models, ...
- **Cheminformatics**: QSAR, virtual screening, molecular property prediction, ...
```

**`.serena/project.yml`** — update the project name:
```yaml
project_name: "my-new-project"
```

### 3. Set up the Python environment

```bash
uv init
uv add jupyter pandas numpy scikit-learn
uv add --dev pytest ruff mypy nbstripout
```

### 4. Copy and fill `.env`

```bash
cp .env.example .env
# fill in JUPYTER_TOKEN, BRAVE_API_KEY, and any data-source keys
```

`BRAVE_API_KEY` and `JUPYTER_TOKEN` must also be set as Windows user environment variables
(Settings → Environment Variables, or PowerShell):

```powershell
[System.Environment]::SetEnvironmentVariable("BRAVE_API_KEY", "your-key", "User")
[System.Environment]::SetEnvironmentVariable("JUPYTER_TOKEN", "123456", "User")
```

### 5. Open in VS Code

```bash
code .
```

- VS Code fires the auto-start task → Jupyter Lab starts on port 8888
- First time only: click **Allow** when VS Code asks about automatic tasks
- Claude Code loads → all 6 MCP servers connect

### 6. Run the docs agent

Once you have initial code, let Claude onboard the project:

> *"Use the docs agent to onboard this project"*

It will scan the codebase, write `ONBOARDING.md`, update `CLAUDE.md` with the discovered
architecture, and populate the memory MCP — so every future session starts with full context.

### 7. Commit the initial customization

```bash
git add CLAUDE.md .serena/project.yml
git commit -m "Init: configure project name and domain"
git push
```

---

## What you get out of the box

- All MCP servers pre-configured and auto-approved — no per-tool confirmation prompts
- 7 sub-agents ready to use (see `CLAUDE.md` for routing guide)
- Jupyter Lab auto-starts when you open the folder in VS Code
- Serena LSP active for Python, YAML, TOML from the first file
- Permissions allowlist covers all routine git, Python, uv, and PowerShell operations

## MCP server setup notes

| Server | Requirement |
|--------|-------------|
| `brave-search` | `BRAVE_API_KEY` env var — [get a free key](https://brave.com/search/api/) (2k req/month free) |
| `jupyter` | `JUPYTER_TOKEN` env var — must match the token in `.vscode/tasks.json` (`123456` by default) |
| `github` | `GITHUB_TOKEN` env var — [create a PAT](https://github.com/settings/tokens) with `repo` + `read:org` scopes |
| `duckdb` | No setup for local use; set `MOTHERDUCK_TOKEN` to also query MotherDuck cloud |
| `openproject` | `OPENPROJECT_URL` + `OPENPROJECT_API_KEY` — My Account → Access Tokens → Generate |
| `serena` | Installed on first use via `uvx` from GitHub — first startup takes ~5s to cache |
| `filesystem`, `memory`, `context7` | No setup required |

## Sub-agents

| Agent | Use for |
|-------|---------|
| `data-analyst` | EDA, data quality, feature distributions |
| `ml-trainer` | Training loops, hyperparameter tuning, evaluation |
| `backtesting-quant` | Strategy backtesting, risk metrics, walk-forward validation |
| `molecule-analyst` | SMILES, RDKit descriptors, ADMET, docking prep |
| `experiment-tracker` | MLflow logging, run comparison, model registration |
| `code-reviewer` | Leakage, look-ahead bias, wrong CV strategy, reproducibility |
| `docs` | Generate ONBOARDING.md, update CLAUDE.md, populate memory |
