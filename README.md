# Python Project Setup with Conventional Commits & Linting

This setup includes:

- **Git hooks** for conventional commits validation
- **Pre-commit hooks** for automatic code formatting and linting
- **GitHub Actions** workflow for CI/CD checks on PRs

## Quick Setup

### 1. Install Dependencies

```bash
pip install -r requirements-dev.txt
```

and install [cuda v12.4](https://developer.nvidia.com/cuda-12-4-1-download-archive) if you have a compatible GPU for faster-whisper

### 2. Install Pre-commit Hooks

```bash
pre-commit install --hook-type commit-msg --hook-type pre-commit
```

This installs two types of hooks:

- **pre-commit**: Runs linters/formatters before each commit
- **commit-msg**: Validates commit messages follow conventional commit format

### 3. Make Your First Commit

Use commitizen for interactive commit messages:

```bash
cz commit
```

Or write conventional commits manually:

```bash
git commit -m "feat: add new feature"
git commit -m "fix: resolve bug in module"
git commit -m "docs: update README"
```

## Conventional Commit Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `ci`: CI/CD changes
- `perf`: Performance improvements

**Examples:**

```
feat(auth): add login functionality
fix(api): resolve timeout issue
docs: update installation guide
```

## Linting Tools

### Black (Code Formatter)

Automatically formats your code to conform to PEP 8.

```bash
black .
```

### isort (Import Sorter)

Sorts and organizes imports.

```bash
isort .
```

### flake8 (Linter)

Checks code for style issues and errors.

```bash
flake8 .
```

### mypy (Type Checker)

Performs static type checking.

```bash
mypy .
```

### Run All Checks Manually

```bash
pre-commit run --all-files
```

## GitHub Actions Workflow

The `.github/workflows/lint.yml` workflow runs automatically on:

- Pull requests to `main` or `develop`
- Pushes to `main` or `develop`

It checks:

- Code formatting (Black)
- Import sorting (isort)
- Linting (flake8)
- Type checking (mypy)
- Tests (pytest, if tests exist)

## Project Structure

```
your-project/
├── .github/
│   └── workflows/
│       └── lint.yml          # CI/CD workflow
├── src/                       # Source code
├── tests/                     # Test files
├── .pre-commit-config.yaml   # Pre-commit configuration
├── pyproject.toml            # Project configuration
├── requirements.txt          # Production dependencies
└── requirements-dev.txt      # Development dependencies
```

## Tips

1. **Bypass hooks** (not recommended):

   ```bash
   git commit --no-verify -m "message"
   ```

2. **Update pre-commit hooks**:

   ```bash
   pre-commit autoupdate
   ```

3. **Skip specific files**:
   Add to `.pre-commit-config.yaml`:

   ```yaml
   exclude: ^(path/to/file\.py|another/file\.py)$
   ```

4. **Configure in IDE**:
   - VS Code: Install Python, Black, and isort extensions
   - PyCharm: Enable Black and configure in Settings → Tools

## Troubleshooting

**Hook installation failed:**

```bash
pre-commit clean
pre-commit install --hook-type commit-msg --hook-type pre-commit
```

**Commit message rejected:**
Ensure your commit follows conventional commit format: `type: description`

**Linting failures:**
Run `pre-commit run --all-files` to see all issues and fix them before committing.
