# Code Style and Conventions

## Python Code Style
- **Python Version**: 3.11+
- **Code Style**: OOP with docstrings
- **Type Hints**: Fully typed Python code with mypy checking
- **Docstrings**: Required for all classes and methods
- **Imports**: Use absolute paths within packages (not relative imports)
- **Path Handling**: Never append to Python's system path
- **Error Handling**: Delete broken code and restart instead of patching

## Project Structure
- Use uv for dependency management
- All packages created with pyproject.toml
- Dependencies listed in pyproject.toml
- Development dependencies in [dependency-groups.dev]

## Database
- Use psycopg (psycopg3) NOT psycopg2
- Connection via POSTGRES_URI environment variable

## Best Practices
- Add tqdm for any long-running operations
- Handle deprecation warnings immediately
- Create tests for all Python code
- Check mypy for faster feedback loops
- Always implement full solutions (no simplified versions)

## Git Workflow
- Add and commit after every significant change
- Pre-commit hooks must pass
- Use absolute file paths in responses