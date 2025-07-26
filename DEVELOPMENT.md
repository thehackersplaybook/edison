# Edison Development Guide

🚀 **Welcome to Edison development!** This guide covers the development workflow and tools for contributing to Edison.

## 🛠 Development Commands

Edison provides a comprehensive set of Make targets to streamline your workflow. All commands are designed with proper error handling, colorful output, and professional messaging.

### Quick Start

```bash
# Run all validation checks (recommended before commits)
make validate

# Show help with Edison branding
make help
```

### Available Commands

| Command       | Description                        | Usage              |
| ------------- | ---------------------------------- | ------------------ |
| `help`        | Show help with Edison ASCII art    | `make help`        |
| `build`       | Build the package for distribution | `make build`       |
| `test`        | Run test suite with coverage       | `make test`        |
| `lint`        | Run static code analysis           | `make lint`        |
| `format`      | Format code with black/isort       | `make format`      |
| `validate`    | Run complete validation pipeline   | `make validate`    |
| `clean`       | Clean build artifacts              | `make clean`       |
| `install`     | Install in dev mode                | `make install`     |
| `install-dev` | Install with dev dependencies      | `make install-dev` |

## 📦 Dependencies

### Production Dependencies

Defined in `requirements.txt`:

- `python-dotenv` - Environment variable management
- `pydantic` - Data validation
- `rich` - Terminal formatting
- `openai` - OpenAI API client
- `openai-agents` - Agent framework

### Development Dependencies

Defined in `requirements.dev.txt`:

- **Testing**: `pytest`, `pytest-cov`, `pytest-mock`
- **Code Quality**: `black`, `isort`, `flake8`, `mypy`
- **Build Tools**: `build`, `twine`, `wheel`
- **Documentation**: `sphinx`, `sphinx-rtd-theme`

### Installation

```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements.dev.txt

# Or install package with dev dependencies
make install-dev
```

## 🔧 Development Workflow

### 1. Setup Development Environment

```bash
# Clone and navigate to project
git clone <repository-url>
cd edison

# Install development dependencies
make install-dev
```

### 2. Development Cycle

```bash
# Make your changes...

# Format code
make format

# Run tests
make test

# Run linting
make lint

# Or run everything at once
make validate
```

### 3. Pre-commit Validation

Before committing changes, always run:

```bash
make validate
```

This runs the complete validation pipeline:

1. **Format**: Auto-formats code with `black` and `isort`
2. **Lint**: Checks code quality with `flake8` and `mypy`
3. **Test**: Runs test suite with coverage reporting
4. **Build**: Validates package can be built successfully

### 4. Building for Release

```bash
# Build package
make build

# Built files will be in dist/
ls dist/
```

## 📊 Code Quality Standards

- **Line Length**: 88 characters (Black standard)
- **Import Sorting**: isort with Black profile
- **Type Checking**: mypy with strict settings
- **Test Coverage**: Aim for >90% coverage
- **Documentation**: Docstrings for all public APIs

## 🔍 Make Features

### Error Handling

- All targets use proper error handling and exit codes
- Graceful fallbacks for missing dependencies
- Clear error messages with context

### Visual Feedback

- Colorful output with consistent color scheme
- Progress indicators and status icons (✓ ✗ → ⚙️ 🧪 🔍 ✨ 🛡️)
- Professional formatting with Edison ASCII art banner
- Beautiful borders and separators

### Dependency Management

- Automatic dependency installation when missing
- Consistent dependency resolution from requirements files
- Isolation between production and development dependencies

### Coverage Reporting

- HTML coverage reports in `htmlcov/`
- XML coverage reports for CI/CD
- Terminal coverage summaries

## 🏗 Project Structure

```
edison/
├── edison/                 # Main package source
├── tests/                  # Test suite
├── pyproject.toml         # Single source: packaging, dependencies & tool configuration
├── Makefile               # Development automation
└── DEVELOPMENT.md         # This file
```

## 🚨 Troubleshooting

### Make Not Available

```bash
# Install make if not available (macOS)
xcode-select --install

# Or use Homebrew
brew install make
```

### Missing Dependencies

```bash
# Install missing development tools
make install-dev
```

### Clean Build Issues

```bash
# Clean all artifacts
make clean
```

### Permission Issues

```bash
# Ensure Python and pip are accessible
which python
which pip
```

## 🤝 Contributing

1. **Setup**: Follow the development setup above
2. **Code**: Make your changes following our quality standards
3. **Test**: Ensure all tests pass with `make test`
4. **Validate**: Run full validation with `make validate`
5. **Commit**: Commit your changes with descriptive messages
6. **Pull Request**: Submit PR with validation passing

## 🎨 Edison Development Experience

Every command shows the beautiful Edison ASCII art banner:

```
  ███████╗██████╗ ██╗███████╗ ██████╗ ███╗   ██╗
  ██╔════╝██╔══██╗██║██╔════╝██╔═══██╗████╗  ██║
  █████╗  ██║  ██║██║███████╗██║   ██║██╔██╗ ██║
  ██╔══╝  ██║  ██║██║╚════██║██║   ██║██║╚██╗██║
  ███████╗██████╔╝██║███████║╚██████╔╝██║ ╚████║
  ╚══════╝╚═════╝ ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝

Deep Research Intelligence for Python
═══════════════════════════════════════
```

---

**Happy coding!** 🎉 For questions, check the main README or open an issue.
