# ==========================================
# Edison Development Makefile
# ==========================================
# Complete development environment for Edison

.PHONY: help build test lint format validate clean install install-dev banner

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
NC := \033[0m

# Icons
CHECKMARK := ✓
CROSS := ✗
ARROW := →
ROCKET := 🚀
TEST := 🧪
MAGNIFYING := 🔍
SPARKLES := ✨
SHIELD := 🛡
GEAR := ⚙

# Default target
help: banner
	@printf "\n"
	@printf "$(BLUE)Available Commands:$(NC)\n"
	@printf "  $(GREEN)help$(NC)        Show this help message\n"
	@printf "  $(GREEN)build$(NC)       Build the Edison package for distribution\n"
	@printf "  $(GREEN)test$(NC)        Run the test suite with coverage reporting\n"
	@printf "  $(GREEN)lint$(NC)        Run static code analysis and linting\n"
	@printf "  $(GREEN)format$(NC)      Format code using black and isort\n"
	@printf "  $(GREEN)validate$(NC)    Run comprehensive validation (format + lint + test + build)\n"
	@printf "  $(GREEN)clean$(NC)       Clean build artifacts\n"
	@printf "  $(GREEN)install$(NC)     Install package in development mode\n"
	@printf "  $(GREEN)install-dev$(NC) Install with development dependencies\n"
	@printf "\n"
	@printf "$(YELLOW)Examples:$(NC)\n"
	@printf "  make test\n"
	@printf "  make validate\n"
	@printf "  make build\n"
	@printf "\n"
	@printf "$(BLUE)$(ARROW) For the best development experience, run 'make validate' before commits$(NC)\n"

banner:
	@printf "$(CYAN)\n"
	@printf "  ███████╗██████╗ ██╗███████╗ ██████╗ ███╗   ██╗\n"
	@printf "  ██╔════╝██╔══██╗██║██╔════╝██╔═══██╗████╗  ██║\n"
	@printf "  █████╗  ██║  ██║██║███████╗██║   ██║██╔██╗ ██║\n"
	@printf "  ██╔══╝  ██║  ██║██║╚════██║██║   ██║██║╚██╗██║\n"
	@printf "  ███████╗██████╔╝██║███████║╚██████╔╝██║ ╚████║\n"
	@printf "  ╚══════╝╚═════╝ ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝\n"
	@printf "$(NC)\n"
	@printf "$(PURPLE)Deep Research Intelligence for Python$(NC)\n"
	@printf "$(PURPLE)═══════════════════════════════════════$(NC)\n"

build: banner
	@printf "$(BLUE)$(GEAR) Edison Build Script$(NC)\n"
	@printf "$(PURPLE)═══════════════════════════════════════$(NC)\n"
	@printf "$(YELLOW)$(ARROW) Checking dependencies...$(NC)\n"
	@test -f pyproject.toml || (printf "$(RED)$(CROSS) pyproject.toml not found$(NC)\n" && exit 1)
	@printf "$(YELLOW)$(ARROW) Cleaning previous builds...$(NC)\n"
	@rm -rf build/ dist/ *.egg-info/
	@printf "$(GREEN)$(CHECKMARK) Build directories cleaned$(NC)\n"
	@printf "$(YELLOW)$(ARROW) Installing build dependencies...$(NC)\n"
	@python -m pip install --upgrade pip build wheel twine
	@printf "$(GREEN)$(CHECKMARK) Build dependencies installed$(NC)\n"
	@printf "$(YELLOW)$(ARROW) Building Edison package...$(NC)\n"
	@python -m build
	@printf "$(GREEN)$(CHECKMARK) Package built successfully$(NC)\n"
	@printf "$(YELLOW)$(ARROW) Verifying build...$(NC)\n"
	@test -d dist && test $$(ls -1 dist/ | wc -l) -gt 0 || (printf "$(RED)$(CROSS) Build verification failed$(NC)\n" && exit 1)
	@printf "$(GREEN)$(CHECKMARK) Build verification passed$(NC)\n"
	@printf "$(BLUE)$(ARROW) Built files:$(NC)\n"
	@ls -la dist/
	@printf "$(GREEN)$(CHECKMARK) Build completed successfully!$(NC)\n"
	@printf "$(PURPLE)═══════════════════════════════════════$(NC)\n"

test: banner
	@printf "$(BLUE)$(TEST) Edison Test Suite$(NC)\n"
	@printf "$(PURPLE)═══════════════════════════════════════$(NC)\n"
	@printf "$(YELLOW)$(ARROW) Checking test dependencies...$(NC)\n"
	@which pytest > /dev/null || (printf "$(YELLOW)$(ARROW) Installing test dependencies...$(NC)\n" && python -m pip install -e .[dev])
	@test -d tests || (printf "$(YELLOW)$(ARROW) Creating tests directory...$(NC)\n" && mkdir -p tests && printf "$(GREEN)$(CHECKMARK) Tests directory created$(NC)\n")
	@printf "$(YELLOW)$(ARROW) Running tests with coverage...$(NC)\n"
	@pytest tests/ --cov=edison --cov-report=term-missing --cov-report=html --cov-report=xml -v || (printf "$(RED)$(CROSS) Tests failed!$(NC)\n" && exit 1)
	@printf "$(GREEN)$(CHECKMARK) All tests passed!$(NC)\n"
	@test -f htmlcov/index.html && printf "$(BLUE)$(ARROW) Coverage report generated: htmlcov/index.html$(NC)\n" || true
	@test -f coverage.xml && printf "$(BLUE)$(ARROW) XML coverage report: coverage.xml$(NC)\n" || true
	@printf "$(PURPLE)═══════════════════════════════════════$(NC)\n"

lint: banner
	@printf "$(BLUE)$(MAGNIFYING) Edison Code Linting$(NC)\n"
	@printf "$(PURPLE)═══════════════════════════════════════$(NC)\n"
	@printf "$(YELLOW)$(ARROW) Checking linting dependencies...$(NC)\n"
	@which flake8 > /dev/null || (printf "$(YELLOW)$(ARROW) Installing linting dependencies...$(NC)\n" && python -m pip install -e .[dev])
	@which mypy > /dev/null || (printf "$(YELLOW)$(ARROW) Installing linting dependencies...$(NC)\n" && python -m pip install -e .[dev])
	@printf "$(YELLOW)$(ARROW) Running flake8...$(NC)\n"
	@flake8 edison/ tests/ --max-line-length=88 --extend-ignore=E203,W503 && printf "$(GREEN)$(CHECKMARK) flake8 passed$(NC)\n" || (printf "$(RED)$(CROSS) flake8 found issues$(NC)\n" && exit 1)
	@printf "$(YELLOW)$(ARROW) Running mypy...$(NC)\n"
	@mypy edison/ --ignore-missing-imports --show-error-codes && printf "$(GREEN)$(CHECKMARK) mypy passed$(NC)\n" || (printf "$(RED)$(CROSS) mypy found issues$(NC)\n" && exit 1)
	@test -d edison && (printf "$(YELLOW)$(ARROW) Checking Python syntax...$(NC)\n" && python -m py_compile edison/*.py 2>/dev/null && printf "$(GREEN)$(CHECKMARK) Python syntax check passed$(NC)\n") || true
	@printf "$(GREEN)$(CHECKMARK) All linting checks passed!$(NC)\n"
	@printf "$(PURPLE)═══════════════════════════════════════$(NC)\n"

format: banner
	@printf "$(BLUE)$(SPARKLES) Edison Code Formatting$(NC)\n"
	@printf "$(PURPLE)═══════════════════════════════════════$(NC)\n"
	@printf "$(YELLOW)$(ARROW) Checking formatting dependencies...$(NC)\n"
	@which black > /dev/null || (printf "$(YELLOW)$(ARROW) Installing formatting dependencies...$(NC)\n" && python -m pip install -e .[dev])
	@which isort > /dev/null || (printf "$(YELLOW)$(ARROW) Installing formatting dependencies...$(NC)\n" && python -m pip install -e .[dev])
	@printf "$(YELLOW)$(ARROW) Sorting imports with isort...$(NC)\n"
	@isort edison/ tests/ --profile black --line-length 88 && printf "$(GREEN)$(CHECKMARK) Import sorting completed$(NC)\n" || (printf "$(RED)$(CROSS) Import sorting failed$(NC)\n" && exit 1)
	@printf "$(YELLOW)$(ARROW) Formatting code with black...$(NC)\n"
	@black edison/ tests/ --line-length 88 && printf "$(GREEN)$(CHECKMARK) Code formatting completed$(NC)\n" || (printf "$(RED)$(CROSS) Code formatting failed$(NC)\n" && exit 1)
	@printf "$(YELLOW)$(ARROW) Verifying formatting...$(NC)\n"
	@black edison/ tests/ --check --line-length 88 && isort edison/ tests/ --check-only --profile black --line-length 88 && printf "$(GREEN)$(CHECKMARK) Code is properly formatted$(NC)\n" || (printf "$(RED)$(CROSS) Code formatting verification failed$(NC)\n" && exit 1)
	@printf "$(GREEN)$(CHECKMARK) Code formatting completed successfully!$(NC)\n"
	@printf "$(PURPLE)═══════════════════════════════════════$(NC)\n"

validate: banner
	@printf "$(BLUE)$(SHIELD) Edison Comprehensive Validation$(NC)\n"
	@printf "$(PURPLE)═══════════════════════════════════════$(NC)\n"
	@printf "$(YELLOW)$(ARROW) Installing all dependencies...$(NC)\n"
	@python -m pip install -e .[dev]
	@printf "$(GREEN)$(CHECKMARK) Dependencies installed$(NC)\n"
	@printf "\n"
	@printf "$(YELLOW)$(ARROW) Running Format validation...$(NC)\n"
	@$(MAKE) format --no-print-directory
	@printf "$(GREEN)$(CHECKMARK) Format validation passed$(NC)\n"
	@printf "\n"
	@printf "$(YELLOW)$(ARROW) Running Lint validation...$(NC)\n"
	@$(MAKE) lint --no-print-directory
	@printf "$(GREEN)$(CHECKMARK) Lint validation passed$(NC)\n"
	@printf "\n"
	@printf "$(YELLOW)$(ARROW) Running Test validation...$(NC)\n"
	@$(MAKE) test --no-print-directory
	@printf "$(GREEN)$(CHECKMARK) Test validation passed$(NC)\n"
	@printf "\n"
	@printf "$(YELLOW)$(ARROW) Running Build validation...$(NC)\n"
	@$(MAKE) build --no-print-directory
	@printf "$(GREEN)$(CHECKMARK) Build validation passed$(NC)\n"
	@printf "\n"
	@printf "$(BLUE)$(SHIELD) Validation Summary$(NC)\n"
	@printf "$(PURPLE)═══════════════════════════════════════$(NC)\n"
	@printf "$(BLUE)$(ARROW) Validation Steps Run: 4$(NC)\n"
	@printf "  $(GREEN)$(CHECKMARK) Format$(NC)\n"
	@printf "  $(GREEN)$(CHECKMARK) Lint$(NC)\n"
	@printf "  $(GREEN)$(CHECKMARK) Test$(NC)\n"
	@printf "  $(GREEN)$(CHECKMARK) Build$(NC)\n"
	@printf "\n"
	@printf "$(GREEN)$(CHECKMARK) All validation steps passed! Edison is ready for release.$(NC)\n"
	@printf "$(PURPLE)═══════════════════════════════════════$(NC)\n"

clean:
	@printf "$(BLUE)🧹 Cleaning build artifacts...$(NC)\n"
	@rm -rf build/ dist/ *.egg-info/ .pytest_cache/ htmlcov/ .coverage coverage.xml
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@printf "$(GREEN)$(CHECKMARK) Clean completed$(NC)\n"

install:
	@printf "$(BLUE)📦 Installing Edison in development mode...$(NC)\n"
	@pip install -e .
	@printf "$(GREEN)$(CHECKMARK) Installation completed$(NC)\n"

install-dev:
	@printf "$(BLUE)📦 Installing Edison with development dependencies...$(NC)\n"
	@pip install -e .[dev]
	@printf "$(GREEN)$(CHECKMARK) Development installation completed$(NC)\n" 