#!/bin/bash

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}→${NC} $1"
}

format() {
    print_info "Running Black formatter..."
    cd backend && uv run black . --config ../pyproject.toml
    print_success "Code formatted successfully"
}

lint() {
    print_info "Running Ruff linter..."
    cd backend && uv run ruff check .
    print_success "Linting completed"
}

typecheck() {
    print_info "Running MyPy type checker..."
    cd backend && uv run mypy . --config-file ../pyproject.toml || true
    print_success "Type checking completed"
}

isort_check() {
    print_info "Running isort import sorter..."
    cd backend && uv run isort . --settings-path ../pyproject.toml
    print_success "Import sorting completed"
}

test() {
    print_info "Running tests..."
    cd backend && uv run pytest -v
    print_success "Tests completed"
}

check() {
    print_info "Running all quality checks..."
    format
    isort_check
    lint
    typecheck
    test
    print_success "All quality checks passed!"
}

quick_check() {
    print_info "Running quick quality checks (no tests)..."
    format
    isort_check
    lint
    typecheck
    print_success "Quick checks completed!"
}

install_hooks() {
    print_info "Installing pre-commit hooks..."
    cd backend && uv run pre-commit install
    print_success "Pre-commit hooks installed"
}

case "${1:-check}" in
    format)
        format
        ;;
    lint)
        lint
        ;;
    typecheck)
        typecheck
        ;;
    isort)
        isort_check
        ;;
    test)
        test
        ;;
    check)
        check
        ;;
    quick)
        quick_check
        ;;
    install-hooks)
        install_hooks
        ;;
    *)
        echo "Usage: $0 {format|lint|typecheck|isort|test|check|quick|install-hooks}"
        echo ""
        echo "Commands:"
        echo "  format        - Format code with Black"
        echo "  lint          - Run Ruff linter"
        echo "  typecheck     - Run MyPy type checker"
        echo "  isort         - Sort imports with isort"
        echo "  test          - Run tests with pytest"
        echo "  check         - Run all quality checks (default)"
        echo "  quick         - Run all checks except tests"
        echo "  install-hooks - Install pre-commit hooks"
        exit 1
        ;;
esac