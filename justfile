default:
    @just --list

install:
    uv sync

run config="configs/default.yaml":
    uv run --group ml main.py --config {{config}}

test:
    uv run pytest

fmt:
    uv run ruff format .

lint:
    uv run ruff check --fix .

typecheck:
    uv run pyright .

check: fmt lint typecheck test

lock:
    uv lock

clean:
    rm -rf .pytest_cache
    rm -rf .ruff_cache
    rm -rf __pycache__
    find . -type d -name "__pycache__" -exec rm -rf {} +
