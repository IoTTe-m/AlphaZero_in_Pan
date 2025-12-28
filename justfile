default:
    @just --list

install:
    uv sync

run:
    uv run main.py

test:
    uv run pytest

fmt:
    uv run ruff format .

lint:
    uv run ruff check .

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
