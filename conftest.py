# Pins the pytest rootdir to the repository root and puts it on sys.path, so
# the suites under tests/ can import the top-level packages (ground, orbital,
# inference, ...) whether they are run as `pytest tests/` or `python -m pytest`.
