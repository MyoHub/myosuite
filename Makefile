# MyoSuite — developer convenience targets.
# Mirrors the verification checklist in CLAUDE.md so it runs with a single command.

.PHONY: lint test-tier1 test-tier2 test-core parity-check verify

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------

lint:
	pre-commit run --all-files

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Fast unit / contract / regression tests (CI pre-merge gate)
test-tier1:
	pytest myosuite/tests/ -m "tier1" -v

# Extended backend / parity tests (MJX, mjlab) — slower
test-tier2:
	pytest myosuite/tests/ -m "tier2" -v

# Core test suite from CLAUDE.md verification checklist
test-core:
	pytest myosuite/tests/test_model_builder.py \
	       myosuite/tests/test_terms_cpu.py \
	       myosuite/tests/test_fragment_compat.py \
	       myosuite/tests/test_parity.py \
	       -v

# Numerical parity only (fast regression check after env migrations)
parity-check:
	pytest myosuite/tests/test_parity.py -v

# Full regression suite
test-regression:
	pytest myosuite/tests/test_regression.py -v

# ---------------------------------------------------------------------------
# Composite: what CI should pass before any merge
# ---------------------------------------------------------------------------

verify: lint test-core
