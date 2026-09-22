# ADA-Step-Entropy Automation Makefile
# A simple wrapper to avoid memorizing long commands.

.PHONY: test build clean save deploy

# The Python interpreter in the virtual environment
PYTHON := .venv/bin/python

# -----------------------------------------------------------------------------
# Development Commands
# -----------------------------------------------------------------------------

## build: Rebuild the Ada Kernel (.so)
build:
	gprbuild -P garlic_core.gpr

## test: Run all tests (Ada Bridge + System-Router)
test: test-ada test-router

## test-ada: Run the root Ada Bridge smoke tests
test-ada:
	$(PYTHON) ada_bridge.py

## test-router: Run the System-Router regression suite
test-router:
	$(PYTHON) -m pytest System-Router/tests/test_system_router.py -q

## clean: Remove build artifacts and cache directories
clean:
	rm -rf obj/* lib/*.so
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

# -----------------------------------------------------------------------------
# Git Automation for "Non-Masters"
# -----------------------------------------------------------------------------

## save: Quickly commit all current work with a generic WIP message and push
save:
	git add .
	git commit -m "chore: save WIP state" || true
	git push

## status: Check what files have changed
status:
	git status

# -----------------------------------------------------------------------------
# Help Menu
# -----------------------------------------------------------------------------

## help: Show this help menu
help:
	@echo "Available commands:"
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## //' | awk -F':' '{printf "\033[36m  %-15s\033[0m %s\n", $$1, $$2}'
