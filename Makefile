PYTHON = python3
LEGACY_CYTHON_VERSION ?= 0.29.36
LEGACY_PYTHON_SITE ?= $(CURDIR)/python_legacy

.PHONY: default release debug complex complex_debug interface complex_interface

default: release

release:
	./build.sh

.PHONY: legacy_cython
legacy_cython:
	@${PYTHON} tools/ensure_legacy_cython.py "$(LEGACY_PYTHON_SITE)" "$(LEGACY_CYTHON_VERSION)"

.PHONY: interface
interface: legacy_cython
	PYTHONPATH=$(LEGACY_PYTHON_SITE)$${PYTHONPATH:+:$$PYTHONPATH} ${PYTHON} setup.py build_ext --inplace

.PHONY: complex_interface
complex_interface: legacy_cython
	PYTHONPATH=$(LEGACY_PYTHON_SITE)$${PYTHONPATH:+:$$PYTHONPATH} ${PYTHON} setup.py build_ext --inplace --define PSPACE_USE_COMPLEX
