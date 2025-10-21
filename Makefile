PYTHON = python3

.PHONY: default release debug complex complex_debug interface complex_interface

default: release

release:
	./build.sh

debug:
	./build.sh debug

complex:
	./build.sh complex

complex_debug:
	./build.sh complex_debug

interface:
	${PYTHON} setup.py build_ext --inplace

complex_interface:
	PSPACE_COMPLEX=1 ${PYTHON} setup.py build_ext --inplace
