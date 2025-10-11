PYTHON = python3

default:
	./build.sh

interface:
	${PYTHON} setup.py build_ext --inplace

complex_interface:
	${PYTHON} setup.py build_ext --inplace --define PSPACE_USE_COMPLEX

venv:
	./scripts/setup_env.sh
