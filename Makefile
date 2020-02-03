PYTHON = python

interface:
	${PYTHON} setup.py build_ext --inplace

complex_interface:
	${PYTHON} setup.py build_ext --inplace --define PSPACE_USE_COMPLEX
