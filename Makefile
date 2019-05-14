init:
	pip install -r requirements.txt --user

test:
	python -m tests.test_basic
