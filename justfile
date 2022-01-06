min_python := "3.7" # only used for poetry install!

install:
	poetry install

kernel:
	python -m ipykernel install --user
