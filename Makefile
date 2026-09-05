install:
	uv add -r requirements.txt
	uv sync 

run:
	clear
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	uv cache clean
	rm -rf data/output
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E "(.venv)" | xargs rm -rf
	find . | grep -E "(.mypy_cache)" | xargs rm -rf

lint:
	flake8 src/.
	mypy src/. --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
