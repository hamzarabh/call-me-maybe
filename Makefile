install:
	uv add ./llm_sdk 

run:
	clear
	time uv run python -m src

debug:

clean:
	rm -rf data/output

lint:

lint-strict:
