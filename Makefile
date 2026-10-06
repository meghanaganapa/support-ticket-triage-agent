.PHONY: install test eval demo api mcp data

install:
	pip install -e ".[dev,anthropic,openai]"

test:
	pytest -q

eval:
	python -m triage.evaluate

demo:
	triage "Nobody in our company can log in - the whole app is down with a 503 error."

api:
	uvicorn triage.api:app --reload

mcp:
	python -m triage.mcp_server

data:
	python scripts/generate_dataset.py --n 400 --seed 7
