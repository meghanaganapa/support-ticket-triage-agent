.PHONY: install test eval eval-test eval-claude tune demo api mcp data

install:
	pip install -e ".[dev,anthropic,openai,agent]"

test:
	pytest -q

eval:
	python -m triage.evaluate --dataset heldout
	python -m triage.evaluate --dataset dev

# Report-only: never tune on this set.
eval-test:
	python -m triage.evaluate --dataset test

# Claude Agent SDK on 20 test tickets (~$0.40-1.00 with Haiku; needs ANTHROPIC_API_KEY)
eval-claude:
	python -m triage.evaluate --dataset test --backend agent-sdk --limit 20

tune:
	python scripts/tune_thresholds.py

demo:
	triage "Nobody in our company can log in - the whole app is down with a 503 error."

api:
	uvicorn triage.api:app --reload

mcp:
	python -m triage.mcp_server

data:
	python scripts/generate_dataset.py --n 400 --seed 7
