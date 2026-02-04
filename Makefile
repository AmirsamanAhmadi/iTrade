install:
	./scripts/setup_env.sh

run-api:
	. .venv/bin/activate && uvicorn backend.api.app:app --reload

run-ui:
	. .venv/bin/activate && streamlit run ui/streamlit/app.py

test:
	pytest -q
