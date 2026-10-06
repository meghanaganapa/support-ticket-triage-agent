FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY kb ./kb
COPY data ./data
RUN pip install --no-cache-dir -e ".[anthropic,openai]"

ENV TRIAGE_LLM=offline
EXPOSE 8000
CMD ["uvicorn", "triage.api:app", "--host", "0.0.0.0", "--port", "8000"]
