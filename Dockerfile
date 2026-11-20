FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --wheel-dir /wheels ".[azure-ops]"

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN useradd --create-home --uid 10001 app
COPY --from=build /wheels /wheels
RUN pip install /wheels/*.whl && rm -rf /wheels
WORKDIR /app
COPY --chown=app:app reports/latest/summary.json reports/latest/summary.json
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["uvicorn", "archeval.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers"]
