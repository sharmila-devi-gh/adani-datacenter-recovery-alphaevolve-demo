ARG BASE_IMAGE=python:3.12-slim-bookworm
FROM ${BASE_IMAGE}

COPY user_examples/dc_recovery_slm/ /app/experiment/
RUN if [ -f /app/experiment/requirements.txt ]; then \
      uv pip install -r /app/experiment/requirements.txt; \
    fi

RUN id -u evaluser >/dev/null 2>&1 || useradd -m -u 1000 evaluser \
    && chown -R evaluser:evaluser /app/experiment

WORKDIR /app/src/alpha_evolve
USER evaluser
