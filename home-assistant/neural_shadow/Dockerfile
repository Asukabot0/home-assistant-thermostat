FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN groupadd --gid 10001 shadow && \
    useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin shadow

WORKDIR /app

COPY home-assistant/neural_shadow/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.5,<2.8" && \
    pip install --no-cache-dir "numpy>=1.26,<3.0" && \
    rm -f /tmp/requirements.txt

COPY home-assistant/shadow/model.py \
     home-assistant/shadow/cadence.py \
     home-assistant/shadow/topology.py \
     home-assistant/shadow/house_policy.py \
     home-assistant/shadow/house_policy.json \
     /app/home-assistant/shadow/
COPY home-assistant/telemetry/evidence.py /app/home-assistant/telemetry/
COPY home-assistant/neural_shadow /app/home-assistant/neural_shadow/

ENV PYTHONPATH=/app/home-assistant

USER shadow

ENTRYPOINT ["python3", "-B", "-m", "neural_shadow"]
CMD ["serve", "--config", "/config/neural_shadow.json"]
