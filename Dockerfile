FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_VISIBLE_DEVICES=-1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_OFFLINE=1 \
    HF_HOME=/data/huggingface

WORKDIR /app

COPY requirements.txt requirements-ui.txt ./
# CPU wheel avoids installing CUDA packages on a CPU-only VPS.
RUN pip install --no-cache-dir "torch>=2.1,<3" --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt -r requirements-ui.txt \
    && pip check \
    && useradd --create-home --uid 10001 wayang

COPY main.py .
COPY app ./app
COPY streamlit_app.py ui_client.py ui_media.py ui_history.py ./
COPY ui_history_component ./ui_history_component
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY scripts/container_entrypoint.py ./scripts/container_entrypoint.py

USER wayang

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=3).close(); urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3).close()"

CMD ["python", "scripts/container_entrypoint.py"]
