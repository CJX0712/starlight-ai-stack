# Starlight AI Stack 运行镜像
# 作者: 晨星
#
# 设计说明：镜像内只装 Python 依赖与本项目代码，模型服务独立成容器（见 docker-compose.yml）。
# 依赖安装沿用 --only-binary=:all:，保证基础镜像里没有编译器也能装成功。

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    STARLIGHT_HOME=/data \
    STARLIGHT_OLLAMA_URL=http://ollama:11434 \
    STARLIGHT_PROFILE=balanced

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
RUN pip install --only-binary=:all: -r requirements.txt

COPY src/ ./src/
COPY web/ ./web/
COPY scripts/ ./scripts/
COPY eval/ ./eval/

RUN mkdir -p /data
VOLUME ["/data"]
EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=6s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8787/health',timeout=5)" || exit 1

CMD ["python", "-m", "uvicorn", "starlight.server:app", "--host", "0.0.0.0", "--port", "8787"]
