FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ARG PIP_INDEX_URL=https://pypi.org/simple

WORKDIR /app

COPY backend/pyproject.toml ./pyproject.toml

# 先安装只由依赖清单决定的第三方包，使业务源码变化不会使依赖层失效。
# 临时空包仅用于让 setuptools 构建项目元数据；运行时使用随后复制的源码。
RUN mkdir -p app \
    && touch app/__init__.py \
    && python -m pip install --index-url "${PIP_INDEX_URL}" . \
    && rm -rf app

COPY backend/app ./app
COPY backend/scripts ./scripts
COPY backend/user_factors ./user_factors

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
