# ========================================
# Stage 1: Builder (dependências)
# ========================================
FROM python:3.10-slim as builder

# Instalar dependências de sistema para build
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Criar ambiente virtual
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copiar e instalar dependências Python
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt


# ========================================
# Stage 2: Runtime (imagem final)
# ========================================
FROM python:3.10-slim

# Metadados
LABEL maintainer="seu_email@example.com"
LABEL description="AI Image Detector API - Forensic Analysis"

# Instalar apenas libs runtime (sem compiladores)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar venv do builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Criar usuário não-root para segurança
RUN useradd -m -u 1000 appuser && \
    mkdir -p /code/uploads /code/logs && \
    chown -R appuser:appuser /code

# Configurar diretório de trabalho
WORKDIR /code

# Copiar código da aplicação
COPY --chown=appuser:appuser ./app /code/app

# Trocar para usuário não-root
USER appuser

# Expor porta
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Rodar a API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]