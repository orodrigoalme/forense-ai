# Use uma versão leve do Python
FROM python:3.11-slim

# ✅ CORRIGIR ESTA LINHA (linha 6)
# Instalar dependências de sistema para o OpenCV (GL lib)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Configurar diretório de trabalho
WORKDIR /code

# Instalar dependências Python
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copiar o código
COPY ./app /code/app

# Criar pasta de uploads
RUN mkdir -p /code/uploads

# Rodar a API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
