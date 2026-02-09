# Forense AI - API de Detecção Forense de Imagens Geradas por IA

## 📋 O que é este aplicativo?

**Forense AI** é uma API REST desenvolvida em FastAPI que realiza análise forense de imagens para detectar se foram geradas ou manipuladas por Inteligência Artificial. A aplicação utiliza múltiplas técnicas de análise digital forense combinadas com IA generativa (Google Gemini) para fornecer um veredicto consolidado sobre a autenticidade de uma imagem.

### 🆕 Versão 2.0 - Novidades

- ✅ **Autenticação Anônima** - Use sem cadastro via tokens JWT
- ✅ **Limites Dinâmicos** - Aumente limites usando sua própria chave Gemini
- ✅ **Proteção Anti-Abuso** - Rate limiting inteligente por IP/sessão
- ✅ **Sistema de Quotas** - Controle de uso por API key e sessões anônimas
- ✅ **Budget Caps** - Proteção automática de custos Gemini
- ✅ **Auth Flexível** - API Key OU Token Anônimo

---

## 🎯 Funcionalidades Principais

### 1. **Análise de Ruído (NOISE)**
Examina o padrão de ruído natural que sensores de câmeras produzem. Imagens geradas por IA tendem a ter:
- Ruído anormalmente baixo ou perfeitamente consistente
- Regiões "lisas demais" (pele, céu, fundos)
- Ausência de padrão de ruído natural de sensores

### 2. **Análise de Espectro de Fourier (FFT)**
Analisa o espectro de frequências da imagem para detectar:
- Simetria excessiva no espectro (IA gera padrões simétricos perfeitos)
- Picos anômalos periódicos (grid artifacts, checkerboard patterns)
- Uniformidade espectral não natural
- Padrões de grade em alta frequência (upscaling artifacts)

### 3. **Error Level Analysis (ELA)**
Técnica que recomprime a imagem JPEG e analisa as diferenças para detectar:
- Regiões com níveis de erro inconsistentes (manipulação seletiva)
- Áreas com erro anormalmente baixo (inserções de IA)
- Bordas com erro inconsistente (splicing, copy-move)
- Padrões de erro uniforme (geração IA completa)

### 4. **Análise com Gemini AI**
Integra a API do Google Gemini para análise contextual avançada:
- Interpreta os resultados das análises técnicas
- Fornece explicação em linguagem acessível para não-técnicos
- Gera veredicto final com nível de confiança
- Identifica indicadores-chave em formato simples

### 5. **Imagens Anotadas**
Gera visualizações anotadas que destacam:
- Áreas suspeitas identificadas por cada método
- Mapas de calor de anomalias
- Score de risco por região

---

## 🔧 Arquitetura Técnica

### Tecnologias Utilizadas
- **Framework:** FastAPI 0.109.0
- **Processamento de Imagens:** OpenCV, NumPy, Pillow
- **Análise Científica:** SciPy
- **IA Generativa:** Google Gemini (google-genai 0.3.0)
- **Autenticação:** JWT (PyJWT)
- **Rate Limiting:** SlowAPI
- **Servidor:** Uvicorn

### Estrutura do Projeto
```
forense-ai/
├── app/
│   ├── main.py                       # Endpoints da API
│   ├── services/
│   │   ├── analysis_service.py       # Orquestração de análises + Gemini
│   │   └── image_annotator.py        # Geração de imagens anotadas
│   ├── analyzers/
│   │   ├── noise.py                  # Análise de ruído
│   │   ├── fft.py                    # Análise FFT
│   │   └── ela.py                    # Error Level Analysis
│   ├── middleware/
│   │   ├── anonymous_auth.py         # Sistema JWT anônimo
│   │   ├── auth.py                   # Autenticação por API Key
│   │   ├── rate_limiter.py           # Rate limiting dinâmico
│   │   ├── quota.py                  # Sistema de quotas
│   │   ├── cost_tracker.py           # Rastreamento de custos Gemini
│   │   └── captcha.py                # Verificação reCAPTCHA (opcional)
│   └── utils.py                      # Validação e utilitários
├── uploads/                          # Diretório temporário para uploads
├── cost_tracking.json                # Registro de custos (auto-gerado)
├── .env                              # Variáveis de ambiente
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🔐 Autenticação

A API v2.0 oferece **2 modos de autenticação flexíveis**:

### Opção 1: API Key (Recomendado para Integração)

**Vantagens:**
- ✅ Sem limitações de sessão anônima
- ✅ Quotas personalizadas por cliente
- ✅ Rate limits mais altos
- ✅ Ideal para aplicações em produção

**Como usar:**
```bash
curl -X POST "http://localhost:8001/api/analyze-image" \
  -H "X-API-Key: aidet_demo_hackathon_2026" \
  -F "file=@imagem.jpg"
```

**API Key Demo (para testes):**
- **Chave:** `aidet_demo_hackathon_2026`
- **Rate Limit:** 20 req/min, 200 req/hora
- **Quota:** Ilimitada

### Opção 2: Token Anônimo (Acesso Público)

**Vantagens:**
- ✅ Sem necessidade de cadastro
- ✅ Acesso imediato
- ✅ Ideal para testes e demos públicas

**Limitações (sem chave Gemini própria):**
- 📊 50 requisições por sessão
- 📊 5.000 créditos de quota
- 📊 3 requisições/minuto

**Limitações (COM chave Gemini própria):**
- 📊 200 requisições por sessão (4x mais!)
- 📊 Quota ilimitada
- 📊 20 requisições/minuto (6x mais!)

**Fluxo de uso:**

```bash
# 1. Obter tokens (access + refresh)
TOKEN_DATA=$(curl -X POST "http://localhost:8001/api/auth/anonymous")
ACCESS_TOKEN=$(echo $TOKEN_DATA | jq -r .access_token)
REFRESH_TOKEN=$(echo $TOKEN_DATA | jq -r .refresh_token)

# 2. Usar access token (válido por 1h)
curl -X POST "http://localhost:8001/api/analyze-image" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -F "file=@imagem.jpg"

# 3. Renovar quando expirar (após 1h)
NEW_TOKENS=$(curl -X POST "http://localhost:8001/api/auth/refresh" \
  -H "X-Refresh-Token: $REFRESH_TOKEN")
```

---

## 🚀 Como Executar

### Pré-requisitos
- Python 3.10+
- Variável de ambiente `GEMINI_API_KEY` (para análise Gemini do servidor)
- Arquivo `.env` configurado

### Instalação

```bash
# 1. Clonar repositório
git clone <repo-url>
cd forense-ai

# 2. Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Criar arquivo .env (ver seção abaixo)
```

### Configuração do `.env`

Crie um arquivo `.env` na raiz do projeto:

```env
# ========================================
# AUTENTICAÇÃO
# ========================================

# Chaves de API válidas (separadas por vírgula)
API_KEYS=aidet_demo_hackathon_2026,aidet_prod_key_xyz123

# Chaves premium (quotas maiores)
PREMIUM_API_KEYS=aidet_prod_key_xyz123

# Secret para JWT (gere com: openssl rand -hex 32)
JWT_SECRET=sua_chave_secreta_muito_longa_e_aleatoria_aqui

# Tempo de vida dos tokens anônimos
ACCESS_TOKEN_LIFETIME_MINUTES=60
SESSION_LIFETIME_DAYS=7

# ========================================
# QUOTAS E LIMITES
# ========================================

# Quotas diárias por tier
FREE_TIER_DAILY_LIMIT=10
PREMIUM_TIER_DAILY_LIMIT=100

# Limites de sessões anônimas SEM chave Gemini própria
ANON_REQUESTS_LIMIT=50
ANON_QUOTA_LIMIT=5000

# Limites de sessões anônimas COM chave Gemini própria
ANON_REQUESTS_LIMIT_CUSTOM_KEY=200
ANON_QUOTA_LIMIT_CUSTOM_KEY=0  # 0 = ilimitado

# ========================================
# RATE LIMITING
# ========================================

# Análise completa (com Gemini)
RATE_LIMIT_ANALYZE_SERVER_KEY=3/minute
RATE_LIMIT_ANALYZE_CUSTOM_KEY=20/minute

# Análises individuais (FFT, NOISE, ELA)
RATE_LIMIT_INDIVIDUAL_SERVER_KEY=10/minute
RATE_LIMIT_INDIVIDUAL_CUSTOM_KEY=30/minute

# ========================================
# PROTEÇÃO ANTI-ABUSO
# ========================================

# Limite de criação de sessões por IP
MAX_SESSIONS_PER_IP_HOUR=3
MAX_SESSIONS_PER_IP_DAY=10
MAX_ACTIVE_SESSIONS_PER_IP=5

# ========================================
# GOOGLE GEMINI
# ========================================

# Chave da API Gemini do SERVIDOR (opcional)
GEMINI_API_KEY=sua_chave_gemini_aqui

# Budget caps (proteção de custos)
MAX_DAILY_GEMINI_COST=5.0
MAX_MONTHLY_GEMINI_COST=50.0

# ========================================
# reCAPTCHA (Opcional)
# ========================================

# Enforcement: "required", "optional", "disabled"
CAPTCHA_ENFORCEMENT=optional
RECAPTCHA_SECRET_KEY=
RECAPTCHA_MIN_SCORE=0.5
```

### Execução Local

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

A API estará disponível em: `http://localhost:8001`

Documentação interativa (Swagger): `http://localhost:8001/docs`

### Docker

```bash
# Build
docker build -t forense-ai .

# Run
docker run -p 8001:8001 --env-file .env forense-ai
```

---

## 🌐 Endpoints da API

### 🔐 Autenticação

#### **POST /api/auth/anonymous**
Gera tokens JWT para acesso anônimo (sem cadastro).

**Request:**
```bash
curl -X POST "http://localhost:8001/api/auth/anonymous"
```

**Response (200):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "Bearer",
  "access_expires_in": 3600,
  "refresh_expires_in": 604800,
  "access_expires_at": "2026-02-09T14:27:00",
  "refresh_expires_at": "2026-02-16T13:27:00",
  "session_id": "anon_a1b2c3d4e5f6",
  "limits": {
    "default": {
      "requests_limit": 50,
      "quota_limit": 5000,
      "description": "Limites ao usar chave Gemini do servidor"
    },
    "custom_key": {
      "requests_limit": 200,
      "quota_limit": "unlimited",
      "description": "Limites ao usar sua própria chave Gemini (X-Gemini-Key)"
    },
    "current_usage": {
      "requests_used": 0,
      "quota_used": 0
    }
  }
}
```

---

#### **POST /api/auth/refresh**
Renova access token usando refresh token.

**Request:**
```bash
curl -X POST "http://localhost:8001/api/auth/refresh" \
  -H "X-Refresh-Token: eyJhbGciOiJIUzI1NiIs..."
```

**Response (200):**
```json
{
  "access_token": "novo_access_token_aqui",
  "refresh_token": "novo_refresh_token_aqui",
  "token_type": "Bearer",
  "access_expires_in": 3600,
  ...
}
```

**Erros:**
- `401 Unauthorized` - Refresh token expirado ou inválido
- `401 Unauthorized` - Sessão não encontrada

---

#### **GET /api/auth/session**
Consulta estatísticas da sessão anônima atual.

**Request:**
```bash
curl -X GET "http://localhost:8001/api/auth/session" \
  -H "Authorization: Bearer <access_token>"
```

**Response (200):**
```json
{
  "session_id": "anon_a1b2c3d4e5f6",
  "type": "anonymous",
  "stats": {
    "requests_used": 12,
    "requests_remaining": 38,
    "requests_limit": 50,
    "quota_used": 1200,
    "quota_remaining": 3800,
    "quota_limit": 5000,
    "created_at": "2026-02-09T10:00:00",
    "session_age_hours": 3.45,
    "limit_type": "server_key",
    "tip": "Use header X-Gemini-Key com sua chave para limites maiores"
  }
}
```

---

#### **DELETE /api/auth/session**
Encerra a sessão anônima atual.

**Request:**
```bash
curl -X DELETE "http://localhost:8001/api/auth/session" \
  -H "Authorization: Bearer <access_token>"
```

**Response (200):**
```json
{
  "message": "Sessão encerrada com sucesso",
  "session_id": "anon_a1b2c3d4e5f6",
  "stats": {
    "requests_used": 15,
    "quota_used": 1500
  }
}
```

---

### 🔍 Análise de Imagens

#### **POST /api/analyze-image** ⭐ (Endpoint Principal)
Executa análise COMPLETA consolidada (FFT + NOISE + ELA + Gemini).

**Autenticação (escolha UMA):**

**Opção 1 - API Key:**
```bash
curl -X POST "http://localhost:8001/api/analyze-image" \
  -H "X-API-Key: aidet_demo_hackathon_2026" \
  -F "file=@imagem.jpg"
```

**Opção 2 - Token Anônimo:**
```bash
curl -X POST "http://localhost:8001/api/analyze-image" \
  -H "Authorization: Bearer <access_token>" \
  -F "file=@imagem.jpg"
```

**Headers Opcionais:**
- `X-Gemini-Key` - Sua chave Gemini (aumenta limites e usa seus créditos)
- `X-Captcha-Token` - Token reCAPTCHA (se CAPTCHA estiver habilitado)

**Response (200):**
```json
{
  "automated_analysis": {
    "final_score": 0.72,
    "interpretation": "Provavelmente IA",
    "confidence": "high",
    "methods_used": ["FFT", "NOISE", "ELA"],
    "individual_scores": {
      "fft": 0.42,
      "noise": 0.85,
      "ela": 0.68
    },
    "key_evidence": [
      "NOISE: Ruído sintético detectado (consistency=0.85)",
      "ELA: Uniformidade excessiva (mean_error=0.012)"
    ],
    "recommendation": "⚠️ ANÁLISE MANUAL - Evidências ambíguas"
  },
  "gemini_analysis": {
    "verdict": "IA",
    "full_analysis": "Texto completo da análise do Gemini...",
    "explanation": "Esta imagem apresenta características típicas...",
    "confidence": "high",
    "key_indicators": [
      "Padrão de ruído uniforme típico de geradores",
      "Ausência de artefatos de compressão JPEG natural"
    ]
  },
  "annotated_image": "base64_encoded_annotated_image",
  "details": {
    "fft": { /* Resultado completo do FFT */ },
    "noise": { /* Resultado completo do NOISE */ },
    "ela": { /* Resultado completo do ELA */ }
  },
  "session_usage": {
    "requests_used": 13,
    "requests_remaining": 37,
    "quota_used": 1300,
    "quota_remaining": 3700,
    "limit_type": "server_key"
  }
}
```

**Rate Limits:**
- **API Key demo:** 20 req/min
- **Token anônimo (sem chave Gemini):** 3 req/min
- **Token anônimo (com chave Gemini):** 20 req/min

**Erros:**
- `401 Unauthorized` - Token/API key inválido ou ausente
- `429 Too Many Requests` - Rate limit ou quota excedida
- `400 Bad Request` - Arquivo inválido
- `500 Internal Server Error` - Erro na análise

---

## 📊 Tabela Comparativa de Limites

| Característica | API Key Demo | Token Anônimo (Servidor) | Token Anônimo (Chave Própria) |
|---|---|---|---|
| **Autenticação** | `X-API-Key: aidet_demo_...` | `Authorization: Bearer ...` | `Authorization: Bearer ...` + `X-Gemini-Key` |
| **Requisições/sessão** | Ilimitadas | 50 | 200 |
| **Quota de créditos** | Ilimitada | 5.000 | Ilimitada |
| **Rate Limit** | 20 req/min | 3 req/min | 20 req/min |
| **Duração da sessão** | Permanente | 7 dias | 7 dias |
| **Custo Gemini** | Servidor | Servidor | Cliente |
| **Budget cap** | N/A | $5/dia, $50/mês | N/A |
| **Ideal para** | Integração prod | Testes rápidos | Uso intenso |

---

## 🛡️ Proteção Anti-Abuso

### Limites por IP (Sessões Anônimas)

**Por hora:**
- Máximo 3 novas sessões criadas por IP

**Por dia:**
- Máximo 10 novas sessões criadas por IP
- Máximo 5 sessões ativas simultâneas por IP

**Sessões ativas:**
- Sessões são limpas automaticamente após 7 dias
- Use `DELETE /api/auth/session` para encerrar manualmente

**Bypass:**
- Use API Key para evitar limites de criação de sessões

---

## 💰 Sistema de Custos e Quotas

### Budget Caps (Chave Gemini do Servidor)

Proteção automática de custos quando clientes usam a chave Gemini do servidor:

- **Limite diário:** $5.00 USD
- **Limite mensal:** $50.00 USD
- **Custo por requisição:** ~$0.002 USD

**Arquivo de rastreamento:** `cost_tracking.json` (auto-gerado)

**Limpeza automática:**
- Mantém últimos 7 dias de dados diários
- Mantém últimos 3 meses de dados mensais

**Bypass:**
- Use `X-Gemini-Key` com sua chave para evitar budget cap do servidor

---

## 🧪 Casos de Uso

### 1. Teste Rápido (Sem Cadastro)

```bash
# 1. Obter token
TOKEN=$(curl -s -X POST "http://localhost:8001/api/auth/anonymous" | jq -r .access_token)

# 2. Analisar imagem
curl -X POST "http://localhost:8001/api/analyze-image" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@foto_suspeita.jpg" \
  | jq .automated_analysis.interpretation
```

### 2. Integração em Produção

```python
import requests

API_KEY = "aidet_demo_hackathon_2026"
API_URL = "http://localhost:8001/api/analyze-image"

def analyze_image(image_path):
    with open(image_path, 'rb') as f:
        response = requests.post(
            API_URL,
            headers={"X-API-Key": API_KEY},
            files={"file": f}
        )
    
    if response.status_code == 200:
        result = response.json()
        return result["automated_analysis"]["interpretation"]
    else:
        raise Exception(f"Error: {response.status_code}")

# Uso
verdict = analyze_image("imagem.jpg")
print(f"Veredicto: {verdict}")
```

### 3. Usando Chave Gemini Própria (Limites Maiores)

```bash
# Obter token anônimo
TOKEN=$(curl -s -X POST "http://localhost:8001/api/auth/anonymous" | jq -r .access_token)

# Analisar com sua chave Gemini (200 req/sessão ao invés de 50!)
curl -X POST "http://localhost:8001/api/analyze-image" \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Gemini-Key: SUA_CHAVE_GEMINI_AQUI" \
  -F "file=@imagem.jpg"
```

---

## 📊 Interpretando Resultados

### Risk Score (0.0 - 1.0)
- **0.00 - 0.15:** Muito provavelmente REAL
- **0.15 - 0.35:** Provavelmente REAL
- **0.35 - 0.55:** INCONCLUSIVO - Análise manual recomendada
- **0.55 - 0.75:** Provavelmente IA
- **0.75 - 1.00:** Muito provavelmente IA

### Confidence Levels
- **very_high:** Todos os 3 métodos concordam + score distante da zona cinzenta
- **high:** Todos os métodos analisaram + resultados consistentes
- **medium:** Alguns métodos falharam ou resultados parcialmente conflitantes
- **low:** Apenas 1-2 métodos funcionaram
- **very_low:** Análise comprometida ou dados insuficientes

### Gemini Verdict
- **REAL:** Imagem autêntica, capturada por câmera
- **IA:** Imagem gerada ou manipulada por IA
- **INCONCLUSIVO:** Evidências conflitantes ou insuficientes
- **DISABLED:** Gemini não configurado
- **ERROR:** Erro na análise Gemini

---

## ⚠️ Limitações

1. **Gemini Desabilitado sem API Key**
   - Se `GEMINI_API_KEY` não estiver configurada E cliente não enviar `X-Gemini-Key`, o campo `gemini_analysis.verdict` será `"DISABLED"`

2. **Formatos de Imagem**
   - ELA funciona melhor com imagens JPEG (imagens PNG são convertidas temporariamente)

3. **Imagens Muito Comprimidas**
   - Compressão pesada pode gerar falsos positivos em todos os métodos

4. **Screenshots e Edições Legítimas**
   - Capturas de tela e edições básicas (crop, resize) podem ser marcadas como suspeitas

5. **Budget Caps**
   - Ao usar chave Gemini do servidor, há limites de $5/dia e $50/mês
   - Use sua própria chave (`X-Gemini-Key`) para evitar esses limites

---

## 🔧 Configurações Avançadas

### Variáveis de Ambiente Completas

```env
# Autenticação
API_KEYS=key1,key2,key3
PREMIUM_API_KEYS=key2
JWT_SECRET=generate_with_openssl_rand_hex_32
ACCESS_TOKEN_LIFETIME_MINUTES=60
SESSION_LIFETIME_DAYS=7

# Quotas
FREE_TIER_DAILY_LIMIT=10
PREMIUM_TIER_DAILY_LIMIT=100
ANON_REQUESTS_LIMIT=50
ANON_QUOTA_LIMIT=5000
ANON_REQUESTS_LIMIT_CUSTOM_KEY=200
ANON_QUOTA_LIMIT_CUSTOM_KEY=0

# Rate Limiting
RATE_LIMIT_ANALYZE_SERVER_KEY=3/minute
RATE_LIMIT_ANALYZE_CUSTOM_KEY=20/minute
RATE_LIMIT_INDIVIDUAL_SERVER_KEY=10/minute
RATE_LIMIT_INDIVIDUAL_CUSTOM_KEY=30/minute

# Anti-Abuso
MAX_SESSIONS_PER_IP_HOUR=3
MAX_SESSIONS_PER_IP_DAY=10
MAX_ACTIVE_SESSIONS_PER_IP=5

# Gemini
GEMINI_API_KEY=your_key_here
MAX_DAILY_GEMINI_COST=5.0
MAX_MONTHLY_GEMINI_COST=50.0

# reCAPTCHA (Opcional)
CAPTCHA_ENFORCEMENT=optional  # required, optional, disabled
RECAPTCHA_SECRET_KEY=
RECAPTCHA_MIN_SCORE=0.5
```

---

## 🐛 Troubleshooting

### Erro: "Limite de sessões atingido"

**Causa:** IP criou muitas sessões em pouco tempo (proteção anti-abuso).

**Soluções:**
1. Aguarde 1 hora (reset automático)
2. Use API Key demo (`X-API-Key: aidet_demo_hackathon_2026`)
3. Encerre sessões antigas: `DELETE /api/auth/session`

### Erro: "Budget cap atingido"

**Causa:** Limites de custo Gemini do servidor excedidos ($5/dia ou $50/mês).

**Soluções:**
1. Use sua própria chave Gemini: `-H "X-Gemini-Key: SUA_CHAVE"`
2. Aguarde reset (meia-noite UTC para diário)
3. Aumente limites no `.env` se você administra o servidor

### Erro: "Token expirado"

**Causa:** Access token válido por 1h expirou.

**Solução:**
```bash
# Renovar com refresh token
curl -X POST "/api/auth/refresh" \
  -H "X-Refresh-Token: SEU_REFRESH_TOKEN"
```

### Erro: "Sessão não encontrada"

**Causa:** Sessão expirou (7 dias) ou foi encerrada.

**Solução:**
```bash
# Criar nova sessão
curl -X POST "/api/auth/anonymous"
```

---

## 📝 Changelog

### v2.0 (2026-02-09)
- ✅ Sistema de autenticação anônima com JWT
- ✅ Limites dinâmicos baseados em chave Gemini própria
- ✅ Proteção anti-abuso por IP
- ✅ Sistema de quotas e cost tracking
- ✅ Rate limiting inteligente
- ✅ Suporte a reCAPTCHA (opcional)

### v1.0 (2025-12-01)
- ✅ Análises forenses: FFT, NOISE, ELA
- ✅ Integração com Gemini AI
- ✅ Geração de imagens anotadas
- ✅ API básica com FastAPI

---

## 📧 Contato e Suporte

- **Documentação Interativa:** `/docs` (Swagger UI)
- **Health Check:** `/health`
- **Repositório:** [GitHub](https://github.com/seu-repo)

---

## 📄 Licença

Este projeto é fornecido como está, sem garantias. Use por sua conta e risco.

---

## 🤝 Contribuições

Contribuições são bem-vindas! Áreas de melhoria:
- Novos métodos de análise (DWT, CFA, Metadata Analysis)
- Melhorias nos limiares de detecção
- Suporte a vídeos e GIFs
- Interface web para upload
- Sistema de cache de análises
- Integração com Redis para quotas distribuídas
