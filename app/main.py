from dotenv import load_dotenv
load_dotenv()
import os
import uuid
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Request, HTTPException,Depends,Header
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from typing import Optional

# Importar validador e analisadores
from app.utils import validate_file_content, resize_if_too_big
from app.analyzers.noise import NoiseMapAnalyzer
from app.analyzers.fft import FFTAnalyzer
from app.analyzers.ela import ELAAnalyzer
from app.services.analysis_service import AnalysisService
from app.middleware.auth import api_key_auth
from app.middleware.rate_limiter import limiter, get_dynamic_rate_limit
from app.middleware.cost_tracker import CostTracker
from app.middleware.quota import quota_manager
from app.middleware.auth import api_key_auth
from app.middleware.anonymous_auth import anon_auth

# Inicializar rastreador de custos
cost_tracker = CostTracker()

app = FastAPI(
    title="AI Image Detector API",
    description="API para detecção forense de imagens geradas por IA",
    version="1.5.0"
)

# ✅ ADICIONAR CORS ANTES DE QUALQUER ROTA
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir qualquer origem (ou especificar domínios)
    allow_credentials=True,
    allow_methods=["*"],  # Permitir POST, GET, OPTIONS, etc
    allow_headers=["*"],  # Permitir qualquer header
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
analysis_service = AnalysisService()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_bytes_to_disk(content: bytes, original_filename: str) -> str:
    # 1. Gerar nome único
    file_extension = original_filename.split(".")[-1] if "." in original_filename else "jpg"
    unique_name = f"{uuid.uuid4()}"
    filename = f"{unique_name}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    print(f"💾 Salvando arquivo temporário: {filename} ({len(content)} bytes)") # DEBUG
    
    # 2. Gravar os bytes no disco (Isso estava faltando no seu snippet visualizado)
    with open(file_path, "wb") as buffer:
        buffer.write(content)
        
    # 3. Redimensionar para evitar travar o servidor (Segurança)
    # Certifique-se de ter importado a função no topo do arquivo:
    # from app.utils import resize_if_too_big 
    try:
        resize_if_too_big(file_path)
    except Exception as e:
        print(f"⚠️ Aviso: Falha ao redimensionar imagem (prosseguindo mesmo assim): {e}")

    return file_path


# Função para evitar cache do navegador
def no_cache_response(data: dict):
    response = JSONResponse(data)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# ========================================
# AUTENTICAÇÃO FLEXÍVEL
# ========================================

def verify_flexible_auth(
    request: Request,
    authorization: Optional[str] = Header(None),
    api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> dict:
    """
    Verifica autenticação - aceita token anônimo OU API key direta
    
    Prioridade:
    1. API Key → validação direta (sem limites de sessão anônima)
    2. Bearer Token → validação de token anônimo (com limites)
    3. Nenhum → erro 401
    
    Returns:
        dict com tipo de autenticação e dados relevantes
    """
    
    # Tentativa 1: API Key direta (clientes registrados ou demo)
    if api_key:
        try:
            api_key_auth.verify_api_key(api_key)
            print(f"🔑 Autenticado via API Key: {api_key[:8]}...")
            return {
                "auth_type": "api_key",
                "api_key": api_key,
                "has_limits": False  # API keys não têm limites de sessão anônima
            }
        except HTTPException as e:
            # API key inválida, mas pode ter token
            if authorization:
                pass  # Tentar token
            else:
                raise e  # Sem token alternativo, propagar erro
    
    # Tentativa 2: Token anônimo (público)
    if authorization:
        try:
            token_payload = anon_auth.verify_anonymous_token(request, authorization)
            session_id = token_payload["session_id"]
            print(f"🎫 Autenticado via token anônimo: {session_id}")
            return {
                "auth_type": "anonymous",
                "session_id": session_id,
                "session": token_payload["session"],
                "has_limits": True
            }
        except HTTPException as e:
            # Token inválido e sem API key válida
            if not api_key:
                raise e
            # Se tinha API key mas também falhou, propagar erro da API key
            raise HTTPException(
                status_code=401,
                detail="API Key e Token inválidos"
            )
    
    # Nenhuma autenticação fornecida
    raise HTTPException(
        status_code=401,
        detail={
            "error": "Autenticação obrigatória",
            "options": [
                "Opção 1: Use API Key demo - Header: X-API-Key: aidet_demo_hackathon_2026",
                "Opção 2: Gere token anônimo - POST /api/auth/anonymous"
            ]
        }
    )



# --- Endpoints ---

# ========================================
# AUTENTICAÇÃO ANÔNIMA (sem cadastro)
# ========================================

@app.post("/api/auth/anonymous", tags=["Authentication"])
def generate_anonymous_token(request: Request):
    """
    🎯 Gera token de acesso anônimo (SEM necessidade de API key)
    
    **Como funciona:**
    1. Faça esta chamada sem enviar nada
    2. Receba tokens JWT (access + refresh)
    3. Use o access_token nas próximas requisições
    
    **Limites da sessão anônima:**
    - ✅ 50 requisições por sessão
    - ✅ 5.000 créditos de quota
    - ✅ Access token válido por 1h (renováveis)
    - ✅ Sessão válida por 7 dias
    
    **Para acesso ilimitado:** Use API key demo ou solicite chave permanente.
    
    **Exemplo de uso:**
    ```bash
    # 1. Obter tokens
    curl -X POST "https://seu-dominio.com/api/auth/anonymous"
    # Resposta: {"access_token": "...", "refresh_token": "..."}
    
    # 2. Usar access token
    curl -X POST "https://seu-dominio.com/api/analyze-image" \\
      -H "Authorization: Bearer <access_token>" \\
      -F "file=@imagem.jpg"
    
    # 3. Renovar quando expirar (após 1h)
    curl -X POST "https://seu-dominio.com/api/auth/refresh" \\
      -H "X-Refresh-Token: <refresh_token>"
    ```
    """
    client_ip = anon_auth._get_real_ip(request)
    fingerprint = request.headers.get("X-Browser-Fingerprint")
    
    token_data = anon_auth.generate_anonymous_token(
        client_ip=client_ip,
        fingerprint=fingerprint
    )
    
    return token_data


@app.post("/api/auth/refresh", tags=["Authentication"])
def refresh_anonymous_token(request: Request):
    """
    🔄 Renova access token usando refresh token
    
    **Quando usar:**
    - Access token expirou (após 1h)
    - Preventivamente antes de expirar
    
    **Exemplo:**
    ```bash
    curl -X POST "https://seu-dominio.com/api/auth/refresh" \\
      -H "X-Refresh-Token: <seu_refresh_token>"
    ```
    
    **Retorna:** Novos access_token + refresh_token
    """
    return anon_auth.refresh_access_token(request)


@app.get("/api/auth/session", tags=["Authentication"])
def get_session_stats(
    token_payload: dict = Depends(anon_auth.verify_anonymous_token)
):
    """
    📊 Consulta estatísticas da sessão anônima atual
    
    Retorna informações sobre uso de quota, limites e idade da sessão.
    
    **Requer:** Header `Authorization: Bearer <access_token>`
    """
    session_id = token_payload["session_id"]
    stats = anon_auth.get_session_stats(session_id)
    
    return {
        "session_id": session_id,
        "type": "anonymous",
        "stats": stats
    }

@app.delete("/api/auth/session", tags=["Authentication"])
def delete_session(
    token_payload: dict = Depends(anon_auth.verify_anonymous_token)
):
    """
    🗑️ Encerra a sessão anônima atual
    
    Use antes de criar nova sessão para trocar de modo (com/sem chave Gemini).
    Isso libera espaço na quota de sessões ativas por IP.
    
    **Exemplo:**
    ```bash
    # 1. Encerrar sessão atual
    curl -X DELETE "/api/auth/session" \\
      -H "Authorization: Bearer <access_token>"
    
    # 2. Criar nova sessão (agora no modo desejado)
    curl -X POST "/api/auth/anonymous"
    ```
    """
    session_id = token_payload["session_id"]
    
    if session_id in anon_auth.anonymous_sessions:
        # Remover sessão
        session = anon_auth.anonymous_sessions[session_id]
        del anon_auth.anonymous_sessions[session_id]
        
        print(f"🗑️ Sessão encerrada: {session_id} | Req usadas: {session['requests_count']}")
        
        return {
            "message": "Sessão encerrada com sucesso",
            "session_id": session_id,
            "stats": {
                "requests_used": session["requests_count"],
                "quota_used": session["quota_used"]
            }
        }
    else:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")

# ========================================
# ANÁLISE
# ========================================

@app.post("/api/analyze-image", tags=["Analysis"])
async def analyze_image_full(
    request: Request,
    file: UploadFile = File(...),
    auth: dict = Depends(verify_flexible_auth),
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key"),
    captcha_token: Optional[str] = Header(None, alias="X-Captcha-Token")
):
    """
    🔍 Análise completa de imagem: FFT + NOISE + ELA + Gemini AI
    
    ## Autenticação (escolha uma):
    
    **Opção 1 - API Key (recomendado para integração):**
    ```bash
    curl -X POST "/api/analyze-image" \\
      -H "X-API-Key: aidet_demo_hackathon_2026" \\
      -F "file=@imagem.jpg"
    ```
    
    **Opção 2 - Token Anônimo (público, limitado):**
    ```bash
    # Obter token primeiro
    TOKEN=$(curl -X POST "/api/auth/anonymous" | jq -r .access_token)
    
    # Usar token
    curl -X POST "/api/analyze-image" \\
      -H "Authorization: Bearer $TOKEN" \\
      -F "file=@imagem.jpg"
    ```
    
    ## Headers opcionais:
    - **X-Gemini-Key:** Sua chave Gemini (aumenta limites e usa seus créditos)
    - **X-Captcha-Token:** Token reCAPTCHA (se CAPTCHA estiver habilitado)
    
    ## Limites por tipo de autenticação:
    
    ### API Key demo:
    - Rate limit: 20 req/min, 200 req/hora
    - Quota: Ilimitada
    
    ### Token anônimo SEM chave Gemini própria:
    - Requisições: 50 por sessão
    - Quota: 5.000 créditos por sessão
    - Rate limit: 3 req/min
    
    ### Token anônimo COM chave Gemini própria:
    - Requisições: 200 por sessão (4x mais!)
    - Quota: Ilimitada
    - Rate limit: 20 req/min (6x mais!)
    
    💡 **Dica:** Use sua própria chave Gemini (X-Gemini-Key) para limites muito maiores!
    
    ## Budget Caps (chave Gemini do servidor):
    - $5/dia | $50/mês (protege custos do servidor)
    """
    
    auth_type = auth["auth_type"]
    has_custom_gemini_key = x_gemini_key is not None
    
    # 1. Verificar quota baseado no tipo de autenticação
    if auth_type == "api_key":
        api_key = auth["api_key"]
        
        # Verificar quota da API key
        can_proceed, message = quota_manager.check_quota(api_key)
        if not can_proceed:
            raise HTTPException(status_code=429, detail=message)
        
        print(f"📊 Análise via API Key: {api_key[:8]}... | {file.filename}")
        
    elif auth_type == "anonymous":
        session_id = auth["session_id"]
        session = auth["session"]
        limit_type = auth.get("limit_type", "server_key")
        current_limits = auth.get("current_limits", {})
        
        # Limites já foram verificados em verify_anonymous_token
        # (incrementou requests_count automaticamente)
        
        req_limit = current_limits.get("requests_limit", "?")
        quota_limit = current_limits.get("quota_limit", "?")
        
        print(f"📊 Análise anônima: {session_id}")
        print(f"   └─ Tipo: {limit_type} | Req: {session['requests_count']}/{req_limit} | Quota: {session['quota_used']}/{quota_limit}")
        
        if has_custom_gemini_key:
            print(f"   └─ 💡 Cliente usando chave Gemini própria (limites estendidos)")
    
    # 2. Se usar chave do servidor, verificar budget
    if not has_custom_gemini_key:
        can_proceed, message = cost_tracker.can_make_request()
        if not can_proceed:
            raise HTTPException(
                status_code=429,
                detail=f"{message}. 💡 Use sua própria chave Gemini (header X-Gemini-Key) para continuar sem limite de budget."
            )
    
    # 3. Log de rate limit
    rate_limit = get_dynamic_rate_limit(request, "analyze_full")
    if has_custom_gemini_key:
        print(f"   └─ Rate limit: {rate_limit} (chave própria)")
    else:
        print(f"   └─ Rate limit: {rate_limit} (chave do servidor)")
    
    # 4. Validar e salvar arquivo
    content = await file.read()
    validate_file_content(content, file.filename)
    path = save_bytes_to_disk(content, file.filename)
    
    try:
        # 5. Executar análise
        analysis_service = AnalysisService(custom_gemini_key=x_gemini_key)
        result = await analysis_service.analyze_full(path)
        
        # 6. Consumir quota baseado no tipo de autenticação
        if auth_type == "api_key":
            quota_manager.consume_quota(api_key)
            print(f"✅ Quota API key consumida")
            
        elif auth_type == "anonymous":
            # Consumir créditos da quota
            # Se estiver usando chave própria, consumir menos (custo simbólico)
            if has_custom_gemini_key:
                cost = result.get("estimated_cost", 10)  # Custo menor (cliente paga Gemini)
            else:
                cost = result.get("estimated_cost", 100)  # Custo normal (servidor paga)
            
            anon_auth.consume_quota(session_id, cost)
            
            # ✅ NOVO: Adicionar info de uso com tipo de limite correto
            result["session_usage"] = anon_auth.get_session_stats(
                session_id, 
                has_custom_key=has_custom_gemini_key
            )
            
            print(f"✅ Quota consumida: {cost} créditos | Restante: {result['session_usage']['quota_remaining']}")
        
        # 7. Rastrear custo (apenas se usou chave do servidor E Gemini foi usado)
        if not has_custom_gemini_key:
            gemini_used = result.get("gemini_analysis", {}).get("verdict") not in ["DISABLED", "ERROR"]
            if gemini_used:
                cost_tracker.track_request()
                print(f"💰 Custo do servidor registrado. Uso atual: {cost_tracker.get_current_usage()}")
        else:
            print(f"💡 Cliente usando chave própria - sem custo para o servidor")
        
        return no_cache_response(result)
        
    except Exception as e:
        print(f"❌ Erro na análise: {e}")
        import traceback
        traceback.print_exc()
        
        # NÃO consumir quota se deu erro
        raise HTTPException(status_code=500, detail=f"Erro na análise: {str(e)}")
    
    finally:
        # 8. Limpar arquivo temporário
        if os.path.exists(path):
            os.remove(path)



@app.get("/")
def home():
    return {
        "message": "AI Detector API v2.0",
        "docs": "/docs",
        "authentication": "Required (X-API-Key header)",
        "custom_gemini_key": "Optional (X-Gemini-Key header)"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}