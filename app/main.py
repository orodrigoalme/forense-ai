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
# Import validator and analyzers
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
from fastapi.middleware.cors import CORSMiddleware

# Inicializar rastreador de custos
# Initialize cost tracker
cost_tracker = CostTracker()

app = FastAPI(
    title="AI Image Detector API",
    description="API para detecção forense de imagens geradas por IA",
    version="1.5.0"
)

# Adicionar CORS antes de qualquer rota
# Add CORS before any route
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir qualquer origem / Allow any origin
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos os métodos / Allow all methods
    allow_headers=["*"],  # Permitir qualquer header / Allow any header
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
analysis_service = AnalysisService()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_bytes_to_disk(content: bytes, original_filename: str) -> str:
    """
    Salva bytes da imagem no disco.
    Saves image bytes to disk.
    """
    # Gerar nome único / Generate unique name
    file_extension = original_filename.split(".")[-1] if "." in original_filename else "jpg"
    unique_name = f"{uuid.uuid4()}"
    filename = f"{unique_name}.{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    # Gravar os bytes no disco / Write bytes to disk
    with open(file_path, "wb") as buffer:
        buffer.write(content)
        
    # Redimensionar para evitar travar o servidor
    # Resize to prevent server overload
    try:
        resize_if_too_big(file_path)
    except Exception as e:
        pass  # Prosseguir mesmo com falha / Proceed even on failure

    return file_path


# Função para evitar cache do navegador
# Function to prevent browser caching
def no_cache_response(data: dict):
    response = JSONResponse(data)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# ========================================
# AUTENTICAÇÃO FLEXÍVEL / FLEXIBLE AUTHENTICATION
# ========================================

def verify_flexible_auth(
    request: Request,
    authorization: Optional[str] = Header(None),
    api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> dict:
    """
    Verifica autenticação - aceita token anônimo OU API key direta.
    Verifies authentication - accepts anonymous token OR direct API key.
    
    Prioridade / Priority:
    1. API Key → validação direta / direct validation (no anonymous session limits)
    2. Bearer Token → validação de token anônimo / anonymous token validation (with limits)
    3. Nenhum / None → erro 401
    
    Returns:
        dict com tipo de autenticação e dados relevantes / dict with auth type and relevant data
    """
    
    # Tentativa 1: API Key direta (clientes registrados ou demo)
    # Attempt 1: Direct API Key (registered clients or demo)
    if api_key:
        try:
            api_key_auth.verify_api_key(api_key)
            return {
                "auth_type": "api_key",
                "api_key": api_key,
                "has_limits": False  # API keys não têm limites / API keys have no limits
            }
        except HTTPException as e:
            # API key inválida, mas pode ter token
            # Invalid API key, but may have token
            if authorization:
                pass  # Tentar token / Try token
            else:
                raise e  # Sem token alternativo / No fallback token
    
    # Tentativa 2: Token anônimo (público)
    # Attempt 2: Anonymous token (public)
    if authorization:
        try:
            token_payload = anon_auth.verify_anonymous_token(request, authorization)
            session_id = token_payload["session_id"]
            return {
                "auth_type": "anonymous",
                "session_id": session_id,
                "session": token_payload["session"],
                "has_limits": True
            }
        except HTTPException as e:
            # Token inválido e sem API key válida
            # Invalid token and no valid API key
            if not api_key:
                raise e
            # Se tinha API key mas também falhou, propagar erro
            # If had API key but also failed, propagate error
            raise HTTPException(
                status_code=401,
                detail="API Key e Token inválidos / Invalid API Key and Token"
            )
    
    # Nenhuma autenticação fornecida
    # No authentication provided
    raise HTTPException(
        status_code=401,
        detail={
            "error": "Autenticação obrigatória / Authentication required",
            "options": [
                "Option 1: Use demo API Key - Header: X-API-Key: aidet_demo_hackathon_2026",
                "Option 2: Generate anonymous token - POST /api/auth/anonymous"
            ]
        }
    )



# --- Endpoints / API Routes ---

# ========================================
# AUTENTICAÇÃO ANÔNIMA / ANONYMOUS AUTHENTICATION
# ========================================

@app.post("/api/auth/anonymous", tags=["Authentication"])
def generate_anonymous_token(request: Request):
    """
    🎯 Gera token de acesso anônimo (SEM necessidade de API key).
    🎯 Generates anonymous access token (NO API key needed).
    
    **Como funciona / How it works:**
    1. Faça esta chamada sem enviar nada / Make this call without sending anything
    2. Receba tokens JWT (access + refresh) / Receive JWT tokens (access + refresh)
    3. Use o access_token nas próximas requisições / Use access_token in subsequent requests
    
    **Limites da sessão anônima / Anonymous session limits:**
    - ✅ 50 requisições por sessão / requests per session
    - ✅ 5.000 créditos de quota / quota credits
    - ✅ Access token válido por 1h (renováveis) / Access token valid for 1h (renewable)
    - ✅ Sessão válida por 7 dias / Session valid for 7 days
    
    **Para acesso ilimitado / For unlimited access:** Use API key demo ou solicite chave permanente / Use demo API key or request a permanent one.
    
    **Exemplo de uso / Usage example:**
    ```bash
    # 1. Obter tokens / 1. Get tokens
    curl -X POST "https://seu-dominio.com/api/auth/anonymous"
    # Resposta: {"access_token": "...", "refresh_token": "..."} / Response: ...
    
    # 2. Usar access token / 2. Use access token
    curl -X POST "https://seu-dominio.com/api/analyze-image" \\
      -H "Authorization: Bearer <access_token>" \\
      -F "file=@imagem.jpg"
    
    # 3. Renovar quando expirar (após 1h) / 3. Renew when expired (after 1h)
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
    🔄 Renova access token usando refresh token.
    🔄 Refreshes access token using refresh token.
    
    **Quando usar / When to use:**
    - Access token expirou (após 1h) / Access token expired
    - Preventivamente antes de expirar / Preventively before expiring
    
    **Exemplo / Example:**
    ```bash
    curl -X POST "https://seu-dominio.com/api/auth/refresh" \\
      -H "X-Refresh-Token: <seu_refresh_token>"
    ```
    
    **Retorna / Returns:** Novos access_token + refresh_token / New access_token + refresh_token
    """
    return anon_auth.refresh_access_token(request)


@app.get("/api/auth/session", tags=["Authentication"])
def get_session_stats(
    token_payload: dict = Depends(anon_auth.verify_anonymous_token)
):
    """
    📊 Consulta estatísticas da sessão anônima atual.
    📊 View current anonymous session statistics.
    
    Retorna informações sobre uso de quota, limites e idade da sessão.
    Returns info about quota usage, limits, and session age.
    
    **Requer / Requires:** Header `Authorization: Bearer <access_token>`
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
    🗑️ Encerra a sessão anônima atual.
    🗑️ Ends the current anonymous session.
    
    Use antes de criar nova sessão para trocar de modo (com/sem chave Gemini).
    Use before creating a new session to switch modes (with/without Gemini key).
    Isso libera espaço na quota de sessões ativas por IP.
    This frees up space in the active session quota per IP.
    
    **Exemplo / Example:**
    ```bash
    # 1. Encerrar sessão atual / 1. End current session
    curl -X DELETE "/api/auth/session" \\
      -H "Authorization: Bearer <access_token>"
    
    # 2. Criar nova sessão / 2. Create new session
    curl -X POST "/api/auth/anonymous"
    ```
    """
    session_id = token_payload["session_id"]
    
    if session_id in anon_auth.anonymous_sessions:
        # Remover sessão / Remove session
        session = anon_auth.anonymous_sessions[session_id]
        del anon_auth.anonymous_sessions[session_id]
        
        return {
            "message": "Sessão encerrada com sucesso / Session ended successfully",
            "session_id": session_id,
            "stats": {
                "requests_used": session["requests_count"],
                "quota_used": session["quota_used"]
            }
        }
    else:
        raise HTTPException(status_code=404, detail="Sessão não encontrada / Session not found")

# ========================================
# ANÁLISE / ANALYSIS
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
    🔍 Análise completa de imagem: FFT + NOISE + ELA + Gemini AI.
    🔍 Full image analysis: FFT + NOISE + ELA + Gemini AI.
    
    ## Autenticação / Authentication (choose one):
    
    **Opção 1 - API Key (recomendado / recommended):**
    ```bash
    curl -X POST "/api/analyze-image" \\
      -H "X-API-Key: aidet_demo_hackathon_2026" \\
      -F "file=@imagem.jpg"
    ```
    
    **Opção 2 - Token Anônimo / Optional 2 - Anonymous Token (public, limited):**
    ```bash
    # Obter token primeiro / Get token first
    TOKEN=$(curl -X POST "/api/auth/anonymous" | jq -r .access_token)
    
    # Usar token / Use token
    curl -X POST "/api/analyze-image" \\
      -H "Authorization: Bearer $TOKEN" \\
      -F "file=@imagem.jpg"
    ```
    
    ## Headers opcionais / Optional Headers:
    - **X-Gemini-Key:** Sua chave Gemini / Your Gemini key (increases limits and uses your credits)
    - **X-Captcha-Token:** Token reCAPTCHA / reCAPTCHA token (if enabled)
    
    ## Limites por tipo de autenticação / Limits by auth type:
    
    ### API Key demo:
    - Rate limit: 20 req/min, 200 req/hora
    - Quota: Ilimitada / Unlimited
    
    ### Token anônimo SEM chave Gemini própria / Anonymous token WITHOUT own Gemini key:
    - Requisições: 50 por sessão / Requests per session
    - Quota: 5.000 créditos por sessão / credits per session
    - Rate limit: 3 req/min
    
    ### Token anônimo COM chave Gemini própria / Anonymous token WITH own Gemini key:
    - Requisições: 200 por sessão / Requests per session (4x more!)
    - Quota: Ilimitada / Unlimited
    - Rate limit: 20 req/min (6x more!)
    
    💡 **Dica / Tip:** Use sua própria chave Gemini (X-Gemini-Key) para limites muito maiores! / Use your own Gemini key for much higher limits.
    
    ## Budget Caps (chave Gemini do servidor / Server Gemini key):
    - $5/dia | $50/mês (protege custos do servidor / protects server costs)
    """
    
    auth_type = auth["auth_type"]
    has_custom_gemini_key = x_gemini_key is not None
    
    # Verificar quota baseado no tipo de autenticação
    # Check quota based on authentication type
    if auth_type == "api_key":
        api_key = auth["api_key"]
        
        # Verificar quota da API key / Check API key quota
        can_proceed, message = quota_manager.check_quota(api_key)
        if not can_proceed:
            raise HTTPException(status_code=429, detail=message)
        
    elif auth_type == "anonymous":
        session_id = auth["session_id"]
        session = auth["session"]
        limit_type = auth.get("limit_type", "server_key")
        current_limits = auth.get("current_limits", {})
        
        # Limites já foram verificados em verify_anonymous_token
        # Limits already checked in verify_anonymous_token
    
    # Se usar chave do servidor, verificar budget
    # If using server key, check budget
    if not has_custom_gemini_key:
        can_proceed, message = cost_tracker.can_make_request()
        if not can_proceed:
            raise HTTPException(
                status_code=429,
                detail=f"{message}. Use your own Gemini key (X-Gemini-Key header) to continue without budget limits."
            )
    
    # Validar e salvar arquivo / Validate and save file
    content = await file.read()
    validate_file_content(content, file.filename)
    path = save_bytes_to_disk(content, file.filename)
    
    try:
        # Executar análise / Execute analysis
        analysis_service = AnalysisService(custom_gemini_key=x_gemini_key)
        result = await analysis_service.analyze_full(path)
        
        # Consumir quota baseado no tipo de autenticação
        # Consume quota based on authentication type
        if auth_type == "api_key":
            quota_manager.consume_quota(api_key)
            
        elif auth_type == "anonymous":
            # Consumir créditos da quota
            # Consume quota credits
            if has_custom_gemini_key:
                cost = result.get("estimated_cost", 10)  # Custo menor / Lower cost
            else:
                cost = result.get("estimated_cost", 100)  # Custo normal / Normal cost
            
            anon_auth.consume_quota(session_id, cost)
            
            # Adicionar info de uso com tipo de limite correto
            # Add usage info with correct limit type
            result["session_usage"] = anon_auth.get_session_stats(
                session_id, 
                has_custom_key=has_custom_gemini_key
            )
        
        # Rastrear custo (apenas se usou chave do servidor E Gemini foi usado)
        # Track cost (only if using server key AND Gemini was used)
        if not has_custom_gemini_key:
            gemini_used = result.get("gemini_analysis", {}).get("verdict") not in ["DISABLED", "ERROR"]
            if gemini_used:
                cost_tracker.track_request()
        
        return no_cache_response(result)
        
    except Exception as e:
        # Não consumir quota se deu erro
        # Don't consume quota on error
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")
    
    finally:
        # Limpar arquivo temporário / Clean up temporary file
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