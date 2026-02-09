import jwt
import uuid
import os
import time
import threading
from datetime import datetime, timedelta
from fastapi import HTTPException, Header, Request
from typing import Optional



class AnonymousAuthManager:
    """
    Gera tokens JWT para usuários anônimos com renovação automática,
    limites dinâmicos e proteção anti-abuso.
    Generates JWT tokens for anonymous users with auto-renewal,
    dynamic limits, and anti-abuse protection.
    """
    
    def __init__(self):
        self.secret = os.getenv("JWT_SECRET")
        if not self.secret:
            raise ValueError("JWT_SECRET não configurado no .env")
        
        # Token curto (access) / Short-lived access token
        self.access_token_lifetime = int(os.getenv("ACCESS_TOKEN_LIFETIME_MINUTES", "60"))
        
        # Sessão longa (refresh) / Long-lived session (refresh)
        self.session_lifetime_days = int(os.getenv("SESSION_LIFETIME_DAYS", "7"))
        
        self.algorithm = "HS256"
        
        # Armazenar sessões anônimas (usar Redis em produção)
        # Store anonymous sessions (use Redis in production)
        self.anonymous_sessions = {}
        
        # Limites SEM chave Gemini própria / Limits WITHOUT own Gemini key
        self.default_requests_limit = int(os.getenv("ANON_REQUESTS_LIMIT", "50"))
        self.default_quota_limit = int(os.getenv("ANON_QUOTA_LIMIT", "5000"))
        
        # Limites COM chave Gemini própria / Limits WITH own Gemini key
        self.custom_key_requests_limit = int(os.getenv("ANON_REQUESTS_LIMIT_CUSTOM_KEY", "200"))
        self.custom_key_quota_limit = int(os.getenv("ANON_QUOTA_LIMIT_CUSTOM_KEY", "0"))  # 0 = ilimitado
        
        # Proteção 1: Rastrear criação de sessões por IP
        # Protection 1: Track session creation per IP
        self.session_creation_tracker = {}
        self.max_sessions_per_ip_per_hour = int(os.getenv("MAX_SESSIONS_PER_IP_HOUR", "3"))
        self.max_sessions_per_ip_per_day = int(os.getenv("MAX_SESSIONS_PER_IP_DAY", "10"))
        
        # Proteção 2: Limite de sessões ativas simultâneas
        # Protection 2: Concurrent active sessions limit
        self.max_active_sessions_per_ip = int(os.getenv("MAX_ACTIVE_SESSIONS_PER_IP", "5"))
        
        # Proteção 3: Limpeza automática / Protection 3: Auto-cleanup
        self._start_cleanup_task()
    
    # ========================================
    # Proteção 1: Rate Limit de Criação
    # Protection 1: Creation Rate Limit
    # ========================================
    
    def can_create_session(self, client_ip: str) -> tuple[bool, str]:
        """
        Verifica se IP pode criar nova sessão.
        Checks if IP can create a new session.
        
        Args:
            client_ip: IP do cliente / Client IP
        
        Returns:
            (can_create, error_message)
        """
        now = datetime.utcnow()
        
        # Limpar registros antigos (> 24h)
        if client_ip in self.session_creation_tracker:
            self.session_creation_tracker[client_ip] = [
                ts for ts in self.session_creation_tracker[client_ip]
                if (now - ts).total_seconds() < 86400  # 24 horas
            ]
        
        # Obter timestamps de criação deste IP / Get creation timestamps for this IP
        timestamps = self.session_creation_tracker.get(client_ip, [])
        
        # Verificar sessões ativas simultâneas / Check concurrent active sessions
        active_sessions = self.get_active_sessions_for_ip(client_ip)
        if active_sessions >= self.max_active_sessions_per_ip:
            return False, f"Limite de {self.max_active_sessions_per_ip} sessões ativas simultâneas atingido. Aguarde sessões expirarem ou use API Key."
        
        # Verificar limite por hora / Check hourly limit
        last_hour = [ts for ts in timestamps if (now - ts).total_seconds() < 3600]
        if len(last_hour) >= self.max_sessions_per_ip_per_hour:
            return False, f"Limite de {self.max_sessions_per_ip_per_hour} sessões/hora atingido. Tente novamente em alguns minutos ou use API Key."
        
        # Verificar limite por dia / Check daily limit
        if len(timestamps) >= self.max_sessions_per_ip_per_day:
            return False, f"Limite de {self.max_sessions_per_ip_per_day} sessões/dia atingido. Retorne amanhã ou use API Key."
        
        return True, ""
    
    def track_session_creation(self, client_ip: str):
        """
        Registra criação de sessão.
        Records session creation.
        """
        if client_ip not in self.session_creation_tracker:
            self.session_creation_tracker[client_ip] = []
        
        self.session_creation_tracker[client_ip].append(datetime.utcnow())
    
    # ========================================
    # Proteção 2: Sessões Ativas por IP
    # Protection 2: Active Sessions per IP
    # ========================================
    
    def get_active_sessions_for_ip(self, client_ip: str) -> int:
        """
        Conta sessões ativas (não expiradas) deste IP.
        Counts active (non-expired) sessions for this IP.
        
        Args:
            client_ip: IP do cliente / Client IP
        
        Returns:
            Número de sessões ativas / Number of active sessions
        """
        count = 0
        now = datetime.utcnow()
        
        for session in self.anonymous_sessions.values():
            # Sessão deste IP e não expirada?
            if session["ip"] == client_ip:
                age = now - session["created_at"]
                if age <= timedelta(days=self.session_lifetime_days):
                    count += 1
        
        return count
    
    # ========================================
    # Proteção 3: Limpeza Automática
    # Protection 3: Automatic Cleanup
    # ========================================
    
    def _start_cleanup_task(self):
        """
        Inicia tarefa de limpeza de sessões expiradas.
        Starts expired session cleanup task.
        """
        def cleanup():
            while True:
                time.sleep(3600)
                self._cleanup_expired_sessions()
        
        thread = threading.Thread(target=cleanup, daemon=True)
        thread.start()
    
    def _cleanup_expired_sessions(self):
        """
        Remove sessões expiradas e limpa tracker.
        Removes expired sessions and cleans tracker.
        """
        now = datetime.utcnow()
        expired_sessions = []
        
        # Limpar sessões expiradas / Clean expired sessions
        for session_id, session in list(self.anonymous_sessions.items()):
            age = now - session["created_at"]
            
            # Sessão expirou?
            if age > timedelta(days=self.session_lifetime_days):
                expired_sessions.append(session_id)
        
        # Remover sessões expiradas / Remove expired sessions
        for session_id in expired_sessions:
            del self.anonymous_sessions[session_id]
        
        # Limpar tracker de criação (> 24h)
        for ip in list(self.session_creation_tracker.keys()):
            self.session_creation_tracker[ip] = [
                ts for ts in self.session_creation_tracker[ip]
                if (now - ts).total_seconds() < 86400
            ]
            
            # Remover IPs sem registros / Remove IPs without records
            if not self.session_creation_tracker[ip]:
                del self.session_creation_tracker[ip]
        

    
    # ========================================
    # Geração de Tokens / Token Generation
    # ========================================
    
    def generate_anonymous_token(
        self, 
        client_ip: str, 
        fingerprint: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> dict:
        """
        Gera tokens JWT para usuário anônimo.
        Generates JWT tokens for anonymous user.
        
        Args:
            client_ip: IP do cliente / Client IP
            fingerprint: Browser fingerprint (opcional / optional)
            session_id: Se fornecido, renova sessão existente / If provided, renews existing session
        
        Returns:
            Dict com access_token, refresh_token e metadados / Dict with tokens and metadata
        """
        now = datetime.utcnow()
        
        # Renovação de sessão existente / Existing session renewal
        if session_id and session_id in self.anonymous_sessions:
            session = self.anonymous_sessions[session_id]
            
            # Verificar se sessão não expirou / Check if session hasn't expired
            session_age = now - session["created_at"]
            if session_age > timedelta(days=self.session_lifetime_days):
                raise HTTPException(
                    status_code=401,
                    detail="Sessão expirada completamente. Crie nova sessão."
                )
            
        else:
            # Nova sessão: Verificar proteções / New session: Check protections
            can_create, error_msg = self.can_create_session(client_ip)
            if not can_create:
                raise HTTPException(
                    status_code=429,
                    detail=error_msg + " 💡 Dica: Use API Key permanente para acesso ilimitado."
                )
            
            # Criar nova sessão / Create new session
            session_id = f"anon_{uuid.uuid4().hex[:12]}"
            self.anonymous_sessions[session_id] = {
                "created_at": now,
                "ip": client_ip,
                "fingerprint": fingerprint,
                "requests_count": 0,
                "quota_used": 0
            }
            
            # Registrar criação / Record creation
            self.track_session_creation(client_ip)
        
        session = self.anonymous_sessions[session_id]
        
        # Gerar ACCESS TOKEN (curto) / Generate ACCESS TOKEN (short-lived)
        access_exp = now + timedelta(minutes=self.access_token_lifetime)
        access_token = jwt.encode({
            "sub": session_id,
            "type": "access",
            "ip": client_ip,
            "fingerprint": fingerprint,
            "iat": now.timestamp(),
            "exp": access_exp.timestamp()
        }, self.secret, algorithm=self.algorithm)
        
        # Gerar REFRESH TOKEN (longo) / Generate REFRESH TOKEN (long-lived)
        refresh_exp = now + timedelta(days=self.session_lifetime_days)
        refresh_token = jwt.encode({
            "sub": session_id,
            "type": "refresh",
            "ip": client_ip,
            "fingerprint": fingerprint,
            "iat": now.timestamp(),
            "exp": refresh_exp.timestamp()
        }, self.secret, algorithm=self.algorithm)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "Bearer",
            "access_expires_in": self.access_token_lifetime * 60,  # em segundos
            "refresh_expires_in": self.session_lifetime_days * 24 * 3600,  # em segundos
            "access_expires_at": access_exp.isoformat(),
            "refresh_expires_at": refresh_exp.isoformat(),
            "session_id": session_id,
            "limits": {
                "default": {
                    "requests_limit": self.default_requests_limit,
                    "quota_limit": self.default_quota_limit,
                    "description": "Limites ao usar chave Gemini do servidor"
                },
                "custom_key": {
                    "requests_limit": self.custom_key_requests_limit,
                    "quota_limit": "unlimited" if self.custom_key_quota_limit == 0 else self.custom_key_quota_limit,
                    "description": "Limites ao usar sua própria chave Gemini (X-Gemini-Key)"
                },
                "current_usage": {
                    "requests_used": session["requests_count"],
                    "quota_used": session["quota_used"]
                }
            }
        }
    
    def refresh_access_token(
        self,
        request: Request,
        refresh_token: str = Header(None, alias="X-Refresh-Token")
    ) -> dict:
        """
        Renova access token usando refresh token.
        Refreshes access token using refresh token.
        
        Args:
            refresh_token: Refresh token válido / Valid refresh token in X-Refresh-Token header
        
        Returns:
            Novos tokens / New access_token + refresh_token
        """
        if not refresh_token:
            raise HTTPException(
                status_code=400,
                detail="Refresh token obrigatório no header X-Refresh-Token"
            )
        
        try:
            # Decodificar refresh token / Decode refresh token
            payload = jwt.decode(refresh_token, self.secret, algorithms=[self.algorithm])
            
            # Validar tipo / Validate type
            if payload.get("type") != "refresh":
                raise HTTPException(
                    status_code=401,
                    detail="Token inválido. Esperado refresh token."
                )
            
            session_id = payload["sub"]
            
            # Sessão existe? / Session exists?
            if session_id not in self.anonymous_sessions:
                raise HTTPException(
                    status_code=401,
                    detail="Sessão não encontrada. Crie nova sessão em POST /api/auth/anonymous"
                )
            
            # Validar IP / Validate IP
            client_ip = self._get_real_ip(request)
            token_ip = payload.get("ip")
            

            
            # Obter fingerprint do payload / Get fingerprint from payload
            fingerprint = payload.get("fingerprint")
            
            # Gerar novos tokens (sem verificar limites de criação)
            # Generate new tokens (skip creation limit checks)
            return self.generate_anonymous_token(
                client_ip=client_ip,
                fingerprint=fingerprint,
                session_id=session_id  # Passa session_id para indicar renovação
            )
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Refresh token expirado. Crie nova sessão em POST /api/auth/anonymous"
            )
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=401,
                detail=f"Refresh token inválido: {str(e)}"
            )
    
    def verify_anonymous_token(
        self,
        request: Request,
        authorization: Optional[str] = Header(None)
    ) -> dict:
        """
        Verifica access token anônimo com limites dinâmicos.
        Verifies anonymous access token with dynamic limits.
        
        Args:
            authorization: Header Authorization com Bearer token
        
        Returns:
            Payload do token com dados da sessão / Token payload with session data
        """
        # Extrair token / Extract token
        token = None
        if authorization:
            if authorization.startswith("Bearer "):
                token = authorization[7:]
            else:
                token = authorization
        
        if not token:
            raise HTTPException(
                status_code=401,
                detail="Token obrigatório. Obtenha em: POST /api/auth/anonymous"
            )
        
        # Decodificar / Decode
        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            
            # Verificar se é access token / Check if it's an access token
            if payload.get("type") != "access":
                raise HTTPException(
                    status_code=401,
                    detail="Token inválido. Use access token (não refresh token)."
                )
            
            session_id = payload["sub"]
            
            # Sessão existe? / Session exists?
            if session_id not in self.anonymous_sessions:
                raise HTTPException(
                    status_code=401,
                    detail="Sessão expirada ou inválida. Crie nova sessão."
                )
            
            session = self.anonymous_sessions[session_id]
            
            # Validar IP (anti-roubo de token) / Validate IP (token theft prevention)
            client_ip = self._get_real_ip(request)
            token_ip = payload.get("ip")
            
            
            # Verificar se cliente usa chave Gemini própria
            # Check if client is using own Gemini key
            has_custom_gemini_key = request.headers.get("X-Gemini-Key") is not None
            
            # Aplicar limites baseado em chave própria ou não
            # Apply limits based on own key or not
            if has_custom_gemini_key:
                requests_limit = self.custom_key_requests_limit
                quota_limit = self.custom_key_quota_limit
                limit_type = "custom_key"
            else:
                requests_limit = self.default_requests_limit
                quota_limit = self.default_quota_limit
                limit_type = "server_key"
            
            # Verificar limite de requisições / Check request limit
            if session["requests_count"] >= requests_limit:
                extra_msg = ""
                if limit_type == "server_key":
                    extra_msg = f" 💡 Use sua própria chave Gemini (X-Gemini-Key) para {self.custom_key_requests_limit} req/sessão."
                
                raise HTTPException(
                    status_code=429,
                    detail=f"Limite de requisições atingido ({requests_limit}).{extra_msg}"
                )
            
            # Verificar quota (0 = ilimitado) / Check quota (0 = unlimited)
            if quota_limit > 0 and session["quota_used"] >= quota_limit:
                extra_msg = ""
                if limit_type == "server_key":
                    quota_msg = "ilimitada" if self.custom_key_quota_limit == 0 else f"{self.custom_key_quota_limit} créditos"
                    extra_msg = f" 💡 Use sua própria chave Gemini (X-Gemini-Key) para quota {quota_msg}."
                
                raise HTTPException(
                    status_code=429,
                    detail=f"Quota esgotada ({quota_limit} créditos).{extra_msg}"
                )
            
            # Incrementar contador / Increment counter
            session["requests_count"] += 1
            
            return {
                "session_id": session_id,
                "type": "anonymous",
                "session": session,
                "limit_type": limit_type,
                "current_limits": {
                    "requests_limit": requests_limit,
                    "quota_limit": quota_limit if quota_limit > 0 else "unlimited"
                }
            }
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=401,
                detail="Access token expirado. Renove em: POST /api/auth/refresh"
            )
        except jwt.InvalidTokenError as e:
            raise HTTPException(
                status_code=401,
                detail=f"Token inválido: {str(e)}"
            )
    
    def consume_quota(self, session_id: str, amount: int):
        """
        Consome quota da sessão anônima.
        Consumes quota from anonymous session.
        
        Args:
            session_id: ID da sessão / Session ID
            amount: Quantidade de créditos / Amount of credits
        """
        if session_id in self.anonymous_sessions:
            self.anonymous_sessions[session_id]["quota_used"] += amount
    
    def get_session_stats(self, session_id: str, has_custom_key: bool = False) -> dict:
        """
        Retorna estatísticas da sessão.
        Returns session statistics.
        
        Args:
            session_id: ID da sessão / Session ID
            has_custom_key: Se usa chave Gemini própria / If using own Gemini key
        
        Returns:
            Dict com estatísticas de uso / Dict with usage statistics
        """
        if session_id not in self.anonymous_sessions:
            return {"error": "Sessão não encontrada"}
        
        session = self.anonymous_sessions[session_id]
        
        # Aplicar limites corretos / Apply correct limits
        if has_custom_key:
            requests_limit = self.custom_key_requests_limit
            quota_limit = self.custom_key_quota_limit
            limit_type = "custom_key"
        else:
            requests_limit = self.default_requests_limit
            quota_limit = self.default_quota_limit
            limit_type = "server_key"
        
        # Calcular restantes / Calculate remaining
        requests_remaining = max(0, requests_limit - session["requests_count"])
        
        if quota_limit == 0:
            quota_remaining = "unlimited"
            quota_limit_display = "unlimited"
        else:
            quota_remaining = max(0, quota_limit - session["quota_used"])
            quota_limit_display = quota_limit
        
        return {
            "requests_used": session["requests_count"],
            "requests_remaining": requests_remaining,
            "requests_limit": requests_limit,
            "quota_used": session["quota_used"],
            "quota_remaining": quota_remaining,
            "quota_limit": quota_limit_display,
            "created_at": session["created_at"].isoformat(),
            "session_age_hours": round((datetime.utcnow() - session["created_at"]).total_seconds() / 3600, 2),
            "limit_type": limit_type,
            "tip": "Use header X-Gemini-Key com sua chave para limites maiores" if limit_type == "server_key" else "Usando limites estendidos (chave própria)"
        }
    
    def _get_real_ip(self, request: Request) -> str:
        """
        Obtém IP real do cliente considerando proxies.
        Gets client's real IP considering proxies.
        
        Args:
            request: FastAPI Request object
        
        Returns:
            IP address string
        """
        # Tentar headers de proxy primeiro / Try proxy headers first
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback para IP direto / Fallback to direct IP
        return request.client.host if request.client else "unknown"


# Instância global / Global instance
anon_auth = AnonymousAuthManager()
