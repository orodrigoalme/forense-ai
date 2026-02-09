# app/middleware/anonymous_auth.py

import jwt
import uuid
import os
from datetime import datetime, timedelta
from fastapi import HTTPException, Header, Request
from typing import Optional


# app/middleware/anonymous_auth.py

import jwt
import uuid
import os
from datetime import datetime, timedelta
from fastapi import HTTPException, Header, Request
from typing import Optional

# app/middleware/anonymous_auth.py

import jwt
import uuid
import os
import time
import threading
from datetime import datetime, timedelta
from fastapi import HTTPException, Header, Request
from typing import Optional


class AnonymousAuthManager:
    """Gera tokens JWT para usuários anônimos com renovação automática, limites dinâmicos e proteção anti-abuso"""
    
    def __init__(self):
        self.secret = os.getenv("JWT_SECRET")
        if not self.secret:
            raise ValueError("JWT_SECRET não configurado no .env")
        
        # Token curto (access)
        self.access_token_lifetime = int(os.getenv("ACCESS_TOKEN_LIFETIME_MINUTES", "60"))
        
        # Sessão longa (refresh)
        self.session_lifetime_days = int(os.getenv("SESSION_LIFETIME_DAYS", "7"))
        
        self.algorithm = "HS256"
        
        # Armazenar sessões anônimas (em produção, usar Redis)
        self.anonymous_sessions = {}
        
        # Limites SEM chave Gemini própria (servidor paga)
        self.default_requests_limit = int(os.getenv("ANON_REQUESTS_LIMIT", "50"))
        self.default_quota_limit = int(os.getenv("ANON_QUOTA_LIMIT", "5000"))
        
        # Limites COM chave Gemini própria (cliente paga)
        self.custom_key_requests_limit = int(os.getenv("ANON_REQUESTS_LIMIT_CUSTOM_KEY", "200"))
        self.custom_key_quota_limit = int(os.getenv("ANON_QUOTA_LIMIT_CUSTOM_KEY", "0"))  # 0 = ilimitado
        
        # ✅ PROTEÇÃO 1: Rastrear criação de sessões por IP
        self.session_creation_tracker = {}  # {ip: [timestamps]}
        self.max_sessions_per_ip_per_hour = int(os.getenv("MAX_SESSIONS_PER_IP_HOUR", "3"))
        self.max_sessions_per_ip_per_day = int(os.getenv("MAX_SESSIONS_PER_IP_DAY", "10"))
        
        # ✅ PROTEÇÃO 2: Limite de sessões ativas simultâneas
        self.max_active_sessions_per_ip = int(os.getenv("MAX_ACTIVE_SESSIONS_PER_IP", "5"))
        
        print(f"✅ Anonymous Auth inicializado | Access: {self.access_token_lifetime}min | Session: {self.session_lifetime_days}d")
        print(f"   📊 Limites SEM chave própria: {self.default_requests_limit} req, {self.default_quota_limit} créditos")
        print(f"   📊 Limites COM chave própria: {self.custom_key_requests_limit} req, {'ilimitado' if self.custom_key_quota_limit == 0 else self.custom_key_quota_limit} créditos")
        print(f"   🛡️ Proteção anti-abuso: {self.max_sessions_per_ip_per_hour}/hora, {self.max_sessions_per_ip_per_day}/dia, {self.max_active_sessions_per_ip} ativas/IP")
        
        # ✅ PROTEÇÃO 3: Iniciar limpeza automática
        self._start_cleanup_task()
    
    # ========================================
    # PROTEÇÃO 1: Rate Limit de Criação
    # ========================================
    
    def can_create_session(self, client_ip: str) -> tuple[bool, str]:
        """
        Verifica se IP pode criar nova sessão
        
        Args:
            client_ip: IP do cliente
        
        Returns:
            (pode_criar, mensagem_erro)
        """
        now = datetime.utcnow()
        
        # Limpar registros antigos (> 24h)
        if client_ip in self.session_creation_tracker:
            self.session_creation_tracker[client_ip] = [
                ts for ts in self.session_creation_tracker[client_ip]
                if (now - ts).total_seconds() < 86400  # 24 horas
            ]
        
        # Obter timestamps de criação deste IP
        timestamps = self.session_creation_tracker.get(client_ip, [])
        
        # Verificar sessões ativas simultâneas
        active_sessions = self.get_active_sessions_for_ip(client_ip)
        if active_sessions >= self.max_active_sessions_per_ip:
            return False, f"Limite de {self.max_active_sessions_per_ip} sessões ativas simultâneas atingido. Aguarde sessões expirarem ou use API Key."
        
        # Verificar limite por hora
        last_hour = [ts for ts in timestamps if (now - ts).total_seconds() < 3600]
        if len(last_hour) >= self.max_sessions_per_ip_per_hour:
            return False, f"Limite de {self.max_sessions_per_ip_per_hour} sessões/hora atingido. Tente novamente em alguns minutos ou use API Key."
        
        # Verificar limite por dia
        if len(timestamps) >= self.max_sessions_per_ip_per_day:
            return False, f"Limite de {self.max_sessions_per_ip_per_day} sessões/dia atingido. Retorne amanhã ou use API Key."
        
        return True, ""
    
    def track_session_creation(self, client_ip: str):
        """Registra criação de sessão com alertas de segurança"""
        if client_ip not in self.session_creation_tracker:
            self.session_creation_tracker[client_ip] = []
        
        self.session_creation_tracker[client_ip].append(datetime.utcnow())
        count = len(self.session_creation_tracker[client_ip])
        
        print(f"📝 Sessões criadas por {client_ip}: {count} (última 24h)")
        
        # Alertas de segurança
        if count >= 5:
            print(f"⚠️ ALERTA: IP {client_ip} criou {count} sessões em 24h (possível abuso)")
        
        if count >= 8:
            print(f"🚨 CRÍTICO: IP {client_ip} criou {count} sessões em 24h (provável ataque)")
    
    # ========================================
    # PROTEÇÃO 2: Sessões Ativas por IP
    # ========================================
    
    def get_active_sessions_for_ip(self, client_ip: str) -> int:
        """
        Conta sessões ativas (não expiradas) deste IP
        
        Args:
            client_ip: IP do cliente
        
        Returns:
            Número de sessões ativas
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
    # PROTEÇÃO 3: Limpeza Automática
    # ========================================
    
    def _start_cleanup_task(self):
        """Inicia tarefa de limpeza de sessões expiradas"""
        def cleanup():
            while True:
                time.sleep(3600)  # A cada 1 hora
                self._cleanup_expired_sessions()
        
        thread = threading.Thread(target=cleanup, daemon=True)
        thread.start()
        print("   🧹 Limpeza automática de sessões iniciada (a cada 1h)")
    
    def _cleanup_expired_sessions(self):
        """Remove sessões expiradas e limpa tracker"""
        now = datetime.utcnow()
        expired_sessions = []
        
        # Limpar sessões expiradas
        for session_id, session in list(self.anonymous_sessions.items()):
            age = now - session["created_at"]
            
            # Sessão expirou?
            if age > timedelta(days=self.session_lifetime_days):
                expired_sessions.append(session_id)
        
        # Remover sessões expiradas
        for session_id in expired_sessions:
            del self.anonymous_sessions[session_id]
        
        # Limpar tracker de criação (> 24h)
        for ip in list(self.session_creation_tracker.keys()):
            self.session_creation_tracker[ip] = [
                ts for ts in self.session_creation_tracker[ip]
                if (now - ts).total_seconds() < 86400
            ]
            
            # Remover IPs sem registros
            if not self.session_creation_tracker[ip]:
                del self.session_creation_tracker[ip]
        
        if expired_sessions:
            print(f"🧹 Limpeza automática: {len(expired_sessions)} sessões expiradas removidas")
            print(f"   └─ Sessões ativas: {len(self.anonymous_sessions)}")
            print(f"   └─ IPs rastreados: {len(self.session_creation_tracker)}")
    
    # ========================================
    # GERAÇÃO DE TOKENS
    # ========================================
    
    def generate_anonymous_token(
        self, 
        client_ip: str, 
        fingerprint: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> dict:
        """
        Gera tokens JWT para usuário anônimo (sem API key)
        
        Args:
            client_ip: IP do cliente
            fingerprint: Browser fingerprint (opcional, para evitar abuso)
            session_id: Se fornecido, renova token da sessão existente
        
        Returns:
            dict com access_token, refresh_token e metadados
        """
        now = datetime.utcnow()
        
        # Renovação de sessão existente?
        if session_id and session_id in self.anonymous_sessions:
            session = self.anonymous_sessions[session_id]
            
            # Verificar se sessão não expirou completamente
            session_age = now - session["created_at"]
            if session_age > timedelta(days=self.session_lifetime_days):
                raise HTTPException(
                    status_code=401,
                    detail="Sessão expirada completamente. Crie nova sessão."
                )
            
            print(f"🔄 Renovando tokens para sessão: {session_id}")
        else:
            # ✅ NOVA SESSÃO: Verificar proteções anti-abuso
            can_create, error_msg = self.can_create_session(client_ip)
            if not can_create:
                raise HTTPException(
                    status_code=429,
                    detail=error_msg + " 💡 Dica: Use API Key permanente para acesso ilimitado."
                )
            
            # Criar nova sessão
            session_id = f"anon_{uuid.uuid4().hex[:12]}"
            self.anonymous_sessions[session_id] = {
                "created_at": now,
                "ip": client_ip,
                "fingerprint": fingerprint,
                "requests_count": 0,
                "quota_used": 0
            }
            
            # Registrar criação
            self.track_session_creation(client_ip)
            
            print(f"🆕 Nova sessão anônima criada: {session_id} | IP: {client_ip}")
        
        session = self.anonymous_sessions[session_id]
        
        # Gerar ACCESS TOKEN (curto - 1h)
        access_exp = now + timedelta(minutes=self.access_token_lifetime)
        access_token = jwt.encode({
            "sub": session_id,
            "type": "access",
            "ip": client_ip,
            "fingerprint": fingerprint,
            "iat": now.timestamp(),
            "exp": access_exp.timestamp()
        }, self.secret, algorithm=self.algorithm)
        
        # Gerar REFRESH TOKEN (longo - 7 dias)
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
        Renova access token usando refresh token
        
        Args:
            refresh_token: Refresh token válido no header X-Refresh-Token
        
        Returns:
            Novos access_token + refresh_token
        """
        if not refresh_token:
            raise HTTPException(
                status_code=400,
                detail="Refresh token obrigatório no header X-Refresh-Token"
            )
        
        try:
            # Decodificar refresh token
            payload = jwt.decode(refresh_token, self.secret, algorithms=[self.algorithm])
            
            # Validar tipo
            if payload.get("type") != "refresh":
                raise HTTPException(
                    status_code=401,
                    detail="Token inválido. Esperado refresh token."
                )
            
            session_id = payload["sub"]
            
            # Sessão existe?
            if session_id not in self.anonymous_sessions:
                raise HTTPException(
                    status_code=401,
                    detail="Sessão não encontrada. Crie nova sessão em POST /api/auth/anonymous"
                )
            
            # Validar IP (segurança)
            client_ip = self._get_real_ip(request)
            token_ip = payload.get("ip")
            
            if token_ip != client_ip:
                print(f"⚠️ IP mudou para sessão {session_id} | Token: {token_ip} | Atual: {client_ip}")
            
            # Obter fingerprint do payload original
            fingerprint = payload.get("fingerprint")
            
            # Gerar NOVOS tokens (access + refresh) - SEM verificar limites de criação
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
        Verifica access token anônimo com limites dinâmicos baseados em chave Gemini
        
        Args:
            authorization: Header Authorization com Bearer token
        
        Returns:
            Payload do token com dados da sessão
        """
        # Extrair token
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
        
        # Decodificar
        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            
            # Verificar se é access token
            if payload.get("type") != "access":
                raise HTTPException(
                    status_code=401,
                    detail="Token inválido. Use access token (não refresh token)."
                )
            
            session_id = payload["sub"]
            
            # Sessão existe?
            if session_id not in self.anonymous_sessions:
                raise HTTPException(
                    status_code=401,
                    detail="Sessão expirada ou inválida. Crie nova sessão."
                )
            
            session = self.anonymous_sessions[session_id]
            
            # Validar IP (anti-roubo de token)
            client_ip = self._get_real_ip(request)
            token_ip = payload.get("ip")
            
            if token_ip != client_ip:
                print(f"⚠️ IP mismatch para sessão {session_id} | Token: {token_ip} | Atual: {client_ip}")
            
            # Verificar se cliente está usando chave Gemini própria
            has_custom_gemini_key = request.headers.get("X-Gemini-Key") is not None
            
            # Aplicar limites baseado em chave própria ou não
            if has_custom_gemini_key:
                requests_limit = self.custom_key_requests_limit
                quota_limit = self.custom_key_quota_limit
                limit_type = "custom_key"
            else:
                requests_limit = self.default_requests_limit
                quota_limit = self.default_quota_limit
                limit_type = "server_key"
            
            # Verificar limite de requisições
            if session["requests_count"] >= requests_limit:
                extra_msg = ""
                if limit_type == "server_key":
                    extra_msg = f" 💡 Use sua própria chave Gemini (X-Gemini-Key) para {self.custom_key_requests_limit} req/sessão."
                
                raise HTTPException(
                    status_code=429,
                    detail=f"Limite de requisições atingido ({requests_limit}).{extra_msg}"
                )
            
            # Verificar quota (0 = ilimitado)
            if quota_limit > 0 and session["quota_used"] >= quota_limit:
                extra_msg = ""
                if limit_type == "server_key":
                    quota_msg = "ilimitada" if self.custom_key_quota_limit == 0 else f"{self.custom_key_quota_limit} créditos"
                    extra_msg = f" 💡 Use sua própria chave Gemini (X-Gemini-Key) para quota {quota_msg}."
                
                raise HTTPException(
                    status_code=429,
                    detail=f"Quota esgotada ({quota_limit} créditos).{extra_msg}"
                )
            
            # Incrementar contador de requisições
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
        Consome quota da sessão anônima
        
        Args:
            session_id: ID da sessão
            amount: Quantidade de créditos a consumir
        """
        if session_id in self.anonymous_sessions:
            self.anonymous_sessions[session_id]["quota_used"] += amount
            print(f"💰 Quota consumida: {amount} | Sessão: {session_id} | Total: {self.anonymous_sessions[session_id]['quota_used']}")
    
    def get_session_stats(self, session_id: str, has_custom_key: bool = False) -> dict:
        """
        Retorna estatísticas da sessão considerando tipo de limite
        
        Args:
            session_id: ID da sessão
            has_custom_key: Se está usando chave Gemini própria
        
        Returns:
            dict com estatísticas de uso
        """
        if session_id not in self.anonymous_sessions:
            return {"error": "Sessão não encontrada"}
        
        session = self.anonymous_sessions[session_id]
        
        # Aplicar limites corretos baseado no tipo
        if has_custom_key:
            requests_limit = self.custom_key_requests_limit
            quota_limit = self.custom_key_quota_limit
            limit_type = "custom_key"
        else:
            requests_limit = self.default_requests_limit
            quota_limit = self.default_quota_limit
            limit_type = "server_key"
        
        # Calcular remainings
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
        Obtém IP real do cliente considerando proxies
        
        Args:
            request: FastAPI Request object
        
        Returns:
            IP address string
        """
        # Tentar headers de proxy primeiro
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback para IP direto
        return request.client.host if request.client else "unknown"


# Instância global
anon_auth = AnonymousAuthManager()
