from fastapi import Header, HTTPException, Request
from typing import Optional
import os


class APIKeyAuth:
    """Sistema de autenticação por API Key"""
    
    def __init__(self):
        # Carregar chaves válidas do ambiente
        self.master_keys = os.getenv("API_KEYS", "").split(",")
        self.master_keys = [k.strip() for k in self.master_keys if k.strip()]
        
        if not self.master_keys:
            print("⚠️ WARNING: Nenhuma API_KEY configurada! Endpoint desprotegido.")
    
    def verify_api_key(self, api_key: Optional[str] = Header(None, alias="X-API-Key")) -> str:
        """Verifica se a API key é válida"""
        if not self.master_keys:
            # Modo desenvolvimento sem chave
            return "dev_mode"
        
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="API Key obrigatória. Envie no header X-API-Key"
            )
        
        if api_key not in self.master_keys:
            raise HTTPException(
                status_code=403,
                detail="API Key inválida"
            )
        
        return api_key


# Instância global
api_key_auth = APIKeyAuth()