from fastapi import Header, HTTPException, Request
from typing import Optional
import os


class APIKeyAuth:
    """
    Sistema de autenticação por API Key.
    API Key authentication system.
    """
    
    def __init__(self):
        # Carregar chaves válidas do ambiente / Load valid keys from environment
        self.master_keys = os.getenv("API_KEYS", "").split(",")
        self.master_keys = [k.strip() for k in self.master_keys if k.strip()]
    
    def verify_api_key(self, api_key: Optional[str] = Header(None, alias="X-API-Key")) -> str:
        """
        Verifica se a API key é válida.
        Verifies if the API key is valid.
        """
        if not self.master_keys:
            # Modo desenvolvimento sem chave / Development mode without key
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


# Instância global / Global instance
api_key_auth = APIKeyAuth()