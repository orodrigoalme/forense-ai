from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request
import os


def get_rate_limit_key(request: Request) -> str:
    """
    Define o rate limit baseado em:
    1. Se tem chave Gemini customizada = limite alto
    2. Se usa chave do servidor = limite baixo (configurável)
    """
    
    # Verificar se tem chave Gemini customizada no header
    custom_gemini_key = request.headers.get("X-Gemini-Key")
    
    if custom_gemini_key:
        # Cliente com chave própria = limite generoso
        return f"custom:{get_remote_address(request)}"
    else:
        # Usando chave do servidor = limite rigoroso
        return f"server:{get_remote_address(request)}"


# Configurar limiter
limiter = Limiter(key_func=get_rate_limit_key)


def get_rate_limit_for_endpoint(endpoint: str) -> str:
    """
    Retorna string de rate limit baseada em variáveis de ambiente
    
    Formato: "N/period" onde period pode ser: second, minute, hour, day
    Exemplo: "10/minute" ou "100/hour"
    """
    
    if endpoint == "analyze_full":
        # Rate limit para análise completa (usa Gemini)
        server_key_limit = os.getenv("RATE_LIMIT_ANALYZE_SERVER_KEY", "3/minute")
        custom_key_limit = os.getenv("RATE_LIMIT_ANALYZE_CUSTOM_KEY", "20/minute")
        
        return {
            "server_key": server_key_limit,
            "custom_key": custom_key_limit
        }
    
    elif endpoint == "analyze_individual":
        # Rate limit para análises individuais (FFT, NOISE, ELA)
        server_key_limit = os.getenv("RATE_LIMIT_INDIVIDUAL_SERVER_KEY", "10/minute")
        custom_key_limit = os.getenv("RATE_LIMIT_INDIVIDUAL_CUSTOM_KEY", "30/minute")
        
        return {
            "server_key": server_key_limit,
            "custom_key": custom_key_limit
        }
    
    return {
        "server_key": "10/minute",
        "custom_key": "30/minute"
    }


def get_dynamic_rate_limit(request: Request, endpoint: str) -> str:
    """Retorna o rate limit apropriado para a requisição"""
    limits = get_rate_limit_for_endpoint(endpoint)
    
    # Verificar se tem chave Gemini customizada
    custom_gemini_key = request.headers.get("X-Gemini-Key")
    
    if custom_gemini_key:
        return limits["custom_key"]
    else:
        return limits["server_key"]