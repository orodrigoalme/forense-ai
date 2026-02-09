from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request
import os


def get_rate_limit_key(request: Request) -> str:
    """
    Define o rate limit baseado em:
    Defines rate limit based on:
    1. Se tem chave Gemini customizada = limite alto / Custom Gemini key = high limit
    2. Se usa chave do servidor = limite baixo / Server key = low limit
    """
    
    # Verificar se tem chave Gemini customizada / Check for custom Gemini key
    custom_gemini_key = request.headers.get("X-Gemini-Key")
    
    if custom_gemini_key:
        # Cliente com chave própria = limite generoso / Own key = generous limit
        return f"custom:{get_remote_address(request)}"
    else:
        # Usando chave do servidor = limite rigoroso / Server key = strict limit
        return f"server:{get_remote_address(request)}"


# Configurar limiter / Configure limiter
limiter = Limiter(key_func=get_rate_limit_key)


def get_rate_limit_for_endpoint(endpoint: str) -> str:
    """
    Retorna string de rate limit baseada em variáveis de ambiente.
    Returns rate limit string based on environment variables.
    
    Formato / Format: "N/period" (second, minute, hour, day)
    """
    
    if endpoint == "analyze_full":
        # Rate limit para análise completa / Full analysis rate limit
        server_key_limit = os.getenv("RATE_LIMIT_ANALYZE_SERVER_KEY", "3/minute")
        custom_key_limit = os.getenv("RATE_LIMIT_ANALYZE_CUSTOM_KEY", "20/minute")
        
        return {
            "server_key": server_key_limit,
            "custom_key": custom_key_limit
        }
    
    elif endpoint == "analyze_individual":
        # Rate limit para análises individuais / Individual analysis rate limit
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
    """
    Retorna o rate limit apropriado para a requisição.
    Returns the appropriate rate limit for the request.
    """
    limits = get_rate_limit_for_endpoint(endpoint)
    
    # Verificar se tem chave Gemini customizada / Check for custom Gemini key
    custom_gemini_key = request.headers.get("X-Gemini-Key")
    
    if custom_gemini_key:
        return limits["custom_key"]
    else:
        return limits["server_key"]