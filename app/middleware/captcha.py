import os
import requests
from typing import Optional
from fastapi import Header, HTTPException


class CaptchaVerifier:
    """
    Verificador de reCAPTCHA com múltiplos modos de enforcement.
    reCAPTCHA verifier with multiple enforcement modes.
    """
    
    def __init__(self):
        self.secret_key = os.getenv("RECAPTCHA_SECRET_KEY")
        self.enabled = bool(self.secret_key)
        self.min_score = float(os.getenv("RECAPTCHA_MIN_SCORE", "0.5"))
        
        # Modo de enforcement / Enforcement mode
        # "required" = sempre obrigatório / always required
        # "optional" = apenas se fornecido / only if provided
        # "disabled" = ignorar sempre / always skip
        self.enforcement_mode = os.getenv("CAPTCHA_ENFORCEMENT", "optional")
    
    def verify_captcha(
        self, 
        captcha_token: Optional[str] = Header(None, alias="X-Captcha-Token")
    ) -> dict:
        """
        Verifica token do reCAPTCHA.
        Verifies reCAPTCHA token.
        """
        
        # Se desabilitado / If disabled
        if not self.enabled or self.enforcement_mode == "disabled":
            return {
                "success": True,
                "score": 1.0,
                "action": "disabled",
                "message": "CAPTCHA disabled"
            }
        
        # Se opcional e não foi enviado / If optional and not provided
        if self.enforcement_mode == "optional" and not captcha_token:
            return {
                "success": True,
                "score": 1.0,
                "action": "skipped",
                "message": "CAPTCHA not provided (optional mode)"
            }
        
        # Se obrigatório e não foi enviado / If required and not provided
        if self.enforcement_mode == "required" and not captcha_token:
            raise HTTPException(
                status_code=400,
                detail="CAPTCHA token obrigatório. Envie no header X-Captcha-Token"
            )
        
        # Se foi fornecido, validar / If provided, validate
        if captcha_token:
            return self._validate_token(captcha_token)
        
        return {"success": True, "score": 1.0, "action": "skipped"}
    
    def _validate_token(self, token: str) -> dict:
        """
        Valida token com Google.
        Validates token with Google.
        """
        try:
            response = requests.post(
                "https://www.google.com/recaptcha/api/siteverify",
                data={
                    "secret": self.secret_key,
                    "response": token
                },
                timeout=5
            )
            
            result = response.json()
            
            if not result.get("success"):
                error_codes = result.get("error-codes", [])
                raise HTTPException(
                    status_code=400,
                    detail=f"CAPTCHA inválido: {', '.join(error_codes)}"
                )
            
            score = result.get("score", 0.0)
            action = result.get("action", "unknown")
            
            if score < self.min_score:
                raise HTTPException(
                    status_code=403,
                    detail=f"CAPTCHA score muito baixo ({score:.2f}). Possível bot."
                )
            
            
            return {
                "success": True,
                "score": score,
                "action": action,
                "message": "Human verified"
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Erro ao verificar CAPTCHA: {str(e)}"
            )
