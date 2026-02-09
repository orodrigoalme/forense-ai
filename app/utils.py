import filetype
from fastapi import HTTPException
import cv2
import numpy as np

# Configurações
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_MIME_TYPES = ["image/jpeg", "image/png", "image/webp", "image/tiff"]

def resize_if_too_big(image_path, max_dimension=2048):
    """Protege o servidor contra imagens gigantes"""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(image_path, img) # Sobrescreve com versão menor
        print(f"⚠️ Imagem redimensionada de {w}x{h} para {new_w}x{new_h}")
    
    return img


def validate_file_content(content: bytes, filename: str):
    # 1. Verificar Tamanho
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"Arquivo muito grande. Limite é {MAX_FILE_SIZE/1024/1024}MB"
        )

    # 2. Verificar Tipo Real (Magic Numbers)
    kind = filetype.guess(content)
    if kind is None or kind.mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400, 
            detail=f"Tipo de arquivo não permitido ({filename}). Apenas JPEG, PNG, WEBP e TIFF."
        )
    
    return True