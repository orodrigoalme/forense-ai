import filetype
from fastapi import HTTPException
import cv2
import numpy as np

# Configurações / Settings
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_MIME_TYPES = ["image/jpeg", "image/png", "image/webp", "image/tiff"]

def resize_if_too_big(image_path, max_dimension=2048):
    """
    Protege o servidor contra imagens gigantes.
    Protects the server against very large images.
    """
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(image_path, img)  # Sobrescreve com versão menor / Overwrite with smaller version
    
    return img

def validate_file_content(content: bytes, filename: str):
    """
    Valida o conteúdo do arquivo.
    Validates file content.
    """
    # Verificar Tamanho / Check file size
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Limit is {MAX_FILE_SIZE/1024/1024}MB"
        )

    # Verificar Tipo Real (Magic Numbers) / Check real type using magic numbers
    kind = filetype.guess(content)
    if kind is None or kind.mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not allowed ({filename}). Only JPEG, PNG, WEBP and TIFF."
        )
    
    return True