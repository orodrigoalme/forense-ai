import cv2
import numpy as np
from typing import Dict, List, Tuple
from PIL import Image
import io
import base64


def remove_accents(text: str) -> str:
    """Remove acentos para compatibilidade com cv2.putText"""
    replacements = {
        'á': 'a', 'à': 'a', 'ã': 'a', 'â': 'a',
        'é': 'e', 'ê': 'e',
        'í': 'i',
        'ó': 'o', 'ô': 'o', 'õ': 'o',
        'ú': 'u', 'ü': 'u',
        'ç': 'c',
        'Á': 'A', 'À': 'A', 'Ã': 'A', 'Â': 'A',
        'É': 'E', 'Ê': 'E',
        'Í': 'I',
        'Ó': 'O', 'Ô': 'O', 'Õ': 'O',
        'Ú': 'U', 'Ü': 'U',
        'Ç': 'C'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

class ImageAnnotator:
    """Anota imagens usando dados REAIS das análises"""
    
    def __init__(self):
        self.colors = {
            "HIGH_RISK": (0, 0, 255),
            "MEDIUM_RISK": (0, 165, 255),
            "LOW_RISK": (0, 255, 255),
            "SUCCESS": (0, 255, 0)
        }


    def annotate_full_analysis(self, 
                              image_path: str,
                              fft_result: Dict,
                              noise_result: Dict,
                              ela_result: Dict,
                              automated: Dict) -> str:
        """Cria imagem anotada usando mapas reais das análises"""
        
        # Carregar imagem original
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Não foi possível carregar a imagem")
        
        # Redimensionar para visualização
        max_size = 1200
        h, w = img.shape[:2]
        scale = 1.0
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
        else:
            new_w, new_h = w, h
        
        annotated = img.copy()
        
        # Coletar anotações reais
        annotations = []
        
        # NOISE: Usar variance_map
        if noise_result.get("status") == "success" and noise_result.get("variance_map"):
            noise_annotations = self._annotate_noise_real(
                img, noise_result, scale
            )
            annotations.extend(noise_annotations)
        
        # ELA: Usar ela_map
        if ela_result.get("status") == "success" and ela_result.get("ela_map"):
            ela_annotations = self._annotate_ela_real(
                img, ela_result, scale
            )
            annotations.extend(ela_annotations)
        
        # FFT: Marcação geral (não tem mapa espacial)
        if fft_result.get("status") == "success":
            fft_annotations = self._annotate_fft_general(img, fft_result)
            annotations.extend(fft_annotations)
        
        # Desenhar anotações
        for annotation in annotations:
            annotated = self._draw_annotation(annotated, annotation)
        
        # Header e legenda
        annotated = self._add_header(annotated, automated)
        annotated = self._add_legend(annotated, annotations)
        
        return self._to_base64(annotated)
    
    def _annotate_noise_real(self, img: np.ndarray, noise_result: Dict, scale: float) -> List[Dict]:
        """Detecta áreas reais com ruído anormal usando variance_map"""
        annotations = []
        
        # Reconstruir variance_map
        variance_map = np.array(noise_result["variance_map"])
        
        # Redimensionar para tamanho da imagem atual
        h, w = img.shape[:2]
        variance_resized = cv2.resize(variance_map, (w, h))
        
        risk_score = noise_result.get("risk_score", 0)
        
        if risk_score > 0.6:
            # IA: Encontrar regiões muito uniformes (baixa variância)
            threshold = np.percentile(variance_resized, 20)  # 20% mais uniformes
            suspicious_mask = variance_resized < threshold
            
        elif risk_score < 0.2:
            # Real: Marcar regiões com ruído natural alto
            threshold = np.percentile(variance_resized, 80)
            suspicious_mask = variance_resized > threshold
        else:
            return annotations
        
        # Encontrar contornos das regiões
        suspicious_mask_uint8 = (suspicious_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            suspicious_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filtrar contornos pequenos e pegar os 3 maiores
        min_area = (w * h) * 0.02  # Mínimo 2% da imagem
        large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        large_contours = sorted(large_contours, key=cv2.contourArea, reverse=True)[:3]
        
        for contour in large_contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Criar anotação
            if risk_score > 0.6:
                label = "Ruído Sintético"
                detail = "Uniformidade anormal"
                risk = "HIGH_RISK" if risk_score > 0.8 else "MEDIUM_RISK"
            else:
                label = "Ruído Natural"
                detail = "Padrão de sensor"
                risk = "SUCCESS"
            
            annotations.append({
                "box": (x, y, x+cw, y+ch),
                "label": label,
                "detail": detail,
                "risk": risk
            })
        
        return annotations
    
    def _annotate_ela_real(self, img: np.ndarray, ela_result: Dict, scale: float) -> List[Dict]:
        """Detecta áreas com inconsistências de compressão usando ELA map"""
        annotations = []
        
        # Reconstruir ELA map
        ela_map = np.array(ela_result["ela_map"])
        
        # Redimensionar
        h, w = img.shape[:2]
        ela_resized = cv2.resize(ela_map, (w, h))
        
        risk_score = ela_result.get("risk_score", 0)
        
        if risk_score > 0.6:
            # IA: Muito uniforme (baixo erro)
            threshold = np.percentile(ela_resized, 30)
            suspicious_mask = ela_resized < threshold
            
            label = "Compressão Uniforme"
            detail = "Típico de IA"
            risk = "MEDIUM_RISK"
            
        elif risk_score < 0.3:
            # Real com variações normais
            return annotations
        else:
            return annotations
        
        # Encontrar contornos
        suspicious_mask_uint8 = (suspicious_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            suspicious_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Pegar os 2 maiores
        min_area = (w * h) * 0.03
        large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        large_contours = sorted(large_contours, key=cv2.contourArea, reverse=True)[:2]
        
        for contour in large_contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            
            annotations.append({
                "box": (x, y, x+cw, y+ch),
                "label": label,
                "detail": detail,
                "risk": risk
            })
        
        return annotations
    
    def _annotate_fft_general(self, img: np.ndarray, fft_result: Dict) -> List[Dict]:
        """FFT não tem mapa espacial - marcação geral"""
        annotations = []
        h, w = img.shape[:2]
        
        risk_score = fft_result.get("risk_score", 0)
        
        if risk_score > 0.6:
            peak_ratio = fft_result.get("metrics", {}).get("peak_ratio", 0)
            
            # Marcação no canto superior direito (convenção visual)
            annotation = {
                "box": (int(w*0.7), 10, int(w-10), int(h*0.12)),
                "label": "Padrão de Grid",
                "detail": f"Peak: {peak_ratio:.2f}",
                "risk": "HIGH_RISK" if risk_score > 0.8 else "MEDIUM_RISK"
            }
            annotations.append(annotation)
        
        return annotations
    
    def _draw_annotation(self, img: np.ndarray, annotation: Dict) -> np.ndarray:
        """Desenha retângulo e texto"""
        box = annotation["box"]
        label = annotation["label"]
        detail = annotation.get("detail", "")
        label = remove_accents(annotation["label"])
        detail = remove_accents(annotation.get("detail", ""))
        risk = annotation["risk"]
        color = self.colors[risk]
        
        # Retângulo com borda mais grossa
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 3)
        
        # Fundo para texto
        text_lines = [label, detail] if detail else [label]
        text_h = 22 * len(text_lines) + 10
        
        # Garantir que caixa de texto não saia da imagem
        text_y = max(box[1] - text_h, 0)
        
        overlay = img.copy()
        cv2.rectangle(overlay, (box[0], text_y), (box[2], box[1]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Texto
        y_offset = text_y + 18
        for line in text_lines:
            cv2.putText(img, line, (box[0]+5, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 22
        
        return img
    
    def _add_header(self, img: np.ndarray, automated: Dict) -> np.ndarray:
        """Header com veredicto"""
        h, w = img.shape[:2]
        header_h = 80
        header = np.zeros((header_h, w, 3), dtype=np.uint8)
        
        # Gradient
        for i in range(header_h):
            intensity = int(30 + (i / header_h) * 40)
            header[i, :] = [intensity, intensity, intensity]
        
        score = automated["final_score"]
        
        # ✅ ALINHADO COM _interpret_score
        if score < 0.15:
            verdict_color = (0, 255, 0)
            verdict_text = remove_accents("MUITO PROVAVEL REAL")
        elif score < 0.35:
            verdict_color = (0, 255, 255)
            verdict_text = remove_accents("PROVAVEL REAL")
        elif score < 0.45:  # ← AJUSTADO DE 0.55 PARA 0.45
            verdict_color = (0, 165, 255)
            verdict_text = remove_accents("INCONCLUSIVO")
        elif score < 0.70:  # ← AGORA 0.45-0.70 = "PROVÁVEL IA"
            verdict_color = (255, 100, 0)  # Laranja
            verdict_text = remove_accents("PROVAVEL IA")
        else:  # >= 0.70
            verdict_color = (0, 0, 255)
            verdict_text = remove_accents("MUITO PROVAVEL IA")
        
        # Texto principal (SIMPLEX com thickness 3 = efeito negrito)
        cv2.putText(header, verdict_text, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, verdict_color, 3, cv2.LINE_AA)
        
        cv2.putText(header, f"Score: {score:.2f}/1.0", (20, 68), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Scores individuais
        scores_text = f"FFT: {automated['individual_scores']['fft']:.2f} | " \
                    f"NOISE: {automated['individual_scores']['noise']:.2f} | " \
                    f"ELA: {automated['individual_scores']['ela']:.2f}"
        
        cv2.putText(header, scores_text, (int(w*0.45), 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        
        # Combinar header com imagem
        result = np.vstack([header, img])
        
        return result

    
    def _add_legend(self, img: np.ndarray, annotations: List[Dict]) -> np.ndarray:
        """Legenda"""
        h, w = img.shape[:2]
        legend_h = 35
        legend = np.zeros((legend_h, w, 3), dtype=np.uint8)
        legend[:] = [30, 30, 30]
        
        text = f"{len(annotations)} area(s) identificada(s) | "
        text += "Vermelho: Alto Risco | Laranja: Medio | Verde: Normal"
        
        cv2.putText(legend, text, (15, 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        return np.vstack([img, legend])
    
    def _to_base64(self, img: np.ndarray) -> str:
        """Converte para base64"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        buffer.seek(0)
        
        return base64.b64encode(buffer.read()).decode('utf-8')
