import cv2
import numpy as np
from PIL import Image, ExifTags
import base64
import io
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class NoiseAnalysisResult:
    """
    Estrutura de dados para resultado da análise de ruído.
    Data structure for noise analysis result.
    """
    method: str
    status: str
    image_base64: str
    metrics: Dict
    risk_score: float
    warnings: List[str]


class NoiseMapAnalyzer:
    """
    Analisador de Mapa de Ruído para detecção de imagens geradas por IA.
    Noise Map Analyzer for AI-generated image detection.
    
    Principais características de imagens IA / Main AI image characteristics:
    - Ruído anormalmente baixo ou consistente / Abnormally low or consistent noise
    - Regiões "lisas demais" / Overly smooth regions (skin, sky, backgrounds)
    - Ausência de padrão de ruído natural / Absence of natural sensor noise pattern
    """
    
    def __init__(self, 
                 block_size: int = 8,
                 low_noise_threshold: float = 5.0,
                 high_noise_threshold: float = 30.0):
        """
        Args:
            block_size: Tamanho do bloco para análise local / Block size for local analysis
            low_noise_threshold: Limiar inferior de ruído / Low noise threshold
            high_noise_threshold: Limiar superior de ruído / High noise threshold
        """
        self.block_size = block_size
        self.low_noise_threshold = low_noise_threshold
        self.high_noise_threshold = high_noise_threshold
        
    def _convert_to_base64(self, image: np.ndarray, format: str = 'PNG') -> str:
        """
        Converte imagem numpy array para base64.
        Converts numpy array image to base64.
        """
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        pil_image = Image.fromarray(image_rgb)
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode('utf-8')
        return img_str
    
    def _extract_iso_from_metadata(self, image_path: str) -> Optional[int]:
        """
        Extrai valor ISO dos metadados EXIF da imagem.
        Extracts ISO value from image EXIF metadata.
        """
        try:
            img = Image.open(image_path)
            exif_data = img._getexif()
            
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    if tag in ['ISOSpeedRatings', 'PhotographicSensitivity', 'ISO']:
                        # Pode ser int ou tuple
                        if isinstance(value, tuple):
                            return int(value[0])
                        elif isinstance(value, int):
                            return value
            return None
        except Exception:
            return None
    
    def _estimate_iso_from_image_characteristics(self, image: np.ndarray) -> Tuple[int, str, str]:
        """
        Estima ISO equivalente baseado nas características da imagem.
        Estimates equivalent ISO based on image characteristics.
        
        Returns: (estimated_iso, confidence, method_used)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 1. Analisar ruído atual
        noise_metrics = self._analyze_current_noise(gray)
        
        # 2. Analisar sharpness
        sharpness = self._estimate_sharpness(gray)
        
        # 3. Estimar tipo de dispositivo
        device_type = self._estimate_device_type(image, noise_metrics)
        
        # 4. Calcular estimativa
        estimated_iso, confidence, method = self._calculate_iso_estimate(
            noise_metrics, sharpness, device_type
        )
        
        return estimated_iso, confidence, method
    
    def _analyze_current_noise(self, gray: np.ndarray) -> Dict:
        """
        Analisa ruído presente na imagem usando múltiplas técnicas.
        Analyzes noise in the image using multiple techniques.
        """
        h, w = gray.shape
        
        # Método 1: Variância do Laplaciano
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = np.var(laplacian)
        
        # Método 2: Ruído em regiões planas
        blurred = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
        smooth_mask = np.abs(gray.astype(np.float32) - blurred) < 5
        
        if np.sum(smooth_mask) > 100:
            smooth_regions = gray[smooth_mask]
            noise_in_smooth = np.var(smooth_regions)
        else:
            noise_in_smooth = lap_var * 0.1
        
        # Método 3: Alta frequência (wavelet-like)
        gauss_small = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0.5)
        gauss_large = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 2.0)
        high_freq = gauss_small - gauss_large
        wavelet_noise = np.std(high_freq)
        
        # Combinar métricas
        composite = (lap_var * 0.3 + noise_in_smooth * 0.5 + wavelet_noise * 0.2)
        
        return {
            'laplacian_variance': float(lap_var),
            'smooth_region_noise': float(noise_in_smooth),
            'high_frequency_std': float(wavelet_noise),
            'composite_noise_score': float(composite)
        }
    
    def _estimate_sharpness(self, gray: np.ndarray) -> float:
        """
        Estima nível de nitidez da imagem.
        Estimates image sharpness level.
        """
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        sharpness_score = np.percentile(sobel_magnitude, 95)
        return float(np.clip(sharpness_score / 255.0, 0, 1))
    
    def _estimate_device_type(self, image: np.ndarray, noise_metrics: Dict) -> str:
        """
        Estima tipo de dispositivo / Estimates device type:
        professional_camera, smartphone, likely_ai_generated, digital_art_processed, unknown
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        has_sensor_pattern = self._detect_sensor_pattern(gray)
        has_artificial_sharpening = self._detect_sharpening_halos(image)
        histogram_score = self._analyze_histogram_naturalness(gray)
        
        # Lógica de classificação
        if not has_sensor_pattern and not has_artificial_sharpening and histogram_score > 0.8:
            if noise_metrics['composite_noise_score'] < 10:
                return "likely_ai_generated"
            else:
                return "digital_art_processed"
        elif has_artificial_sharpening and noise_metrics['composite_noise_score'] < 50:
            return "smartphone"
        elif has_sensor_pattern:
            return "professional_camera"
        else:
            return "unknown"
    
    def _detect_sharpening_halos(self, image: np.ndarray) -> bool:
        """
        Detecta halos de sharpening artificial.
        Detects artificial sharpening halos.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edge_dilated = cv2.dilate(edges, kernel, iterations=1)
        edge_mask = edge_dilated > 0
        
        if np.sum(edge_mask) < 100:
            return False
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=1)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=1)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        
        edge_gradient_mean = np.mean(gradient[edge_mask])
        return edge_gradient_mean > 80
    
    def _analyze_histogram_naturalness(self, gray: np.ndarray) -> float:
        """
        Analisa se histograma parece natural (1) ou processado/IA (0).
        Analyzes if histogram looks natural (1) or processed/AI (0).
        """
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / np.sum(hist)
        
        hist_nonzero = hist[hist > 0]
        entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero + 1e-10))
        normalized_entropy = entropy / 8.0
        
        zero_bins = np.sum(hist == 0)
        gap_ratio = zero_bins / 256.0
        
        if normalized_entropy > 0.9 and gap_ratio < 0.1:
            return 1.0
        elif normalized_entropy < 0.5 or gap_ratio > 0.3:
            return 0.0
        else:
            return normalized_entropy * (1 - gap_ratio)
    
    def _calculate_iso_estimate(self, 
                                  noise_metrics: Dict, 
                                  sharpness: float,
                                  device_type: str) -> Tuple[int, str, str]:
        """
        Calcula estimativa final de ISO.
        Calculates final ISO estimate.
        """
        noise_score = noise_metrics['composite_noise_score']
        
        if device_type == "likely_ai_generated":
            base_iso = 400
            confidence = "low"
            method = "ai_pattern_detected"
            
        elif device_type == "smartphone":
            if noise_score < 20:
                estimated_iso = 100
            elif noise_score < 50:
                estimated_iso = 400
            elif noise_score < 100:
                estimated_iso = 800
            else:
                estimated_iso = 1600
            confidence = "medium"
            method = "smartphone_heuristics"
            
        elif device_type == "professional_camera":
            if noise_score < 10:
                estimated_iso = 100
            elif noise_score < 30:
                estimated_iso = 400
            elif noise_score < 80:
                estimated_iso = 800
            elif noise_score < 150:
                estimated_iso = 1600
            else:
                estimated_iso = 3200
            confidence = "medium-high"
            method = "pro_camera_model"
            
        else:
            if noise_score < 15:
                estimated_iso = 200
            elif noise_score < 40:
                estimated_iso = 400
            elif noise_score < 90:
                estimated_iso = 800
            else:
                estimated_iso = 1600
            confidence = "low"
            method = "generic_heuristics"
        
        # Ajuste por sharpness
        if sharpness > 0.8 and noise_score > 50:
            estimated_iso = max(400, estimated_iso)
            confidence = "low"
            method += "_sharpness_adjusted"
        
        estimated_iso = max(100, min(12800, estimated_iso))
        
        return estimated_iso, confidence, method
    
    def _get_expected_noise(self, 
                           image_path: str, 
                           image: np.ndarray) -> Tuple[float, Optional[int], str, str]:
        """
        Obtém ruído esperado de forma robusta, com ou sem EXIF.
        Gets expected noise robustly, with or without EXIF.
        
        Returns: (expected_noise, iso_value, source, confidence)
        """
        # Tentar EXIF primeiro
        iso_from_exif = self._extract_iso_from_metadata(image_path)
        
        if iso_from_exif:
            expected_noise = self._estimate_expected_noise_from_iso(iso_from_exif)
            return expected_noise, iso_from_exif, "exif", "high"
        
        # Fallback: estimar da imagem
        estimated_iso, confidence, method = self._estimate_iso_from_image_characteristics(image)
        expected_noise = self._estimate_expected_noise_from_iso(estimated_iso)
        
        # Ajustar por confiança baixa
        if confidence == "low":
            expected_noise = expected_noise * 1.5  # Margem maior
        
        return expected_noise, None, method, confidence
    
    def _estimate_expected_noise_from_iso(self, iso_value: int) -> float:
        """
        Estima nível de ruído esperado baseado no ISO.
        Estimates expected noise level based on ISO.
        """
        import math
        noise = 0.01 + 0.025 * math.log2(max(iso_value, 100) / 100)
        return min(noise, 0.30)
    
    def _calculate_local_variance(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Calcula variância local em blocos.
        Calculates local variance in blocks.
        """
        h, w = gray_image.shape
        block_h, block_w = self.block_size, self.block_size
        
        # OTIMIZAÇÃO: Garantir divisibilidade em vez de padding complexo
        h_crop = (h // block_h) * block_h
        w_crop = (w // block_w) * block_w
        
        # Cortar se necessário (pequena perda de borda, ganho enorme de performance)
        cropped = gray_image[:h_crop, :w_crop]
        
        # Reshape vetorizado: (H, W) -> (H/block, block, W/block, block)
        blocks = cropped.reshape(h_crop // block_h, block_h,
                                w_crop // block_w, block_w)
        
        # Variância por bloco - vetorizada
        variances = blocks.var(axis=(1, 3))
        
        # Interpolação bilinear para tamanho original (OpenCV otimizado em C++)
        variance_map_resized = cv2.resize(
            variances.astype(np.float32), 
            (w, h), 
            interpolation=cv2.INTER_LINEAR
        )
        
        return variance_map_resized

    def _detect_sensor_pattern(self, gray: np.ndarray) -> bool:
        """
        Detecta padrão de sensor Bayer ou banding.
        Detects Bayer sensor pattern or banding.
        """
        h, w = gray.shape
        
        if h < 100 or w < 100:
            return False
    
        # OTIMIZAÇÃO: Amostragem estratificada em vez de análise completa para velocidade
        step = max(1, min(h, w) // 500)  # Amostrar ~500x500 pontos máximo
    
        # Padrão 2x2 usando slicing com step
        rows_0 = gray[0::2*step, ::step]
        rows_1 = gray[1::2*step, ::step]
    
        if rows_0.size == 0 or rows_1.size == 0:
            return False
    
        # Diferença média entre linhas alternadas (padrão Bayer)
        diff_h = np.abs(rows_0[:min(len(rows_0), len(rows_1))] - 
                        rows_1[:min(len(rows_0), len(rows_1))])
    
        # Análise de banding em amostra
        row_means_sample = np.mean(gray[::step, ::step], axis=1)
        row_variance = np.var(row_means_sample)
    
        pattern_strength = np.mean(diff_h)
        has_banding = 0.5 < row_variance < 20
    
        return pattern_strength > 1.0 or has_banding
    
    def _identify_low_noise_regions(self, 
                                    variance_map: np.ndarray, 
                                    original_image: np.ndarray) -> List[str]:
        """
        Identifica regiões com ruído anormalmente baixo.
        Identifies regions with abnormally low noise.
        """
        regions = []
        low_noise_mask = variance_map < self.low_noise_threshold
    
        if not np.any(low_noise_mask):
            return regions
    
        hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
        h, w = original_image.shape[:2]
        top_third = h // 3  # Apenas o terço superior para céu
    
        # ==== SKY - APENAS NO TOPO ====
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
        # Criar máscara apenas para região superior
        blue_mask_top = np.zeros_like(blue_mask)
        blue_mask_top[:top_third] = blue_mask[:top_third]
    
        sky_low_noise = np.sum(low_noise_mask & (blue_mask_top > 0))
        min_pixels_sky = top_third * w * 0.10  # 10% da área do topo
    
        if sky_low_noise > min_pixels_sky:
            regions.append("sky")
    
        # ==== SKIN ====
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 255, 255])
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
        min_pixels = 1000
        if np.sum(low_noise_mask & (skin_mask > 0)) > min_pixels:
            regions.append("skin")
    
        # ==== VEGETATION ====
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
        if np.sum(low_noise_mask & (green_mask > 0)) > min_pixels:
            regions.append("vegetation")
    
        # ==== GENERIC SMOOTH ====
        if not regions and np.sum(low_noise_mask) > (h * w * 0.05):
            regions.append("smooth_regions")
    
        return regions
    
    def _calculate_noise_consistency(self, variance_map: np.ndarray) -> float:
        """
        Calcula consistência do ruído, calibrado para IA moderna.
        Calculates noise consistency, calibrated for modern AI.
        """
    
        variance_normalized = variance_map / (np.max(variance_map) + 1e-10)
    
        mean_var = np.mean(variance_normalized)
        std_var = np.std(variance_normalized)
    
        if mean_var < 1e-10:
            return 1.0
    
        cv = std_var / mean_var
    
        # Análise Espacial
        h, w = variance_map.shape
        region_h, region_w = h // 4, w // 4
    
        region_means = []
        for i in range(4):
            for j in range(4):
                region = variance_map[i*region_h:(i+1)*region_h, j*region_w:(j+1)*region_w]
                if region.size > 0:
                    region_means.append(np.mean(region))
    
        region_variance = np.var(region_means) / (np.mean(region_means)**2 + 1e-10)
    
    
        # Lógica recalibrada / Recalibrated logic
        # IA moderna: cv entre 2.0-3.5 / Modern AI: cv between 2.0-3.5
        # Real: cv > 3.5 / Real: cv > 3.5
    
        if cv < 1.0:  # Muito consistente (IA antiga)
            consistency = 0.95
        elif cv < 2.0:  # Consistente moderado
            consistency = 0.8
        elif cv < 3.0:  # IA moderna (faixa 2.5-3.0)
            consistency = 0.65
        elif cv < 3.5:  # Limiar crítico
            consistency = 0.4
        else:  # > 3.5 = Câmera real
            consistency = 0.1
    
        # Ajuste por region_variance (Se CV está na zona cinzenta 2.5-3.5)
        if 2.5 < cv < 3.5:
            if region_variance < 0.3:  # Uniforme espacialmente = mais suspeito
                consistency = min(1.0, consistency * 1.3)
    
        return float(consistency)
    
    def _create_noise_heatmap(self, 
                              variance_map: np.ndarray, 
                              original_image: np.ndarray,
                              iso_source: str,
                              iso_confidence: str) -> np.ndarray:
        """
        Cria heatmap colorizado com informações de estimativa.
        Creates colorized heatmap with estimation info.
        """
        variance_norm = cv2.normalize(
            variance_map, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        
        heatmap = cv2.applyColorMap(variance_norm, cv2.COLORMAP_JET)
        
        original_resized = cv2.resize(original_image, (heatmap.shape[1], heatmap.shape[0]))
        overlay = cv2.addWeighted(original_resized, 0.3, heatmap, 0.7, 0)
        
        h, w = overlay.shape[:2]
        
        # Criar barra de legenda com info de estimativa
        legend_height = 50
        legend = np.zeros((legend_height, w, 3), dtype=np.uint8)
        
        # Gradiente
        for i in range(w):
            color_val = int(255 * (i / w))
            color = cv2.applyColorMap(
                np.array([[color_val]], dtype=np.uint8), 
                cv2.COLORMAP_JET
            )[0][0]
            legend[:, i] = color
        
        # Textos
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(legend, "Low Noise (Suspicious)", (10, 20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(legend, "High Noise (Natural)", (w-200, 20), font, 0.5, (255, 255, 255), 1)
        
        # Info de estimativa ISO
        source_text = f"ISO Source: {iso_source.upper()} ({iso_confidence})"
        cv2.putText(legend, source_text, (10, 40), font, 0.4, (255, 255, 0), 1)
        
        result = np.vstack([overlay, legend])
        
        return result
    
    def _calculate_risk_score(self,
                          mean_noise: float, 
                          noise_consistency: float,
                          low_noise_regions: List[str],
                          expected_noise: float,
                            iso_confidence: str) -> float:
        """
        Calcula score de risco composto.
        Calculates composite risk score.
        """
        scores = []
        
        # Fator 1: Diferença entre esperado e observado
        # Factor 1: Difference between expected and observed
        noise_ratio = mean_noise / expected_noise if expected_noise > 0 else 1.0
        
        if noise_ratio < 0.5:
            # Ruído muito baixo
            if noise_consistency < 0.2:
                scores.append(0.05)  # Câmera boa
            else:
                scores.append(0.7)   # Suspeito
        elif noise_ratio < 0.7:
            scores.append(0.4)
        elif noise_ratio > 2.5:  # MUITO mais ruído (IA fake noise)
            scores.append(0.85)  # AUMENTADO de 0.7
        elif noise_ratio > 1.5:
            scores.append(0.6)
        else:
            scores.append(0.1)
        
        # Fator 2: Consistência / Factor 2: Consistency
        if noise_consistency > 0.95:
            scores.append(0.98)
        elif noise_consistency > 0.85:
            scores.append(0.9)
        elif noise_consistency > 0.6:
            scores.append(0.85)
        elif noise_consistency > 0.4:
            scores.append(0.6)
        elif noise_consistency > 0.2:
            scores.append(0.3)
        else:
            scores.append(0.05)
        
        # Fator 3: Regiões / Factor 3: Regions
        if "artificial_uniformity" in low_noise_regions:
            scores.append(0.85)
        elif len(low_noise_regions) >= 2:
            scores.append(0.5)
        elif len(low_noise_regions) >= 1:
            scores.append(0.25)
        else:
            scores.append(0.0)
        
        # Fator 4: Padrão de Ruído Sintético / Factor 4: Synthetic Noise Pattern
        # Detecta IA adicionando ruído fake / Detects AI adding fake noise
        synthetic_noise_score = 0.0
        if noise_ratio > 2.0 and 0.5 < noise_consistency < 0.8:
            # IA moderna: adiciona MUITO ruído mas com padrão
            synthetic_noise_score = 0.95
        elif noise_ratio > 1.5 and noise_consistency > 0.7:
            synthetic_noise_score = 0.8
        
        scores.append(synthetic_noise_score)
        
        # Pesos / Weights: [Factor1, Factor2, Factor3, Factor4]
        weights = [0.10, 0.45, 0.10, 0.35]  # Fator 4 pesa 35%
        final_score = sum(s * w for s, w in zip(scores, weights))
        
        # Ajuste por confiança / Confidence adjustment
        if iso_confidence == "low":
            final_score = final_score * 0.95
        
        return min(1.0, max(0.0, final_score))



    def _get_interpretation(self, risk_score: float) -> str:
        """
        Interpreta o risk score.
        Interprets the risk score.
        """
        if risk_score < 0.15:
            return "Muito provavelmente real"
        elif risk_score < 0.35:
            return "Provavelmente real"
        elif risk_score < 0.55:
            return "Inconclusivo - requer análise adicional"
        elif risk_score < 0.75:
            return "Provavelmente gerada por IA"
        else:
            return "Muito provavelmente gerada por IA"

    def _generate_warnings(self,
                      mean_noise: float,
                      noise_consistency: float,
                      low_noise_regions: List[str],
                      expected_noise: float,
                      iso_value: Optional[int],
                      iso_source: str,
                      iso_confidence: str,
                      risk_score: float) -> List[str]:
        """
        Gera lista de avisos contextualizada.
        Generates context-filtered warning list.
        """
        warnings = []
        
        # Aviso sobre metadados (sempre relevante)
        # Metadata warning (always relevant)
        if iso_source != "exif":
            if iso_confidence == "low":
                warnings.append(
                    f"ISO estimado por análise de padrão ({iso_source}) - "
                    f"confiança baixa, interpretar com cautela"
                )
            else:
                warnings.append(
                    f"ISO estimado por análise de imagem ({iso_source}) - "
                    f"sem metadados EXIF originais"
                )
        
        # Filtros contextuais - só gerar warnings se risco > 0.35
        # Contextual filters - only generate warnings if risk > 0.35
        
        # Aviso sobre ruído baixo / Low noise warning
        if mean_noise < expected_noise * 0.5 and risk_score > 0.35:
            warnings.append(
                f"Ruído anormalmente baixo ({mean_noise:.3f} vs esperado {expected_noise:.3f})"
            )
        
        # Aviso sobre consistência / Consistency warning
        if noise_consistency > 0.95:
            warnings.append("Ruído perfeitamente consistente - padrão típico de IA")
        elif noise_consistency > 0.85:
            warnings.append("Alta consistência no padrão de ruído")
        elif noise_consistency > 0.6 and risk_score > 0.5:
            warnings.append("Padrão de ruído com consistência moderada-alta")
        
        # Avisos de regiões, filtrados por score / Region warnings, filtered by score
        if risk_score > 0.35:  # Só avisar se suspeito
            if "skin" in low_noise_regions:
                warnings.append("Textura de pele anormalmente lisa")
            
            if "sky" in low_noise_regions:
                warnings.append("Céu com padrão artificialmente uniforme")
            
            if "artificial_uniformity" in low_noise_regions:
                warnings.append("Regiões com uniformidade artificial detectadas")
        
        # Aviso específico de ISO alto
        if iso_value and iso_value > 800 and mean_noise < 0.05:
            warnings.append(f"ISO alto ({iso_value}) mas ruído inexistente - altamente suspeito")
        
        # Aviso de padrão IA detectado na estimativa
        if iso_source == "ai_pattern_detected":
            warnings.append("Padrão visual sugere imagem gerada por IA (uniformidade excessiva)")
        
        # Aviso de Ruído Sintético / Synthetic Noise Warning
        noise_ratio = mean_noise / expected_noise if expected_noise > 0 else 1.0
        if noise_ratio > 2.0 and 0.5 < noise_consistency < 0.8:
            warnings.append("Ruído sintético detectado - típico de IA adicionando ruído fake")
        
        return warnings

    
    def analyze(self, image_path: str) -> Dict:
        """
        Executa análise completa de Mapa de Ruído.
        Runs complete Noise Map analysis.
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {
                    "method": "NOISE",
                    "status": "error",
                    "image_base64": "",
                    "metrics": {},
                    "risk_score": 0.0,
                    "warnings": ["Falha ao carregar imagem"]
                }
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Obter ruído esperado / Get expected noise
            expected_noise, iso_value, iso_source, iso_confidence = self._get_expected_noise(
                image_path, image
            )
            
            # Calcular mapa de variância / Compute variance map
            variance_map = self._calculate_local_variance(gray)
            
            # Métricas / Metrics
            mean_noise = float(np.mean(variance_map))
            max_noise = float(np.max(variance_map))
            mean_noise_normalized = mean_noise / 255.0
            
            noise_consistency = self._calculate_noise_consistency(variance_map)
            low_noise_regions = self._identify_low_noise_regions(variance_map, image)
            variance_map_normalized = variance_map / (np.max(variance_map) + 1e-10)
            
            # Criar heatmap / Create heatmap
            heatmap_image = self._create_noise_heatmap(
                variance_map, image, iso_source, iso_confidence
            )
            heatmap_base64 = self._convert_to_base64(heatmap_image)
            
            # Calcular risco / Calculate risk
            risk_score = self._calculate_risk_score(
                mean_noise_normalized, 
                noise_consistency, 
                low_noise_regions,
                expected_noise,
                iso_confidence
            )
            
            # Gerar avisos / Generate warnings
            warnings_list = self._generate_warnings(
                mean_noise_normalized,
                noise_consistency,
                low_noise_regions,
                expected_noise,
                iso_value,
                iso_source,
                iso_confidence,
                risk_score
            )
            
            # Montar métricas / Build metrics
            metrics = {
                "mean_noise_level": round(mean_noise_normalized, 4),
                "noise_consistency": round(noise_consistency, 4),
                "regions_with_low_noise": low_noise_regions,
                "expected_noise_for_iso": round(expected_noise, 4),
                "max_noise_value": round(max_noise / 255.0, 4),
                "iso_metadata": iso_value,
                "iso_metadata_source": iso_source,
                "iso_confidence": iso_confidence
            }
            
            return {
                "method": "NOISE",
                "status": "success",
                "image_base64": heatmap_base64,
                "metrics": metrics,
                "risk_score": round(risk_score, 2),
                "warnings": warnings_list,
                "interpretation": self._get_interpretation(risk_score),
                "variance_map": variance_map_normalized.tolist() 
            }
            
        except Exception as e:
            return {
                "method": "NOISE",
                "status": "error",
                "image_base64": "",
                "metrics": {},
                "risk_score": 0.0,
                "warnings": [f"Erro na análise: {str(e)}"]
            }


# Função de conveniência / Convenience function
def analyze_noise(image_path: str) -> Dict:
    """
    Função standalone para análise de mapa de ruído.
    Standalone function for noise map analysis.
    """
    analyzer = NoiseMapAnalyzer()
    return analyzer.analyze(image_path)