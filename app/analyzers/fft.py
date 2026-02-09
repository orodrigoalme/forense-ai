import cv2
import numpy as np
from PIL import Image
import base64
import io
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class FFTAnalysisResult:
    """Estrutura de dados para resultado da análise FFT"""
    method: str
    status: str
    image_base64: str
    metrics: Dict
    risk_score: float
    warnings: List[str]


class FFTAnalyzer:
    """
    Analisador de Espectro de Fourier (FFT) para detecção de imagens geradas por IA.
    
    Principais características de imagens IA detectáveis via FFT:
    - Simetria excessiva no espectro (IA gera padrões simétricos perfeitos)
    - Picos anômalos periódicos (grid artifacts, checkerboard patterns)
    - Uniformidade espectral não natural (falta de complexidade orgânica)
    - Padrões de grade (grid) em alta frequência (upscaling artifacts)
    """
    
    def __init__(self,
                 magnitude_threshold: float = 0.1,
                 symmetry_threshold: float = 0.9,
                 peak_detection_threshold: float = 0.3):
        """
        Args:
            magnitude_threshold: Limiar para detectar picos significativos
            symmetry_threshold: Limiar para considerar simetria excessiva
            peak_detection_threshold: Limiar para detecção de picos locais
        """
        self.magnitude_threshold = magnitude_threshold
        self.symmetry_threshold = symmetry_threshold
        self.peak_detection_threshold = peak_detection_threshold
        
    def _convert_to_base64(self, image: np.ndarray, format: str = 'PNG') -> str:
        """Converte imagem numpy array para base64"""
        # Garantir que está em formato válido para PIL
        if len(image.shape) == 2:
            # Grayscale
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # BGR para RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        # Normalizar se necessário
        if image_rgb.dtype != np.uint8:
            image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
            
        pil_image = Image.fromarray(image_rgb)
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode('utf-8')
        return img_str
    
    def _compute_fft_spectrum(self, gray_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula a FFT 2D e retorna espectro de magnitude e fase.
        Aplica shift para centralizar baixas frequências.
        """
        # Converter para float e normalizar
        f = np.fft.fft2(gray_image.astype(np.float32))
        fshift = np.fft.fftshift(f)
        
        # Magnitude (logarítmica para visualização)
        magnitude = np.abs(fshift)
        magnitude_log = np.log(magnitude + 1e-10)
        
        # Fase (menos relevante para detecção de IA, mas incluída)
        phase = np.angle(fshift)
        
        return magnitude, magnitude_log, phase
    
    def _create_spectrum_visualization(self, 
                                        magnitude_log: np.ndarray,
                                        magnitude: np.ndarray) -> np.ndarray:
        """
        Cria visualização colorida do espectro de Fourier.
        Inclui: espectro bruto, espectro logarítmico, e análise de quadrantes.
        """
        h, w = magnitude_log.shape
        
        # Normalizar para 0-255
        mag_norm = cv2.normalize(magnitude_log, None, 0, 255, cv2.NORM_MINMAX)
        mag_vis = mag_norm.astype(np.uint8)
        
        # Aplicar colormap (INFERNO: bom para destacar anomalias)
        spectrum_colored = cv2.applyColorMap(mag_vis, cv2.COLORMAP_INFERNO)
        
        # Criar máscara de análise (círculos de frequência)
        center_y, center_x = h // 2, w // 2
        max_radius = min(center_y, center_x)
        
        # Visualização em 4 partes
        vis_height = h
        vis_width = w * 2
        
        # Esquerda: espectro original
        # Direita: espectro com anotações
        composite = np.zeros((vis_height, vis_width, 3), dtype=np.uint8)
        
        # Colocar espectro original (esquerda)
        composite[:, :w] = spectrum_colored
        
        # Criar versão anotada (direita)
        annotated = spectrum_colored.copy()
        
        # Desenhar círculos de frequência
        for radius in [max_radius // 4, max_radius // 2, 3 * max_radius // 4]:
            cv2.circle(annotated, (center_x, center_y), radius, (0, 255, 0), 1)
        
        # Desenhar eixos de simetria
        cv2.line(annotated, (center_x, 0), (center_x, h), (255, 0, 0), 1)
        cv2.line(annotated, (0, center_y), (w, center_y), (255, 0, 0), 1)
        
        # Destacar picos anômalos
        peaks_mask = self._detect_peaks(magnitude)
        peaks_coords = np.argwhere(peaks_mask)
        
        for y, x in peaks_coords[:20]:  # Limitar a 20 picos para não poluir
            cv2.circle(annotated, (x, y), 3, (0, 0, 255), -1)
        
        composite[:, w:] = annotated
        
        # Adicionar legendas
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(composite, "FFT Spectrum", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(composite, "Analyzed (Green: Freq bands, Red: Peaks)", (w + 10, 30), font, 0.7, (255, 255, 255), 2)
        
        return composite
    
    def _detect_peaks(self, magnitude: np.ndarray) -> np.ndarray:
        """
        Detecta picos locais anômalos no espectro.
        Picos periódicos indicam padrões artificiais (grid artifacts).
        """
        # Normalizar
        mag_norm = magnitude / (np.max(magnitude) + 1e-10)
        
        # Aplicar threshold
        threshold_mask = mag_norm > self.peak_detection_threshold
        
        # Encontrar máximos locais
        from scipy import ndimage
        # Nota: se scipy não disponível, implementação alternativa abaixo
        
        try:
            # Usar scipy para máximos locais
            local_max = ndimage.maximum_filter(mag_norm, size=5) == mag_norm
            peaks = threshold_mask & local_max
            
            # Excluir centro (DC component - sempre máximo)
            h, w = magnitude.shape
            center_y, center_x = h // 2, w // 2
            exclude_radius = min(h, w) // 20
            Y, X = np.ogrid[:h, :w]
            center_mask = (X - center_x)**2 + (Y - center_y)**2 > exclude_radius**2
            peaks = peaks & center_mask
            
        except ImportError:
            # Implementação alternativa sem scipy
            peaks = self._detect_peaks_simple(mag_norm, threshold_mask)
        
        return peaks
    
    def _detect_peaks_simple(self, 
                              magnitude_norm: np.ndarray, 
                              threshold_mask: np.ndarray) -> np.ndarray:
        """
        Implementação simples de detecção de picos - OTIMIZADA.
        Substitui loop duplo por operações de convolução.
        """
        h, w = magnitude_norm.shape
    
        # OTIMIZAÇÃO: Usar maximum_filter do scipy.ndimage ou implementar com OpenCV
        # Criar kernel 3x3 para máximo local
        kernel = np.ones((3, 3), dtype=np.uint8)
    
        # Dilatação = máximo local na vizinhança 3x3 (OpenCV otimizado)
        local_max = cv2.dilate(magnitude_norm.astype(np.float32), kernel)
    
        # Picos são pixels iguais ao máximo local E acima do threshold
        peaks = (magnitude_norm == local_max) & threshold_mask
    
        # Excluir centro (DC component) - vetorizado
        center_y, center_x = h // 2, w // 2
        exclude_radius = min(h, w) // 20
    
        Y, X = np.ogrid[:h, :w]
        center_mask = (X - center_x)**2 + (Y - center_y)**2 <= exclude_radius**2
        peaks[center_mask] = False
    
        return peaks

    def _calculate_symmetry_score(self, magnitude: np.ndarray) -> float:
        """
        Calcula score de simetria - OTIMIZADO.
        """
        h, w = magnitude.shape
    
        # Normalizar
        mag = magnitude / (np.max(magnitude) + 1e-10)
    
        # OTIMIZAÇÃO: Operações vetorizadas em vez de slicing manual
        # Simetria horizontal
        w_half = w // 2
        left = mag[:, :w_half]
        right = np.fliplr(mag[:, w_half + (w % 2):])
    
        # Garantir mesmo tamanho
        min_w = min(left.shape[1], right.shape[1])
        sym_horizontal = 1 - np.mean(np.abs(left[:, :min_w] - right[:, :min_w]))
    
        # Simetria vertical
        h_half = h // 2
        top = mag[:h_half, :]
        bottom = np.flipud(mag[h_half + (h % 2):, :])
    
        min_h = min(top.shape[0], bottom.shape[0])
        sym_vertical = 1 - np.mean(np.abs(top[:min_h, :] - bottom[:min_h, :]))
    
        # Simetria diagonal (simplificada e vetorizada)
        # Usar diagonal principal e anti-diagonal
        diag_size = min(h, w)
        diag_indices = np.arange(diag_size)
    
        main_diag = mag[diag_indices, diag_indices]
        anti_diag = mag[diag_indices, w - 1 - diag_indices]
    
        sym_diagonal = 1 - np.mean(np.abs(main_diag - np.flip(anti_diag)))
    
        # Score combinado
        combined = sym_horizontal * 0.4 + sym_vertical * 0.4 + sym_diagonal * 0.2
    
        return float(np.clip(combined, 0.0, 1.0))
    
    def _calculate_spectral_uniformity(self, magnitude: np.ndarray) -> float:
        """
        Calcula uniformidade do espectro.
        Espectros naturais têm distribuição complexa e não uniforme.
        Espectros de IA tendem a ser mais uniformes/organizados.
        """
        # Normalizar
        mag = magnitude / (np.max(magnitude) + 1e-10)
        
        # Calcular entropia do espectro (menos entropia = mais uniforme/artificial)
        # Histograma do espectro
        hist, _ = np.histogram(mag.flatten(), bins=256, range=(0, 1), density=True)
        hist = hist[hist > 0]  # Remover zeros para log
        
        # Entropia de Shannon
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Normalizar: entropia máxima teórica é log2(256) = 8
        max_entropy = 8.0
        normalized_entropy = entropy / max_entropy
        
        # Inverter: alta entropia = natural (baixa uniformidade)
        # baixa entropia = artificial (alta uniformidade)
        uniformity = 1.0 - normalized_entropy
        
        return float(np.clip(uniformity, 0.0, 1.0))
    
    def _detect_grid_artifacts(self, 
                                  magnitude: np.ndarray,
                                  peaks_mask: np.ndarray) -> bool:
        """
        Detecta artefatos de grade (grid) típicos de upscaling IA.
        
        Características:
        - Picos periódicos em padrão de grade
        - Simetria quadrada/retangular nos picos
        """
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        # Coordenadas dos picos
        peaks_coords = np.argwhere(peaks_mask)
        
        if len(peaks_coords) < 4:
            return False
        
        # Calcular distâncias dos picos ao centro
        distances = np.sqrt(
            (peaks_coords[:, 0] - center_y)**2 + 
            (peaks_coords[:, 1] - center_x)**2
        )
        
        # Agrupar por distância similar (anéis de frequência)
        from collections import defaultdict
        distance_groups = defaultdict(list)
        
        for i, dist in enumerate(distances):
            # Agrupar com tolerância de 5%
            key = round(dist / (max(h, w) * 0.05))
            distance_groups[key].append(peaks_coords[i])
        
        # Verificar se há grupos com simetria quadrada
        for group in distance_groups.values():
            if len(group) >= 4:
                # Verificar se formam padrão aproximadamente quadrado/retangular
                angles = []
                for y, x in group:
                    angle = np.arctan2(y - center_y, x - center_x)
                    angles.append(angle)
                
                # Ordenar ângulos
                angles_sorted = np.sort(angles)
                
                # Verificar espaçamento regular (simetria)
                if len(angles_sorted) >= 4:
                    diffs = np.diff(np.concatenate([angles_sorted, [angles_sorted[0] + 2*np.pi]]))
                    angle_variance = np.var(diffs)
                    
                    # Baixa variância nos ângulos = simetria perfeita (suspeito)
                    if angle_variance < 0.1:
                        return True
        
        # Verificar padrão de checkerboard (alternância perfeita)
        # Analisar distribuição de energia em quadrantes
        q1 = magnitude[:center_y, :center_x].mean()
        q2 = magnitude[:center_y, center_x:].mean()
        q3 = magnitude[center_y:, :center_x].mean()
        q4 = magnitude[center_y:, center_x:].mean()
        
        quadrants = [q1, q2, q3, q4]
        quad_variance = np.var(quadrants) / (np.mean(quadrants)**2 + 1e-10)
        
        # Quadrantes muito similares = padrão artificial
        if quad_variance < 0.01:
            return True
        
        return False
    
    def _count_significant_peaks(self, peaks_mask: np.ndarray) -> int:
        """Conta número de picos significativos detectados"""
        return int(np.sum(peaks_mask))

    def _detect_simple_image(self, gray_image: np.ndarray) -> bool:
        """
        Detecta se a imagem é naturalmente simples (monocromática, sem detalhes).
        Imagens simples têm espectro uniforme naturalmente, não é sinal de IA.
        """
        # Calcular complexidade visual (Variância de gradientes)
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Calcular média de gradiente (quanto mais baixo, mais simples)
        mean_gradient = np.mean(gradient_magnitude)
        
        # Calcular variância de cor (imagem monocromática tem baixa variância)
        color_variance = np.var(gray_image)
        
        # DEBUG: Ver os valores reais
        print(f"🔍 FFT Complexity Check: gradient={mean_gradient:.2f}, variance={color_variance:.2f}")
        
        is_simple = (mean_gradient < 15.0) and (color_variance < 6000.0)
        
        print(f"🔍 Is Simple Image? {is_simple}")
        
        return is_simple
  
    def _compute_spatial_fft_map(self, gray: np.ndarray) -> np.ndarray:
        """
        Calcula FFT por blocos para criar mapa espacial.
        Retorna mapa onde valores altos = padrões suspeitos (grid patterns)
        """
        h, w = gray.shape
        block_size = 64  # Blocos de 64x64
        
        # Ajustar dimensões
        h_blocks = h // block_size
        w_blocks = w // block_size
        
        fft_map = np.zeros((h_blocks, w_blocks), dtype=np.float32)
        
        for i in range(h_blocks):
            for j in range(w_blocks):
                # Extrair bloco
                y_start = i * block_size
                x_start = j * block_size
                block = gray[y_start:y_start+block_size, x_start:x_start+block_size]
                
                # FFT do bloco
                f = np.fft.fft2(block)
                fshift = np.fft.fftshift(f)
                magnitude = np.abs(fshift)
                
                # Detectar picos (grid pattern)
                center_y, center_x = block_size // 2, block_size // 2
                
                # Criar máscara: ignorar centro (baixas frequências)
                y, x = np.ogrid[:block_size, :block_size]
                mask = (x - center_x)**2 + (y - center_y)**2 > (block_size // 4)**2
                
                # Calcular "suspiciousness" do bloco
                magnitude_filtered = magnitude * mask
                peak_strength = np.percentile(magnitude_filtered[mask], 98)
                avg_strength = np.mean(magnitude_filtered[mask])
                
                # Ratio: quanto os picos se destacam
                if avg_strength > 0:
                    suspiciousness = peak_strength / (avg_strength + 1e-10)
                else:
                    suspiciousness = 0
                
                fft_map[i, j] = suspiciousness
        
        # Normalizar
        if np.max(fft_map) > 0:
            fft_map = fft_map / np.max(fft_map)
        
        return fft_map

    
    def _calculate_risk_score(self,
                               symmetry: float,
                               uniformity: float,
                               grid_artifacts: bool,
                               peak_count: int,
                               is_simple_image: bool = False) -> float:
        """
        Calcula score de risco composto (0-1).
        
        Fatores:
        - Simetria excessiva (>0.9 suspeito)
        - Uniformidade alta (>0.8 suspeito)
        - Grid artifacts (booleano, peso alto)
        - Número de picos anômalos
        """
        scores = []
        
        # Fator 1: Simetria (AJUSTADO)
        # Se a imagem é simples, simetria alta é NORMAL
        if is_simple_image:
            scores.append(0.05) # Ignora simetria em imagens simples
        else:
            if symmetry > 0.98: # Aumentei de 0.95 para 0.98 (mais tolerante)
                scores.append(0.9)
            elif symmetry > 0.95:
                scores.append(0.6)
            elif symmetry > 0.9:
                scores.append(0.4)
            else:
                scores.append(0.1)
        
        # Fator 2: Uniformidade (AJUSTADO)
        if is_simple_image:
            scores.append(0.05) # Ignora uniformidade em imagens simples
        else:
            if uniformity > 0.95: # Aumentei de 0.9 para 0.95
                scores.append(0.85)
            elif uniformity > 0.85:
                scores.append(0.5)
            elif uniformity > 0.75:
                scores.append(0.2)
            else:
                scores.append(0.05)
        
        # Fator 3: Grid artifacts (mantido - esse é confiável)
        if grid_artifacts:
            scores.append(0.95)
        else:
            scores.append(0.0)
        
        # Fator 4: Picos (mantido)
        if peak_count > 50:
            scores.append(0.8)
        elif peak_count > 30:
            scores.append(0.5)
        elif peak_count > 15:
            scores.append(0.3)
        else:
            scores.append(0.1)
        
        # Média ponderada
        weights = [0.25, 0.25, 0.35, 0.15]
        final_score = sum(s * w for s, w in zip(scores, weights))
        
        return min(1.0, max(0.0, final_score))
    
    def _generate_warnings(self,
                          symmetry: float,
                          uniformity: float,
                          grid_artifacts: bool,
                          peak_count: int,
                          is_simple_image: bool = False) -> List[str]:
        """Gera lista de avisos - VERSÃO COM FILTRO DE CONTEXTO"""
        warnings = []
        
        # Se a imagem é simples, ignora avisos de simetria/uniformidade
        if is_simple_image:
            if grid_artifacts:
                warnings.append("Padrões de grade detectados - típico de upscaling/GANs")
            
            if peak_count > 50:
                warnings.append(f"Múltiplos picos periódicos ({peak_count}) - artefatos de geração")
            
            # Mensagem positiva para imagem simples sem problemas
            if len(warnings) == 0:
                return ["Imagem de baixa complexidade (monocromática). Espectro normal para este tipo de conteúdo."]
            
            return warnings
        
        # ==== LÓGICA NORMAL (Imagens complexas) ====
        
        if symmetry > 0.98:
            warnings.append("Simetria espectral quase perfeita - altamente artificial")
        elif symmetry > 0.95:
            warnings.append("Simetria anômala no espectro de frequência")
        
        if uniformity > 0.95:
            warnings.append("Espectro excessivamente uniforme - falta complexidade orgânica")
        elif uniformity > 0.85:
            warnings.append("Distribuição espectral suspeitamente regular")
        
        if grid_artifacts:
            warnings.append("Padrões de grade detectados - típico de upscaling/GANs")
        
        if peak_count > 50:
            warnings.append(f"Múltiplos picos periódicos ({peak_count}) - artefatos de geração")
        elif peak_count > 30:
            warnings.append(f"Picos de frequência anômalos detectados ({peak_count})")
        
        if symmetry > 0.9 and grid_artifacts:
            warnings.append("Combinação crítica: simetria perfeita + grid artifacts - provável IA")
        
        # Mensagem positiva se não houver problemas
        if len(warnings) == 0:
            return ["✅ Nenhuma anomalia espectral detectada."]
        
        return warnings

    
    def analyze(self, image_path: str) -> Dict:
        """
        Executa análise completa de Espectro de Fourier.
        
        Args:
            image_path: Caminho para a imagem
            
        Returns:
            Dict no formato especificado do padrão de resposta
        """
        try:
            # Carregar imagem
            image = cv2.imread(image_path)
            if image is None:
                return {
                    "method": "FFT",
                    "status": "error",
                    "image_base64": "",
                    "metrics": {},
                    "risk_score": 0.0,
                    "warnings": ["Falha ao carregar imagem"]
                }
            
            # Converter para grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            is_simple = self._detect_simple_image(gray)
            if is_simple:
                print("📷 Imagem simples detectada (parede/céu/monocromática)")
            
            # Redimensionar para potência de 2 (otimização FFT) se muito grande
            max_size = 1024
            h, w = gray.shape
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                # Garantir dimensões pares para simetria perfeita na análise
                new_h = (new_h // 2) * 2
                new_w = (new_w // 2) * 2
                gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Calcular FFT
            magnitude, magnitude_log, phase = self._compute_fft_spectrum(gray)
            
            # Métricas
            symmetry_score = self._calculate_symmetry_score(magnitude)
            spectral_uniformity = self._calculate_spectral_uniformity(magnitude)
            
            # Detectar picos
            peaks_mask = self._detect_peaks(magnitude)
            peak_count = self._count_significant_peaks(peaks_mask)
            
            # Detectar grid artifacts
            grid_artifacts = self._detect_grid_artifacts(magnitude, peaks_mask)
            
            # Criar visualização
            spectrum_vis = self._create_spectrum_visualization(magnitude_log, magnitude)
            spectrum_base64 = self._convert_to_base64(spectrum_vis)

            # Gerar mapa espacial (NOVO)
            fft_spatial_map = self._compute_spatial_fft_map(gray)
            fft_map_base64 = self._convert_to_base64(fft_spatial_map)
            
            # Calcular risco
            risk_score = self._calculate_risk_score(
                symmetry_score,
                spectral_uniformity,
                grid_artifacts,
                peak_count,
                is_simple
            )
            
            # Gerar avisos
            warnings_list = self._generate_warnings(
                symmetry_score,
                spectral_uniformity,
                grid_artifacts,
                peak_count,
                is_simple
            )
            
            return {
                "method": "FFT",
                "status": "success",
                "image_base64": spectrum_base64,
                "metrics": {
                    "spectral_uniformity": round(spectral_uniformity, 4),
                    "peak_frequency_count": peak_count,
                    "symmetry_score": round(symmetry_score, 4),
                    "grid_artifacts": grid_artifacts,
                    "dominant_frequency": self._find_dominant_frequency(magnitude),
                    "high_frequency_energy_ratio": self._calculate_hf_energy_ratio(magnitude),
                    "fft_spatial_map": fft_spatial_map.tolist()
                },
                "risk_score": round(risk_score, 2),
                "warnings": warnings_list
            }
            
        except Exception as e:
            return {
                "method": "FFT",
                "status": "error",
                "image_base64": "",
                "metrics": {},
                "risk_score": 0.0,
                "warnings": [f"Erro na análise: {str(e)}"]
            }
    
    def _find_dominant_frequency(self, magnitude: np.ndarray) -> Tuple[int, int]:
        """Encontra a frequência dominante (excluindo DC component)"""
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        # Criar máscara para excluir centro
        mask = np.ones_like(magnitude, dtype=bool)
        exclude_radius = min(h, w) // 20
        Y, X = np.ogrid[:h, :w]
        center_mask = (X - center_x)**2 + (Y - center_y)**2 <= exclude_radius**2
        mask[center_mask] = False
        
        # Aplicar máscara
        masked_magnitude = magnitude.copy()
        masked_magnitude[~mask] = 0
        
        # Encontrar máximo
        max_idx = np.unravel_index(np.argmax(masked_magnitude), magnitude.shape)
        
        return (int(max_idx[0] - center_y), int(max_idx[1] - center_x))
    
    def _calculate_hf_energy_ratio(self, magnitude: np.ndarray) -> float:
        """
        Calcula razão de energia em alta frequência vs baixa frequência.
        Imagens IA tendem a ter distribuição anômala de energia.
        """
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        # Definir regiões
        radius_low = min(h, w) // 8
        radius_high = min(h, w) // 2
        
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        
        # Máscaras
        low_freq_mask = dist_from_center <= radius_low
        high_freq_mask = (dist_from_center > radius_low) & (dist_from_center <= radius_high)
        
        # Energias
        energy_low = np.sum(magnitude[low_freq_mask])
        energy_high = np.sum(magnitude[high_freq_mask])
        
        # Razão
        total_energy = energy_low + energy_high
        if total_energy > 0:
            ratio = energy_high / total_energy
        else:
            ratio = 0.0
        
        return round(float(ratio), 4)


# Função de conveniência para uso direto
def analyze_fft(image_path: str) -> Dict:
    """
    Função standalone para análise de Espectro de Fourier.
    
    Args:
        image_path: Caminho para a imagem
        
    Returns:
        Dict com resultado da análise no padrão especificado
    """
    analyzer = FFTAnalyzer()
    return analyzer.analyze(image_path)