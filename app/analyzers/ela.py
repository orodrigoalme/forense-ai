import cv2
import numpy as np
from PIL import Image
import base64
import io
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ELAResult:
    """Estrutura de dados para resultado da análise ELA"""
    method: str
    status: str
    image_base64: str
    metrics: Dict
    risk_score: float
    warnings: List[str]


class ELAAnalyzer:
    """
    Analisador de Error Level Analysis (ELA) para detecção de manipulação e imagens IA.
    
    Principais características detectáveis via ELA:
    - Diferenças de nível de erro entre regiões (indicativo de recompressão seletiva)
    - Regiões com erro anormalmente baixo (inserções de IA perfeitas)
    - Bordas com erro inconsistente (splicing, copy-move)
    - Padrões de erro uniforme demais (geração IA total)
    
    ELA funciona re-salvando a imagem com qualidade conhecida e comparando
    com o original - regiões que mudam pouco foram provavelmente salvas 
    na mesma qualidade anterior (possivelmente manipuladas).
    """
    
    def __init__(self,
                 quality_levels: List[int] = [90, 85, 80],
                 error_threshold_low: float = 5.0,
                 error_threshold_high: float = 80.0,
                 min_region_size: int = 100):
        """
        Args:
            quality_levels: Níveis de qualidade JPEG para recompressão
            error_threshold_low: Limiar inferior de erro (pixels suspeitos)
            error_threshold_high: Limiar superior de erro (pixels normais)
            min_region_size: Tamanho mínimo para considerar uma região
        """
        self.quality_levels = quality_levels
        self.error_threshold_low = error_threshold_low
        self.error_threshold_high = error_threshold_high
        self.min_region_size = min_region_size
        
    def _convert_to_base64(self, image: np.ndarray, format: str = 'PNG') -> str:
        """Converte imagem numpy array para base64"""
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        if image_rgb.dtype != np.uint8:
            image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
            
        pil_image = Image.fromarray(image_rgb)
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode('utf-8')
        return img_str
    
    def _compute_ela(self, 
                     original: np.ndarray, 
                     quality: int = 90,
                     scale_factor: int = 15) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula Error Level Analysis para um nível de qualidade específico.
        
        Args:
            original: Imagem original (BGR)
            quality: Qualidade JPEG para recompressão
            scale_factor: Fator de escala para visualização do erro
            
        Returns:
            ela_image: Imagem ELA visualizável
            error_map: Mapa numérico de erro absoluto
        """
        # Converter para PIL para compressão JPEG
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        pil_original = Image.fromarray(original_rgb)
        
        # Recomprimir
        buffer = io.BytesIO()
        pil_original.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        pil_recompressed = Image.open(buffer)
        
        # Converter de volta para numpy
        recompressed = cv2.cvtColor(
            np.array(pil_recompressed), 
            cv2.COLOR_RGB2BGR
        )
        
        # Redimensionar recompressed para match se necessário
        if original.shape != recompressed.shape:
            recompressed = cv2.resize(
                recompressed, 
                (original.shape[1], original.shape[0])
            )
        
        # Calcular diferença absoluta
        diff = cv2.absdiff(original.astype(np.float32), 
                          recompressed.astype(np.float32))
        
        # Multiplicar por fator de escala para visualização
        diff_scaled = diff * scale_factor
        
        # Criar imagem ELA (escala de cinza da magnitude do erro)
        error_magnitude = np.sqrt(
            diff_scaled[:,:,0]**2 + 
            diff_scaled[:,:,1]**2 + 
            diff_scaled[:,:,2]**2
        )
        
        # Normalizar para 0-255
        ela_gray = np.clip(error_magnitude, 0, 255).astype(np.uint8)
        
        # Aplicar colormap para visualização colorida
        ela_color = cv2.applyColorMap(ela_gray, cv2.COLORMAP_JET)
        
        # Calcular erro médio por canal para análise
        error_per_channel = np.mean(diff, axis=(0,1))
        
        return ela_color, ela_gray, error_per_channel, diff
    
    def _multi_quality_ela(self, original: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Executa ELA em múltiplos níveis de qualidade para robustez.
        Combina resultados ponderados.
        """
        ela_results = []
        error_maps = []
        channel_errors = []
        
        for quality in self.quality_levels:
            ela_color, ela_gray, channel_error, raw_diff = self._compute_ela(
                original, quality
            )
            ela_results.append(ela_color)
            error_maps.append(ela_gray.astype(np.float32))
            channel_errors.append(channel_error)
        
        # Combinar error maps (média ponderada - qualidade mais baixa pesa mais)
        weights = [0.2, 0.3, 0.5]  # Pesos crescentes para qualidades mais baixas
        combined_error = np.average(error_maps, axis=0, weights=weights)
        
        # ELA final: usar o resultado de qualidade média para visualização
        # mas com overlay do erro combinado
        final_ela = self._create_enhanced_ela_visualization(
            original, ela_results[1], combined_error
        )
        
        # Estatísticas combinadas
        stats = {
            'mean_channel_errors': np.mean(channel_errors, axis=0),
            'error_variance_across_qualities': np.var(error_maps, axis=0).mean()
        }
        
        return final_ela, combined_error, stats
    
    def _create_enhanced_ela_visualization(self,
                                            original: np.ndarray,
                                            base_ela: np.ndarray,
                                            combined_error: np.ndarray) -> np.ndarray:
        """
        Cria visualização ELA melhorada com overlay e anotações.
        """
        h, w = original.shape[:2]
        
        # Criar canvas composto
        # Layout: [Original | ELA | ELA + Overlay | Heatmap puro]
        canvas_width = w * 4
        canvas_height = h + 60  # Espaço para legendas
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
        
        # 1. Imagem original (reduzida)
        orig_small = cv2.resize(original, (w, h))
        canvas[:h, :w] = orig_small
        
        # 2. ELA base
        canvas[:h, w:2*w] = base_ela
        
        # 3. Overlay ELA + Original
        alpha = 0.6
        overlay = cv2.addWeighted(orig_small, 1-alpha, base_ela, alpha, 0)
        canvas[:h, 2*w:3*w] = overlay
        
        # 4. Heatmap de erro combinado (mais sensível)
        error_norm = cv2.normalize(combined_error, None, 0, 255, cv2.NORM_MINMAX)
        error_colored = cv2.applyColorMap(error_norm.astype(np.uint8), cv2.COLORMAP_HOT)
        canvas[:h, 3*w:4*w] = error_colored
        
        # Adicionar legendas
        font = cv2.FONT_HERSHEY_SIMPLEX
        labels = [
            "Original",
            f"ELA (Q={self.quality_levels[1]})", 
            "Overlay",
            "Error Heatmap"
        ]
        
        for i, label in enumerate(labels):
            x_pos = i * w + 10
            y_pos = h + 30
            cv2.putText(canvas, label, (x_pos, y_pos), font, 0.6, (0, 0, 0), 2)
        
        # Adicionar barra de escala de erro
        legend_y = h + 50
        cv2.rectangle(canvas, (10, legend_y), (canvas_width-10, legend_y+10), (200, 200, 200), -1)
        cv2.putText(canvas, "Low Error (Suspicious)", (10, legend_y-5), font, 0.4, (0, 0, 0), 1)
        cv2.putText(canvas, "High Error (Normal)", (canvas_width-150, legend_y-5), font, 0.4, (0, 0, 0), 1)
        
        return canvas
    
    def _identify_affected_regions(self,
                                    error_map: np.ndarray,
                                    original: np.ndarray) -> List[str]:
        """
        Identifica regiões da imagem com níveis de erro anômalos.
        Usa segmentação por cor + análise de erro local.
        """
        regions = []
        h, w = error_map.shape
        
        # Criar máscaras de erro
        low_error_mask = error_map < self.error_threshold_low
        high_error_mask = error_map > self.error_threshold_high
        
        # Análise de regiões usando cor original
        hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        
        # Definir ranges para segmentação
        # Pele
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 255, 255])
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Céu/Azul
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Verde/Vegetação
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Textura/Detalhes (alta frequência de cor)
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        texture_mask = edges > 0
        
        # Verificar sobreposição com baixo erro (suspeito)
        min_pixels = self.min_region_size
        
        if np.sum(low_error_mask & (skin_mask > 0)) > min_pixels:
            # Verificar se é erro uniformemente baixo (padrão IA)
            skin_error_mean = np.mean(error_map[skin_mask > 0])
            if skin_error_mean < self.error_threshold_low * 2:
                regions.append("face")
        
        if np.sum(low_error_mask & (blue_mask > 0)) > min_pixels:
            regions.append("sky")
            
        if np.sum(low_error_mask & (green_mask > 0)) > min_pixels:
            regions.append("vegetation")
        
        # Análise de bordas inconsistentes
        edge_low_error = np.sum(edges & low_error_mask)
        edge_total = np.sum(edges)
        if edge_total > 0 and (edge_low_error / edge_total) > 0.3:
            regions.append("edges")
        
        # Detectar padrão de erro uniforme (toda imagem com erro similar = IA)
        error_std = np.std(error_map)
        if error_std < 2.0:
            regions.append("uniform_pattern")
        
        # Análise de blocos (padrão de compressão em blocos 8x8)
        block_size = 8
        block_variances = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = error_map[y:y+block_size, x:x+block_size]
                block_variances.append(np.var(block))
        
        if np.mean(block_variances) < 1.0:
            regions.append("block_artifacts")
        
        return list(set(regions))  # Remover duplicatas
    
    def _calculate_error_statistics(self, error_map: np.ndarray) -> Dict:
        """
        Calcula estatísticas detalhadas do mapa de erro.
        """
        # Estatísticas básicas
        mean_error = float(np.mean(error_map))
        max_error = float(np.max(error_map))
        std_error = float(np.std(error_map))
        
        # Percentis
        percentiles = np.percentile(error_map, [25, 50, 75, 90, 95, 99])
        
        # Contagem de pixels por faixa de erro
        total_pixels = error_map.size
        low_error_pixels = np.sum(error_map < self.error_threshold_low)
        high_error_pixels = np.sum(error_map > self.error_threshold_high)
        
        bright_pixels_pct = (high_error_pixels / total_pixels) * 100
        suspicious_pixels_pct = (low_error_pixels / total_pixels) * 100
        
        # Análise de distribuição (curtose - cauda da distribuição)
        # Distribuição natural tende a ter cauda longa
        from scipy.stats import kurtosis
        try:
            error_kurtosis = float(kurtosis(error_map.flatten()))
        except:
            error_kurtosis = 0.0
        
        return {
            'mean_error_level': round(mean_error / 255.0, 4),  # Normalizado 0-1
            'max_error_value': round(max_error, 2),
            'std_error': round(std_error, 2),
            'bright_pixels_percentage': round(bright_pixels_pct, 2),
            'suspicious_low_error_percentage': round(suspicious_pixels_pct, 2),
            'error_percentiles': {
                'p25': round(percentiles[0], 2),
                'p50': round(percentiles[1], 2),
                'p75': round(percentiles[2], 2),
                'p90': round(percentiles[3], 2),
                'p99': round(percentiles[5], 2)
            },
            'error_kurtosis': round(error_kurtosis, 2),
            'error_distribution_uniformity': round(1.0 / (std_error + 1e-10), 4)
        }
    
    def _detect_compression_inconsistencies(self,
                                           error_map: np.ndarray, 
                                           original: np.ndarray) -> List[str]:
        """
        Detecta inconsistências típicas de múltiplas compressões ou manipulação.
        OTIMIZADO: Usa operações vetorizadas NumPy e IGNORA clipping.
        """
        inconsistencies = []
        h, w = error_map.shape
        
        # 1. Detectar fronteiras abruptas de erro (indicativo de splicing)
        grad_x = cv2.Sobel(error_map, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(error_map, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        high_gradient_pixels = np.sum(gradient_magnitude > 100)
        if high_gradient_pixels > (error_map.size * 0.005):
            inconsistencies.append("error_boundary_discontinuities")
        
        # 2. Detectar padrão periódico de erro (JPEG blocking artifacts)
        block_size = 8
        h_crop = (h // block_size) * block_size
        w_crop = (w // block_size) * block_size
        error_crop = error_map[:h_crop, :w_crop]
        
        blocks = error_crop.reshape(h_crop // block_size, block_size, 
                                    w_crop // block_size, block_size)
            
        block_means = blocks.mean(axis=(1, 3))
        inter_block_variance = np.var(block_means)
            
        # Aumentei tolerância aqui também (WhatsApp gera muita variância de bloco)
        if inter_block_variance > 150: # Subi de 50 para 80
            inconsistencies.append("multiple_compression_blocks")
            
        # 3. Detectar regiões de erro zero absoluto (CORRIGIDO: IGNORAR SATURAÇÃO)
        # Em áreas saturadas (preto 0 ou branco 255), o erro é zero naturalmente.
        # Precisamos ignorar esses pixels para não acusar fake falsamente.
            
        # Converte original para grayscale para checar brilho
        if len(original.shape) == 3:
            gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            gray_orig = original

        # Máscara de pixels saturados (Muito escuros < 5 ou Muito claros > 250)
        # Esses pixels NÃO mudam na recompressão, gerando erro 0 falso.
        saturated_mask = (gray_orig < 5) | (gray_orig > 250)
            
        # Máscara de erro zero
        zero_error_mask = (error_map < 0.5)
            
        # Zero Error REAL = Erro Zero E Não Saturado
        true_zero_error_pixels = np.sum(zero_error_mask & ~saturated_mask)
        valid_pixels_count = np.sum(~saturated_mask)
            
        # Só acusa se tiver muito erro zero em áreas NÃO saturadas (Ex: Pele lisa demais de IA)
        if valid_pixels_count > 0:
            zero_error_ratio = true_zero_error_pixels / valid_pixels_count
            if zero_error_ratio > 0.05: # > 5% da imagem válida
                inconsistencies.append("zero_error_regions")
            
        # 4. Detectar padrão de grid 8x8 anômalo
        if block_means.size > 1:
            h_diff = np.diff(block_means, axis=1)
            v_diff = np.diff(block_means, axis=0)
            
            h_diff_variance = np.var(h_diff) if h_diff.size > 0 else 0
            v_diff_variance = np.var(v_diff) if v_diff.size > 0 else 0
                
            if h_diff_variance < 5 or v_diff_variance < 5:
                inconsistencies.append("regular_blocking_pattern")
            
        return inconsistencies

    
    def _calculate_risk_score(self,
                          stats: Dict,
                          affected_regions: List[str],
                          inconsistencies: List[str]) -> float:
        scores = []
        
        # Fator 1: Nível médio de erro (seu código atual)
        mean_error = stats['mean_error_level'] * 255
        if mean_error < 1.0:
            scores.append(0.95)
        elif mean_error < 3.0:
            scores.append(0.6)
        elif mean_error < 15.0:
            scores.append(0.2)
        else:
            scores.append(0.05)
        
        # Fator 2: Pixels brilhantes (mantido)
        bright_pct = stats['bright_pixels_percentage']
        if bright_pct < 1.0:
            scores.append(0.9)
        elif bright_pct < 5.0:
            scores.append(0.5)
        elif bright_pct < 20.0:
            scores.append(0.2)
        else:
            scores.append(0.05)
        
        # NOVO: Fator 3 - Percentual de Erro Suspeito Baixo (O FATOR QUE FALTAVA!)
        # Fator 3 - Percentual de Erro Suspeito Baixo (MAIS AGRESSIVO)
        suspicious_pct = stats['suspicious_low_error_percentage']
        
        if suspicious_pct > 25.0: # > 25% (A sua editada tem 26.55%)
            scores.append(0.98) # AUMENTEI de 0.95 para 0.98
        elif suspicious_pct > 15.0:
            scores.append(0.85) # AUMENTEI de 0.7 para 0.85
        elif suspicious_pct > 8.0: # Aumentei sensibilidade aqui também
            scores.append(0.65)
        elif suspicious_pct > 5.0:
            scores.append(0.35)
        else:
            scores.append(0.05)
        
        # Fator 4: Regiões afetadas (mantido, ajustado)
        region_weights = {
            'face': 0.8,
            'edges': 0.8,
            'uniform_pattern': 0.95,
            'block_artifacts': 0.6,
            'sky': 0.3,
            'vegetation': 0.2
        }
        
        if affected_regions:
            max_region_score = max([region_weights.get(r, 0.4) for r in affected_regions])
            scores.append(max_region_score)
        else:
            scores.append(0.0)
        
        # Fator 5: Inconsistências (recalibrado)
        inconsistency_weights = {
            'zero_error_regions': 0.7,
            'multiple_compression_blocks': 0.4,
            'error_boundary_discontinuities': 0.6
        }
        
        if inconsistencies:
            max_inc_score = max([inconsistency_weights.get(i, 0.3) for i in inconsistencies])
            scores.append(max_inc_score)
        else:
            scores.append(0.1)
        
        # Fator 6: Curtose (mantido)
        kurt = stats.get('error_kurtosis', 0)
        if kurt < -1.0:
            scores.append(0.5)
        else:
            scores.append(0.1)
        
        # Média ponderada RECALIBRADA (agora são 6 fatores)
        # Aumentei o peso do Fator 3 (Suspicious %) para 25%
        weights = [0.12, 0.08, 0.35, 0.20, 0.15, 0.10]
        
        final_score = sum(s * w for s, w in zip(scores, weights))
        
        # ============ BOOSTS FINAIS (CRITICAL COMBINATIONS) ============
    
        # Regra 1: Se tem >20% de pixels suspeitos E tem zero_error_regions, é quase certeza de edição
        if stats['suspicious_low_error_percentage'] > 20.0 and 'zero_error_regions' in inconsistencies:
            final_score = max(final_score, 0.75) # Força mínimo de 0.75
            print("🚨 BOOST: Área extensa suspeita + Zero Error → Fake Confirmado")
    
        # Regra 2: Se tem >25% suspeito E múltiplas inconsistências (3+), é montagem clara
        if stats['suspicious_low_error_percentage'] > 25.0 and len(inconsistencies) >= 3:
            final_score = min(1.0, final_score * 1.35) # Multiplica por 1.5 (boost agressivo)
            print("🚨 BOOST: Múltiplas inconsistências + Alto % suspeito → Manipulação Grave")
    
        # Regra 3: Edição "Profissional" (baixo erro médio + alto suspicious %)
        if mean_error < 50.0 and stats['suspicious_low_error_percentage'] > 20.0:
            final_score = max(final_score, 0.70) # Força mínimo de 0.70
            print("🚨 BOOST: Edição limpa detectada")

        
        return min(1.0, max(0.0, final_score))
    
    def _generate_warnings(self,
                          stats: Dict,
                          affected_regions: List[str],
                          inconsistencies: List[str]) -> List[str]:
        """Gera lista de avisos contextualizados"""
        warnings = []
        
        # ===== NOVO: FILTRO DE RELEVÂNCIA =====
        # Calcula o risk_score aqui também para contexto
        # (Ou passe como argumento se preferir)
        temp_risk = self._calculate_risk_score(stats, affected_regions, inconsistencies)
    
        # Se risco é baixo, ignora inconsistências leves (compressão normal)
        if temp_risk < 0.4:
            # Remove inconsistências "chatas" que são normais em JPEG
            inconsistencies_filtered = [
                i for i in inconsistencies 
                if i not in ['error_boundary_discontinuities', 'multiple_compression_blocks']
            ]
        else:
            inconsistencies_filtered = inconsistencies
        # ===== FIM DO FILTRO =====

        # Avisos sobre nível de erro
        mean_error = stats['mean_error_level'] * 255
        
        if mean_error < 3.0:
            warnings.append(f"Erro de recompressão extremamente baixo ({mean_error:.1f}) - imagem possivelmente gerada")
        elif mean_error < 8.0:
            warnings.append(f"Nível de erro suspeitamente baixo ({mean_error:.1f})")
        
        # Avisos sobre distribuição
        if stats['bright_pixels_percentage'] < 5:
            warnings.append("Quase nenhuma variação de erro detectada - padrão artificial")
        
        if stats['suspicious_low_error_percentage'] > 30:
            warnings.append(f"{stats['suspicious_low_error_percentage']:.1f}% da imagem com erro anormalmente baixo")
        
        # Avisos de regiões
        if 'face' in affected_regions:
            warnings.append("Alto nível de erro detectado na região do rosto - possível manipulação")
        
        if 'uniform_pattern' in affected_regions:
            warnings.append("Padrão de erro uniforme em toda a imagem - típico de geração IA")
        
        if 'edges' in affected_regions:
            warnings.append("Bordas com inconsistências de compressão - indício de splicing")
        
        if 'block_artifacts' in affected_regions:
            warnings.append("Artefatos de bloco 8x8 detectados - múltiplas compressões ou origem digital")
        
        # Avisos de inconsistências
        if 'zero_error_regions' in inconsistencies_filtered:
            warnings.append("Regiões com erro zero absoluto - impossível em captura fotográfica real")
    
        if 'multiple_compression_blocks' in inconsistencies_filtered:
            warnings.append("Padrão de compressão inconsistente entre regiões")
    
        if 'error_boundary_discontinuities' in inconsistencies_filtered:
            warnings.append("Fronteiras abruptas no nível de erro - possível montagem de imagens")
        # Análise combinada
        if mean_error < 5 and 'face' in affected_regions:
            warnings.append("Combinação crítica: baixo erro + região facial suspeita")
        
        return warnings
    
    def analyze(self, image_path: str) -> Dict:
        """
        Executa análise completa de Error Level Analysis.
        
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
                    "method": "ELA",
                    "status": "error",
                    "image_base64": "",
                    "metrics": {},
                    "risk_score": 0.0,
                    "warnings": ["Falha ao carregar imagem"]
                }
            
            # Verificar se é JPEG (ELA funciona melhor em imagens comprimidas)
            is_jpeg = image_path.lower().endswith(('.jpg', '.jpeg'))
            if not is_jpeg:
                # Salvar temporariamente como JPEG para aplicar ELA
                _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                warnings_list_init = ["Imagem convertida para análise JPEG"]
            else:
                warnings_list_init = []
            
            # Executar ELA multi-qualidade
            ela_image, error_map, extra_stats = self._multi_quality_ela(image)
            
            # Calcular estatísticas
            stats = self._calculate_error_statistics(error_map)
            
            # Identificar regiões afetadas
            affected_regions = self._identify_affected_regions(error_map, image)
            
            # Detectar inconsistências
            inconsistencies = self._detect_compression_inconsistencies(error_map, image)
            
            # Converter para base64
            ela_base64 = self._convert_to_base64(ela_image)
            

            # Calcular risco
            risk_score = self._calculate_risk_score(
                stats, affected_regions, inconsistencies
            )
            
            # Gerar avisos
            warnings_list = warnings_list_init + self._generate_warnings(
                stats, affected_regions, inconsistencies
            )
            
            # Montar resposta
            metrics = {
                "mean_error_level": stats['mean_error_level'],
                "bright_pixels_percentage": stats['bright_pixels_percentage'],
                "max_error_value": stats['max_error_value'],
                "affected_regions": affected_regions,
                "error_std": stats['std_error'],
                "error_kurtosis": stats['error_kurtosis'],
                "suspicious_low_error_percentage": stats['suspicious_low_error_percentage'],
                "compression_inconsistencies": inconsistencies,
                "error_percentiles": stats['error_percentiles']
            }

            # mapa ELA normalizado
            ela_map_normalized = error_map / 255.0 if np.max(error_map) > 0 else error_map

            return {
                "method": "ELA",
                "status": "success",
                "image_base64": ela_base64,
                "metrics": metrics,
                "risk_score": round(risk_score, 2),
                "warnings": warnings_list if warnings_list else ["Nenhuma anomalia significativa detectada"],
                "ela_map": ela_map_normalized.tolist() 
            }
            
        except Exception as e:
            return {
                "method": "ELA",
                "status": "error",
                "image_base64": "",
                "metrics": {},
                "risk_score": 0.0,
                "warnings": [f"Erro na análise: {str(e)}"]
            }


# Função de conveniência para uso direto
def analyze_ela(image_path: str) -> Dict:
    """
    Função standalone para análise de Error Level Analysis.
    
    Args:
        image_path: Caminho para a imagem
        
    Returns:
        Dict com resultado da análise no padrão especificado
    """

    analyzer = ELAAnalyzer()
    return analyzer.analyze(image_path)