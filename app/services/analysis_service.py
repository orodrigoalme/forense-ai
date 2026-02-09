import asyncio
from typing import Dict, List, Optional
from google import genai
from google.genai import types
import os
from PIL import Image
import io
import base64
import tempfile
from datetime import datetime

# Importar analyzers da estrutura existente
from app.analyzers.fft import FFTAnalyzer
from app.analyzers.noise import NoiseMapAnalyzer
from app.analyzers.ela import ELAAnalyzer
from app.services.image_annotator import ImageAnnotator


class CostTracker:
    """Rastreia custos da API Gemini"""
    
    def __init__(self):
        # Usar Redis para rastreamento distribuído (ou arquivo local)
        self.max_daily_cost = float(os.getenv("MAX_DAILY_GEMINI_COST", "5.0"))  # $5/dia
        self.max_monthly_cost = float(os.getenv("MAX_MONTHLY_GEMINI_COST", "50.0"))  # $50/mês
        
        # Custo aproximado do Gemini (ajustar conforme modelo)
        self.cost_per_request = 0.002  # ~$0.002 por análise
    
    def can_make_request(self) -> tuple[bool, str]:
        """Verifica se pode fazer requisição baseado no budget"""
        today_cost = self._get_today_cost()
        month_cost = self._get_month_cost()
        
        if today_cost >= self.max_daily_cost:
            return False, f"Limite diário atingido (${today_cost:.2f}/${self.max_daily_cost:.2f})"
        
        if month_cost >= self.max_monthly_cost:
            return False, f"Limite mensal atingido (${month_cost:.2f}/${self.max_monthly_cost:.2f})"
        
        return True, "OK"
    
    def track_request(self):
        """Registra custo de uma requisição"""
        # Incrementar contador
        # Implementação depende se usa Redis, SQLite, ou arquivo
        pass
    
    def _get_today_cost(self) -> float:
        # Ler custos do dia
        # Implementação simplificada
        return 0.0
    
    def _get_month_cost(self) -> float:
        # Ler custos do mês
        return 0.0

class AnalysisService:
    """Serviço consolidado de análise de imagem com integração Gemini"""
    
    def __init__(self, custom_gemini_key: Optional[str] = None):
        self.fft_analyzer = FFTAnalyzer()
        self.noise_analyzer = NoiseMapAnalyzer()
        self.ela_analyzer = ELAAnalyzer()
        self.annotator = ImageAnnotator() 
        
        # Configurar Gemini com chave customizada ou do servidor
        if custom_gemini_key:
            # Cliente forneceu sua própria chave
            try:
                self.client = genai.Client(api_key=custom_gemini_key)
                self.gemini_enabled = True
                self.using_custom_key = True
                print("✅ Usando chave Gemini customizada do cliente")
            except Exception as e:
                print(f"⚠️ Chave Gemini customizada inválida: {e}")
                self.gemini_enabled = False
                self.using_custom_key = False
        else:
            # Usar chave do servidor
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("⚠️ WARNING: GEMINI_API_KEY não configurada. Análise Gemini desabilitada.")
                self.gemini_enabled = False
                self.using_custom_key = False
            else:
                self.client = genai.Client(api_key=api_key)
                self.gemini_enabled = True
                self.using_custom_key = False
                print("✅ Usando chave Gemini do servidor")
    
    async def analyze_full(self, image_path: str) -> Dict:
        """
        Executa análise completa:
        1. FFT, Noise e ELA em paralelo
        2. Consolida resultados
        3. Envia para Gemini (se habilitado)
        4. Retorna veredicto final
        """
        
        print(f"🔍 Iniciando análise completa: {image_path}")
        
        # Executar análises em paralelo
        fft_result, noise_result, ela_result = await asyncio.gather(
            asyncio.to_thread(self.fft_analyzer.analyze, image_path),
            asyncio.to_thread(self.noise_analyzer.analyze, image_path),
            asyncio.to_thread(self.ela_analyzer.analyze, image_path),
            return_exceptions=True
        )
        
        # Tratar exceções
        if isinstance(fft_result, Exception):
            print(f"❌ FFT Error: {fft_result}")
            fft_result = self._empty_result("FFT", str(fft_result))
        if isinstance(noise_result, Exception):
            print(f"❌ NOISE Error: {noise_result}")
            noise_result = self._empty_result("NOISE", str(noise_result))
        if isinstance(ela_result, Exception):
            print(f"❌ ELA Error: {ela_result}")
            ela_result = self._empty_result("ELA", str(ela_result))
        
        print(f"✅ Análises concluídas: FFT={fft_result['risk_score']:.2f}, NOISE={noise_result['risk_score']:.2f}, ELA={ela_result['risk_score']:.2f}")
        
        # Consolidar análise automatizada
        automated = self._consolidate_results(fft_result, noise_result, ela_result)
        
        # Enviar para Gemini (se habilitado)
        if self.gemini_enabled:
            gemini_analysis = await self._analyze_with_gemini(
                automated, fft_result, noise_result, ela_result
            )
        else:
            gemini_analysis = {
                "verdict": "DISABLED",
                "explanation": "Gemini API não configurada",
                "confidence": "none"
            }
        

        # Gerar imagem anotada
        try:
            annotated_image_base64 = self.annotator.annotate_full_analysis(
                image_path, fft_result, noise_result, ela_result, automated
            )
            print("✅ Imagem anotada gerada")
        except Exception as e:
            print(f"⚠️ Erro ao gerar imagem anotada: {e}")
            annotated_image_base64 = ""

        # Remover mapas espaciais dos resultados originais (se existirem)
        noise_result.pop("variance_map", None)
        ela_result.pop("ela_map", None)

        # Agora criar os objetos limpos
        fft_clean = {
            "method": fft_result.get("method"),
            "status": fft_result.get("status"),
            "image_base64": fft_result.get("image_base64"),
            "risk_score": fft_result.get("risk_score"),
            "metrics": fft_result.get("metrics"),
            "warnings": fft_result.get("warnings")
        }

        if isinstance(fft_clean["metrics"], dict):
            fft_clean["metrics"].pop("fft_spatial_map", None)

        noise_clean = {
            "method": noise_result.get("method"),
            "status": noise_result.get("status"),
            "image_base64": noise_result.get("image_base64"),
            "risk_score": noise_result.get("risk_score"),
            "metrics": noise_result.get("metrics"),
            "warnings": noise_result.get("warnings")
        }

        ela_clean = {
            "method": ela_result.get("method"),
            "status": ela_result.get("status"),
            "image_base64": ela_result.get("image_base64"),
            "risk_score": ela_result.get("risk_score"),
            "metrics": ela_result.get("metrics"),
            "warnings": ela_result.get("warnings")
        }

        return {
            "automated_analysis": automated,
            "gemini_analysis": gemini_analysis,
            "annotated_image": annotated_image_base64,
            "details": {
                "fft": fft_clean,
                "noise": noise_clean,
                "ela": ela_clean
            }
        }
            
    def _empty_result(self, method: str, error_msg: str) -> Dict:
        """Retorna resultado vazio em caso de erro"""
        return {
            "method": method,
            "status": "error",
            "risk_score": 0.0,
            "metrics": {},
            "warnings": [f"Análise falhou: {error_msg}"],
            "image_base64": ""
        }
    

    def _extract_explanation(self, text: str) -> str:
        """Extrai apenas a seção EXPLANATION"""
        lines = text.split('\n')
        explanation = []
        capturing = False
        
        for line in lines:
            if 'EXPLANATION' in line.upper():
                capturing = True
                continue
            elif capturing:
                if line.strip().startswith('**') or line.strip().startswith('#'):
                    break  # Próxima seção
                if line.strip():
                    explanation.append(line.strip())
        
        return ' '.join(explanation) if explanation else text[:200]

    def _extract_key_indicators(self, text: str) -> List[str]:
        """Extrai bullet points de KEY INDICATORS"""
        lines = text.split('\n')
        indicators = []
        capturing = False
        
        for line in lines:
            if 'KEY INDICATORS' in line.upper():
                capturing = True
                continue
            elif capturing:
                if line.strip().startswith('**') or line.strip().startswith('#'):
                    break
                if line.strip().startswith('•') or line.strip().startswith('-'):
                    indicators.append(line.strip()[1:].strip())
        
        return indicators[:3]  # Máximo 3


    def _consolidate_results(self, fft: Dict, noise: Dict, ela: Dict) -> Dict:
        """Consolida os 3 resultados em um veredicto automatizado"""
        
        # Pesos por confiabilidade
        weights = {
            "FFT": 0.25,
            "NOISE": 0.50,
            "ELA": 0.25
        }
        
        scores = []
        valid_methods = []
        
        if fft.get("status") == "success":
            scores.append(fft["risk_score"] * weights["FFT"])
            valid_methods.append("FFT")
        
        if noise.get("status") == "success":
            scores.append(noise["risk_score"] * weights["NOISE"])
            valid_methods.append("NOISE")
        
        if ela.get("status") == "success":
            scores.append(ela["risk_score"] * weights["ELA"])
            valid_methods.append("ELA")
        
        if scores:
            total_weight = sum(weights[m] for m in valid_methods)
            final_score = sum(scores) / total_weight
        else:
            final_score = 0.0
        
        interpretation = self._interpret_score(final_score)
        evidence = self._extract_evidence(fft, noise, ela, final_score)
        confidence = self._calculate_confidence(valid_methods, final_score)

        # DEBUG FINAL (REMOVE DEPOIS)
        print(f"🎯 CONSOLIDATED RESULT:")
        print(f"   Final Score: {final_score}")
        print(f"   Interpretation: {interpretation}")
        print(f"   Confidence: {confidence}")
        
        return {
            "final_score": round(final_score, 2),
            "interpretation": interpretation,
            "confidence": confidence,
            "methods_used": valid_methods,
            "individual_scores": {
                "fft": round(fft.get("risk_score", 0.0), 2),
                "noise": round(noise.get("risk_score", 0.0), 2),
                "ela": round(ela.get("risk_score", 0.0), 2)
            },
            "key_evidence": evidence,
            "recommendation": self._generate_recommendation(final_score, confidence)
        }
        
    def _interpret_score(self, score: float) -> str:
        """Interpreta o score final"""
        print(f"🔍🔍🔍 DEBUG: Score recebido = {score:.4f}")
        
        if score < 0.15:
            interpretation = "Muito provavelmente REAL"
        elif score < 0.35:
            interpretation = "Provavelmente REAL"
        elif score < 0.45:  # ← BAIXEI DE 0.50 PARA 0.45
            interpretation = "INCONCLUSIVO - Análise manual recomendada"
        elif score < 0.70:
            interpretation = "Provavelmente IA"
        else:
            interpretation = "Muito provavelmente IA"
        
        print(f"🔍🔍🔍 DEBUG: Interpretação = {interpretation}")
        return interpretation
    
    def _calculate_confidence(self, valid_methods: List[str], score: float) -> str:
        """Calcula confiança geral"""
        base_confidence = len(valid_methods) / 3.0
        
        if 0.40 <= score <= 0.60:
            base_confidence *= 0.7
        
        if base_confidence > 0.85:
            return "very_high"
        elif base_confidence > 0.65:
            return "high"
        elif base_confidence > 0.45:
            return "medium"
        elif base_confidence > 0.25:
            return "low"
        else:
            return "very_low"
    
    def _extract_evidence(self, fft: Dict, noise: Dict, ela: Dict, final_score: float) -> List[str]:
        """Extrai evidências-chave"""
        evidence = []
        
        # FFT
        if fft.get("status") == "success" and fft["risk_score"] > 0.5:
            peak_ratio = fft.get("metrics", {}).get("peak_ratio", 0)
            grid_score = fft.get("metrics", {}).get("grid_score", 0)
            evidence.append(f"FFT: Padrão de grid detectado (peak_ratio={peak_ratio:.2f}, grid_score={grid_score:.2f})")
        
        # NOISE
        if noise.get("status") == "success":
            consistency = noise.get("metrics", {}).get("noise_consistency", 0)
            mean_noise = noise.get("metrics", {}).get("mean_noise_level", 0)
            
            if noise["risk_score"] > 0.6:
                evidence.append(f"NOISE: Ruído sintético detectado (consistency={consistency:.2f})")
            elif noise["risk_score"] < 0.15:
                evidence.append(f"NOISE: Ruído natural de sensor confirmado")
        
        # ELA
        if ela.get("status") == "success" and ela["risk_score"] > 0.5:
            mean_error = ela.get("metrics", {}).get("mean_error", 0)
            ela_consistency = ela.get("metrics", {}).get("consistency", 0)
            evidence.append(f"ELA: Uniformidade excessiva (mean_error={mean_error:.2f})")
        
        # Warnings importantes
        for result in [fft, noise, ela]:
            if result.get("status") == "success":
                for warning in result.get("warnings", [])[:2]:
                    if any(kw in warning.lower() for kw in ["ia", "sintético", "artificial", "grid"]):
                        evidence.append(f"{result['method']}: {warning}")
        
        return evidence[:6]
    
    def _generate_recommendation(self, score: float, confidence: str) -> str:
        """Gera recomendação"""
        if score > 0.75 and confidence in ["high", "very_high"]:
            return "🚫 REJEITAR - Alta probabilidade de IA"
        elif score > 0.55:
            return "⚠️ ANÁLISE MANUAL - Evidências ambíguas"
        elif score > 0.35:
            return "✅ APROVAÇÃO CONDICIONAL - Baixo risco"
        else:
            return "✅ APROVAR - Características naturais"
    
    async def _analyze_with_gemini(self, automated: Dict, fft: Dict, noise: Dict, ela: Dict) -> Dict:
        """Envia para Gemini usando nova API"""
        
        prompt = self._build_gemini_prompt(automated, fft, noise, ela)
        
        try:
            print("🤖 Consultando Gemini...")
            
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model='gemini-3-flash-preview',
                contents=prompt
            )
            
            text = response.text.strip()
            verdict = self._parse_gemini_verdict(text)
            
            # Extrair confiança da resposta
            confidence = "medium"
            if "CONFIDENCE:" in text.upper():
                if "HIGH" in text.upper():
                    confidence = "high"
                elif "LOW" in text.upper():
                    confidence = "low"
            
            print(f"✅ Gemini respondeu: {verdict} (confiança: {confidence})")
            
            return {
                "verdict": verdict,
                "full_analysis": text,  # ← TEXTO COMPLETO para mostrar ao usuário
                "explanation": self._extract_explanation(text),  # ← Extrai só a parte "EXPLANATION"
                "confidence": confidence,
                "key_indicators": self._extract_key_indicators(text)  # ← Extrai bullet points
            }
            
        except Exception as e:
            print(f"❌ Gemini Error: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "verdict": "ERROR",
                "full_analysis": f"Erro ao consultar Gemini: {str(e)}",
                "explanation": "Analysis service temporarily unavailable",
                "confidence": "none",
                "key_indicators": []
            }
    
    def _build_gemini_prompt(self, automated: Dict, fft: Dict, noise: Dict, ela: Dict) -> str:
        """Constrói prompt otimizado para Gemini 3"""
        
        # Construir lista de evidências
        evidence_list = "\n".join([f"• {ev}" for ev in automated['key_evidence']]) if automated['key_evidence'] else "• No significant evidence detected"
        
        # Warnings consolidados
        all_warnings = []
        for method_name, result in [("FFT", fft), ("NOISE", noise), ("ELA", ela)]:
            warnings = result.get('warnings', [])
            if warnings:
                all_warnings.extend([f"[{method_name}] {w}" for w in warnings[:2]])
        
        warnings_text = "\n".join([f"• {w}" for w in all_warnings]) if all_warnings else "• No warnings"
        
        prompt = f"""You are an expert in AI-generated image detection using forensic analysis techniques.

        **TASK**: Analyze the forensic data below and provide a clear, accessible verdict for non-technical users (journalists, fact-checkers, general public).

        ---

        ## AUTOMATED ANALYSIS RESULTS

        **Overall Risk Score**: {automated['final_score']:.2f}/1.0
        **Automated Interpretation**: {automated['interpretation']}
        **Analysis Confidence**: {automated['confidence']}

        **Individual Scores**:
        - FFT (Frequency Analysis): {automated['individual_scores']['fft']:.2f}/1.0
        - NOISE (Sensor Pattern): {automated['individual_scores']['noise']:.2f}/1.0
        - ELA (Compression Analysis): {automated['individual_scores']['ela']:.2f}/1.0

        **Key Evidence Detected**:
        {evidence_list}

        **Technical Warnings**:
        {warnings_text}

        ---

        ## YOUR ANALYSIS

        Provide your assessment in the following structured format:

        **VERDICT**: [Choose ONE: REAL | AI-GENERATED | INCONCLUSIVE]

        **CONFIDENCE**: [Choose ONE: HIGH | MEDIUM | LOW]

        **EXPLANATION FOR NON-TECHNICAL USERS** (2-3 sentences in simple language):
        [Explain what the analysis found and why it suggests the image is real/AI-generated/inconclusive. Use analogies if helpful. Avoid jargon.]

        **KEY INDICATORS** (bullet points):
        - [List 2-3 specific findings that support your verdict]
        - [Example: "Noise pattern is too uniform, like a computer-generated texture instead of camera sensor noise"]

        **RECOMMENDATION**:
        [One sentence: Should this image be trusted, flagged for review, or rejected?]

        ---

        **IMPORTANT GUIDELINES**:
        1. Prioritize NOISE analysis (weight: 50%) as it's most reliable for AI detection
        2. If scores are contradictory (e.g., FFT says AI but NOISE says REAL), explain the discrepancy
        3. Be honest about uncertainty - say INCONCLUSIVE if evidence is mixed
        4. Translate technical findings into plain English (e.g., "FFT grid pattern" → "repeating mathematical patterns typical of AI generators")
        5. For scores 0.40-0.60, lean toward INCONCLUSIVE unless evidence is very clear

        Begin your analysis now:"""
        
        return prompt
    
    def _parse_gemini_verdict(self, text: str) -> str:
        """Extrai veredicto do formato estruturado"""
        text_upper = text.upper()
        
        # Buscar na linha VERDICT:
        if "VERDICT:" in text_upper or "VERDICT**:" in text_upper:
            # Pegar linha após VERDICT
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if 'VERDICT' in line.upper():
                    # Pegar o conteúdo da mesma linha ou próxima
                    verdict_text = line.upper()
                    if i + 1 < len(lines):
                        verdict_text += " " + lines[i + 1].upper()
                    
                    # Verificar em ordem de prioridade
                    if "AI-GENERATED" in verdict_text or "AI GENERATED" in verdict_text:
                        return "IA"
                    elif "REAL" in verdict_text and "AI" not in verdict_text:
                        return "REAL"
                    elif "INCONCLUSIVE" in verdict_text:
                        return "INCONCLUSIVO"
                    break
        
        # Fallback: buscar nos primeiros 200 caracteres
        header = text_upper[:200]
        
        if "AI-GENERATED" in header or ("AI" in header and "GENERATED" in header):
            return "IA"
        elif "REAL" in header and "AI" not in header.replace("REAL", ""):
            return "REAL"
        elif "INCONCLUSIVE" in header:
            return "INCONCLUSIVO"
        
        # Se não achou nada claro, marca como inconclusivo
        print(f"⚠️ Não foi possível extrair veredicto claro. Texto: {text[:100]}...")
        return "INCONCLUSIVO"
    
    def _base64_to_pil(self, base64_str: str) -> Image.Image:
        """Converte base64 para PIL"""
        try:
            img_bytes = base64.b64decode(base64_str)
            img = Image.open(io.BytesIO(img_bytes))
            
            # Reduzir tamanho
            if img.width > 768 or img.height > 768:
                img.thumbnail((768, 768), Image.Resampling.LANCZOS)
            
            return img
        except Exception:
            return None
