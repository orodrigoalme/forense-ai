import { AnalysisResult } from './types';

// Using placeholders for base64 images to keep file size manageable for the demo
const PLACEHOLDER_IMAGE = "https://picsum.photos/800/600";
const PLACEHOLDER_NOISE = "https://picsum.photos/800/600?grayscale&blur=2";
const PLACEHOLDER_ELA = "https://picsum.photos/800/600?contrast=2";
const PLACEHOLDER_FFT = "https://picsum.photos/800/600?invert=1";

export const MOCK_RESULT: AnalysisResult = {
  automated_analysis: {
    final_score: 0.82,
    interpretation: "Altamente Provável IA",
    confidence: "high",
    methods_used: ["FFT", "NOISE", "ELA"],
    individual_scores: {
      fft: 0.88,
      noise: 0.82,
      ela: 0.24
    },
    key_evidence: [
      "NOISE: Ruído sintético detectado (consistency=0.85)",
      "FFT: Simetria espectral quase perfeita - altamente artificial",
      "NOISE: Padrão de ruído com consistência moderada-alta",
      "ELA: Nenhuma anomalia significativa detectada (falso negativo comum em modelos high-end)"
    ],
    recommendation: "⚠️ ANÁLISE MANUAL - Fortes indícios de geração sintética"
  },
  gemini_analysis: {
    verdict: "IA",
    full_analysis: "**VERDICT**: AI-GENERATED\n\n**CONFIDENCE**: HIGH\n\n**EXPLANATION**...",
    explanation: "Embora esta imagem possa parecer convincente a olho nu, a análise forense encontrou 'impressões digitais' que não existem no mundo real. Especificamente, a textura da imagem é muito consistente e matematicamente perfeita.",
    confidence: "high",
    key_indicators: [
      "Padrão de ruído uniforme típico de geradores",
      "Ausência de artefatos de compressão JPEG natural",
      "Simetria perfeita no espectro de frequência"
    ]
  },
  annotated_image: PLACEHOLDER_IMAGE,
  details: {
    fft: {
      method: "FFT",
      status: "success",
      image_base64: PLACEHOLDER_FFT,
      risk_score: 0.88,
      metrics: {
        "spectral_uniformity": 1.0,
        "peak_frequency_count": 0,
        "symmetry_score": 0.9999,
        "grid_artifacts": false,
        "dominant_frequency": [-30, 0],
        "high_frequency_energy_ratio": 0.5736
      },
      warnings: [
        "Simetria espectral quase perfeita - altamente artificial",
        "Espectro excessivamente uniforme - falta complexidade orgânica"
      ]
    },
    noise: {
      method: "NOISE",
      status: "success",
      image_base64: PLACEHOLDER_NOISE,
      risk_score: 0.82,
      metrics: {
        "mean_noise_level": 0.3716,
        "noise_consistency": 0.65,
        "regions_with_low_noise": ["skin"],
        "expected_noise_for_iso": 0.135,
        "iso_confidence": "medium-high"
      },
      warnings: [
        "Padrão de ruído com consistência moderada-alta",
        "Textura de pele anormalmente lisa",
        "Ruído sintético detectado - típico de IA adicionando ruído fake"
      ]
    },
    ela: {
      method: "ELA",
      status: "success",
      image_base64: PLACEHOLDER_ELA,
      risk_score: 0.24,
      metrics: {
        "mean_error_level": 0.1836,
        "bright_pixels_percentage": 16.35,
        "error_kurtosis": 4.09,
        "suspicious_low_error_percentage": 6.46
      },
      warnings: [
        "Nenhuma anomalia significativa detectada"
      ]
    }
  }
};