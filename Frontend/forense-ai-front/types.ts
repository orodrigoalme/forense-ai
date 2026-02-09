
export type VerdictType = "REAL" | "IA" | "INCONCLUSIVO";
export type ConfidenceLevel = "very_high" | "high" | "medium" | "low" | "very_low";

// Auth & Session Types
export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  expires_in: number; // seconds
  expires_at: number; // timestamp ms
}

export interface SessionStats {
  session_id?: string;
  requests_used: number;
  requests_remaining: number;
  quota_used: number;
  quota_remaining: number;
  limit_type: 'anonymous_default' | 'server_key' | 'custom_key' | 'offline_demo';
  session_age_hours?: number;
}

export interface AutomatedAnalysis {
  final_score: number;
  interpretation: string;
  confidence: ConfidenceLevel;
  methods_used: string[];
  individual_scores: {
    fft: number;
    noise: number;
    ela: number;
  };
  key_evidence: string[];
  recommendation: string;
}

export interface GeminiAnalysis {
  verdict: VerdictType;
  full_analysis: string;
  explanation: string;
  confidence: ConfidenceLevel;
  key_indicators: string[];
}

export interface MethodMetrics {
  [key: string]: number | string | boolean | (number | string)[] | null;
}

export interface MethodDetail {
  method: string;
  status: "success" | "failed";
  image_base64?: string; // Prefix with data:image/png;base64,
  risk_score: number;
  metrics: MethodMetrics;
  warnings: string[];
}

export interface AnalysisDetails {
  fft: MethodDetail;
  noise: MethodDetail;
  ela: MethodDetail;
}

export interface AnalysisResult {
  automated_analysis: AutomatedAnalysis;
  gemini_analysis: GeminiAnalysis;
  annotated_image: string; // Base64
  details: AnalysisDetails;
}
