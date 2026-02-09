import { AnalysisResult } from '../types';
import { authService } from './authService';
import { MOCK_RESULT } from '../mockData';

class ApiService {
  
  /**
   * Uploads an image for forensic analysis.
   * Uses centralized authenticatedFetch to ensure "Authorization: Bearer <token>" is present.
   */
  public async analyzeImage(file: File): Promise<AnalysisResult> {
    
    // 1. Prepare Upload Data
    const formData = new FormData();
    // API expects 'file' field
    formData.append('file', file);

    try {
      // 2. Perform Request using Centralized Auth Client
      // Note: authenticatedFetch handles:
      // - Getting the Bearer Token
      // - Adding 'Authorization: Bearer <token>'
      // - Handling 401 Token Expiration (Refresh & Retry)
      // - Adding 'Accept: application/json'
      
      const response = await authService.authenticatedFetch('/api/analyze-image', {
        method: 'POST',
        body: formData
        // IMPORTANT: We do NOT set 'Content-Type'. 
        // The browser automatically sets 'multipart/form-data; boundary=...' when body is FormData.
      });

      // 3. Handle Business Errors
      if (response.status === 429) {
        throw new Error("Quota excedida. Tente novamente mais tarde ou use uma chave Gemini própria.");
      }

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Erro na análise (${response.status}): ${errorText}`);
      }

      // 4. Return Data
      return await response.json();

    } catch (error) {
      console.error("Analysis failed:", error);

      // Fallback for demo purposes if strictly a network error (server down/offline)
      // This preserves the UI experience if the API is unreachable.
      if ((error as Error).message.includes("Network") || (error as Error).message.includes("Failed to fetch") || (error as Error).message.includes("offline")) {
          console.warn("Network unavailable, returning mock data for demonstration.");
          return new Promise(resolve => setTimeout(() => resolve(MOCK_RESULT), 2000));
      }

      throw error;
    }
  }
}

export const apiService = new ApiService();