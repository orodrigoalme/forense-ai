import { AuthTokens, SessionStats } from '../types';

const BASE_URL = 'https://api.orodrigoalme.com';

const STORAGE_KEYS = {
  TOKENS: 'forense_ai_tokens',
  CUSTOM_KEY: 'forense_ai_gemini_key'
};

class AuthService {
  private tokens: AuthTokens | null = null;
  private customKey: string | null = null;
  private isOfflineMode: boolean = false;

  constructor() {
    this.loadFromStorage();
  }

  private loadFromStorage() {
    try {
      const storedTokens = localStorage.getItem(STORAGE_KEYS.TOKENS);
      const storedKey = localStorage.getItem(STORAGE_KEYS.CUSTOM_KEY);
      
      if (storedTokens) {
        this.tokens = JSON.parse(storedTokens);
      }
      if (storedKey && storedKey.trim() !== '') {
        this.customKey = storedKey.trim();
      }
    } catch (e) {
      console.error("Error loading tokens:", e);
      this.logout();
    }
  }

  private saveTokens(data: any) {
    try {
      const now = Date.now();
      // Use expires_in (seconds) to calculate absolute expiration
      const expiresIn = data.access_expires_in || data.expires_in || 3600;
      
      this.tokens = {
        access_token: data.access_token,
        refresh_token: data.refresh_token,
        expires_in: expiresIn,
        expires_at: now + (expiresIn * 1000)
      };
      localStorage.setItem(STORAGE_KEYS.TOKENS, JSON.stringify(this.tokens));
    } catch (e) {
      console.error("Error saving tokens:", e);
    }
  }

  private mapResponseToStats(data: any): SessionStats {
    // Determine limit type based on local custom key presence or server response
    const isCustomKey = !!this.customKey || data.limit_type === 'custom_key' || data.limit_type === 'server_key';
    
    // Flattened API v2 structure
    if (typeof data.requests_remaining === 'number') {
        return {
            session_id: data.session_id,
            requests_used: data.requests_used,
            requests_remaining: data.requests_remaining,
            quota_used: data.quota_used,
            quota_remaining: data.quota_remaining,
            limit_type: isCustomKey ? 'custom_key' : (data.limit_type || 'anonymous_default'),
            session_age_hours: data.session_age_hours || 0
        };
    }

    // Fallback to 'stats' object
    if (data.stats && typeof data.stats.requests_remaining === 'number') {
        return {
            session_id: data.session_id,
            requests_used: data.stats.requests_used,
            requests_remaining: data.stats.requests_remaining,
            quota_used: data.stats.quota_used,
            quota_remaining: data.stats.quota_remaining,
            limit_type: isCustomKey ? 'custom_key' : (data.limit_type || 'anonymous_default'),
            session_age_hours: data.session_age_hours || 0
        };
    }

    // Legacy fallback
    const limits = data.limits || {};
    const usage = limits.current_usage || { requests_used: 0, quota_used: 0 };
    const maxRequests = 50;
    const maxQuota = 5000;

    return {
      session_id: data.session_id,
      requests_used: usage.requests_used,
      requests_remaining: Math.max(0, maxRequests - usage.requests_used),
      quota_used: usage.quota_used,
      quota_remaining: Math.max(0, maxQuota - usage.quota_used),
      limit_type: isCustomKey ? 'custom_key' : 'anonymous_default',
      session_age_hours: 0
    };
  }

  public setCustomKey(key: string | null) {
    const trimmedKey = key ? key.trim() : null;
    this.customKey = trimmedKey;
    
    if (trimmedKey) {
      localStorage.setItem(STORAGE_KEYS.CUSTOM_KEY, trimmedKey);
    } else {
      localStorage.removeItem(STORAGE_KEYS.CUSTOM_KEY);
    }
  }

  public getCustomKey(): string | null {
    return this.customKey;
  }

  /**
   * Checks if the current token is valid.
   * Logic: Date.now() < expires_at
   * Adds a small buffer (10s) to avoid edge cases.
   */
  public isAuthenticated(): boolean {
    return !!this.tokens && Date.now() < (this.tokens.expires_at - 10000);
  }

  public isOffline(): boolean {
    return this.isOfflineMode;
  }

  private getOfflineStats(): SessionStats {
    return {
      requests_used: 0,
      requests_remaining: 50,
      quota_used: 0,
      quota_remaining: 0,
      limit_type: 'offline_demo'
    };
  }

  /**
   * Main HTTP Client with Auth Injection & Retry Logic
   */
  public async authenticatedFetch(endpoint: string, options: RequestInit = {}): Promise<Response> {
    if (this.isOfflineMode) {
      throw new Error("Network offline: Cannot perform authenticated fetch.");
    }

    // 1. Check/Init Session
    if (!this.isAuthenticated()) {
      try {
        await this.initSession();
      } catch (e) {
        if (this.isOfflineMode) {
           throw new Error("Network offline: Cannot perform authenticated fetch.");
        }
        console.error("Failed to init session before fetch:", e);
        throw new Error("Falha de autenticação. Não foi possível iniciar uma sessão.");
      }
    }

    // 2. Prepare Headers (Auth + Custom Key)
    const authHeaders: Record<string, string> = {
      'Authorization': `Bearer ${this.tokens?.access_token}`,
      'Accept': 'application/json',
    };

    if (this.customKey) {
      authHeaders['X-Gemini-Key'] = this.customKey;
    }

    const config: RequestInit = {
      ...options,
      headers: {
        ...authHeaders,
        ...(options.headers || {})
      }
    };

    try {
      // 3. Request & Retry
      let response = await fetch(`${BASE_URL}${endpoint}`, config);

      if (response.status === 401) {
        console.warn("Token expired (401). Attempting refresh...");
        try {
          await this.refreshToken();
          
          // Update headers with new token
          const newAuthHeaders: Record<string, string> = {
            'Authorization': `Bearer ${this.tokens?.access_token}`,
            'Accept': 'application/json',
          };
          if (this.customKey) {
            newAuthHeaders['X-Gemini-Key'] = this.customKey;
          }
          config.headers = { ...newAuthHeaders, ...(options.headers || {}) };

          response = await fetch(`${BASE_URL}${endpoint}`, config);
        } catch (refreshError) {
          console.error("Token refresh failed:", refreshError);
          this.logout();
          throw new Error("Sessão expirada. Por favor, recarregue a página.");
        }
      }

      return response;
    } catch (error) {
      throw error;
    }
  }

  /**
   * Initializes session based on LocalStorage rules.
   * - If Token valid + Custom Key exists: Returns synthetic stats (No API call).
   * - If Token valid + No Key: Fetches stats (GET).
   * - If Token invalid: Creates new session (POST).
   */
  public async initSession(): Promise<SessionStats> {
    this.isOfflineMode = false;

    try {
      // RULE: If token exists and is valid, DO NOT create a new session.
      if (this.isAuthenticated()) {
         
         // OPTIMIZATION: If using custom key, we don't need real server stats 
         // because UI hides counters. We just need the valid token for analysis.
         if (this.customKey) {
             console.log("Custom Key active + Valid Token. Skipping stats fetch.");
             return {
                 requests_used: 0,
                 requests_remaining: 9999, // Dummy value, UI will hide it
                 quota_used: 0,
                 quota_remaining: 9999,
                 limit_type: 'custom_key',
                 session_id: 'custom-key-session'
             };
         }

         // If anonymous, we need to fetch stats to show the counter
         try {
            return await this.fetchSessionStats();
         } catch (e) {
            // If fetch fails but not because of offline/429, we might need a new session
            const errString = String(e);
            if (errString.includes('429')) {
                this.isOfflineMode = true;
                return this.getOfflineStats();
            }
            // Fallthrough to create new session if GET fails hard
         }
      } else if (this.tokens?.refresh_token) {
        // Try refresh if access token expired but refresh exists
        try {
           await this.refreshToken();
           // Recurse to handle stats logic
           return await this.initSession(); 
        } catch (e) {
           this.logout();
        }
      }

      // RULE: Only create new session if no valid token exists
      return await this.createAnonymousSession();

    } catch (error) {
       // ... existing error handling ...
       const errString = String(error);
       if (errString.includes('429')) {
         this.isOfflineMode = true;
         return this.getOfflineStats();
       }
       if (error instanceof TypeError || errString.includes('Failed to fetch')) {
         this.isOfflineMode = true;
         return this.getOfflineStats();
       }
       
       this.isOfflineMode = true;
       return this.getOfflineStats();
    }
  }

  public async createAnonymousSession(): Promise<SessionStats> {
    try {
      const headers: Record<string, string> = { 
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      };

      if (this.customKey) {
        headers['X-Gemini-Key'] = this.customKey;
      }

      const response = await fetch(`${BASE_URL}/api/auth/anonymous`, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify({})
      });

      if (!response.ok) {
        throw new Error(`Auth Error: ${response.status}`);
      }

      const data = await response.json();
      this.saveTokens(data);
      
      return this.mapResponseToStats(data);
    } catch (error) {
      throw error; 
    }
  }

  public async refreshToken(): Promise<void> {
    if (!this.tokens?.refresh_token) throw new Error('No refresh token available');

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'X-Refresh-Token': this.tokens.refresh_token
    };

    try {
      const response = await fetch(`${BASE_URL}/api/auth/refresh`, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify({})
      });

      if (!response.ok) {
        throw new Error(`Refresh failed: ${response.status}`);
      }

      const data = await response.json();
      this.saveTokens(data);
    } catch (error) {
      this.logout();
      throw error;
    }
  }

  public async fetchSessionStats(): Promise<SessionStats> {
    if (this.isOfflineMode) return this.getOfflineStats();

    // Optimization: If Custom Key, no need to fetch real stats
    if (this.customKey) {
         return {
             requests_used: 0,
             requests_remaining: 9999,
             quota_used: 0,
             quota_remaining: 9999,
             limit_type: 'custom_key',
             session_id: 'custom-key-session'
         };
    }

    try {
      const response = await this.authenticatedFetch('/api/auth/session', {
        method: 'GET',
        headers: { 'Cache-Control': 'no-cache', 'Pragma': 'no-cache' }
      });

      if (!response.ok) throw new Error(`Session fetch failed: ${response.status}`);

      const data = await response.json();
      return this.mapResponseToStats(data);
    } catch (error) {
      if (error instanceof TypeError || (error as Error).message.includes('Failed to fetch')) {
          this.isOfflineMode = true;
          return this.getOfflineStats();
      }
      throw error;
    }
  }

  public async deleteSession(): Promise<void> {
    if (!this.isOfflineMode && this.isAuthenticated()) {
      try {
        await this.authenticatedFetch('/api/auth/session', { method: 'DELETE' });
      } catch (e) {
        console.warn("Error deleting session remote:", e);
      }
    }
    this.logout();
  }

  public logout() {
    this.tokens = null;
    localStorage.removeItem(STORAGE_KEYS.TOKENS);
  }
}

export const authService = new AuthService();