export interface ResearchRequest {
  query: string;
  promptType?: string;
  apiKeys: {
    gemini?: string;
    serper?: string;
  };
}

export interface ResearchResult {
  report: string;
  sources: string[];
  wordCount: number;
  citations: string[];
  timestamp: string;
  sessionId?: string; // Optional session ID for server-side downloads
}

export interface ActivityLog {
  id: string;
  timestamp: string;
  level: 'info' | 'success' | 'warning' | 'error';
  message: string;
  details?: any;
}

export interface ResearchState {
  isLoading: boolean;
  result: ResearchResult | null;
  error: string | null;
  logs: ActivityLog[];
  progress: number;
  currentStep: string;
}

export interface Citation {
  id: number;
  url: string;
  title: string;
}

// Authentication types
export interface AuthState {
  isAuthenticated: boolean;
  apiKey: string | null;
  userInfo: UserInfo | null;
  error: string | null;
}

export interface UserInfo {
  user_id: string;
  is_admin: boolean;
  rate_limits: {
    requests_per_hour: number;
    requests_per_minute: number;
  };
  authenticated_at: string;
}

export interface AuthContextType {
  authState: AuthState;
  setApiKey: (apiKey: string) => Promise<boolean>;
  clearAuth: () => void;
  checkAuth: () => Promise<boolean>;
}