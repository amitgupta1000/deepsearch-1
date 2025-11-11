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

