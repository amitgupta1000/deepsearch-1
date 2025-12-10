export interface ResearchRequest {
  query: string;
  prompt_type?: string;
  search_mode?: 'fast' | 'ultra';
  retrieval_method?: 'file_search'; // Only file_search supported now
}

export interface ResearchResult {
  analysis_content: string;
  appendix_content: string;
  analysis_filename: string;
  appendix_filename: string;
  session_id: string; 
  created_at: string;
  completed_at?: string;
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
