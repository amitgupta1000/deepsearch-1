import React, { createContext, useContext, useReducer } from 'react';
import type { ReactNode } from 'react';
import type { ResearchState, ResearchRequest, ResearchResult } from '../types';


type ActivityLog = {
  id: string;
  timestamp: string;
  level: 'info' | 'success' | 'warning' | 'error';
  message: string;
  details?: any;
};

type ResearchAction =
  | { type: 'START_RESEARCH' }
  | { type: 'RESEARCH_SUCCESS'; payload: ResearchResult }
  | { type: 'RESEARCH_ERROR'; payload: string }
  | { type: 'CLEAR_RESULTS' }
  | { type: 'ADD_LOG'; payload: ActivityLog }
  | { type: 'CLEAR_LOGS' }
  | { type: 'UPDATE_PROGRESS'; payload: { progress: number; currentStep: string } };

interface ResearchContextType {
  state: ResearchState;
  startResearch: (request: ResearchRequest) => Promise<void>;
  clearResults: () => void;
  clearLogs: () => void;
}

const initialState: ResearchState = {
  isLoading: false,
  result: null,
  error: null,
  logs: [],
  progress: 0,
  currentStep: '',
};

const researchReducer = (state: ResearchState, action: ResearchAction): ResearchState => {
  switch (action.type) {
    case 'START_RESEARCH':
      return { ...state, isLoading: true, error: null, result: null, logs: [], progress: 0, currentStep: 'Starting research...' };
    case 'RESEARCH_SUCCESS':
      return { ...state, isLoading: false, result: action.payload, error: null, progress: 100, currentStep: 'Research completed successfully!' };
    case 'RESEARCH_ERROR':
      return { ...state, isLoading: false, error: action.payload, progress: 0, currentStep: 'Research failed' };
    case 'CLEAR_RESULTS':
      return { ...state, result: null, error: null, logs: [], progress: 0, currentStep: '' };
    case 'ADD_LOG':
      return { ...state, logs: [...state.logs, action.payload] };
    case 'CLEAR_LOGS':
      return { ...state, logs: [] };
    case 'UPDATE_PROGRESS':
      return { ...state, progress: action.payload.progress, currentStep: action.payload.currentStep };
    default:
      return state;
  }
};

const ResearchContext = createContext<ResearchContextType | undefined>(undefined);

export const ResearchProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(researchReducer, initialState);
  // Direct fetch, no authentication required

  const addLog = (level: ActivityLog['level'], message: string, details?: any) => {
    const log: ActivityLog = {
      id: Date.now().toString(),
      timestamp: new Date().toLocaleTimeString(),
      level,
      message,
      details
    };
    dispatch({ type: 'ADD_LOG', payload: log });
    console.log(`[${level.toUpperCase()}] ${message}`, details || '');
  };

  const updateProgress = (progress: number, currentStep: string) => {
    dispatch({ type: 'UPDATE_PROGRESS', payload: { progress, currentStep } });
  };

  const startResearch = async (request: ResearchRequest) => {
    dispatch({ type: 'START_RESEARCH' });
    addLog('info', 'Research session started', { query: request.query, prompt_type: request.prompt_type });
    
    try {
      const apiUrl = import.meta.env.VITE_API_URL;
      addLog('info', `Connecting to API: ${apiUrl}`);
      updateProgress(5, 'Connecting to research backend...');
      // Health check first
      try {
        addLog('info', 'Performing health check...');
        const healthResponse = await fetch(`${apiUrl}/api/health`);
        const healthData = await healthResponse.json();
        addLog('success', 'Backend is healthy', healthData);
      } catch (healthError) {
        addLog('warning', 'Health check failed, proceeding anyway', healthError);
      }
      // Step 1: Start the research
      updateProgress(10, 'Initializing research pipeline...');
      addLog('info', 'Sending research request to backend...');
      const startResponse = await fetch(`${apiUrl}/api/research`, {
        method: 'POST',
        body: JSON.stringify({
          query: request.query,
          prompt_type: request.prompt_type || 'general',
          search_mode: request.search_mode || 'fast',
          retrieval_method: 'file_search'
        }),
        headers: {
          'Content-Type': 'application/json'
        }
      });

      addLog('info', `Research start response: ${startResponse.status}`);
      
      if (!startResponse.ok) {
        const errorText = await startResponse.text();
        addLog('error', `Failed to start research: ${startResponse.status}`, errorText);
        throw new Error(`Failed to start research: ${startResponse.status} - ${errorText}`);
      }

      const startResult = await startResponse.json();
      addLog('success', 'Research request accepted', startResult);
      
      if (!startResult.session_id) {
        addLog('error', 'Invalid server response', startResult);
        throw new Error('Invalid response from server: missing session_id');
      }
      
      const sessionId = startResult.session_id;
      const createdAt = startResult.created_at || new Date().toISOString();
      addLog('info', `Research session created: ${sessionId}`);
      updateProgress(15, 'Research pipeline started, monitoring progress...');

      // Step 2: Poll for completion
      const pollForResult = async (): Promise<ResearchResult> => {
        let attempts = 0;
        const maxAttempts = 150; // 5 minutes max (150 * 2 seconds)
        
        while (attempts < maxAttempts) {
          attempts++;
          const progressPercent = Math.min(15 + (attempts / maxAttempts) * 75, 90);
          
          addLog('info', `Checking research progress (${attempts}/${maxAttempts})`);
          updateProgress(progressPercent, 'Research in progress, analyzing sources...');
          
          const statusResponse = await fetch(`${apiUrl}/api/research/${sessionId}/status`);
          
          if (!statusResponse.ok) {
            addLog('error', `Failed to check status: ${statusResponse.status}`);
            throw new Error(`Failed to check research status: ${statusResponse.status}`);
          }

          const status = await statusResponse.json();
          addLog('info', `Research status: ${status.status}`, { 
            progress: status.progress, 
            step: status.current_step 
          });
          
          if (status.current_step) {
            updateProgress(progressPercent, status.current_step);
          }
          
          if (status.status === 'completed') {
            addLog('success', 'Research completed! Fetching final report...');
            updateProgress(95, 'Retrieving final report...');
            
            // Get the final result
            const resultResponse = await fetch(`${apiUrl}/api/research/${sessionId}/result`);
            if (!resultResponse.ok) {
              addLog('error', `Failed to get result: ${resultResponse.status}`);
              throw new Error(`Failed to get research result: ${resultResponse.status}`);
            }
            
            const result = await resultResponse.json();
            addLog('success', 'Research report retrieved successfully', {
              wordCount: result.analysis_content?.split(' ').length || 0
            });

            // Map backend fields to frontend expected structure
            return {
              analysis_content: result.analysis_content || 'No analysis content available',
              appendix_content: result.appendix_content || 'No appendix content available',
              analysis_filename: result.analysis_filename || '',
              appendix_filename: result.appendix_filename || '',
              session_id: sessionId,
              created_at: createdAt,
              completed_at: result.completed_at || new Date().toISOString()
            };
          } else if (status.status === 'failed') {
            addLog('error', 'Research failed', status.error_message);
            throw new Error(`Research failed: ${status.error_message || 'Unknown error'}`);
          }
          
          // Wait 2 seconds before polling again
          await new Promise(resolve => setTimeout(resolve, 2000));
        }
        
        addLog('error', 'Research timed out after 5 minutes');
        throw new Error('Research timed out after 5 minutes');
      };

      const result = await pollForResult();
      addLog('success', 'Research completed successfully!');
      dispatch({ type: 'RESEARCH_SUCCESS', payload: result });
      
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
      addLog('error', 'Research failed', error);
      dispatch({ 
        type: 'RESEARCH_ERROR', 
        payload: errorMessage
      });
    }
  };

  const clearResults = () => {
    dispatch({ type: 'CLEAR_RESULTS' });
  };

  const clearLogs = () => {
    dispatch({ type: 'CLEAR_LOGS' });
  };

  return (
    <ResearchContext.Provider value={{ state, startResearch, clearResults, clearLogs }}>
      {children}
    </ResearchContext.Provider>
  );
};

export const useResearch = () => {
  const context = useContext(ResearchContext);
  if (context === undefined) {
    throw new Error('useResearch must be used within a ResearchProvider');
  }
  return context;
};
