import React, { createContext, useContext, useReducer } from 'react';
import type { ReactNode } from 'react';
import type { ResearchState, ResearchRequest, ResearchResult } from '../types';

type ResearchAction =
  | { type: 'START_RESEARCH' }
  | { type: 'RESEARCH_SUCCESS'; payload: ResearchResult }
  | { type: 'RESEARCH_ERROR'; payload: string }
  | { type: 'CLEAR_RESULTS' }
  | { type: 'UPDATE_PROGRESS'; payload: { progress: number; currentStep: string } };

interface ResearchContextType {
  state: ResearchState;
  startResearch: (request: ResearchRequest) => Promise<void>;
  clearResults: () => void;
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
      return { ...initialState, isLoading: true, currentStep: 'Starting research...' };
    case 'RESEARCH_SUCCESS':
      return { ...state, isLoading: false, result: action.payload, progress: 100, currentStep: 'Research completed' };
    case 'RESEARCH_ERROR':
      return { ...state, isLoading: false, error: action.payload, currentStep: 'Research failed' };
    case 'CLEAR_RESULTS':
      return initialState;
    case 'UPDATE_PROGRESS':
      return { ...state, progress: action.payload.progress, currentStep: action.payload.currentStep };
    default:
      return state;
  }
};

const ResearchContext = createContext<ResearchContextType | undefined>(undefined);

export const ResearchProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(researchReducer, initialState);

  const startResearch = async (request: ResearchRequest) => {
    dispatch({ type: 'START_RESEARCH' });

    try {
      const apiUrl = import.meta.env.VITE_API_URL || '';

      const startResponse = await fetch(`${apiUrl}/api/research`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      });

      if (!startResponse.ok) {
        const errorData = await startResponse.json();
        throw new Error(errorData.detail || 'Failed to start research');
      }

      const startResult = await startResponse.json();
      const sessionId = startResult.session_id;

      const pollForResult = async (): Promise<ResearchResult> => {
        while (true) {
          const statusResponse = await fetch(`${apiUrl}/api/research/${sessionId}/status`);
          if (!statusResponse.ok) {
            throw new Error('Failed to get research status');
          }

          const status = await statusResponse.json();
          dispatch({ type: 'UPDATE_PROGRESS', payload: { progress: status.progress, currentStep: status.current_step } });

          if (status.status === 'completed') {
            const resultResponse = await fetch(`${apiUrl}/api/research/${sessionId}/result`);
            if (!resultResponse.ok) {
              throw new Error('Failed to get research result');
            }
            return await resultResponse.json();
          } else if (status.status === 'failed') {
            throw new Error(status.error_message || 'Research failed without a specific error message.');
          }

          await new Promise(resolve => setTimeout(resolve, 2000));
        }
      };

      const result = await pollForResult();
      dispatch({ type: 'RESEARCH_SUCCESS', payload: result });

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'An unknown error occurred';
      dispatch({ type: 'RESEARCH_ERROR', payload: errorMessage });
    }
  };

  const clearResults = () => {
    dispatch({ type: 'CLEAR_RESULTS' });
  };

  return (
    <ResearchContext.Provider value={{ state, startResearch, clearResults }}>
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