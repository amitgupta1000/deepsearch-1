import React, { useState, useEffect, FC } from 'react';

// Read API URL from environment variables, with a fallback for development
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// --- Type Definitions ---

type SearchMode = 'fast' | 'ultra';

interface ResearchStatus {
    status: 'pending' | 'running' | 'completed' | 'failed';
    progress: number;
    current_step: string;
}

interface ResearchResult {
    analysis_content: string;
    appendix_content: string;
}

const ResearchComponent: FC = () => {
    const [query, setQuery] = useState<string>('');
    const [searchMode, setSearchMode] = useState<SearchMode>('fast');
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [status, setStatus] = useState<ResearchStatus | null>(null);
    const [result, setResult] = useState<ResearchResult | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [error, setError] = useState<string>('');

    // Poll for status updates when a session is active
    useEffect(() => {
        if (sessionId && (status?.status === 'running' || status?.status === 'pending')) {
            const interval = setInterval(async () => {
                try {
                    const response = await fetch(`${API_BASE_URL}/api/research/${sessionId}/status`);
                    if (!response.ok) {
                        throw new Error('Failed to fetch status');
                    }
                    const data: ResearchStatus = await response.json();
                    setStatus(data);

                    if (data.status === 'completed' || data.status === 'failed') {
                        clearInterval(interval);
                        setIsLoading(false);
                        if (data.status === 'completed') {
                            fetchResult(sessionId);
                        } else {
                            setError(`Research failed: ${data.current_step}`);
                        }
                    }
                } catch (err: any) {
                    setError(err.message);
                    setIsLoading(false);
                    clearInterval(interval);
                }
            }, 3000); // Poll every 3 seconds

            return () => clearInterval(interval); // Cleanup on component unmount
        }
    }, [sessionId, status]);

    const handleSearch = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        if (!query.trim()) {
            setError('Please enter a research query.');
            return;
        }

        setIsLoading(true);
        setError('');
        setResult(null);
        setStatus(null);
        setSessionId(null);

        try {
            const response = await fetch(`${API_BASE_URL}/api/research`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query,
                    search_mode: searchMode, // Pass the selected search mode
                    prompt_type: 'general', // Or make this selectable
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to start research');
            }

            const data: { session_id: string } = await response.json();
            setSessionId(data.session_id);
            setStatus({ status: 'pending', progress: 0, current_step: 'Queued...' });
        } catch (err: any) {
            setError(err.message);
            setIsLoading(false);
        }
    };

    const fetchResult = async (sId: string) => {
        try {
            const response = await fetch(`${API_BASE_URL}/api/research/${sId}/result`);
            if (!response.ok) {
                throw new Error('Failed to fetch result');
            }
            const data: ResearchResult = await response.json();
            setResult(data);
        } catch (err: any) {
            setError(err.message);
        }
    };

    return (
        <div className="research-container">
            <h1>INTELLISEARCH</h1>
            <form onSubmit={handleSearch}>
                <input
                    type="text"
                    value={query}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => setQuery(e.target.value)}
                    placeholder="Enter your research query..."
                    className="search-input"
                    disabled={isLoading}
                />
                <div className="search-options">
                    <label>Search Mode:</label>
                    <div className="toggle-switch">
                        <input 
                            type="radio" 
                            id="fast" 
                            name="searchMode" 
                            value="fast" 
                            checked={searchMode === 'fast'} 
                            onChange={() => setSearchMode('fast')} 
                        />
                        <label htmlFor="fast">Fast</label>
                        <input 
                            type="radio" 
                            id="ultra" 
                            name="searchMode" 
                            value="ultra" 
                            checked={searchMode === 'ultra'} 
                            onChange={() => setSearchMode('ultra')} 
                        />
                        <label htmlFor="ultra">Ultra</label>
                    </div>
                </div>
                <button type="submit" disabled={isLoading} className="search-button">
                    {isLoading ? 'Researching...' : 'Start Research'}
                </button>
            </form>

            {error && <p className="error-message">{error}</p>}

            {isLoading && status && (
                <div className="progress-container">
                    <h3>Progress</h3>
                    <progress value={status.progress} max="100"></progress>
                    <p>{status.progress}% - {status.current_step}</p>
                </div>
            )}

            {result && (
                <div className="result-container">
                    <h2>Research Complete</h2>
                    <h3>Analysis</h3>
                    <pre className="result-content">
                        {result.analysis_content}
                    </pre>
                    <h3>Appendix</h3>
                    <pre className="result-content">
                        {result.appendix_content}
                    </pre>
                </div>
            )}
        </div>
    );
};

export default ResearchComponent;