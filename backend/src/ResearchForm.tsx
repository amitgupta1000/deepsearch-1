import React, { useState } from 'react';
import { useResearch } from '../context/ResearchContext'; // Assuming you have a context like this

type SearchMode = 'fast' | 'ultra';

const ResearchForm: React.FC = () => {
  const [query, setQuery] = useState('');
  const [searchMode, setSearchMode] = useState<SearchMode>('fast');
  const { startResearch, isLoading } = useResearch(); // Using your context hook

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;

    // Call the startResearch function from your context
    startResearch({ query, search_mode: searchMode });
  };

  return (
    <div id="research" className="card p-6 md:p-8">
      <h2 className="text-2xl md:text-3xl font-bold gradient-text mb-4">
        Start Your Research
      </h2>
      <p className="text-gray-600 mb-6">
        Enter a topic, question, or keyword to begin the automated research process.
      </p>
      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label htmlFor="query" className="sr-only">
            Research Query
          </label>
          <textarea
            id="query"
            rows={3}
            className="input-field"
            placeholder="e.g., 'What are the latest advancements in AI-powered drug discovery?'"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={isLoading}
          />
        </div>

        {/* Fast/Ultra Search Toggle */}
        <div className="flex flex-col sm:flex-row justify-between items-center gap-4">
          <div className="flex items-center gap-4">
            <span className="font-medium text-gray-700">Search Depth:</span>
            <div className="flex items-center rounded-full bg-gray-200 p-1">
              <button
                type="button"
                onClick={() => setSearchMode('fast')}
                className={`px-4 py-1 text-sm font-semibold rounded-full transition-colors ${
                  searchMode === 'fast'
                    ? 'bg-primary-600 text-white shadow'
                    : 'text-gray-600 hover:bg-gray-300'
                }`}
              >
                Fast
              </button>
              <button
                type="button"
                onClick={() => setSearchMode('ultra')}
                className={`px-4 py-1 text-sm font-semibold rounded-full transition-colors ${
                  searchMode === 'ultra'
                    ? 'bg-primary-600 text-white shadow'
                    : 'text-gray-600 hover:bg-gray-300'
                }`}
              >
                Ultra
              </button>
            </div>
          </div>

          <button
            type="submit"
            className="btn-primary w-full sm:w-auto disabled:bg-gray-400 disabled:cursor-not-allowed"
            disabled={isLoading || !query.trim()}
          >
            {isLoading ? (
              <>
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Researching...
              </>
            ) : (
              'Start Research'
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

export default ResearchForm;