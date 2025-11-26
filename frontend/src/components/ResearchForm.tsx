import React, { useState } from 'react';
import { MagnifyingGlassIcon } from '@heroicons/react/24/outline';
import { useResearch } from '../context/ResearchContext';
import type { ResearchRequest } from '../types';

const ResearchForm: React.FC = () => {
  const { startResearch, state } = useResearch();
  const [query, setQuery] = useState('');
  const [promptType, setPromptType] = useState('general');
  const [searchMode, setSearchMode] = useState('fast'); // Add searchMode state

  const promptTypes = [
    { value: 'general', label: 'General Research', description: 'Broad research across multiple topics and sources' },
    { value: 'legal', label: 'Legal Research', description: 'Legal analysis and regulatory information' },
    { value: 'macro', label: 'Macro Analysis', description: 'Economic and market trends analysis' },
    { value: 'deepsearch', label: 'Deep Search', description: 'Comprehensive in-depth research' },
    { value: 'person_search', label: 'Person Search', description: 'Research about specific individuals' },
    { value: 'investment', label: 'Investment Research', description: 'Financial and investment analysis' }
  ];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    const request: ResearchRequest = {
      query: query.trim(),
      promptType,
      search_mode: searchMode as 'fast' | 'ultra', // Include search_mode
    };

    await startResearch(request);
  };

  const exampleQueries = [
    "Latest developments in artificial intelligence and machine learning",
    "Impact of climate change on global agriculture",
    "Cryptocurrency market trends and future predictions",
    "Benefits and risks of remote work in 2025"
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center space-y-4">
        <h2 className="text-2xl font-bold text-gray-900 flex items-center justify-center space-x-3">
          <MagnifyingGlassIcon className="w-6 h-6 text-primary-600" />
          <span>Start Your Research</span>
        </h2>
        <p className="text-gray-600">
          Enter your research question and choose your research objective for tailored analysis
        </p>
      </div>

      <div className="card p-8 shadow-lg">
        <form onSubmit={handleSubmit} className="space-y-8">
          {/* Query Input */}
          <div className="space-y-4">
            <label htmlFor="query" className="block text-lg font-semibold text-gray-800">
              What would you like to research?
            </label>
            <div className="relative">
              <textarea
                id="query"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Enter your research question or topic..."
                className="input-field h-32 resize-none text-lg"
                required
                disabled={state.isLoading}
              />
              <div className="absolute bottom-3 right-3 text-sm text-gray-400">
                {query.length}/500
              </div>
            </div>
            
            {/* Example Queries */}
            <div className="space-y-2">
              <p className="text-sm font-medium text-gray-600">Try these examples:</p>
              <div className="flex flex-wrap gap-2">
                {exampleQueries.map((example, index) => (
                  <button
                    key={index}
                    type="button"
                    onClick={() => setQuery(example)}
                    className="text-xs bg-gray-100 hover:bg-primary-50 hover:text-primary-700 px-3 py-1 rounded-full transition-colors"
                    disabled={state.isLoading}
                  >
                    {example.length > 50 ? `${example.substring(0, 50)}...` : example}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Research Objective Selection */}
          <div className="space-y-4">
            <label className="block text-lg font-semibold text-gray-800">
              Choose Research Objective
            </label>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {promptTypes.map((type) => (
                <label key={type.value} className="cursor-pointer">
                  <input
                    type="radio"
                    value={type.value}
                    checked={promptType === type.value}
                    onChange={(e) => setPromptType(e.target.value)}
                    className="sr-only"
                    disabled={state.isLoading}
                  />
                  <div className={`p-4 rounded-lg border-2 transition-all duration-200 ${
                    promptType === type.value
                      ? 'border-primary-500 bg-primary-50 shadow-md'
                      : 'border-gray-300 bg-white hover:border-gray-400 hover:shadow-sm'
                  }`}>
                    <div className="font-semibold text-base mb-1">{type.label}</div>
                    <div className="text-sm text-gray-600">{type.description}</div>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Search Options & Submit Button */}
          <div className="flex flex-col sm:flex-row justify-between items-center gap-4 pt-4 border-t border-gray-200">
            <div className="flex items-center gap-4">
              <span className="font-medium text-gray-700">Search Depth:</span>
              <div className="flex items-center rounded-full bg-gray-200 p-1">
                <button
                  type="button"
                  onClick={() => setSearchMode('fast')}
                  disabled={state.isLoading}
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
                  disabled={state.isLoading}
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
              disabled={!query.trim() || state.isLoading}
              className="btn-primary w-full sm:w-auto py-3 px-6 text-base font-semibold disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-3 shadow-lg hover:shadow-xl transition-all duration-200"
            >
              <MagnifyingGlassIcon className="w-5 h-5" />
              <span>{state.isLoading ? 'Researching...' : 'Start Research'}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ResearchForm;
