import React, { useState } from 'react';
import { MagnifyingGlassIcon, Cog6ToothIcon, KeyIcon, SparklesIcon, DocumentTextIcon, BoltIcon } from '@heroicons/react/24/outline';
import { useResearch } from '../context/ResearchContext';
import type { ResearchRequest } from '../types';

const ResearchForm: React.FC = () => {
  const { startResearch, state } = useResearch();
  const [query, setQuery] = useState('');
  const [reportType, setReportType] = useState<'concise' | 'detailed'>('concise');
  const [promptType, setPromptType] = useState('general');
  const [reasoningMode, setReasoningMode] = useState(true);
  const [showApiKeys, setShowApiKeys] = useState(false);
  const [apiKeys, setApiKeys] = useState({
    gemini: '',
    serper: '',
  });

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
      reportType,
      promptType,
      reasoningMode,
      apiKeys,
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
          <SparklesIcon className="w-6 h-6 text-primary-600" />
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

          {/* Report Type Selection */}
          <div className="space-y-4">
            <label className="block text-lg font-semibold text-gray-800">
              Choose Report Type
            </label>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <label className="cursor-pointer">
                <input
                  type="radio"
                  value="concise"
                  checked={reportType === 'concise'}
                  onChange={(e) => setReportType(e.target.value as 'concise' | 'detailed')}
                  className="sr-only"
                  disabled={state.isLoading}
                />
                <div className={`p-6 rounded-xl border-2 transition-all duration-200 ${
                  reportType === 'concise'
                    ? 'border-primary-500 bg-primary-50 shadow-md'
                    : 'border-gray-300 bg-white hover:border-gray-400 hover:shadow-sm'
                }`}>
                  <div className="flex items-center space-x-3 mb-3">
                    <BoltIcon className={`w-5 h-5 ${reportType === 'concise' ? 'text-primary-600' : 'text-gray-500'}`} />
                    <div className="font-semibold text-lg">Concise Report</div>
                  </div>
                  <div className="text-sm text-gray-600 space-y-1">
                    <div>• ~500 words</div>
                    <div>• Quick insights</div>
                    <div>• Key findings only</div>
                    <div>• 2-3 minutes read</div>
                  </div>
                </div>
              </label>
              
              <label className="cursor-pointer">
                <input
                  type="radio"
                  value="detailed"
                  checked={reportType === 'detailed'}
                  onChange={(e) => setReportType(e.target.value as 'concise' | 'detailed')}
                  className="sr-only"
                  disabled={state.isLoading}
                />
                <div className={`p-6 rounded-xl border-2 transition-all duration-200 ${
                  reportType === 'detailed'
                    ? 'border-primary-500 bg-primary-50 shadow-md'
                    : 'border-gray-300 bg-white hover:border-gray-400 hover:shadow-sm'
                }`}>
                  <div className="flex items-center space-x-3 mb-3">
                    <DocumentTextIcon className={`w-5 h-5 ${reportType === 'detailed' ? 'text-primary-600' : 'text-gray-500'}`} />
                    <div className="font-semibold text-lg">Detailed Report</div>
                  </div>
                  <div className="text-sm text-gray-600 space-y-1">
                    <div>• ~1000 words</div>
                    <div>• Comprehensive analysis</div>
                    <div>• Multiple perspectives</div>
                    <div>• 5-7 minutes read</div>
                  </div>
                </div>
              </label>
            </div>
          </div>

          {/* Reasoning Mode Selection */}
          <div className="space-y-4">
            <label className="block text-lg font-semibold text-gray-800">
              Choose Analysis Mode
            </label>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <label className="cursor-pointer">
                <input
                  type="radio"
                  value="reasoning"
                  checked={reasoningMode === true}
                  onChange={(e) => setReasoningMode(true)}
                  className="sr-only"
                  disabled={state.isLoading}
                />
                <div className={`p-6 rounded-xl border-2 transition-all duration-200 ${
                  reasoningMode === true
                    ? 'border-primary-500 bg-primary-50 shadow-md'
                    : 'border-gray-300 bg-white hover:border-gray-400 hover:shadow-sm'
                }`}>
                  <div className="flex items-center space-x-3 mb-3">
                    <SparklesIcon className={`w-5 h-5 ${reasoningMode === true ? 'text-primary-600' : 'text-gray-500'}`} />
                    <div className="font-semibold text-lg">Reasoning Mode</div>
                  </div>
                  <div className="text-sm text-gray-600 space-y-1">
                    <div>• AI provides opinions & analysis</div>
                    <div>• Interpretive insights</div>
                    <div>• Expert conclusions</div>
                    <div>• "What this means" perspective</div>
                  </div>
                </div>
              </label>
              
              <label className="cursor-pointer">
                <input
                  type="radio"
                  value="research"
                  checked={reasoningMode === false}
                  onChange={(e) => setReasoningMode(false)}
                  className="sr-only"
                  disabled={state.isLoading}
                />
                <div className={`p-6 rounded-xl border-2 transition-all duration-200 ${
                  reasoningMode === false
                    ? 'border-primary-500 bg-primary-50 shadow-md'
                    : 'border-gray-300 bg-white hover:border-gray-400 hover:shadow-sm'
                }`}>
                  <div className="flex items-center space-x-3 mb-3">
                    <DocumentTextIcon className={`w-5 h-5 ${reasoningMode === false ? 'text-primary-600' : 'text-gray-500'}`} />
                    <div className="font-semibold text-lg">Research Mode</div>
                  </div>
                  <div className="text-sm text-gray-600 space-y-1">
                    <div>• No opinions or analysis</div>
                    <div>• Pure facts and data only</div>
                    <div>• Objective information</div>
                    <div>• "Just the facts" approach</div>
                  </div>
                </div>
              </label>
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

          {/* API Keys Section */}
          <div className="border-t pt-6">
            <button
              type="button"
              onClick={() => setShowApiKeys(!showApiKeys)}
              className="flex items-center space-x-2 text-gray-600 hover:text-gray-900 transition-colors"
              disabled={state.isLoading}
            >
              <Cog6ToothIcon className="w-4 h-4" />
              <span>Advanced Settings</span>
              <span className="text-xs bg-gray-100 px-2 py-1 rounded">Optional</span>
            </button>
            
            {showApiKeys && (
              <div className="mt-4 p-4 bg-gray-50 rounded-lg space-y-4">
                <div className="flex items-center space-x-2 mb-3">
                  <KeyIcon className="w-4 h-4 text-gray-500" />
                  <span className="font-medium text-gray-700">Custom API Keys</span>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label htmlFor="gemini-key" className="block text-sm font-medium text-gray-700 mb-1">
                      Google Gemini API Key
                    </label>
                    <input
                      id="gemini-key"
                      type="password"
                      value={apiKeys.gemini}
                      onChange={(e) => setApiKeys(prev => ({ ...prev, gemini: e.target.value }))}
                      placeholder="Enter Gemini API key..."
                      className="input-field"
                      disabled={state.isLoading}
                    />
                  </div>
                  <div>
                    <label htmlFor="serper-key" className="block text-sm font-medium text-gray-700 mb-1">
                      Serper API Key
                    </label>
                    <input
                      id="serper-key"
                      type="password"
                      value={apiKeys.serper}
                      onChange={(e) => setApiKeys(prev => ({ ...prev, serper: e.target.value }))}
                      placeholder="Enter Serper API key..."
                      className="input-field"
                      disabled={state.isLoading}
                    />
                  </div>
                </div>
                <p className="text-xs text-gray-500">
                  Leave empty to use default API keys configured on the server.
                </p>
              </div>
            )}
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            disabled={!query.trim() || state.isLoading}
            className="btn-primary w-full py-4 text-lg font-semibold disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-3 shadow-lg hover:shadow-xl transition-all duration-200"
          >
            <MagnifyingGlassIcon className="w-5 h-5" />
            <span>{state.isLoading ? 'Researching...' : 'Start Research'}</span>
          </button>
        </form>
      </div>
    </div>
  );
};

export default ResearchForm;