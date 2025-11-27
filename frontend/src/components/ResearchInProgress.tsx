import React from 'react';
import { BeakerIcon, MagnifyingGlassIcon, DocumentTextIcon } from '@heroicons/react/24/outline';

const ResearchInProgress: React.FC = () => {
  return (
    <div className="card p-8 shadow-lg text-center">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Research in Progress...</h2>
      <div className="flex justify-center items-center space-x-8">
        <div className="flex flex-col items-center animate-float animation-delay-100">
          <BeakerIcon className="w-12 h-12 text-primary-600" />
          <p className="mt-2 text-sm text-gray-600">Formulating strategy</p>
        </div>
        <div className="flex flex-col items-center animate-float animation-delay-200">
          <MagnifyingGlassIcon className="w-12 h-12 text-primary-600" />
          <p className="mt-2 text-sm text-gray-600">Gathering sources</p>
        </div>
        <div className="flex flex-col items-center animate-float animation-delay-300">
          <DocumentTextIcon className="w-12 h-12 text-primary-600" />
          <p className="mt-2 text-sm text-gray-600">Generating report</p>
        </div>
      </div>
      <div className="mt-8">
        <div className="w-full bg-gray-200 rounded-full h-2.5">
          <div className="bg-primary-600 h-2.5 rounded-full animate-pulse-glow"></div>
        </div>
        <p className="text-sm text-gray-500 mt-2">This may take a minute. Please don't close this window.</p>
      </div>
    </div>
  );
};

export default ResearchInProgress;
