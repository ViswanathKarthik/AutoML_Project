import React, { useState } from "react";
import MainLanding from "./components/MainLanding";
import AutoMLWorkspace from "./components/AutoMLWorkspace";
import { Cpu, HelpCircle, Code, Award, Sparkles } from "lucide-react";

export default function App() {
  const [view, setView] = useState<"landing" | "workspace">("landing");
  const [selectedSampleId, setSelectedSampleId] = useState<string | undefined>(undefined);

  const handleStartWorkspace = (sampleId?: string) => {
    setSelectedSampleId(sampleId);
    setView("workspace");
  };

  const handleGoBack = () => {
    setView("landing");
    setSelectedSampleId(undefined);
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans">
      {/* Top Navigation */}
      <nav className="bg-white border-b border-slate-200 text-slate-800 px-8 py-4 flex items-center justify-between sticky top-0 z-50 shadow-sm">
        <div 
          onClick={handleGoBack}
          className="flex items-center gap-2.5 cursor-pointer hover:opacity-90 select-none"
        >
          <div className="w-8 h-8 bg-indigo-600 rounded flex items-center justify-center">
            <div className="w-4 h-4 bg-white rounded-sm rotate-45"></div>
          </div>
          <div>
            <span className="font-bold text-xl tracking-tight text-slate-800 font-display">
              AutoML<span className="text-indigo-600">Studio</span>
            </span>
          </div>
        </div>

        <div className="flex items-center gap-6 text-sm font-medium text-slate-500">
          <button 
            id="nav-link-capabilities"
            onClick={handleGoBack}
            className={`cursor-pointer transition-all pb-1 ${
              view === "landing" 
                ? "text-indigo-600 border-b-2 border-indigo-600 font-semibold" 
                : "hover:text-slate-800"
            }`}
          >
            Capabilities
          </button>
          
          <button 
            id="nav-link-workbench"
            onClick={() => handleStartWorkspace()}
            className={`cursor-pointer transition-all pb-1 ${
              view === "workspace" 
                ? "text-indigo-600 border-b-2 border-indigo-600 font-semibold" 
                : "hover:text-slate-800"
            }`}
          >
            Project Workbench
          </button>
          
          <div className="h-4 w-px bg-slate-200 hidden sm:block" />
          
          <a
            id="nav-link-build"
            href="https://ai.studio/build"
            target="_blank"
            rel="noopener noreferrer"
            className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-xl text-xs font-semibold shadow-md shadow-indigo-100 transition-all flex items-center gap-1.5 cursor-pointer"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>AI Studio Build</span>
          </a>
        </div>
      </nav>

      {/* Primary Page Router */}
      {view === "landing" ? (
        <MainLanding onStartWorkspace={handleStartWorkspace} />
      ) : (
        <AutoMLWorkspace initialSampleId={selectedSampleId} onGoBack={handleGoBack} />
      )}
    </div>
  );
}
