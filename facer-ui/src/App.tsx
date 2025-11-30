import React, { useState, useRef } from 'react';

import { Upload, Camera, Save, AlertCircle, CheckCircle, Loader2, LayoutGrid } from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import type { AnalysisResponse  } from './types';

import Gallery from './Gallery'; 

// Utility for cleaner tailwind classes
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

function App() {

  const [currentView, setCurrentView] = useState<'analyze' | 'gallery'>('analyze');

  // State
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [imgDim, setImgDim] = useState<{ w: number; h: number } | null>(null);

  // Form Data State
  const [description, setDescription] = useState('');
  const [keywords, setKeywords] = useState('');
  const [classification, setClassification] = useState('');
  const [shouldSave, setShouldSave] = useState(true);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Handle Image Selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreviewUrl(URL.createObjectURL(selectedFile));
      setAnalysis(null);
      setImgDim(null);
    }
  };

  // Get Image Natural Dimensions for Bounding Box Calculation
  const onImgLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const { naturalWidth, naturalHeight } = e.currentTarget;
    setImgDim({ w: naturalWidth, h: naturalHeight });
  };

  // Submit to Python Backend
  const handleAnalyze = async () => {
    if (!file) return;
    setIsAnalyzing(true);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('description', description);
    formData.append('keywords', keywords);
    formData.append('classification', classification);
    formData.append('save', shouldSave.toString());

    try {
      const response = await fetch('/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Analysis failed');

      const data: AnalysisResponse = await response.json();
      setAnalysis(data);
    } catch (error) {
      console.error(error);
      alert('Error analyzing image. Ensure server.py is running.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Render Bounding Boxes
  const renderBoxes = () => {
    if (!analysis || !imgDim) return null;

    return analysis.results.map((face, idx) => {
      const [x1, y1, x2, y2] = face.bbox;
      
      // Convert pixel coords to percentages
      const style = {
        left: `${(x1 / imgDim.w) * 100}%`,
        top: `${(y1 / imgDim.h) * 100}%`,
        width: `${((x2 - x1) / imgDim.w) * 100}%`,
        height: `${((y2 - y1) / imgDim.h) * 100}%`,
      };

      return (
        <div
          key={idx}
          className={cn(
            "absolute border-2 transition-all duration-300 group/box hover:bg-white/10 cursor-pointer",
            face.is_valid_pose ? "border-green-500" : "border-red-500"
          )}
          style={style}
        >
          {/* Tooltip on Hover */}
          {/* ‼️ CHANGED: Added z-index and fixed positioning context for tooltip */}
          <div className="opacity-0 group-hover/box:opacity-100 absolute -top-8 left-0 bg-black/80 text-white text-xs px-2 py-1 rounded whitespace-nowrap z-50 pointer-events-none transition-opacity">
             Face #{idx + 1}: {face.pose.direction_label}
          </div>
        </div>
      );
    });
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans p-6">
      <div className="max-w-6xl mx-auto">
        
        {/* Header */}
        <header className="mb-8 flex items-center justify-between border-b border-slate-800 pb-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-indigo-600 rounded-lg">
              <Camera className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-2xl font-bold tracking-tight">Facer<span className="text-indigo-400">UI</span></h1>
          </div>


          <nav className="flex bg-slate-900 p-1 rounded-lg border border-slate-800">
            <button
              onClick={() => setCurrentView('analyze')}
              className={cn(
                "px-4 py-2 rounded-md text-sm font-medium transition-all flex items-center gap-2",
                currentView === 'analyze' 
                  ? "bg-indigo-600 text-white shadow-lg" 
                  : "text-slate-400 hover:text-white hover:bg-slate-800"
              )}
            >
              <Camera className="w-4 h-4" />
              Analyze
            </button>
            <button
              onClick={() => setCurrentView('gallery')}
              className={cn(
                "px-4 py-2 rounded-md text-sm font-medium transition-all flex items-center gap-2",
                currentView === 'gallery' 
                  ? "bg-indigo-600 text-white shadow-lg" 
                  : "text-slate-400 hover:text-white hover:bg-slate-800"
              )}
            >
              <LayoutGrid className="w-4 h-4" />
              Gallery
            </button>
          </nav>
        </header>


        {currentView === 'gallery' ? (
          <Gallery />
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 animate-in fade-in duration-300">
            
            {/* Left Column: Image Preview */}
            <div className="lg:col-span-2 space-y-4">
              {/* ‼️ CHANGED: Replaced the flex container with a layout that supports accurate absolute positioning overlays.
                  The previous use of object-contain inside a flex container caused the bounding boxes (positioned by %) 
                  to misalign because the coordinate system of the div didn't match the rendered image size.
              */}
              <div className="relative bg-slate-900 rounded-xl overflow-hidden border border-slate-800 shadow-2xl min-h-[400px] flex items-center justify-center group p-4">
                {previewUrl ? (
                  /* ‼️ CRITICAL FIX: "inline-block" ensures this div shrinks to fit the image width exactly.
                     "relative" establishes the coordinate boundary for the bounding boxes.
                  */
                  <div className="relative inline-block">
                    <img 
                      src={previewUrl} 
                      alt="Preview" 
                      onLoad={onImgLoad}
                      /* ‼️ CHANGED: Removed object-contain, allowed height to drive width naturally */
                      className="max-h-[70vh] w-auto block rounded-lg"
                    />
                    {renderBoxes()}
                  </div>
                ) : (
                  <div 
                    onClick={() => fileInputRef.current?.click()}
                    className="text-slate-500 flex flex-col items-center gap-4 cursor-pointer hover:text-indigo-400 transition-colors"
                  >
                    <Upload className="w-12 h-12" />
                    <p className="font-medium">Click to upload an image</p>
                  </div>
                )}
                
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  onChange={handleFileChange} 
                  className="hidden" 
                  accept="image/*"
                />
              </div>

              {/* Analysis Stats */}
              {analysis && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-slate-900 p-4 rounded-lg border border-slate-800">
                    <span className="text-slate-400 text-sm block">Faces Detected</span>
                    <span className="text-2xl font-bold text-indigo-400">{analysis.face_count}</span>
                  </div>
                  <div className="bg-slate-900 p-4 rounded-lg border border-slate-800">
                    <span className="text-slate-400 text-sm block">Valid Poses</span>
                    <span className="text-2xl font-bold text-green-400">
                      {analysis.results.filter(r => r.is_valid_pose).length}
                    </span>
                  </div>
                </div>
              )}
            </div>

            {/* Right Column: Controls & Data */}
            <div className="space-y-6">
              
              {/* Input Form */}
              <div className="bg-slate-900 p-6 rounded-xl border border-slate-800 space-y-4">
                <h2 className="font-semibold text-lg flex items-center gap-2">
                  <Save className="w-5 h-5 text-indigo-400" />
                  Metadata
                </h2>
                
                <div className="space-y-3">
                  <div>
                    <label className="text-xs uppercase text-slate-500 font-bold tracking-wider">Description</label>
                    <input 
                      type="text" 
                      value={description}
                      onChange={(e) => setDescription(e.target.value)}
                      placeholder="e.g. Employee ID photo"
                      className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-2 mt-1 focus:outline-none focus:border-indigo-500 transition-colors"
                    />
                  </div>
                  
                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-xs uppercase text-slate-500 font-bold tracking-wider">Classification</label>
                      <input 
                        type="text" 
                        value={classification}
                        onChange={(e) => setClassification(e.target.value)}
                        placeholder="e.g. Security"
                        className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-2 mt-1 focus:outline-none focus:border-indigo-500"
                      />
                    </div>
                    <div>
                      <label className="text-xs uppercase text-slate-500 font-bold tracking-wider">Keywords</label>
                      <input 
                        type="text" 
                        value={keywords}
                        onChange={(e) => setKeywords(e.target.value)}
                        placeholder="e.g. front, daylight"
                        className="w-full bg-slate-950 border border-slate-800 rounded px-3 py-2 mt-1 focus:outline-none focus:border-indigo-500"
                      />
                    </div>
                  </div>

                  <div className="flex items-center gap-2 pt-2">
                    <input 
                      type="checkbox" 
                      id="saveDb"
                      checked={shouldSave}
                      onChange={(e) => setShouldSave(e.target.checked)}
                      className="accent-indigo-500 w-4 h-4"
                    />
                    <label htmlFor="saveDb" className="text-sm cursor-pointer select-none">Save to Database</label>
                  </div>
                </div>

                <button
                  onClick={handleAnalyze}
                  disabled={!file || isAnalyzing}
                  className={cn(
                    "w-full py-3 rounded-lg font-bold flex items-center justify-center gap-2 transition-all",
                    !file || isAnalyzing 
                      ? "bg-slate-800 text-slate-500 cursor-not-allowed"
                      : "bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-900/20"
                  )}
                >
                  {isAnalyzing ? <Loader2 className="animate-spin" /> : "Analyze & Process"}
                </button>
              </div>

              {/* Detailed Results List */}
              {analysis && (
                <div className="space-y-3">
                  <h3 className="font-semibold text-slate-400 uppercase text-xs tracking-wider">Detection Details</h3>
                  <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2 custom-scrollbar">
                    {analysis.results.map((face, i) => (
                      <div key={i} className="bg-slate-900 border border-slate-800 p-3 rounded-lg flex items-start justify-between">
                        <div>
                          <div className="flex items-center gap-2 mb-1">
                            <span className="font-bold text-indigo-300">Face #{i + 1}</span>
                            <span className="text-xs bg-slate-800 px-2 py-0.5 rounded text-slate-300">
                              {face.pose.direction_label}
                            </span>
                          </div>
                          <div className="text-xs text-slate-500 space-y-0.5">
                            <p>Yaw: {face.pose.yaw.toFixed(1)}°</p>
                            <p>Pitch: {face.pose.pitch.toFixed(1)}°</p>
                          </div>
                        </div>
                        
                        {face.is_valid_pose ? (
                          <CheckCircle className="text-green-500 w-5 h-5 mt-1" />
                        ) : (
                          <AlertCircle className="text-red-500 w-5 h-5 mt-1" />
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
