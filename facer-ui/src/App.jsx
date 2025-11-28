import React, { useState, useRef } from 'react';
import { Upload, Camera, CheckCircle, XCircle, Database, Server, RefreshCw, Box } from 'lucide-react';

const API_URL = '/analyze';

export default function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });
  const fileInputRef = useRef(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      processFile(file);
    }
    // Optional: reset value to allow selecting the same file again if needed
    event.target.value = ''; 
  };

  const processFile = (file) => {
    setLoading(true);
    setAnalysisData(null);
    setError(null);

    // Create preview
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
    
    // Get dimensions for SVG overlay mapping
    const img = new Image();
    img.onload = () => {
      setImageDimensions({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.src = objectUrl;

    // Upload to API
    const formData = new FormData();
    formData.append('file', file);

    fetch(API_URL, {
      method: 'POST',
      body: formData,
    })
      .then(res => {
        if (!res.ok) throw new Error("API Connection Failed. Is server.py running?");
        return res.json();
      })
      .then(data => {
        setAnalysisData(data);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setError(err.message);
        setLoading(false);
      });
  };

  // Drag and Drop handlers
  const onDragOver = (e) => e.preventDefault();
  const onDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-indigo-500/30">
      {/* Header */}
      <header className="bg-slate-900 border-b border-slate-800 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-indigo-600 p-2 rounded-lg">
              <Camera className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-xl font-bold tracking-tight text-white">Facer <span className="text-slate-500 font-normal text-sm ml-2">v0.1.0</span></h1>
          </div>
          <div className="flex items-center gap-4 text-sm text-slate-400">
              <div className="flex items-center gap-2 px-3 py-1 bg-slate-800 rounded-full border border-slate-700">
                <div className={`w-2 h-2 rounded-full ${analysisData ? 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.6)]' : 'bg-slate-500'}`}></div>
                <span>API Status</span>
              </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        
        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-red-900/20 border border-red-800 rounded-lg flex items-center gap-3 text-red-200">
            <XCircle className="w-5 h-5" />
            <p>{error}</p>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Left Column: Input & Visualization */}
          <div className="lg:col-span-7 space-y-6">
            
            {/* ‼️ MOVED: Input element moved OUTSIDE the clickable div to prevent event bubbling loop */}
            <input 
              type="file" 
              ref={fileInputRef} 
              onChange={handleFileSelect} 
              className="hidden" 
              accept="image/*"
            />

            {/* Upload Zone */}
            <div 
              onDragOver={onDragOver}
              onDrop={onDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`
                relative group cursor-pointer rounded-2xl border-2 border-dashed transition-all duration-300 overflow-hidden
                ${previewUrl ? 'border-slate-700 bg-slate-900' : 'border-slate-700 hover:border-indigo-500 hover:bg-slate-900/50 h-64 flex items-center justify-center'}
              `}
            >
              {/* ‼️ REMOVED: Input element was here previously */}
              
              {!previewUrl ? (
                <div className="text-center space-y-4">
                  <div className="w-16 h-16 bg-slate-800 rounded-full flex items-center justify-center mx-auto group-hover:scale-110 transition-transform">
                    <Upload className="w-8 h-8 text-indigo-400" />
                  </div>
                  <div>
                    <p className="text-lg font-medium text-slate-200">Drop image here</p>
                    <p className="text-sm text-slate-500">or click to browse</p>
                  </div>
                </div>
              ) : (
                <div className="relative w-full h-full flex items-center justify-center bg-slate-950">
                  {/* The Image */}
                  <img 
                    src={previewUrl} 
                    alt="Preview" 
                    className="max-h-[600px] w-auto mx-auto object-contain" 
                  />
                  
                  {/* SVG Overlay for Bounding Boxes */}
                  {analysisData && imageDimensions.width > 0 && (
                    <svg 
                      viewBox={`0 0 ${imageDimensions.width} ${imageDimensions.height}`} 
                      className="absolute inset-0 w-full h-full pointer-events-none"
                    >
                      {analysisData.results.map((face, idx) => {
                        const [x1, y1, x2, y2] = face.bbox;
                        const width = x2 - x1;
                        const height = y2 - y1;
                        const color = face.is_valid_pose ? '#22c55e' : '#ef4444'; // Green vs Red

                        return (
                          <g key={idx}>
                            <rect
                              x={x1} y={y1} width={width} height={height}
                              fill="none"
                              stroke={color}
                              strokeWidth="3"
                              vectorEffect="non-scaling-stroke"
                            />
                            <rect
                              x={x1} y={y1 - 24} width="80" height="24"
                              fill={color}
                            />
                            <text
                              x={x1 + 4} y={y1 - 7}
                              fill="black"
                              fontSize="14"
                              fontWeight="bold"
                            >
                              Face #{idx + 1}
                            </text>
                          </g>
                        );
                      })}
                    </svg>
                  )}
                  
                  {/* Loading Overlay */}
                  {loading && (
                    <div className="absolute inset-0 bg-slate-950/60 backdrop-blur-sm flex items-center justify-center">
                      <RefreshCw className="w-10 h-10 text-indigo-500 animate-spin" />
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* JSON/DB Preview (Collapsible logic could be added here) */}
            {analysisData && (
               <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
                  <div className="px-4 py-3 border-b border-slate-800 bg-slate-800/50 flex items-center gap-2">
                     <Database className="w-4 h-4 text-slate-400" />
                     <h3 className="text-sm font-medium text-slate-300">Raw Analysis (DB Payload)</h3>
                  </div>
                  <pre className="p-4 overflow-x-auto text-xs font-mono text-indigo-300 bg-slate-950/50">
                     {JSON.stringify(analysisData, null, 2)}
                  </pre>
               </div>
            )}
          </div>

          {/* Right Column: Results & Stats */}
          <div className="lg:col-span-5 space-y-6">
             <div className="flex items-center justify-between">
                <h2 className="text-xl font-bold">Analysis Results</h2>
                {analysisData && (
                   <span className="text-sm px-3 py-1 bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 rounded-full">
                      {analysisData.face_count} Faces Detected
                   </span>
                )}
             </div>

             {!analysisData && !loading && (
                <div className="h-64 flex flex-col items-center justify-center text-slate-600 border border-dashed border-slate-800 rounded-xl bg-slate-900/30">
                   <Box className="w-12 h-12 mb-3 opacity-20" />
                   <p>Upload an image to see details</p>
                </div>
             )}

             {loading && (
                <div className="space-y-4">
                   {[1, 2].map(i => (
                      <div key={i} className="h-32 bg-slate-900 rounded-xl animate-pulse"></div>
                   ))}
                </div>
             )}

             <div className="space-y-4">
                {analysisData?.results.map((face, idx) => (
                   <div 
                     key={idx} 
                     className={`
                       relative overflow-hidden rounded-xl border transition-all duration-300
                       ${face.is_valid_pose 
                          ? 'bg-slate-900 border-slate-800 hover:border-green-500/50' 
                          : 'bg-slate-900 border-slate-800 hover:border-red-500/50'}
                     `}
                   >
                      {/* Validity Stripe */}
                      <div className={`absolute left-0 top-0 bottom-0 w-1 ${face.is_valid_pose ? 'bg-green-500' : 'bg-red-500'}`}></div>

                      <div className="p-5 pl-6">
                         <div className="flex items-center justify-between mb-4">
                            <h3 className="font-semibold text-lg flex items-center gap-2">
                               Face #{idx + 1}
                               {face.is_valid_pose ? (
                                  <span className="text-xs bg-green-500/10 text-green-400 px-2 py-0.5 rounded border border-green-500/20">Enrollable</span>
                               ) : (
                                  <span className="text-xs bg-red-500/10 text-red-400 px-2 py-0.5 rounded border border-red-500/20">Bad Pose</span>
                               )}
                            </h3>
                            <div className="text-xs text-slate-500 font-mono">
                               BBOX: [{face.bbox.map(n => Math.round(n)).join(', ')}]
                            </div>
                         </div>

                         {/* Pose Grid */}
                         <div className="grid grid-cols-3 gap-2 mb-4">
                            <StatBox label="YAW" value={face.pose.yaw} unit="°" />
                            <StatBox label="PITCH" value={face.pose.pitch} unit="°" />
                            <StatBox label="ROLL" value={face.pose.roll} unit="°" />
                         </div>

                         {/* Details */}
                         <div className="space-y-2 text-sm">
                            <div className="flex justify-between p-2 bg-slate-950 rounded border border-slate-800">
                               <span className="text-slate-400">Direction</span>
                               <span className="font-medium text-white capitalize">{face.pose.direction_label}</span>
                            </div>
                            <div className="flex justify-between p-2 bg-slate-950 rounded border border-slate-800">
                               <span className="text-slate-400">Vector Embedding</span>
                               {face.embedding ? (
                                  <span className="font-mono text-green-400 flex items-center gap-1">
                                     <CheckCircle className="w-3 h-3" />
                                     Generated (512d)
                                  </span>
                               ) : (
                                  <span className="font-mono text-slate-500">Skipped</span>
                               )}
                            </div>
                         </div>
                      </div>
                   </div>
                ))}
             </div>
          </div>
        </div>
      </main>
    </div>
  );
}

// Helper Component for Stats
function StatBox({ label, value, unit }) {
   const isZero = Math.abs(value) < 5;
   return (
      <div className={`p-2 rounded bg-slate-950 border border-slate-800 text-center ${!isZero ? 'border-slate-700' : ''}`}>
         <div className="text-[10px] uppercase text-slate-500 font-bold tracking-wider">{label}</div>
         <div className="text-lg font-mono font-medium text-slate-200">
            {value.toFixed(1)}{unit}
         </div>
      </div>
   );
}
