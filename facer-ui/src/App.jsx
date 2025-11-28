import React, { useState, useEffect, useRef } from 'react';
import { Upload, Camera, CheckCircle, XCircle, Database, Server, RefreshCw, Box, Grid, LayoutDashboard, Save, Scan, Tag, Hash } from 'lucide-react'; 

const API_URL = '/analyze';

export default function App() {
  const [currentView, setCurrentView] = useState('analyze'); 
  const [currentFile, setCurrentFile] = useState(null);
  const [description, setDescription] = useState(""); 
  const [keywords, setKeywords] = useState("");
  const [faceClass, setFaceClass] = useState("");
  
  const [previewUrl, setPreviewUrl] = useState(null);
  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [error, setError] = useState(null);
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });
  
  const fileInputRef = useRef(null);

  const handleLocalFile = (file) => {
      setCurrentFile(file);
      setAnalysisData(null); 
      setError(null);
      setSaveSuccess(false);
      
      const objectUrl = URL.createObjectURL(file);
      setPreviewUrl(objectUrl);

      const img = new Image();
      img.onload = () => {
        setImageDimensions({ width: img.naturalWidth, height: img.naturalHeight });
      };
      img.src = objectUrl;
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      handleLocalFile(file);
    }
    event.target.value = ''; 
  };

  const processFile = (file, saveToDb = false) => {
    setLoading(true);
    setSaveSuccess(false);

    if (!saveToDb) {
        setAnalysisData(null);
        setError(null);
    }

    const formData = new FormData();
    formData.append('file', file);
    if (description) formData.append('description', description);
    

    if (keywords) formData.append('keywords', keywords);

    // ‼️ Change: Updated key from 'class' to 'classification' to match backend expectation
    if (faceClass) formData.append('classification', faceClass);
    
    formData.append('save', saveToDb);

    fetch(API_URL, {
      method: 'POST',
      body: formData,
    })
      .then(async res => {
        if (!res.ok) {
            const errorData = await res.json().catch(() => null);
            const errorMessage = errorData?.detail || `Error ${res.status}: ${res.statusText}`;
            throw new Error(errorMessage);
        }
        return res.json();
      })
      .then(data => {
        setAnalysisData(data);
        setLoading(false);
        if (saveToDb) {
            setSaveSuccess(true);
            setTimeout(() => setSaveSuccess(false), 3000);
        }
      })
      .catch(err => {
        console.error(err);
        const msg = err.message.replace(/^Error:\s*/, '');
        setError(msg);
        setLoading(false);
      });
  };

  const handleSave = () => {
      if (currentFile) {
          processFile(currentFile, true);
      }
  };

  const onDragOver = (e) => e.preventDefault();
  
  const onDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      handleLocalFile(file);
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
          
          <div className="flex bg-slate-800 p-1 rounded-lg">
             <button 
                onClick={() => setCurrentView('analyze')}
                className={`flex items-center gap-2 px-4 py-1.5 rounded-md text-sm font-medium transition-all ${currentView === 'analyze' ? 'bg-slate-700 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}
             >
                <LayoutDashboard className="w-4 h-4" />
                Analyze
             </button>
             <button 
                onClick={() => setCurrentView('gallery')}
                className={`flex items-center gap-2 px-4 py-1.5 rounded-md text-sm font-medium transition-all ${currentView === 'gallery' ? 'bg-slate-700 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'}`}
             >
                <Grid className="w-4 h-4" />
                Gallery
             </button>
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
        {error && (
          <div className="mb-6 p-4 bg-red-900/20 border border-red-800 rounded-lg flex items-center gap-3 text-red-200">
            <XCircle className="w-5 h-5 flex-shrink-0" />
            <p>{error}</p>
          </div>
        )}

        {saveSuccess && (
          <div className="mb-6 p-4 bg-green-900/20 border border-green-800 rounded-lg flex items-center gap-3 text-green-200 animate-pulse">
            <CheckCircle className="w-5 h-5" />
            <p>Results and Images saved to database successfully!</p>
          </div>
        )}

        {currentView === 'analyze' ? (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            <div className="lg:col-span-7 space-y-6">
              
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileSelect} 
                className="hidden" 
                accept="image/*"
              />

              {/* Description, Keywords, Class Inputs */}
              <div className="bg-slate-900 p-4 rounded-xl border border-slate-800 flex flex-col gap-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="md:col-span-2">
                        <label className="block text-xs font-semibold uppercase text-slate-500 mb-2 tracking-wider">
                           Image Description
                        </label>
                        <input 
                            type="text" 
                            value={description}
                            onChange={(e) => setDescription(e.target.value)}
                            placeholder="e.g. Employee ID photos batch 1"
                            className="w-full bg-slate-950 border border-slate-700 rounded-lg px-4 py-2.5 text-sm text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/50 transition-all"
                        />
                    </div>
                    
                    <div>
                        <label className="block text-xs font-semibold uppercase text-slate-500 mb-2 tracking-wider">
                           Keywords (CSV)
                        </label>
                        <div className="relative">
                            <Tag className="absolute left-3 top-2.5 w-4 h-4 text-slate-600" />
                            <input 
                                type="text" 
                                value={keywords}
                                onChange={(e) => setKeywords(e.target.value)}
                                placeholder="office, id-card, 2023"
                                className="w-full bg-slate-950 border border-slate-700 rounded-lg pl-10 pr-4 py-2.5 text-sm text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/50 transition-all"
                            />
                        </div>
                    </div>
                    
                    <div>
                        <label className="block text-xs font-semibold uppercase text-slate-500 mb-2 tracking-wider">
                           Class (Category)
                        </label>
                         <div className="relative">
                            <Hash className="absolute left-3 top-2.5 w-4 h-4 text-slate-600" />
                            <input 
                                type="text" 
                                value={faceClass}
                                onChange={(e) => setFaceClass(e.target.value)}
                                placeholder="Personnel"
                                className="w-full bg-slate-950 border border-slate-700 rounded-lg pl-10 pr-4 py-2.5 text-sm text-slate-200 placeholder:text-slate-600 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500/50 transition-all"
                            />
                        </div>
                    </div>
                  </div>
                  
                  <div className="flex gap-4 pt-2 border-t border-slate-800 mt-2">
                      <button
                        onClick={() => processFile(currentFile, false)}
                        disabled={!currentFile || loading}
                        className={`
                            flex-1 flex items-center justify-center gap-2 px-6 py-2.5 rounded-lg font-medium transition-all
                            ${!currentFile || loading 
                                ? 'bg-slate-800 text-slate-500 cursor-not-allowed' 
                                : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-500/20'}
                        `}
                      >
                        {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Scan className="w-4 h-4" />}
                        Analyze Only
                      </button>

                      <button
                        onClick={handleSave}
                        disabled={!analysisData || loading}
                        className={`
                            flex-1 flex items-center justify-center gap-2 px-6 py-2.5 rounded-lg font-medium transition-all
                            ${!analysisData || loading 
                                ? 'bg-slate-800 text-slate-500 cursor-not-allowed' 
                                : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-500/20'}
                        `}
                      >
                          <Save className="w-4 h-4" />
                        Save to DB
                      </button>
                  </div>
              </div>

              <div 
                onDragOver={onDragOver}
                onDrop={onDrop}
                onClick={() => !previewUrl && fileInputRef.current?.click()} 
                className={`
                  relative group rounded-2xl border-2 border-dashed transition-all duration-300 overflow-hidden
                  ${previewUrl ? 'border-slate-700 bg-slate-900' : 'cursor-pointer border-slate-700 hover:border-indigo-500 hover:bg-slate-900/50 h-64 flex items-center justify-center'}
                `}
              >
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
                  <div className="relative w-full min-h-[400px] flex items-center justify-center bg-slate-950 p-4">
                    <div className="grid place-items-center" style={{ width: 'fit-content' }}>
                      <img 
                        src={previewUrl} 
                        alt="Preview" 
                        className="max-h-[600px] w-auto max-w-full object-contain col-start-1 row-start-1 z-10 block" 
                      />
                      {analysisData && imageDimensions.width > 0 && (
                        <svg 
                          viewBox={`0 0 ${imageDimensions.width} ${imageDimensions.height}`} 
                          className="w-full h-full col-start-1 row-start-1 z-20 pointer-events-none"
                        >
                          {analysisData.results.map((face, idx) => {
                            const [x1, y1, x2, y2] = face.bbox;
                            const width = x2 - x1;
                            const height = y2 - y1;
                            const color = face.is_valid_pose ? '#22c55e' : '#ef4444'; 

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
                    </div>
                    {loading && (
                      <div className="absolute inset-0 bg-slate-950/60 backdrop-blur-sm flex items-center justify-center z-30">
                        <RefreshCw className="w-10 h-10 text-indigo-500 animate-spin" />
                      </div>
                    )}
                  </div>
                )}
              </div>

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
                      <p>Upload an image and click Analyze</p>
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
        ) : (
           <GalleryView />
        )}
      </main>
    </div>
  );
}

function GalleryView() {
   const [faces, setFaces] = useState([]);
   const [loading, setLoading] = useState(true);

   useEffect(() => {
      fetch('/faces')
         .then(res => res.json())
         .then(data => {
            setFaces(data);
            setLoading(false);
         })
         .catch(err => {
            console.error(err);
            setLoading(false);
         });
   }, []);

   if (loading) return <div className="text-center p-10 text-slate-500">Loading gallery...</div>;

   return (
      <div className="space-y-6">
         <h2 className="text-2xl font-bold text-white flex items-center gap-2">
            <Database className="w-6 h-6 text-indigo-400" />
            Database Gallery
         </h2>

         {faces.length === 0 ? (
            <div className="text-center p-20 bg-slate-900/50 rounded-2xl border border-slate-800 border-dashed">
               <p className="text-slate-500">No faces enrolled in database yet.</p>
            </div>
         ) : (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6">
               {faces.map(face => (
                  <div key={face.id} className="group bg-slate-900 rounded-xl overflow-hidden border border-slate-800 hover:border-indigo-500/50 transition-all hover:shadow-lg hover:shadow-indigo-500/10">
                      <div className="aspect-square bg-slate-950 relative overflow-hidden">
                         <img 
                            src={`/faces/${face.id}/image`} 
                            alt={face.image_name}
                            className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                            loading="lazy"
                         />
                         <div className={`absolute top-2 right-2 px-2 py-1 text-[10px] font-bold uppercase rounded border backdrop-blur-md
                            ${face.is_valid_pose ? 'bg-green-500/20 text-green-300 border-green-500/30' : 'bg-red-500/20 text-red-300 border-red-500/30'}
                         `}>
                            {face.is_valid_pose ? 'Valid' : 'Invalid'}
                         </div>
                      </div>
                      <div className="p-3">

                         <div className="mb-2 space-y-1">
                            <div className="flex items-center justify-between">
                               {/* ‼️ Change: Updated property from face.class to face.classification to match API response */}
                               {face.classification && (
                                   <span className="text-[10px] font-bold uppercase bg-indigo-500/20 text-indigo-300 px-1.5 py-0.5 rounded border border-indigo-500/30">
                                       {face.classification}
                                   </span>
                               )}
                            </div>
                            
                            {face.description && (
                               <p className="text-sm font-medium text-slate-200 truncate" title={face.description}>
                                   {face.description}
                               </p>
                            )}
                            
                            {face.keywords && (
                               <div className="flex flex-wrap gap-1 mt-1">
                                   {face.keywords.split(',').slice(0, 3).map((kw, i) => (
                                       <span key={i} className="text-[10px] text-slate-400 bg-slate-800 px-1 rounded">
                                           #{kw.trim()}
                                       </span>
                                   ))}
                                   {face.keywords.split(',').length > 3 && (
                                       <span className="text-[10px] text-slate-500">...</span>
                                   )}
                               </div>
                            )}

                            <p className="text-xs text-slate-500 truncate pt-1 border-t border-slate-800 mt-2" title={face.image_name}>
                               {face.image_name}
                            </p>
                         </div>
                      </div>
                  </div>
               ))}
            </div>
         )}
      </div>
   );
}
