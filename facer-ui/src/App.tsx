import React, { useState, useRef } from 'react';
import {
  Upload, Camera, Save, AlertCircle, CheckCircle,
  Loader2, LayoutGrid, X, AlertTriangle, FileImage
} from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import type { AnalysisResponse } from './types';
import Gallery from './Gallery';
import { twMerge } from 'tailwind-merge';

// --- Utility ---
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}


interface QueuedImage {
  id: string; // Unique ID for React keys
  file: File;
  previewUrl: string;
  analysis: AnalysisResponse | null;
  status: 'pending' | 'analyzing' | 'done' | 'error';
  imgDim?: { w: number; h: number };
}

function App() {

  const [currentView, setCurrentView] = useState<'analyze' | 'gallery'>('analyze');


  const [queue, setQueue] = useState<QueuedImage[]>([]);
  const [activeIndex, setActiveIndex] = useState<number>(0);

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessingDrop, setIsProcessingDrop] = useState(false);

  // Form Data State
  const [description, setDescription] = useState('');
  const [keywords, setKeywords] = useState('');
  const [classification, setClassification] = useState('');
  const [shouldSave, setShouldSave] = useState(true);

  const fileInputRef = useRef<HTMLInputElement>(null);


  const createQueuedImages = (files: FileList | File[]): QueuedImage[] => {
    return Array.from(files).map(file => ({
      id: Math.random().toString(36).substring(7),
      file,
      previewUrl: URL.createObjectURL(file),
      analysis: null,
      status: 'pending'
    }));
  };


  const traverseFileTree = (item: any, path = ''): Promise<File[]> => {
    return new Promise((resolve) => {
      if (item.isFile) {
        item.file((file: File) => {
          // Update file path if needed, or just return file
          resolve([file]);
        });
      } else if (item.isDirectory) {
        const dirReader = item.createReader();
        const entries: any[] = [];

        const readEntries = () => {
          dirReader.readEntries((result: any[]) => {
            if (result.length === 0) {
              // Done reading directory, process recursive calls
              const promises = entries.map(entry => traverseFileTree(entry, path + item.name + "/"));
              Promise.all(promises).then(filesArrays => {
                resolve(filesArrays.flat());
              });
            } else {
              entries.push(...result);
              readEntries(); // Continue reading (readEntries returns blocks of entries)
            }
          });
        };
        readEntries();
      } else {
        resolve([]);
      }
    });
  };

  const isValidImage = (file: File) => {
    return file.type.startsWith('image/') || /\.(jpg|jpeg|png|gif|webp|bmp)$/i.test(file.name);
  };

  // Handle Image Selection (Input)
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const newImages = createQueuedImages(e.target.files);
      setQueue(prev => {
        const next = [...prev, ...newImages];
        if (prev.length === 0) setActiveIndex(0);
        return next;
      });
    }
  };

  /* Handler to remove specific image */
  const handleRemoveImage = (e: React.MouseEvent, index: number) => {
    e.stopPropagation();

    setQueue(prev => {
      const toRemove = prev[index];
      URL.revokeObjectURL(toRemove.previewUrl); // Cleanup

      const next = prev.filter((_, i) => i !== index);

      // Adjust active index
      if (index === activeIndex) {
        setActiveIndex(Math.max(0, index - 1));
      } else if (index < activeIndex) {
        setActiveIndex(activeIndex - 1);
      }
      return next;
    });

    // Reset input if queue becomes empty
    if (queue.length <= 1 && fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };


  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const items = e.dataTransfer.items;
    if (!items || items.length === 0) return;

    setIsProcessingDrop(true);

    try {
      const filePromises: Promise<File[]>[] = [];

      for (let i = 0; i < items.length; i++) {
        const item = items[i];
        // webkitGetAsEntry is standard in modern browsers (despite the prefix) for FS access
        const entry = item.webkitGetAsEntry ? item.webkitGetAsEntry() : null;

        if (entry) {
          filePromises.push(traverseFileTree(entry));
        } else if (item.kind === 'file') {
          const file = item.getAsFile();
          if (file) filePromises.push(Promise.resolve([file]));
        }
      }

      const filesArrays = await Promise.all(filePromises);
      const flatFiles = filesArrays.flat().filter(isValidImage);

      if (flatFiles.length > 0) {
        const newImages = createQueuedImages(flatFiles);
        setQueue(prev => {
          const next = [...prev, ...newImages];
          if (prev.length === 0) setActiveIndex(0);
          return next;
        });
      }
    } catch (err) {
      console.error("Error processing drop:", err);
    } finally {
      setIsProcessingDrop(false);
    }
  };

  // Get Image Natural Dimensions for Bounding Box Calculation
  const onImgLoad = (e: React.SyntheticEvent<HTMLImageElement>, index: number) => {
    const { naturalWidth, naturalHeight } = e.currentTarget;
    setQueue(prev => prev.map((item, i) =>
      i === index ? { ...item, imgDim: { w: naturalWidth, h: naturalHeight } } : item
    ));
  };


  const handleAnalyze = async () => {
    if (queue.length === 0) return;
    setIsAnalyzing(true);

    const formData = new FormData();
    // Append all files with the same key 'files'
    queue.forEach(item => {
      if (item.status !== 'done') { // Only retry pending/error items or re-analyze all? Let's analyze all current.
        formData.append('files', item.file);
      }
    });

    // If everything is already done, maybe user wants to re-run? 
    // For simplicity, let's just send everything currently in queue.
    // Reset formData and just send all.
    const finalFormData = new FormData();
    queue.forEach(item => finalFormData.append('files', item.file));

    finalFormData.append('description', description);
    finalFormData.append('keywords', keywords);
    finalFormData.append('classification', classification);
    finalFormData.append('save', shouldSave.toString());

    try {
      const response = await fetch('/analyze', {
        method: 'POST',
        body: finalFormData,
      });

      if (!response.ok) throw new Error('Analysis failed');

      const results: AnalysisResponse[] = await response.json();

      // Map results back to queue items by filename
      // Note: This relies on filenames being unique in the batch or order preservation. 
      // The backend processes in order, so mapping by index is safer if filenames are duplicates in upload.

      setQueue(prev => {
        const next = [...prev];
        results.forEach((res, i) => {
          // We assume backend returns results in same order as files were appended
          if (next[i]) {
            next[i] = {
              ...next[i],
              analysis: res,
              status: res.error ? 'error' : 'done'
            };
          }
        });
        return next;
      });

    } catch (error) {
      console.error(error);
      alert('Error analyzing images. Ensure server.py is running.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Render Bounding Boxes for Active Image
  const renderBoxes = () => {
    const activeItem = queue[activeIndex];
    if (!activeItem || !activeItem.analysis || !activeItem.imgDim) return null;

    return activeItem.analysis.results.map((face, idx) => {
      const [x1, y1, x2, y2] = face.bbox;
      const { w, h } = activeItem.imgDim!;

      const style = {
        left: `${(x1 / w) * 100}%`,
        top: `${(y1 / h) * 100}%`,
        width: `${((x2 - x1) / w) * 100}%`,
        height: `${((y2 - y1) / h) * 100}%`,
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
          <div className="opacity-0 group-hover/box:opacity-100 absolute -top-8 left-0 bg-black/80 text-white text-xs px-2 py-1 rounded whitespace-nowrap z-50 pointer-events-none transition-opacity">
            Face #{idx + 1}: {face.pose.direction_label}
          </div>
        </div>
      );
    });
  };

  const activeItem = queue[activeIndex];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans p-6">
      <div className="max-w-7xl mx-auto">

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
              Batch Analyze
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
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 animate-in fade-in duration-300">

            {/* Left Column: Image Queue & Preview */}
            <div className="lg:col-span-8 space-y-4">

              {/* Main Preview Area */}
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={cn(
                  "relative bg-slate-900 rounded-xl overflow-hidden border transition-all duration-300 shadow-2xl min-h-[500px] flex items-center justify-center group p-4",
                  isDragging ? "border-indigo-500 border-2 bg-slate-800 scale-[1.01]" : "border-slate-800"
                )}
              >

                {isProcessingDrop && (
                  <div className="absolute inset-0 z-50 bg-black/60 backdrop-blur-sm flex flex-col items-center justify-center text-indigo-400 animate-in fade-in">
                    <Loader2 className="w-12 h-12 animate-spin mb-2" />
                    <span className="font-semibold">Scanning folder contents...</span>
                  </div>
                )}

                {activeItem ? (
                  <div className="relative inline-block">
                    <img
                      src={activeItem.previewUrl}
                      alt="Preview"
                      onLoad={(e) => onImgLoad(e, activeIndex)}
                      className="max-h-[60vh] w-auto block rounded-lg shadow-lg"
                    />
                    {renderBoxes()}

                    {/* Error Overlay */}
                    {activeItem.status === 'error' && (
                      <div className="absolute inset-0 bg-red-900/40 backdrop-blur-sm flex items-center justify-center rounded-lg border-2 border-red-500/50">
                        <div className="bg-black/80 text-red-200 px-4 py-2 rounded-lg flex items-center gap-2">
                          <AlertTriangle className="w-5 h-5 text-red-500" />
                          {activeItem.analysis?.error || "Analysis Error"}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div
                    onClick={() => fileInputRef.current?.click()}
                    className="text-slate-500 flex flex-col items-center gap-4 cursor-pointer hover:text-indigo-400 transition-colors py-20"
                  >
                    <Upload className={cn("w-16 h-16 transition-transform", isDragging && "scale-125 text-indigo-400")} />
                    <div className="text-center">
                      <p className="font-medium text-lg text-slate-300">
                        {isDragging ? "Drop images or folders here" : "Click or drag to upload"}
                      </p>
                      <p className="text-sm mt-1">Supports folders & multiple files</p>
                    </div>
                  </div>
                )}

                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                  className="hidden"
                  accept="image/*"
                  multiple
                />
              </div>


              {queue.length > 0 && (
                <div className="flex gap-3 overflow-x-auto pb-2 custom-scrollbar">
                  {/* Add Button in strip */}
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="flex-shrink-0 w-24 h-24 bg-slate-900 border border-dashed border-slate-700 rounded-lg flex flex-col items-center justify-center text-slate-500 hover:text-indigo-400 hover:border-indigo-500 transition-all"
                  >
                    <Upload className="w-6 h-6 mb-1" />
                    <span className="text-xs">Add More</span>
                  </button>

                  {queue.map((item, i) => (
                    <div
                      key={item.id}
                      onClick={() => setActiveIndex(i)}
                      className={cn(
                        "relative flex-shrink-0 w-24 h-24 rounded-lg overflow-hidden border-2 cursor-pointer transition-all group/thumb",
                        i === activeIndex ? "border-indigo-500 ring-2 ring-indigo-500/20" : "border-slate-800 hover:border-slate-600"
                      )}
                    >
                      <img src={item.previewUrl} className="w-full h-full object-cover" />

                      {/* Status Indicators */}
                      <div className="absolute inset-0 bg-black/0 group-hover/thumb:bg-black/20 transition-colors" />

                      {item.status === 'done' && (
                        <div className="absolute bottom-1 right-1 bg-green-500 text-white rounded-full p-0.5">
                          <CheckCircle size={12} />
                        </div>
                      )}
                      {item.status === 'error' && (
                        <div className="absolute bottom-1 right-1 bg-red-500 text-white rounded-full p-0.5">
                          <AlertCircle size={12} />
                        </div>
                      )}

                      {/* Remove Button */}
                      <button
                        onClick={(e) => handleRemoveImage(e, i)}
                        className="absolute top-1 right-1 bg-black/60 text-white p-1 rounded-full opacity-0 group-hover/thumb:opacity-100 transition-opacity hover:bg-red-500"
                      >
                        <X size={12} />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Right Column: Controls & Data */}
            <div className="lg:col-span-4 space-y-6">

              {/* Input Form */}
              <div className="bg-slate-900 p-6 rounded-xl border border-slate-800 space-y-4">
                <h2 className="font-semibold text-lg flex items-center gap-2">
                  <Save className="w-5 h-5 text-indigo-400" />
                  Batch Metadata
                </h2>
                <p className="text-xs text-slate-500">
                  Applies to all {queue.length} images in queue.
                </p>

                <div className="space-y-3">
                  <div>
                    <label className="text-xs uppercase text-slate-500 font-bold tracking-wider">Description</label>
                    <input
                      type="text"
                      value={description}
                      onChange={(e) => setDescription(e.target.value)}
                      placeholder="e.g. Employee ID photos"
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
                        placeholder="e.g. front"
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
                  disabled={queue.length === 0 || isAnalyzing}
                  className={cn(
                    "w-full py-3 rounded-lg font-bold flex items-center justify-center gap-2 transition-all",
                    queue.length === 0 || isAnalyzing
                      ? "bg-slate-800 text-slate-500 cursor-not-allowed"
                      : "bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-900/20"
                  )}
                >
                  {isAnalyzing ? <Loader2 className="animate-spin" /> : `Process ${queue.length} Images`}
                </button>
              </div>

              {/* Detailed Results List for ACTIVE image */}
              {activeItem && activeItem.analysis && !activeItem.analysis.error && (
                <div className="space-y-3 animate-in slide-in-from-right duration-300">
                  <div className="flex items-center justify-between">
                    <h3 className="font-semibold text-slate-400 uppercase text-xs tracking-wider">
                      Results for {activeItem.file.name}
                    </h3>
                    <span className="text-xs text-slate-500">{activeItem.analysis.face_count} faces</span>
                  </div>

                  <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2 custom-scrollbar">
                    {activeItem.analysis.results.map((face, i) => (
                      <div key={i} className="bg-slate-900 border border-slate-800 p-3 rounded-lg flex items-start justify-between hover:border-slate-700 transition-colors">
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
                    {activeItem.analysis.results.length === 0 && (
                      <div className="text-center py-4 text-slate-500 text-sm italic">
                        No faces detected in this image.
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Fallback empty state for right column */}
              {(!activeItem || !activeItem.analysis) && queue.length > 0 && (
                <div className="border border-dashed border-slate-800 rounded-xl p-8 text-center text-slate-600">
                  <FileImage className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">Select an image from the queue to view details.</p>
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