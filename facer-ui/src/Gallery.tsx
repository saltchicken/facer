import React, { useEffect, useState, useRef } from 'react';
import {
  Loader2, Tag, User, Edit2, Check, X,
  ChevronLeft, ChevronRight, RefreshCw, ChevronDown, Download,
  Trash2, Maximize, ScanSearch
} from 'lucide-react';
import type { FaceRecord } from './types';

const ITEMS_PER_PAGE = 8;


interface MultiSelectProps {
  label: string;
  icon: React.ElementType;
  options: string[];
  selected: string[];
  onChange: (selected: string[]) => void;
  noneLabel?: string;
}

function MultiSelect({ label, icon: Icon, options, selected, onChange, noneLabel = "None" }: MultiSelectProps) {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Close when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const toggleOption = (opt: string) => {
    if (selected.includes(opt)) {
      onChange(selected.filter(s => s !== opt));
    } else {
      onChange([...selected, opt]);
    }
  };

  return (
    <div className="relative" ref={containerRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm transition-all ${selected.length > 0
          ? 'bg-indigo-600/20 border-indigo-500 text-indigo-200'
          : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
          }`}
      >
        <Icon className="w-4 h-4" />
        <span className="max-w-[100px] truncate">
          {selected.length === 0 ? label : `${selected.length} selected`}
        </span>
        <ChevronDown className={`w-3 h-3 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <div className="absolute top-full left-0 mt-2 w-56 max-h-80 overflow-y-auto bg-slate-900 border border-slate-700 rounded-lg shadow-xl z-50 p-1 flex flex-col gap-0.5">
          {/* Option for NULL/None */}
          <label className="flex items-center gap-2 px-3 py-2 hover:bg-slate-800 rounded cursor-pointer select-none">
            <input
              type="checkbox"
              className="rounded border-slate-600 bg-slate-950 text-indigo-500 focus:ring-offset-slate-900"
              checked={selected.includes("__NONE__")}
              onChange={() => toggleOption("__NONE__")}
            />
            <span className="text-sm text-slate-300 italic">{noneLabel}</span>
          </label>

          {options.length > 0 && <div className="h-px bg-slate-800 my-1" />}

          {options.map((opt) => (
            <label key={opt} className="flex items-center gap-2 px-3 py-2 hover:bg-slate-800 rounded cursor-pointer select-none">
              <input
                type="checkbox"
                className="rounded border-slate-600 bg-slate-950 text-indigo-500 focus:ring-offset-slate-900"
                checked={selected.includes(opt)}
                onChange={() => toggleOption(opt)}
              />
              <span className="text-sm text-slate-200">{opt}</span>
            </label>
          ))}
          {options.length === 0 && (
            <div className="px-3 py-2 text-xs text-slate-500 text-center">No options available</div>
          )}
        </div>
      )}
    </div>
  );
}


export default function Gallery() {
  const [faces, setFaces] = useState<FaceRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(0);
  const [hasNextPage, setHasNextPage] = useState(false);


  const [selectedImage, setSelectedImage] = useState<FaceRecord | null>(null);

  const [analyzing, setAnalyzing] = useState(false);


  const [filters, setFilters] = useState<{
    keywords: string[];
    classifications: string[];
  }>({ keywords: [], classifications: [] });


  const [filterOptions, setFilterOptions] = useState<{
    keywords: string[];
    classifications: string[];
  }>({ keywords: [], classifications: [] });

  const [editingId, setEditingId] = useState<number | null>(null);
  const [editForm, setEditForm] = useState<{
    description: string;
    classification: string;
    keywords: string;
  }>({ description: '', classification: '', keywords: '' });


  useEffect(() => {
    fetchFilterOptions();
  }, []);


  useEffect(() => {
    fetchFaces();
  }, [page, filters]);


  const fetchFilterOptions = async () => {
    try {
      const res = await fetch('/filters');
      if (res.ok) {
        const data = await res.json();
        setFilterOptions(data);
      }
    } catch (err) {
      console.error("Failed to load filter options", err);
    }
  };

  const fetchFaces = async () => {
    setLoading(true);
    try {
      const offset = page * ITEMS_PER_PAGE;


      const params = new URLSearchParams();

      params.append('limit', (ITEMS_PER_PAGE + 1).toString());
      params.append('offset', offset.toString());


      filters.keywords.forEach(k => params.append('keyword', k));
      filters.classifications.forEach(c => params.append('classification', c));

      const res = await fetch(`/faces?${params.toString()}`);
      const data = await res.json();


      if (data.length > ITEMS_PER_PAGE) {
        setHasNextPage(true);

        setFaces(data.slice(0, ITEMS_PER_PAGE));
      } else {
        setHasNextPage(false);
        setFaces(data);
      }
    } catch (err) {
      console.error("Failed to load faces", err);
    } finally {
      setLoading(false);
    }
  };


  const handleExport = () => {
    const params = new URLSearchParams();
    filters.keywords.forEach(k => params.append('keyword', k));
    filters.classifications.forEach(c => params.append('classification', c));

    // Trigger download by setting window location to the export endpoint
    window.location.href = `/export?${params.toString()}`;
  };

  const startEditing = (face: FaceRecord) => {
    setEditingId(face.id);
    setEditForm({
      description: face.description || '',
      classification: face.classification || '',
      keywords: face.keywords || ''
    });
  };

  const cancelEditing = () => {
    setEditingId(null);
  };

  const saveEdit = async (id: number) => {
    try {
      const res = await fetch(`/faces/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(editForm)
      });

      if (res.ok) {
        setFaces(faces.map(f => f.id === id ? { ...f, ...editForm } : f));
        setEditingId(null);
        // Refresh filter options just in case new tags were added
        fetchFilterOptions();
      } else {
        console.error("Failed to save changes");
      }
    } catch (err) {
      console.error("Failed to update", err);
    }
  };


  const deleteFace = async (id: number) => {
    if (!window.confirm("Are you sure you want to delete this image? This action cannot be undone.")) return;

    try {
      const res = await fetch(`/faces/${id}`, { method: 'DELETE' });
      if (res.ok) {
        setFaces(faces.filter(f => f.id !== id));
        if (selectedImage?.id === id) setSelectedImage(null); // Close modal if open
        if (editingId === id) setEditingId(null); // Exit edit mode
      } else {
        console.error("Failed to delete");
      }
    } catch (err) {
      console.error("Error deleting", err);
    }
  };


  const handleClassificationChange = (newSelected: string[]) => {
    setFilters(prev => ({ ...prev, classifications: newSelected }));
    setPage(0);
  }

  const handleKeywordChange = (newSelected: string[]) => {
    setFilters(prev => ({ ...prev, keywords: newSelected }));
    setPage(0);
  }


  const clearFilters = () => {
    setFilters({ keywords: [], classifications: [] });
    setPage(0);
  };


  const handleReanalyze = async () => {
    if (!selectedImage) return;
    setAnalyzing(true);
    try {
      const res = await fetch(`/faces/${selectedImage.id}/reanalyze`, {
        method: 'POST'
      });

      if (!res.ok) throw new Error("Re-analysis failed");

      const responseData = await res.json();
      const updatedData = responseData.data;

      // Update local state (both the selected image modal and the main list)
      const updatedRecord = {
        ...selectedImage,
        ...updatedData
      };

      setSelectedImage(updatedRecord);
      setFaces(prev => prev.map(f => f.id === selectedImage.id ? updatedRecord : f));

      // Force reload the image in the list to reflect new crop if bbox changed?
      // Since the /image endpoint is just /faces/{id}/image, it might be cached by browser.
      // We can append a timestamp to force refresh if needed, but keeping it simple for now.

    } catch (error) {
      console.error("Re-analyze error", error);
      alert("Failed to re-analyze image.");
    } finally {
      setAnalyzing(false);
    }
  }

  if (loading && faces.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-indigo-400">
        <Loader2 className="w-8 h-8 animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-in fade-in duration-500 pb-12">


      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-slate-900/50 p-4 rounded-xl border border-slate-800">
        <div>
          <h2 className="text-xl font-semibold text-slate-200">Database Records</h2>
          <span className="text-slate-500 text-sm">
            Showing {faces.length} items (Page {page + 1})
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-2">


          <div className="flex items-center gap-2">
            <MultiSelect
              label="Class"
              icon={User}
              options={filterOptions.classifications}
              selected={filters.classifications}
              onChange={handleClassificationChange}
              noneLabel="Unclassified"
            />

            <div className="w-px h-6 bg-slate-800 mx-1"></div>

            <MultiSelect
              label="Keywords"
              icon={Tag}
              options={filterOptions.keywords}
              selected={filters.keywords}
              onChange={handleKeywordChange}
              noneLabel="No Keywords"
            />
          </div>

          {(filters.classifications.length > 0 || filters.keywords.length > 0) && (
            <button
              onClick={clearFilters}
              className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
              title="Clear Filters"
            >
              <X className="w-5 h-5" />
            </button>
          )}


          <button
            onClick={handleExport}
            className="p-2 text-slate-400 hover:text-indigo-400 hover:bg-slate-800 rounded-lg transition-colors"
            title="Export Filtered Results"
          >
            <Download className="w-5 h-5" />
          </button>

          <button
            onClick={() => { fetchFaces(); fetchFilterOptions(); }}
            className="p-2 text-slate-400 hover:text-indigo-400 hover:bg-slate-800 rounded-lg transition-colors"
            title="Refresh"
          >
            <RefreshCw className="w-5 h-5" />
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {faces.length === 0 && !loading && (
          <div className="col-span-full py-12 text-center text-slate-500 bg-slate-900/50 rounded-xl border border-dashed border-slate-800">
            <p>No records found matching your filters.</p>
            <button onClick={clearFilters} className="mt-2 text-indigo-400 hover:underline">Clear Filters</button>
          </div>
        )}

        {faces.map((face) => (
          <div
            key={face.id}
            className="group bg-slate-900 border border-slate-800 rounded-xl overflow-hidden hover:border-indigo-500/50 transition-all hover:shadow-xl hover:shadow-indigo-500/10"
          >
            {/* Image Container */}
            <div className="aspect-square bg-slate-950 relative overflow-hidden">
              <img
                src={`/faces/${face.id}/image`}
                alt={face.image_name}

                onClick={() => setSelectedImage(face)}
                className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110 cursor-pointer"
                loading="lazy"
              />
              <div className="absolute top-2 right-2 pointer-events-none">
                <span className={`px-2 py-1 rounded text-xs font-bold border ${face.is_valid_pose
                  ? 'bg-green-500/20 text-green-400 border-green-500/30'
                  : 'bg-red-500/20 text-red-400 border-red-500/30'
                  }`}>

                  {face.is_valid_pose ? "Valid" : "Invalid"}
                </span>
              </div>
            </div>

            {/* Details */}
            <div className="p-4 space-y-3">
              {editingId === face.id ? (
                // Edit Mode
                <div className="space-y-2 animate-in fade-in">
                  <input
                    value={editForm.description}
                    onChange={e => setEditForm({ ...editForm, description: e.target.value })}
                    className="w-full bg-slate-950 border border-indigo-500 rounded px-2 py-1 text-sm text-slate-200 focus:outline-none"
                    placeholder="Description"
                    autoFocus
                  />
                  <div className="flex gap-2">
                    <input
                      value={editForm.classification}
                      onChange={e => setEditForm({ ...editForm, classification: e.target.value })}
                      className="w-1/2 bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs text-slate-200 focus:border-indigo-500 focus:outline-none"
                      placeholder="Class"
                    />
                    <input
                      value={editForm.keywords}
                      onChange={e => setEditForm({ ...editForm, keywords: e.target.value })}
                      className="w-1/2 bg-slate-950 border border-slate-700 rounded px-2 py-1 text-xs text-slate-200 focus:border-indigo-500 focus:outline-none"
                      placeholder="Keywords"
                    />
                  </div>
                  <div className="flex gap-2 justify-between mt-2 pt-2 border-t border-slate-800">


                    <button
                      onClick={() => deleteFace(face.id)}
                      className="p-1.5 bg-red-900/30 text-red-400 rounded hover:bg-red-900/50 transition-colors"
                      title="Delete Image"
                    >
                      <Trash2 size={14} />
                    </button>

                    <div className="flex gap-2">
                      <button
                        onClick={() => saveEdit(face.id)}
                        className="p-1.5 bg-green-600/20 text-green-400 rounded hover:bg-green-600/30 transition-colors"
                        title="Save"
                      >
                        <Check size={14} />
                      </button>
                      <button
                        onClick={cancelEditing}
                        className="p-1.5 bg-slate-600/20 text-slate-400 rounded hover:bg-slate-600/30 transition-colors"
                        title="Cancel"
                      >
                        <X size={14} />
                      </button>
                    </div>
                  </div>
                </div>
              ) : (
                // Display Mode
                <>
                  <div className="flex items-start justify-between gap-2">
                    <h3 className="font-medium text-slate-200 truncate" title={face.description || face.image_name}>
                      {face.description || face.image_name}
                    </h3>
                    <button
                      onClick={() => startEditing(face)}
                      className="text-slate-600 hover:text-indigo-400 transition-colors opacity-0 group-hover:opacity-100"
                      title="Edit Metadata"
                    >
                      <Edit2 className="w-4 h-4" />
                    </button>
                  </div>

                  <div className="space-y-2 text-xs text-slate-400">
                    <div className="flex items-center gap-2">
                      <User className="w-3 h-3" />
                      <span className="truncate">{face.classification || 'Unclassified'}</span>
                    </div>

                    <div className="flex items-center gap-2">
                      <Tag className="w-3 h-3" />
                      <span className="truncate">{face.keywords || 'No keywords'}</span>
                    </div>


                    <div className="flex items-center gap-2">
                      <Maximize className="w-3 h-3" />
                      <span className="truncate">
                        {face.width && face.height ? `${face.width} x ${face.height} px` : 'Unknown Size'}
                      </span>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="flex items-center justify-center gap-4 mt-8 pt-4 border-t border-slate-800">
        <button
          onClick={() => setPage((p) => Math.max(0, p - 1))}
          disabled={page === 0}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-800 border border-slate-700 hover:bg-slate-700 hover:text-white disabled:opacity-50 disabled:cursor-not-allowed transition-all text-slate-400"
        >
          <ChevronLeft className="w-4 h-4" />
          Previous
        </button>

        <span className="text-slate-500 font-mono text-sm bg-slate-900 px-3 py-1 rounded border border-slate-800">
          Page {page + 1}
        </span>

        <button
          onClick={() => setPage((p) => p + 1)}
          disabled={!hasNextPage}
          className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-800 border border-slate-700 hover:bg-slate-700 hover:text-white disabled:opacity-50 disabled:cursor-not-allowed transition-all text-slate-400"
        >
          Next
          <ChevronRight className="w-4 h-4" />
        </button>
      </div>


      {selectedImage && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200"
          onClick={() => setSelectedImage(null)}
        >
          <div
            className="relative max-w-7xl w-full max-h-[95vh] flex flex-col items-center"
            onClick={e => e.stopPropagation()}
          >
            <button
              onClick={() => setSelectedImage(null)}
              className="absolute -top-12 right-0 p-2 text-slate-400 hover:text-white transition-colors"
            >
              <X size={32} />
            </button>

            <img
              src={`/faces/${selectedImage.id}/full_image`}
              alt={selectedImage.image_name}
              className="max-w-full max-h-[85vh] object-contain rounded-lg shadow-2xl bg-slate-900"
            />

            <div className="mt-4 bg-slate-900/90 px-6 py-3 rounded-full border border-slate-700 shadow-xl backdrop-blur text-slate-200 flex gap-4 items-center">
              <span className="font-medium">{selectedImage.description || selectedImage.image_name}</span>
              {selectedImage.classification && (
                <span className="text-xs bg-indigo-500/20 text-indigo-300 px-2 py-1 rounded border border-indigo-500/30">
                  {selectedImage.classification}
                </span>
              )}

              <div className="w-px h-5 bg-slate-700 mx-2"></div>

              <button
                onClick={handleReanalyze}
                disabled={analyzing}
                className="flex items-center gap-2 text-xs font-semibold bg-indigo-600 hover:bg-indigo-500 text-white px-3 py-1.5 rounded-full transition-colors disabled:opacity-50"
              >
                {analyzing ? (
                  <Loader2 className="w-3 h-3 animate-spin" />
                ) : (
                  <ScanSearch className="w-3 h-3" />
                )}
                Run Analysis
              </button>


              {selectedImage.yaw !== undefined && (
                <div className="text-xs text-slate-400 flex flex-col leading-tight ml-2">
                  <span>Y: {selectedImage.yaw?.toFixed(1)}°</span>
                  <span>P: {selectedImage.pitch?.toFixed(1)}°</span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

    </div>
  );
}