import { useEffect, useState } from 'react';
import {
  Loader2, Tag, User, Edit2, Check, X,
  ChevronLeft, ChevronRight, Filter, RefreshCw
} from 'lucide-react';
import type { FaceRecord } from './types';

const ITEMS_PER_PAGE = 8;

export default function Gallery() {
  const [faces, setFaces] = useState<FaceRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(0);


  const [selectedImage, setSelectedImage] = useState<FaceRecord | null>(null);

  const [filters, setFilters] = useState({
    keyword: '',
    classification: ''
  });


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
      params.append('limit', ITEMS_PER_PAGE.toString());
      params.append('offset', offset.toString());

      if (filters.keyword) params.append('keyword', filters.keyword);
      if (filters.classification) params.append('classification', filters.classification);

      const res = await fetch(`/faces?${params.toString()}`);
      const data = await res.json();
      setFaces(data);
    } catch (err) {
      console.error("Failed to load faces", err);
    } finally {
      setLoading(false);
    }
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


  const handleFilterChange = (key: 'keyword' | 'classification', value: string) => {
    setFilters(prev => ({ ...prev, [key]: value }));
    setPage(0); // Reset to first page on filter change
  };


  const clearFilters = () => {
    setFilters({ keyword: '', classification: '' });
    setPage(0);
  };

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
          <div className="flex items-center gap-2 bg-slate-950 px-3 py-2 rounded-lg border border-slate-800">
            <Filter className="w-4 h-4 text-indigo-400" />

            {/* Classification Filter */}
            <select
              value={filters.classification}
              onChange={(e) => handleFilterChange('classification', e.target.value)}
              className="bg-transparent text-sm text-slate-300 focus:outline-none w-32"
            >
              <option value="">All Classes</option>

              <option value="__NONE__">Unclassified</option>
              {filterOptions.classifications.map((c, i) => (
                <option key={i} value={c}>{c}</option>
              ))}
            </select>

            <div className="w-px h-4 bg-slate-800 mx-2"></div>

            {/* Keyword Filter */}
            <Tag className="w-4 h-4 text-indigo-400" />
            <select
              value={filters.keyword}
              onChange={(e) => handleFilterChange('keyword', e.target.value)}
              className="bg-transparent text-sm text-slate-300 focus:outline-none w-32"
            >
              <option value="">All Keywords</option>

              <option value="__NONE__">No Keywords</option>
              {filterOptions.keywords.map((k, i) => (
                <option key={i} value={k}>{k}</option>
              ))}
            </select>
          </div>

          {(filters.classification || filters.keyword) && (
            <button
              onClick={clearFilters}
              className="p-2 text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-colors"
              title="Clear Filters"
            >
              <X className="w-5 h-5" />
            </button>
          )}

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
                  {face.direction}
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
                  <div className="flex gap-2 justify-end mt-2 pt-2 border-t border-slate-800">
                    <button
                      onClick={() => saveEdit(face.id)}
                      className="p-1.5 bg-green-600/20 text-green-400 rounded hover:bg-green-600/30 transition-colors"
                      title="Save"
                    >
                      <Check size={14} />
                    </button>
                    <button
                      onClick={cancelEditing}
                      className="p-1.5 bg-red-600/20 text-red-400 rounded hover:bg-red-600/30 transition-colors"
                      title="Cancel"
                    >
                      <X size={14} />
                    </button>
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
          disabled={faces.length < ITEMS_PER_PAGE}
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
            </div>
          </div>
        </div>
      )}

    </div>
  );
}