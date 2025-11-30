import { useEffect, useState } from 'react';
import { Loader2, Tag, User, Edit2, Check, X } from 'lucide-react';
import type { FaceRecord } from './types';

export default function Gallery() {
  const [faces, setFaces] = useState<FaceRecord[]>([]);
  const [loading, setLoading] = useState(true);


  const [editingId, setEditingId] = useState<number | null>(null);
  const [editForm, setEditForm] = useState<{
    description: string;
    classification: string;
    keywords: string;
  }>({ description: '', classification: '', keywords: '' });

  useEffect(() => {
    fetchFaces();
  }, []);

  const fetchFaces = async () => {
    try {
      const res = await fetch('/faces?limit=100');
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
        // Optimistic update of local state
        setFaces(faces.map(f => f.id === id ? { ...f, ...editForm } : f));
        setEditingId(null);
      } else {
        console.error("Failed to save changes");
      }
    } catch (err) {
      console.error("Failed to update", err);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 text-indigo-400">
        <Loader2 className="w-8 h-8 animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-slate-200">Database Records</h2>
        <span className="text-slate-500 text-sm">{faces.length} items found</span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
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
                className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
                loading="lazy"
              />
              <div className="absolute top-2 right-2">
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
    </div>
  );
}