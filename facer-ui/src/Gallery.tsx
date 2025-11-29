import { useEffect, useState } from 'react';
import { Loader2, Calendar, Tag, User } from 'lucide-react';
import type { FaceRecord } from './types';

export default function Gallery() {
  const [faces, setFaces] = useState<FaceRecord[]>([]);
  const [loading, setLoading] = useState(true);

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
                <span className={`px-2 py-1 rounded text-xs font-bold border ${
                  face.is_valid_pose 
                    ? 'bg-green-500/20 text-green-400 border-green-500/30' 
                    : 'bg-red-500/20 text-red-400 border-red-500/30'
                }`}>
                  {face.direction}
                </span>
              </div>
            </div>

            {/* Details */}
            <div className="p-4 space-y-3">
              <div className="flex items-start justify-between gap-2">
                <h3 className="font-medium text-slate-200 truncate" title={face.image_name}>
                  {face.description || face.image_name}
                </h3>
              </div>

              <div className="space-y-2 text-xs text-slate-400">
                <div className="flex items-center gap-2">
                  <User className="w-3 h-3" />
                  <span className="truncate">{face.classification || 'Unclassified'}</span>
                </div>
                {face.keywords && (
                  <div className="flex items-center gap-2">
                    <Tag className="w-3 h-3" />
                    <span className="truncate">{face.keywords}</span>
                  </div>
                )}
                <div className="flex items-center gap-2 text-slate-600">
                  <Calendar className="w-3 h-3" />
                  <span>{new Date(face.created_at).toLocaleDateString()}</span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
