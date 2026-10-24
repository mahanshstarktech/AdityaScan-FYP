import React from 'react';
import { X, Activity, Zap, Shield, AlertTriangle, Layers } from 'lucide-react';

export interface ActiveRegionDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  regionId: string;
  flareProbability: number;
  magneticClass: string;
  pilLengthMm: number;
  mcIntoshClass?: string;
  shearAnglePct?: number;
  sympatheticLinks?: string[];
}

export const ActiveRegionDrawer: React.FC<ActiveRegionDrawerProps> = ({
  isOpen,
  onClose,
  regionId,
  flareProbability,
  magneticClass,
  pilLengthMm,
  mcIntoshClass = 'Fkc',
  shearAnglePct = 84.6,
  sympatheticLinks = ['AR13668', 'AR13663'],
}) => {
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    if (isOpen) window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-y-0 right-0 z-50 flex w-full max-w-md flex-col border-l border-slate-800 bg-slate-950/95 p-6 shadow-2xl backdrop-blur-xl animate-in slide-in-from-right duration-300">
      <div className="flex items-center justify-between border-b border-slate-800 pb-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Activity className="h-5 w-5 text-cyan-400" />
            Active Region {regionId}
          </h2>
          <p className="text-xs text-slate-400">SDO/HMI SHARP Vector Magnetogram Analysis</p>
        </div>
        <button
          onClick={onClose}
          className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-white transition"
        >
          <X className="h-5 w-5" />
        </button>
      </div>

      <div className="mt-6 space-y-4">
        <div className="rounded-lg border border-slate-800/80 bg-slate-900/50 p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400">Hale Magnetic Class</span>
            <span className="rounded bg-amber-950/60 px-2 py-0.5 text-[10px] font-mono text-amber-400 border border-amber-800">
              {mcIntoshClass}
            </span>
          </div>
          <p className="text-xl font-mono font-bold text-amber-400 mt-1">{magneticClass}</p>
          <p className="text-xs text-slate-400 mt-2">PIL Length: <span className="font-mono text-cyan-300">{pilLengthMm.toLocaleString()} Mm</span></p>
        </div>

        <div className="rounded-lg border border-slate-800/80 bg-slate-900/50 p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400">Magnetic Shear (&gt;45°)</span>
            <span className="text-xs font-mono font-bold text-rose-400">{shearAnglePct}% Area</span>
          </div>
          <div className="mt-2 h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
            <div className="h-full bg-rose-500 rounded-full" style={{ width: `${shearAnglePct}%` }} />
          </div>
        </div>

        <div className="rounded-lg border border-slate-800/80 bg-slate-900/50 p-4">
          <span className="text-xs text-slate-400">Sympathetic Arcade Links</span>
          <div className="flex gap-2 mt-2">
            {sympatheticLinks.map((ar) => (
              <span key={ar} className="rounded bg-slate-800 px-2 py-0.5 text-[11px] font-mono text-cyan-300 border border-slate-700">
                {ar}
              </span>
            ))}
          </div>
        </div>

        <div className="rounded-lg border border-slate-800/80 bg-slate-900/50 p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400">Operational Risk</span>
            <span className="rounded bg-rose-950/60 px-2 py-0.5 text-[10px] font-mono text-rose-400 border border-rose-800">
              24h Horizon
            </span>
          </div>
          <p className="text-2xl font-mono font-bold text-rose-500 mt-1">{flareProbability}%</p>
        </div>
      </div>
    </div>
  );
};
