import React from 'react';
import { X, Activity, Zap, Shield, AlertTriangle } from 'lucide-react';

interface ActiveRegionProps {
  isOpen: boolean;
  onClose: () => void;
  regionId: string;
  flareProbability: number;
  magneticClass: string;
  pilLengthMm: number;
}

export const ActiveRegionDrawer: React.FC<ActiveRegionProps> = ({
  isOpen,
  onClose,
  regionId,
  flareProbability,
  magneticClass,
  pilLengthMm,
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
          <p className="text-xs text-slate-400">SDO/HMI SHARP Magnetic Complexity</p>
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
          <span className="text-xs text-slate-400">Hale Magnetic Class</span>
          <p className="text-xl font-mono font-bold text-amber-400">{magneticClass}</p>
        </div>

        <div className="rounded-lg border border-slate-800/80 bg-slate-900/50 p-4">
          <span className="text-xs text-slate-400">Polarity Inversion Line (PIL) Length</span>
          <p className="text-xl font-mono font-bold text-cyan-400">{pilLengthMm.toLocaleString()} Mm</p>
        </div>

        <div className="rounded-lg border border-slate-800/80 bg-slate-900/50 p-4">
          <span className="text-xs text-slate-400">24h X-Class Flare Risk</span>
          <p className="text-2xl font-mono font-bold text-rose-500">{flareProbability}%</p>
        </div>
      </div>
    </div>
  );
};
