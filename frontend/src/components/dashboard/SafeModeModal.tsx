import React, { useState } from 'react';
import { Shield, AlertTriangle, CheckSquare, Square, X } from 'lucide-react';

interface SafeModeModalProps {
  isOpen: boolean;
  onClose: () => void;
  constellation: string;
}

export const SafeModeModal: React.FC<SafeModeModalProps> = ({ isOpen, onClose, constellation }) => {
  const [steps, setSteps] = useState([
    { id: 1, text: 'Confirm flare alert classification & arrival ETA with ISRO ISTRAC', done: false },
    { id: 2, text: 'Slew solar panels to feathered low-drag safe orientation', done: false },
    { id: 3, text: 'De-energize high-voltage payload sensor grids (ASPEX/SUIT)', done: false },
    { id: 4, text: 'Upload emergency star-tracker shield orientation vector', done: false },
    { id: 5, text: 'Log emergency telemetry snapshot to mission control repository', done: false },
  ]);

  if (!isOpen) return null;

  const toggleStep = (id: number) => {
    setSteps(prev => prev.map(s => s.id === id ? { ...s, done: !s.done } : s));
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-md animate-in fade-in">
      <div className="w-full max-w-lg rounded-2xl border border-slate-800 bg-slate-950 p-6 shadow-2xl">
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div className="flex items-center gap-3">
            <Shield className="h-6 w-6 text-rose-500" />
            <h3 className="text-lg font-bold text-white">Satellite Safe-Mode Protocol: {constellation}</h3>
          </div>
          <button onClick={onClose} className="rounded p-1 text-slate-400 hover:text-white">
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="mt-4 space-y-3">
          {steps.map(s => (
            <div
              key={s.id}
              onClick={() => toggleStep(s.id)}
              className="flex cursor-pointer items-center gap-3 rounded-lg border border-slate-800/80 bg-slate-900/50 p-3 hover:bg-slate-900 transition"
            >
              {s.done ? (
                <CheckSquare className="h-5 w-5 text-emerald-400" />
              ) : (
                <Square className="h-5 w-5 text-slate-500" />
              )}
              <span className={`text-sm ${s.done ? 'line-through text-slate-500' : 'text-slate-200'}`}>
                {s.text}
              </span>
            </div>
          ))}
        </div>

        <div className="mt-6 flex justify-end gap-3">
          <button onClick={onClose} className="rounded-lg px-4 py-2 text-xs font-semibold text-slate-400 hover:text-white">
            Close
          </button>
          <button onClick={onClose} className="rounded-lg bg-rose-600 px-4 py-2 text-xs font-bold text-white hover:bg-rose-500">
            Acknowledge Protocols
          </button>
        </div>
      </div>
    </div>
  );
};
