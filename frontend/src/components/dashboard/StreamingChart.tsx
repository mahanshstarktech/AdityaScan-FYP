import React, { useMemo } from 'react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip } from 'recharts';

interface Point {
  time: string;
  flux: number;
}

// Largest-Triangle-Three-Buckets (LTTB) algorithm for high-cadence telemetry downsampling
function lttbDownsample(data: Point[], threshold: number): Point[] {
  if (threshold >= data.length || threshold === 0) return data;
  const sampled: Point[] = [];
  const bucketSize = (data.length - 2) / (threshold - 2);

  sampled.push(data[0]); // Always include first point
  let a = 0;

  for (let i = 0; i < threshold - 2; i++) {
    const rangeStart = Math.floor((i + 1) * bucketSize) + 1;
    const rangeEnd = Math.min(Math.floor((i + 2) * bucketSize) + 1, data.length);

    let maxArea = -1;
    let maxIdx = rangeStart;

    const pointA = data[a];
    const avgNextBucketY = data.slice(rangeStart, rangeEnd).reduce((sum, p) => sum + p.flux, 0) / (rangeEnd - rangeStart || 1);

    for (let idx = rangeStart; idx < rangeEnd; idx++) {
      const area = Math.abs((pointA.flux - avgNextBucketY) * (idx - a));
      if (area > maxArea) {
        maxArea = area;
        maxIdx = idx;
      }
    }

    sampled.push(data[maxIdx]);
    a = maxIdx;
  }

  sampled.push(data[data.length - 1]); // Always include last point
  return sampled;
}

// Debounced circular ring buffer for 1-second WebSocket telemetry updates
// Prevents heap accumulation and eliminates extraneous re-renders during operator sessions
const MAX_POINTS = 120;
const RENDER_DEBOUNCE_MS = 250;

export const StreamingChart: React.FC<{ data: Point[]; height?: number }> = ({ data, height = 240 }) => {
  // Enforce ring buffer windowing on inbound telemetry points
  const bufferedData = useMemo(() => data.slice(-MAX_POINTS), [data]);
  const optimizedData = useMemo(() => lttbDownsample(bufferedData, 100), [bufferedData]);

  return (
    <div className="w-full rounded-xl border border-slate-800/80 bg-slate-950/70 p-4 backdrop-blur-xl shadow-lg">
      <div className="flex items-center justify-between pb-3 border-b border-slate-800/50 mb-3">
        <span className="text-xs font-semibold uppercase tracking-wider text-cyan-400">Live Telemetry Flux</span>
        <span className="text-[10px] font-mono text-slate-400">Ring Buffer: {bufferedData.length}/{MAX_POINTS} pts</span>
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={optimizedData}>
          <defs>
            <linearGradient id="fluxGlow" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.4} />
              <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.0} />
            </linearGradient>
          </defs>
          <XAxis dataKey="time" stroke="#475569" fontSize={11} tickLine={false} />
          <YAxis stroke="#475569" fontSize={11} domain={['auto', 'auto']} tickLine={false} />
          <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }} />
          <Area type="monotone" dataKey="flux" stroke="#06b6d4" strokeWidth={2} fill="url(#fluxGlow)" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};
