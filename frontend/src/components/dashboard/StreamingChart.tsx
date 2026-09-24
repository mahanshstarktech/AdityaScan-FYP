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

export const StreamingChart: React.FC<{ data: Point[]; height?: number }> = ({ data, height = 240 }) => {
  const optimizedData = useMemo(() => lttbDownsample(data, 100), [data]);

  return (
    <div className="w-full rounded-xl border border-slate-800 bg-slate-950/60 p-4 backdrop-blur-md">
      <div className="flex items-center justify-between pb-3">
        <span className="text-xs font-semibold uppercase tracking-wider text-cyan-400">Telemetry Flux (LTTB Optimized)</span>
        <span className="text-[10px] text-slate-400">Latency: <120ms | Buffer: {data.length} pts → Rendered: {optimizedData.length} pts</span>
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
