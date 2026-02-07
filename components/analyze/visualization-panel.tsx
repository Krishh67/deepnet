'use client'

import { Card } from '@/components/ui/card'
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

interface VisualizationPanelProps {
  result: any
}

export default function VisualizationPanel({ result }: VisualizationPanelProps) {
  // Mock waveform data
  const waveformData = Array.from({ length: 100 }, (_, i) => ({
    time: (i * 0.1).toFixed(1),
    amplitude: Math.sin(i * 0.3) * Math.cos(i * 0.1) * 50 + 50,
  }))

  // Mock spectrogram data (frequency over time)
  const spectrogramData = Array.from({ length: 50 }, (_, i) => ({
    time: (i * 0.2).toFixed(1),
    frequency: Math.sin(i * 0.2) * 20 + 15,
    energy: Math.sin(i * 0.1) * 40 + 60,
  }))

  // Mock energy data
  const energyData = Array.from({ length: 40 }, (_, i) => ({
    time: (i * 0.1).toFixed(1),
    energy: Math.abs(Math.sin(i * 0.15) * 100 + 50),
    rms: Math.abs(Math.cos(i * 0.1) * 70 + 60),
  }))

  return (
    <div className="space-y-6">
      {/* Waveform */}
      <Card className="p-6 bg-card/50 border-border/50 backdrop-blur">
        <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-primary"></div>
          Waveform Visualization
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={waveformData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(190, 214, 255, 0.1)" />
            <XAxis dataKey="time" stroke="rgba(190, 214, 255, 0.5)" label={{ value: 'Time (s)', position: 'insideBottomRight', offset: -5 }} />
            <YAxis stroke="rgba(190, 214, 255, 0.5)" label={{ value: 'Amplitude', angle: -90, position: 'insideLeft' }} />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(10, 30, 60, 0.8)',
                border: '1px solid rgba(190, 214, 255, 0.3)',
                borderRadius: '8px',
              }}
            />
            <Line
              type="monotone"
              dataKey="amplitude"
              stroke="#5fc3ff"
              dot={false}
              strokeWidth={2}
              isAnimationActive={true}
            />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      {/* Spectrogram - Real Image from Backend */}
      <Card className="p-6 bg-card/50 border-border/50 backdrop-blur">
        <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-accent"></div>
          Filtered Spectrogram (0-50 Hz)
        </h3>
        {result.spectrogramBase64 ? (
          <div className="rounded-lg overflow-hidden border border-border/30 bg-background/50 p-2">
            <img
              src={`data:image/png;base64,${result.spectrogramBase64}`}
              alt="Audio Spectrogram"
              className="w-full h-auto rounded"
            />
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={spectrogramData}>
              <defs>
                <linearGradient id="colorFreq" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#5fc3ff" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="#5fc3ff" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(190, 214, 255, 0.1)" />
              <XAxis dataKey="time" stroke="rgba(190, 214, 255, 0.5)" label={{ value: 'Time (s)', position: 'insideBottomRight', offset: -5 }} />
              <YAxis stroke="rgba(190, 214, 255, 0.5)" label={{ value: 'Frequency (Hz)', angle: -90, position: 'insideLeft' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'rgba(10, 30, 60, 0.8)',
                  border: '1px solid rgba(190, 214, 255, 0.3)',
                  borderRadius: '8px',
                }}
              />
              <Area
                type="monotone"
                dataKey="frequency"
                stroke="#5fc3ff"
                fillOpacity={1}
                fill="url(#colorFreq)"
                isAnimationActive={true}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </Card>

      {/* Energy Graph */}
      <Card className="p-6 bg-card/50 border-border/50 backdrop-blur">
        <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-primary/80"></div>
          RMS Energy Over Time
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={energyData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(190, 214, 255, 0.1)" />
            <XAxis dataKey="time" stroke="rgba(190, 214, 255, 0.5)" label={{ value: 'Time (s)', position: 'insideBottomRight', offset: -5 }} />
            <YAxis stroke="rgba(190, 214, 255, 0.5)" label={{ value: 'Energy (dB)', angle: -90, position: 'insideLeft' }} />
            <Tooltip
              contentStyle={{
                backgroundColor: 'rgba(10, 30, 60, 0.8)',
                border: '1px solid rgba(190, 214, 255, 0.3)',
                borderRadius: '8px',
              }}
            />
            <Line
              type="monotone"
              dataKey="energy"
              stroke="#5fc3ff"
              dot={false}
              strokeWidth={2}
              isAnimationActive={true}
              name="Energy"
            />
            <Line
              type="monotone"
              dataKey="rms"
              stroke="#b4d4ff"
              dot={false}
              strokeWidth={2}
              isAnimationActive={true}
              name="RMS"
              opacity={0.6}
            />
          </LineChart>
        </ResponsiveContainer>
      </Card>
    </div>
  )
}
