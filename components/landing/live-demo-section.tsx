'use client'

import { useEffect, useState } from 'react'
import { Card } from '@/components/ui/card'
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

const spectrogramData = Array.from({ length: 20 }, (_, i) => ({
  time: `${i * 5}s`,
  freq1: Math.sin(i * 0.5) * 40 + 50,
  freq2: Math.cos(i * 0.3) * 35 + 45,
  freq3: Math.sin(i * 0.7) * 30 + 40,
}))

const energyData = Array.from({ length: 30 }, (_, i) => ({
  time: i,
  energy: Math.abs(Math.sin(i * 0.2) * 80 + Math.random() * 20 + 40),
}))

const detectionEvents = [
  { id: 1, time: '14:32:15', type: 'Seismic Event', confidence: 92, risk: 'High' },
  { id: 2, time: '14:28:42', type: 'Marine Animal', confidence: 87, risk: 'Low' },
  { id: 3, time: '14:22:08', type: 'Unknown Source', confidence: 71, risk: 'Medium' },
]

const frequencyPeaks = [
  { freq: '8 Hz', intensity: 85, type: 'Seismic' },
  { freq: '15 Hz', intensity: 62, type: 'Marine' },
  { freq: '32 Hz', intensity: 45, type: 'Noise' },
  { freq: '47 Hz', intensity: 38, type: 'Background' },
]

interface LiveDemoSectionProps {
  onStartListening?: () => void
}

export default function LiveDemoSection({ onStartListening }: LiveDemoSectionProps) {
  const [isLive, setIsLive] = useState(true)
  const [detectionsList, setDetectionsList] = useState(detectionEvents)
  const [detectionCount, setDetectionCount] = useState(2847)
  const [tsunamiRisk, setTsunamiRisk] = useState(34)
  const [avgConfidence, setAvgConfidence] = useState(86)

  useEffect(() => {
    if (!isLive) return

    const interval = setInterval(() => {
      setDetectionsList((prev) => {
        const types = ['Seismic Event', 'Marine Animal', 'Unknown Source', 'Acoustic Anomaly']
        const risks = ['Low', 'Medium', 'High']
        const newEvent = {
          id: Math.random(),
          time: new Date().toLocaleTimeString(),
          type: types[Math.floor(Math.random() * types.length)],
          confidence: Math.floor(Math.random() * 30) + 65,
          risk: risks[Math.floor(Math.random() * risks.length)],
        }
        return [newEvent, ...prev.slice(0, 4)]
      })

      // Update metrics
      setDetectionCount((prev) => prev + Math.floor(Math.random() * 5) + 1)
      setTsunamiRisk((prev) => {
        const change = (Math.random() - 0.5) * 8
        const newRisk = Math.max(10, Math.min(80, prev + change))
        return Math.round(newRisk)
      })
      setAvgConfidence((prev) => {
        const change = (Math.random() - 0.5) * 5
        const newConf = Math.max(60, Math.min(95, prev + change))
        return Math.round(newConf)
      })
    }, 4000)

    return () => clearInterval(interval)
  }, [isLive])

  return (
    <section className="py-20 px-4 relative">
      <div className="max-w-7xl mx-auto">
        {/* Section Header */}
        <div className="text-center mb-16">
          <div className="inline-block px-4 py-2 rounded-full bg-primary/10 border border-primary/30 mb-4">
            <span className="text-primary font-medium text-sm">LIVE MONITORING</span>
          </div>
          <h2 className="text-4xl md:text-5xl font-bold mb-4 text-foreground">Model Performing on Live Data</h2>
          <p className="text-foreground/70 max-w-2xl mx-auto">Watch the AI actively listen and analyze real-time ocean acoustic data with dynamic visualizations</p>
        </div>

        {/* Live Status */}
        <div className="flex items-center justify-center gap-3 mb-8">
          <div className="w-3 h-3 bg-red-500 rounded-full animate-pulse"></div>
          <span className="text-foreground font-semibold">Live Detection Active</span>
          <button
            onClick={() => setIsLive(!isLive)}
            className="ml-4 px-4 py-1.5 rounded-lg bg-primary/20 border border-primary/50 text-primary text-sm hover:bg-primary/30 transition-colors"
          >
            {isLive ? 'Pause' : 'Resume'}
          </button>
        </div>

        {/* Key Metrics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          {/* Detections Counter */}
          <Card className="p-6 bg-gradient-to-br from-primary/20 to-primary/5 border border-primary/50 backdrop-blur">
            <div className="flex items-end justify-between">
              <div>
                <p className="text-foreground/70 text-sm font-medium mb-2">Total Detections</p>
                <p className="text-4xl font-bold text-primary">{detectionCount.toLocaleString()}</p>
              </div>
              <div className="text-3xl opacity-30">📊</div>
            </div>
          </Card>

          {/* Average Confidence */}
          <Card className="p-6 bg-gradient-to-br from-accent/20 to-accent/5 border border-accent/50 backdrop-blur">
            <div className="flex items-end justify-between">
              <div>
                <p className="text-foreground/70 text-sm font-medium mb-2">Avg Confidence</p>
                <p className="text-4xl font-bold text-accent">{avgConfidence}%</p>
              </div>
              <div className="text-3xl opacity-30">🎯</div>
            </div>
          </Card>

          {/* Tsunami Risk Meter */}
          <Card className="p-6 bg-gradient-to-br from-orange-500/20 to-orange-500/5 border border-orange-500/50 backdrop-blur">
            <div className="flex items-end justify-between">
              <div>
                <p className="text-foreground/70 text-sm font-medium mb-2">Tsunami Risk</p>
                <p className="text-4xl font-bold text-orange-400">{tsunamiRisk}%</p>
              </div>
              <div className="text-3xl opacity-30">⚠️</div>
            </div>
          </Card>
        </div>

        {/* Visualizations Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Spectrogram */}
          <Card className="p-6 bg-card/50 border border-border/50 backdrop-blur">
            <div className="mb-4">
              <h3 className="text-lg font-semibold text-foreground">Frequency Spectrogram</h3>
              <p className="text-sm text-foreground/60">Filtered 0-50 Hz underwater acoustic band</p>
            </div>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={spectrogramData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1a4d5c" />
                  <XAxis dataKey="time" tick={{ fill: '#b0d6ff', fontSize: 12 }} />
                  <YAxis tick={{ fill: '#b0d6ff', fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0a2540',
                      border: '1px solid #1a7d9d',
                      borderRadius: '8px',
                    }}
                  />
                  <Line type="monotone" dataKey="freq1" stroke="#5fc3ff" strokeWidth={2} dot={false} isAnimationActive={isLive} />
                  <Line type="monotone" dataKey="freq2" stroke="#2dd9c1" strokeWidth={2} dot={false} isAnimationActive={isLive} />
                  <Line type="monotone" dataKey="freq3" stroke="#64b5f6" strokeWidth={2} dot={false} isAnimationActive={isLive} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>

          {/* Energy Analysis */}
          <Card className="p-6 bg-card/50 border border-border/50 backdrop-blur">
            <div className="mb-4">
              <h3 className="text-lg font-semibold text-foreground">RMS Energy Over Time</h3>
              <p className="text-sm text-foreground/60">Real-time acoustic intensity analysis</p>
            </div>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={energyData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1a4d5c" />
                  <XAxis dataKey="time" tick={{ fill: '#b0d6ff', fontSize: 12 }} />
                  <YAxis tick={{ fill: '#b0d6ff', fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#0a2540',
                      border: '1px solid #1a7d9d',
                      borderRadius: '8px',
                    }}
                  />
                  <Bar dataKey="energy" fill="#2dd9c1" isAnimationActive={isLive} radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </div>

        {/* Frequency Peaks Analysis */}
        <Card className="p-6 bg-card/50 border border-border/50 backdrop-blur mb-8">
          <div className="mb-4">
            <h3 className="text-lg font-semibold text-foreground">Frequency Peaks Analysis</h3>
            <p className="text-sm text-foreground/60">Dominant frequencies detected in real-time</p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {frequencyPeaks.map((peak, idx) => (
              <div key={idx} className="p-4 rounded-lg bg-background/40 border border-border/30">
                <div className="mb-3">
                  <p className="text-sm text-foreground/70 mb-2">{peak.freq}</p>
                  <div className="flex items-end gap-2">
                    <div className="w-full h-16 bg-background/50 rounded relative overflow-hidden">
                      <div
                        className="absolute bottom-0 left-0 w-full bg-gradient-to-t from-primary to-accent opacity-70 transition-all"
                        style={{ height: `${peak.intensity}%` }}
                      ></div>
                    </div>
                    <p className="text-lg font-bold text-primary min-w-fit">{peak.intensity}</p>
                  </div>
                </div>
                <p className="text-xs text-foreground/60 bg-background/50 px-2 py-1 rounded w-fit">{peak.type}</p>
              </div>
            ))}
          </div>
        </Card>

        {/* Listening Visualization */}
        <Card className="p-6 bg-card/50 border border-border/50 backdrop-blur mb-8">
          <div className="mb-6">
            <h3 className="text-lg font-semibold text-foreground">AI Listening State</h3>
            <p className="text-sm text-foreground/60">System actively monitoring hydrophones</p>
          </div>
          <div className="flex flex-col items-center justify-center py-8">
            <div className="relative w-32 h-32 mb-6">
              {/* Pulsing circles */}
              <div className="absolute inset-0 rounded-full border-2 border-primary/30 animate-pulse" style={{ animation: 'pulse 2s infinite' }}></div>
              <div
                className="absolute inset-4 rounded-full border-2 border-accent/40 animate-pulse"
                style={{ animation: 'pulse 2s infinite 0.3s' }}
              ></div>
              <div
                className="absolute inset-8 rounded-full border-2 border-primary/50"
                style={{ animation: 'pulse 2s infinite 0.6s' }}
              ></div>

              {/* Center icon */}
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-4xl">🔊</div>
              </div>
            </div>

            <div className="text-center">
              <p className="text-foreground font-semibold mb-2">
                {isLive ? 'Active Listening' : 'Paused'}
              </p>
              <p className="text-sm text-foreground/60">System connectivity: <span className="text-green-400 font-semibold">Optimal</span></p>
            </div>
          </div>
        </Card>

        {/* Detection Events */}
        <Card className="p-6 bg-card/50 border border-border/50 backdrop-blur">
          <h3 className="text-lg font-semibold text-foreground mb-4">Recent Detections</h3>
          <div className="space-y-3">
            {detectionsList.map((event) => (
              <div key={event.id} className="flex items-center justify-between p-4 rounded-lg bg-background/40 border border-border/30 hover:border-border/60 transition-colors">
                <div className="flex items-center gap-4">
                  <div className="w-2 h-2 bg-primary rounded-full animate-pulse"></div>
                  <div>
                    <p className="font-semibold text-foreground">{event.type}</p>
                    <p className="text-sm text-foreground/60">{event.time}</p>
                  </div>
                </div>
                <div className="flex items-center gap-6">
                  <div className="text-right">
                    <p className="font-semibold text-foreground">{event.confidence}%</p>
                    <p className="text-xs text-foreground/60">Confidence</p>
                  </div>
                  <div
                    className={`px-3 py-1 rounded-full text-sm font-semibold ${event.risk === 'High'
                      ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                      : event.risk === 'Medium'
                        ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
                        : 'bg-green-500/20 text-green-400 border border-green-500/30'
                      }`}
                  >
                    {event.risk} Risk
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* Start Listening CTA */}
        {onStartListening && (
          <div className="mt-12 text-center">
            <div className="max-w-3xl mx-auto mb-8">
              <h3 className="text-3xl font-bold text-foreground mb-4">Ready to Analyze Your Own Data?</h3>
              <p className="text-foreground/70 text-lg">Upload your acoustic recordings and let the AI detect underwater phenomena in real-time</p>
            </div>
            <button
              onClick={onStartListening}
              className="px-12 py-4 rounded-lg bg-gradient-to-r from-primary to-accent hover:from-primary/90 hover:to-accent/90 text-background font-bold text-lg shadow-2xl shadow-primary/50 hover:shadow-primary/80 transition-all hover:scale-105"
            >
              🎙️ Start Listening
            </button>
          </div>
        )}
      </div>
    </section>
  )
}
