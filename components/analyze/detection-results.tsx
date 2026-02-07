'use client'

import { Card } from '@/components/ui/card'
import { AlertTriangle, CheckCircle2 } from 'lucide-react'

interface DetectionResultsProps {
  result: any
}

export default function DetectionResults({ result }: DetectionResultsProps) {
  const getTsunamiRiskColor = (risk: string) => {
    switch (risk.toLowerCase()) {
      case 'low':
        return 'from-green-500 to-green-400'
      case 'medium':
        return 'from-yellow-500 to-yellow-400'
      case 'high':
        return 'from-red-500 to-red-400'
      default:
        return 'from-primary to-accent'
    }
  }

  const getTsunamiRiskBgColor = (risk: string) => {
    switch (risk.toLowerCase()) {
      case 'low':
        return 'bg-green-500/10 border-green-500/30'
      case 'medium':
        return 'bg-yellow-500/10 border-yellow-500/30'
      case 'high':
        return 'bg-red-500/10 border-red-500/30'
      default:
        return 'bg-primary/10 border-primary/30'
    }
  }

  return (
    <Card className="p-8 bg-card/50 border-border/50 backdrop-blur">
      <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
        <CheckCircle2 className="text-primary" size={28} />
        Detection Results
      </h2>

      <div className="grid md:grid-cols-2 gap-6 mb-6">
        {/* Detection Type */}
        <div className="p-6 rounded-lg bg-primary/10 border border-primary/30">
          <p className="text-foreground/70 text-sm mb-2">Detection Type</p>
          <p className="text-2xl font-bold text-primary">{result.detectionType}</p>
        </div>

        {/* Confidence Score */}
        <div className="p-6 rounded-lg bg-accent/10 border border-accent/30">
          <p className="text-foreground/70 text-sm mb-2">Confidence Score</p>
          <div className="flex items-center gap-3">
            <div className="text-3xl font-bold text-accent">{result.confidence}%</div>
            <div className="flex-1 bg-background/50 rounded-full h-2 overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-accent to-primary rounded-full transition-all duration-500"
                style={{ width: `${result.confidence}%` }}
              ></div>
            </div>
          </div>
        </div>
      </div>

      {/* Tsunami Risk */}
      <div className={`p-6 rounded-lg border-2 ${getTsunamiRiskBgColor(result.tsunamiRisk)} mb-6`}>
        <div className="flex items-center gap-3 mb-3">
          <AlertTriangle size={24} className="text-foreground" />
          <p className="text-foreground/70 font-semibold">Tsunami Risk Assessment</p>
        </div>
        <div className={`inline-block px-4 py-2 rounded-lg font-bold text-background bg-gradient-to-r ${getTsunamiRiskColor(result.tsunamiRisk)}`}>
          {result.tsunamiRisk} Risk
        </div>
      </div>

      {/* Additional Metrics */}
      <div className="grid md:grid-cols-3 gap-4">
        <div className="p-4 rounded-lg bg-background/50 border border-border/50">
          <p className="text-foreground/60 text-sm mb-1">Energy Peak</p>
          <p className="text-xl font-bold text-primary">{result.energyPeak} dB</p>
        </div>
        <div className="p-4 rounded-lg bg-background/50 border border-border/50">
          <p className="text-foreground/60 text-sm mb-1">Dominant Frequency</p>
          <p className="text-xl font-bold text-primary">{result.frequency} Hz</p>
        </div>
        <div className="p-4 rounded-lg bg-background/50 border border-border/50">
          <p className="text-foreground/60 text-sm mb-1">Event Duration</p>
          <p className="text-xl font-bold text-primary">{result.duration} s</p>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="mt-6 p-4 rounded-lg bg-muted/30 border border-muted/50">
        <p className="text-foreground/60 text-sm">
          <span className="font-semibold">⚠️ Disclaimer:</span> This system detects seismic-like acoustic events and infers tsunami risk. It does not
          directly detect tsunamis. Always follow official alert systems and government warnings.
        </p>
      </div>
    </Card>
  )
}
