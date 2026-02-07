'use client'

import { Card } from '@/components/ui/card'
import { Lightbulb } from 'lucide-react'

interface ExplainabilitySectionProps {
  result: any
}

export default function ExplainabilitySection({ result }: ExplainabilitySectionProps) {
  return (
    <Card className="p-8 bg-card/50 border-border/50 backdrop-blur">
      <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
        <Lightbulb className="text-accent" size={28} />
        What We Detected & Our Inference
      </h2>

      <div className="space-y-6">
        {/* What was detected */}
        <div>
          <h3 className="text-lg font-semibold text-primary mb-3">What Was Detected</h3>
          <p className="text-foreground/80 leading-relaxed">
            {result.analysis.description}
          </p>
        </div>

        {/* Inference */}
        <div>
          <h3 className="text-lg font-semibold text-accent mb-3">Our Inference</h3>
          <p className="text-foreground/80 leading-relaxed">
            {result.analysis.inference}
          </p>
        </div>

        {/* How the AI works */}
        <div className="border-t border-border/30 pt-6">
          <h3 className="text-lg font-semibold mb-4">How Our AI Analyzed This Audio:</h3>
          <div className="space-y-3">
            <div className="flex gap-3">
              <div className="w-8 h-8 rounded-full bg-primary/20 border border-primary/50 flex items-center justify-center flex-shrink-0 text-sm font-bold text-primary">
                1
              </div>
              <div>
                <p className="font-semibold text-foreground">Audio Conversion</p>
                <p className="text-foreground/60 text-sm">
                  Converted audio to digital signal and extracted low-frequency components (0-50 Hz)
                </p>
              </div>
            </div>
            <div className="flex gap-3">
              <div className="w-8 h-8 rounded-full bg-primary/20 border border-primary/50 flex items-center justify-center flex-shrink-0 text-sm font-bold text-primary">
                2
              </div>
              <div>
                <p className="font-semibold text-foreground">Noise Filtering</p>
                <p className="text-foreground/60 text-sm">
                  Applied advanced filters to remove ocean ambient noise and biological signals
                </p>
              </div>
            </div>
            <div className="flex gap-3">
              <div className="w-8 h-8 rounded-full bg-primary/20 border border-primary/50 flex items-center justify-center flex-shrink-0 text-sm font-bold text-primary">
                3
              </div>
              <div>
                <p className="font-semibold text-foreground">Energy Persistence Analysis</p>
                <p className="text-foreground/60 text-sm">
                  Analyzed energy distribution over time to identify sustained low-frequency signals
                </p>
              </div>
            </div>
            <div className="flex gap-3">
              <div className="w-8 h-8 rounded-full bg-primary/20 border border-primary/50 flex items-center justify-center flex-shrink-0 text-sm font-bold text-primary">
                4
              </div>
              <div>
                <p className="font-semibold text-foreground">Classification & Scoring</p>
                <p className="text-foreground/60 text-sm">
                  Compared patterns to known seismic signatures and generated confidence score
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Card>
  )
}
