'use client'

import { Card } from '@/components/ui/card'
import { Activity, Filter, Radio, Zap } from 'lucide-react'

export default function HowItWorksSection() {
  const steps = [
    {
      icon: Radio,
      title: 'Capture Audio',
      description: 'Hydrophones capture acoustic signals from the ocean floor and water column.',
      number: '1',
    },
    {
      icon: Filter,
      title: 'Process & Filter',
      description: 'AI filters ocean noise and isolates low-frequency signals (0-50 Hz) characteristic of seismic events.',
      number: '2',
    },
    {
      icon: Activity,
      title: 'Analyze Energy',
      description: 'Energy persistence is analyzed to detect patterns characteristic of seismic-like acoustic events.',
      number: '3',
    },
    {
      icon: Zap,
      title: 'Generate Insights',
      description: 'Results are classified and confidence scores are calculated for accurate detection.',
      number: '4',
    },
  ]

  return (
    <section id="how-it-works" className="py-20 px-4 relative">
      <div className="max-w-7xl mx-auto">
        {/* Section header */}
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              How It Works
            </span>
          </h2>
          <p className="text-foreground/60 max-w-2xl mx-auto text-lg">
            Our advanced AI pipeline breaks down the analysis into four intelligent steps.
          </p>
        </div>

        {/* Steps */}
        <div className="grid md:grid-cols-4 gap-6 mb-8">
          {steps.map((step, index) => {
            const Icon = step.icon
            return (
              <div key={index} className="relative">
                {/* Connector line */}
                {index < steps.length - 1 && (
                  <div className="hidden md:block absolute top-20 left-full w-full h-1 bg-gradient-to-r from-primary/50 to-transparent"></div>
                )}

                <Card className="p-6 bg-card/50 border-border/50 hover:border-primary/50 transition-all duration-300 backdrop-blur relative z-10">
                  <div className="flex items-center gap-4 mb-4">
                    <div className="w-12 h-12 rounded-full bg-gradient-to-br from-primary to-accent flex items-center justify-center font-bold text-background">
                      {step.number}
                    </div>
                  </div>
                  <h3 className="text-lg font-semibold mb-2">{step.title}</h3>
                  <p className="text-foreground/60 text-sm">{step.description}</p>
                </Card>
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
