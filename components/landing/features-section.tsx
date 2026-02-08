'use client'

import { Card } from '@/components/ui/card'
import { AlertCircle, Radio, TrendingUp, Zap } from 'lucide-react'

export default function FeaturesSection() {
  const features = [
    {
      icon: Zap,
      title: 'Lightning Fast',
      description: 'Detects and analyzes ocean sounds in real-time with sub-100ms latency.',
      color: 'from-primary to-primary',
    },
    {
      icon: AlertCircle,
      title: 'Seismic Detection',
      description: 'Classifies underwater acoustic events (earthquakes, marine life, explosions) and assesses risk automatically.',
      color: 'from-accent to-primary',
    },
    {
      icon: Radio,
      title: 'Multi-Species Recognition',
      description: 'Recognizes whale calls, dolphin clicks, fish sounds, and unknown marine creatures.',
      color: 'from-primary to-accent',
    },
    {
      icon: TrendingUp,
      title: 'Advanced Analytics',
      description: 'Provides detailed spectrograms, energy graphs, and confidence metrics.',
      color: 'from-accent to-accent',
    },
  ]

  return (
    <section className="py-20 px-4 relative">
      <div className="max-w-7xl mx-auto">
        {/* Section header */}
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              Powerful Features
            </span>
          </h2>
          <p className="text-foreground/60 max-w-2xl mx-auto text-lg">
            Experience the next generation of ocean monitoring with our cutting-edge AI system.
          </p>
        </div>

        {/* Features grid */}
        <div className="grid md:grid-cols-2 gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon
            return (
              <Card
                key={index}
                className="p-6 bg-card/50 border-border/50 hover:border-primary/50 hover:bg-card/80 transition-all duration-300 backdrop-blur"
              >
                <div className="flex gap-4">
                  <div className={`p-3 rounded-lg bg-gradient-to-br ${feature.color} w-fit h-fit`}>
                    <Icon className="text-background" size={24} />
                  </div>
                  <div>
                    <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                    <p className="text-foreground/60">{feature.description}</p>
                  </div>
                </div>
              </Card>
            )
          })}
        </div>
      </div>
    </section>
  )
}
