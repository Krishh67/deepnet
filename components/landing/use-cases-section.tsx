'use client'

import { Card } from '@/components/ui/card'
import { AlertTriangle, Microscope, Radio, Waves } from 'lucide-react'

export default function UseCasesSection() {
  const useCases = [
    {
      icon: AlertTriangle,
      title: 'Early Seismic Awareness',
      description: 'Detect seismic-like acoustic events in real-time for early disaster preparedness.',
      color: 'from-primary to-primary',
    },
    {
      icon: Waves,
      title: 'Tsunami Risk Assessment',
      description: 'Infer tsunami risk levels based on seismic acoustic signatures detected underwater.',
      color: 'from-accent to-primary',
    },
    {
      icon: Radio,
      title: 'Marine Life Monitoring',
      description: 'Track whale migrations, dolphin populations, and marine species behavior patterns.',
      color: 'from-primary to-accent',
    },
    {
      icon: Microscope,
      title: 'Ocean Research',
      description: 'Support scientific research with comprehensive acoustic data and AI-driven insights.',
      color: 'from-accent to-accent',
    },
  ]

  return (
    <section id="use-cases" className="py-20 px-4 relative">
      <div className="max-w-7xl mx-auto">
        {/* Section header */}
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-4">
            <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              Use Cases
            </span>
          </h2>
          <p className="text-foreground/60 max-w-2xl mx-auto text-lg">
            Explore diverse applications of our ocean monitoring system.
          </p>
        </div>

        {/* Use cases grid */}
        <div className="grid md:grid-cols-2 gap-6">
          {useCases.map((useCase, index) => {
            const Icon = useCase.icon
            return (
              <Card
                key={index}
                className="p-8 bg-card/50 border-border/50 hover:border-primary/50 hover:bg-card/80 hover:shadow-lg hover:shadow-primary/20 transition-all duration-300 backdrop-blur group cursor-pointer"
              >
                <div className={`p-4 rounded-lg bg-gradient-to-br ${useCase.color} w-fit mb-4 group-hover:scale-110 transition-transform`}>
                  <Icon className="text-background" size={28} />
                </div>
                <h3 className="text-2xl font-semibold mb-3">{useCase.title}</h3>
                <p className="text-foreground/60">{useCase.description}</p>
              </Card>
            )
          })}
        </div>
      </div>
    </section>
  )
}
