'use client'

import React from "react"

import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Upload, Zap, Brain, CheckCircle } from 'lucide-react'
import { useState } from 'react'

export default function AnalysisUploadSection() {
  const [isDragActive, setIsDragActive] = useState(false)

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragActive(true)
    } else if (e.type === 'dragleave') {
      setIsDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(false)
  }

  return (
    <section className="py-20 px-4 relative">
      <div className="max-w-7xl mx-auto">
        {/* Section Header */}
        <div className="text-center mb-16">
          <div className="inline-block px-4 py-2 rounded-full bg-accent/10 border border-accent/30 mb-4">
            <span className="text-accent font-medium text-sm">CUSTOM ANALYSIS</span>
          </div>
          <h2 className="text-4xl md:text-5xl font-bold mb-4 text-foreground">Test with Your Own Data</h2>
          <p className="text-foreground/70 max-w-2xl mx-auto">Upload your own acoustic recordings to see how the AI analyzes and detects patterns in your specific hydrophone data</p>
        </div>

        <div className="grid lg:grid-cols-2 gap-8 mb-12">
          {/* Upload Card */}
          <Card
            className={`p-8 border-2 border-dashed transition-all duration-300 cursor-pointer ${
              isDragActive
                ? 'border-primary bg-primary/10'
                : 'border-primary/50 bg-card/30 hover:border-primary/80 hover:bg-primary/5'
            } backdrop-blur`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
          >
            <div className="flex flex-col items-center justify-center py-12 text-center">
              <div className="p-4 rounded-full bg-primary/20 mb-4">
                <Upload className="w-8 h-8 text-primary" />
              </div>
              <h3 className="text-xl font-bold text-foreground mb-2">Drag & Drop Your Audio</h3>
              <p className="text-foreground/70 mb-6">or click to select WAV/MP3 files from your computer</p>
              <Button className="bg-primary/20 border border-primary/50 text-primary hover:bg-primary/30">
                Browse Files
              </Button>
              <p className="text-xs text-foreground/50 mt-4">Max 100MB • Supported: WAV, MP3, FLAC</p>
            </div>
          </Card>

          {/* Features of Upload */}
          <div className="space-y-4">
            <Card className="p-6 bg-card/50 border border-border/50 backdrop-blur hover:border-primary/50 transition-colors">
              <div className="flex gap-4">
                <div className="p-2 rounded-lg bg-primary/20 h-fit">
                  <Zap className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Real-time Processing</h4>
                  <p className="text-sm text-foreground/70">Get instant analysis results within seconds of upload</p>
                </div>
              </div>
            </Card>

            <Card className="p-6 bg-card/50 border border-border/50 backdrop-blur hover:border-accent/50 transition-colors">
              <div className="flex gap-4">
                <div className="p-2 rounded-lg bg-accent/20 h-fit">
                  <Brain className="w-6 h-6 text-accent" />
                </div>
                <div>
                  <h4 className="font-semibold text-foreground mb-1">AI Inference</h4>
                  <p className="text-sm text-foreground/70">Advanced neural networks analyze frequency patterns and anomalies</p>
                </div>
              </div>
            </Card>

            <Card className="p-6 bg-card/50 border border-border/50 backdrop-blur hover:border-green-400/50 transition-colors">
              <div className="flex gap-4">
                <div className="p-2 rounded-lg bg-green-400/20 h-fit">
                  <CheckCircle className="w-6 h-6 text-green-400" />
                </div>
                <div>
                  <h4 className="font-semibold text-foreground mb-1">Detailed Report</h4>
                  <p className="text-sm text-foreground/70">Comprehensive breakdown with confidence scores and classifications</p>
                </div>
              </div>
            </Card>
          </div>
        </div>

        {/* How It Works */}
        <Card className="p-8 bg-gradient-to-r from-primary/10 to-accent/10 border border-primary/30 backdrop-blur mb-12">
          <h3 className="text-2xl font-bold mb-8 text-foreground">What Gets Analyzed</h3>
          <div className="grid md:grid-cols-4 gap-6">
            {[
              { num: '01', title: 'Frequency Content', desc: 'Analyzes dominant frequencies from 0-50 Hz underwater band' },
              { num: '02', title: 'Energy Levels', desc: 'Measures acoustic intensity and RMS energy over time' },
              { num: '03', title: 'Event Detection', desc: 'Identifies seismic events, marine life, and anomalies' },
              { num: '04', title: 'Risk Assessment', desc: 'Calculates tsunami risk and event confidence scores' },
            ].map((step, idx) => (
              <div key={idx} className="text-center">
                <div className="text-3xl font-black text-primary mb-3">{step.num}</div>
                <h4 className="font-semibold text-foreground mb-2">{step.title}</h4>
                <p className="text-sm text-foreground/70">{step.desc}</p>
              </div>
            ))}
          </div>
        </Card>

        {/* Call to Action */}
        <div className="text-center">
          <p className="text-foreground/70 mb-6 max-w-2xl mx-auto">
            Ready to explore your own hydrophone data? Upload a sample recording and see the AI in action with real-time visualizations and detailed analysis.
          </p>
          <Button className="bg-gradient-to-r from-primary to-accent hover:from-primary/80 hover:to-accent/80 text-background font-bold px-8 py-6 text-base shadow-lg shadow-primary/30">
            Start Analyzing Now
          </Button>
        </div>
      </div>
    </section>
  )
}
