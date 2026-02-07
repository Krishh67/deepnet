'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { Button } from '@/components/ui/button'

export default function HeroSection() {
  const [isVisible, setIsVisible] = useState(false)
  const [particles, setParticles] = useState<Array<{ left: number; top: number; duration: number; delay: number }>>([])

  useEffect(() => {
    setIsVisible(true)
    // Generate particles only on client side to avoid hydration mismatch
    const generatedParticles = [...Array(8)].map((_, i) => ({
      left: Math.random() * 100,
      top: Math.random() * 100,
      duration: 8 + i * 2,
      delay: i * 0.5,
    }))
    setParticles(generatedParticles)
  }, [])

  return (
    <section className="relative min-h-screen flex items-center justify-center px-4 py-20 overflow-hidden pt-24">
      {/* Extreme animated background - Multiple layered blobs */}
      <div className="absolute inset-0">
        {/* Layer 1 - Large pulsing blobs with varied animations */}
        <div
          className="absolute top-20 left-10 w-96 h-96 bg-primary/40 rounded-full blur-3xl"
          style={{ animation: 'pulse 4s ease-in-out infinite, float 12s ease-in-out infinite' }}
        ></div>
        <div
          className="absolute top-40 right-20 w-80 h-80 bg-accent/30 rounded-full blur-3xl"
          style={{ animation: 'pulse 5s ease-in-out infinite 0.5s, float 14s ease-in-out infinite 1s' }}
        ></div>
        <div
          className="absolute bottom-32 left-1/3 w-96 h-96 bg-primary/25 rounded-full blur-3xl"
          style={{ animation: 'pulse 6s ease-in-out infinite 1s' }}
        ></div>

        {/* Layer 2 - Floating orbs with glow and dynamic movement */}
        <div
          className="absolute top-1/4 right-1/4 w-48 h-48 bg-cyan-400/20 rounded-full blur-2xl pulse-glow"
          style={{ animation: 'float 10s ease-in-out infinite' }}
        ></div>
        <div
          className="absolute bottom-1/4 left-1/4 w-64 h-64 bg-blue-400/15 rounded-full blur-2xl"
          style={{ animation: 'float 16s ease-in-out infinite 2s' }}
        ></div>

        {/* Layer 3 - Additional smaller orbs for depth */}
        <div
          className="absolute top-1/3 left-1/3 w-40 h-40 bg-primary/20 rounded-full blur-2xl"
          style={{ animation: 'float 18s ease-in-out infinite 0.8s' }}
        ></div>
        <div
          className="absolute bottom-20 right-1/3 w-56 h-56 bg-accent/15 rounded-full blur-3xl"
          style={{ animation: 'pulse 7s ease-in-out infinite 1.5s, float 20s ease-in-out infinite 3s' }}
        ></div>

        {/* Grid overlay with animation */}
        <div className="absolute inset-0 opacity-5 bg-grid-pattern"></div>
      </div>

      {/* Floating particles effect */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {particles.map((particle, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 bg-primary rounded-full opacity-50"
            style={{
              left: `${particle.left}%`,
              top: `${particle.top}%`,
              animation: `float ${particle.duration}s ease-in-out infinite`,
              animationDelay: `${particle.delay}s`,
            }}
          ></div>
        ))}
      </div>

      <div className="relative z-10 text-center max-w-6xl mx-auto">
        {/* Animated badge */}
        <div
          className={`inline-block mb-8 px-6 py-3 rounded-full bg-primary/15 border border-primary/40 backdrop-blur-sm transition-all duration-1000 ${isVisible ? 'opacity-100 scale-100' : 'opacity-0 scale-95'
            }`}
        >
          <span className="text-primary font-bold text-sm tracking-wider">NEXT-GEN OCEAN INTELLIGENCE</span>
        </div>

        {/* Main heading with dramatic styling */}
        <h1
          className={`text-5xl md:text-7xl font-black mb-6 leading-tight transition-all duration-1000 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-20'
            }`}
        >
          <span className="bg-gradient-to-r from-primary via-cyan-300 to-primary bg-clip-text text-transparent animate-pulse" style={{ backgroundSize: '200% 200%' }}>
            AI That Listens
          </span>
          <br />
          <span className="bg-gradient-to-l from-accent via-primary to-cyan-400 bg-clip-text text-transparent text-shadow-lg">
            to the Ocean
          </span>
        </h1>

        {/* Subtitle with emphasis */}
        <p
          className={`text-xl md:text-2xl font-bold text-primary mb-6 transition-all duration-1000 delay-200 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
            }`}
        >
          Detecting undersea seismic events using acoustic intelligence
        </p>

        {/* Description with glow effect */}
        <div
          className={`max-w-3xl mx-auto mb-12 p-6 rounded-xl bg-gradient-to-r from-primary/10 to-accent/10 border border-primary/30 backdrop-blur-lg transition-all duration-1000 delay-300 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
            }`}
        >
          <p className="text-lg md:text-xl text-foreground/90">
            Our system analyzes underwater sound to detect seismic-like events and infer tsunami risk using explainable AI.
          </p>
        </div>

        {/* Button group with effects - Both same size */}
        <div
          className={`flex flex-col sm:flex-row gap-4 justify-center items-center transition-all duration-1000 delay-400 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
            }`}
        >
          <Link href="/dashboard" className="w-full sm:w-auto">
            <Button
              size="lg"
              className="w-full sm:w-auto h-16 relative bg-gradient-to-r from-primary via-cyan-400 to-accent hover:from-primary/90 hover:to-accent/90 text-background font-black text-lg px-16 border-2 border-primary/50 shadow-2xl shadow-primary/50 hover:shadow-primary/80 transition-all hover:scale-105"
            >
              <span className="relative z-10">Explore the System</span>
              <div className="absolute inset-0 bg-gradient-to-r from-primary to-accent opacity-0 hover:opacity-20 rounded-lg transition-opacity"></div>
            </Button>
          </Link>

          <button className="w-full sm:w-auto h-16 px-16 rounded-lg border-2 border-accent/50 text-accent font-black text-lg hover:bg-accent/10 hover:text-accent transition-all hover:border-accent hover:shadow-2xl shadow-accent/30 backdrop-blur-sm flex items-center justify-center">
            Watch Demo
          </button>
        </div>

        {/* Live stats indicator */}
        <div
          className={`mt-16 inline-flex items-center gap-3 px-6 py-3 rounded-full bg-background/40 border border-primary/30 backdrop-blur transition-all duration-1000 delay-500 ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'
            }`}
        >
          <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
          <span className="text-foreground/80 font-semibold text-sm">System Live • 2,847 Active Detections</span>
        </div>
      </div>

      {/* Animated bottom gradient */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-background/50 to-transparent"></div>
    </section>
  )
}
