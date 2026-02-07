'use client'

import { useState, useRef } from 'react'
import Navigation from '@/components/navigation'
import LiveDemoSection from '@/components/landing/live-demo-section'
import AnalysisUploadSection from '@/components/landing/analysis-upload-section'
import FeaturesSection from '@/components/landing/features-section'
import HowItWorksSection from '@/components/landing/how-it-works-section'
import UseCasesSection from '@/components/landing/use-cases-section'
import Footer from '@/components/landing/footer'

export default function Dashboard() {
  const [showUpload, setShowUpload] = useState(false)
  const uploadSectionRef = useRef<HTMLDivElement>(null)

  const handleStartListening = () => {
    setShowUpload(true)
    // Scroll to upload section after a brief delay to allow rendering
    setTimeout(() => {
      uploadSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }, 100)
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-background via-[#0a2540] to-background">
      <Navigation />
      <LiveDemoSection onStartListening={handleStartListening} />
      {showUpload && (
        <div ref={uploadSectionRef}>
          <AnalysisUploadSection />
        </div>
      )}
      <FeaturesSection />
      <HowItWorksSection />
      <UseCasesSection />
      <Footer />
    </main>
  )
}
