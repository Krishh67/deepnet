import Navigation from '@/components/navigation'
import AudioAnalysisSection from '@/components/analyze/audio-analysis-section'
import Footer from '@/components/landing/footer'

export default function AnalyzePage() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-background via-[#0a2540] to-background">
      <Navigation />
      <AudioAnalysisSection />
      <Footer />
    </main>
  )
}
