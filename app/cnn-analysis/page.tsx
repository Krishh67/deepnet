import Navigation from '@/components/navigation'
import CnnAnalysisSection from '@/components/cnn-analysis/cnn-analysis-section'
import Footer from '@/components/landing/footer'

export default function CnnAnalysisPage() {
    return (
        <main className="min-h-screen bg-gradient-to-br from-background via-[#0a2540] to-background">
            <Navigation />
            <CnnAnalysisSection />
            <Footer />
        </main>
    )
}
