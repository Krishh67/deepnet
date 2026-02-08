'use client'

import { useState } from 'react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import CnnFileUpload from './cnn-file-upload'
import CnnDetectionResults from './cnn-detection-results'

export default function CnnAnalysisSection() {
    const [isAnalyzing, setIsAnalyzing] = useState(false)
    const [analysisResult, setAnalysisResult] = useState<any>(null)
    const [uploadedFile, setUploadedFile] = useState<File | null>(null)

    const handleFileUpload = (file: File) => {
        setUploadedFile(file)
        setAnalysisResult(null)
    }

    const handleAnalyze = async () => {
        if (!uploadedFile) return

        setIsAnalyzing(true)

        try {
            // Create FormData and upload to backend
            const formData = new FormData()
            formData.append('file', uploadedFile)

            // Call FastAPI backend CNN endpoint
            const response = await fetch('http://localhost:8000/predict-seismic', {
                method: 'POST',
                body: formData,
            })

            if (!response.ok) {
                const error = await response.json()
                throw new Error(error.detail || 'CNN prediction failed')
            }

            const data = await response.json()

            setAnalysisResult({
                detectionType: data.detection_type,
                confidence: data.confidence,
                modelAccuracy: data.model_accuracy,
                description: data.description,
            })
        } catch (error) {
            console.error('CNN Analysis error:', error)
            alert(`Analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
        } finally {
            setIsAnalyzing(false)
        }
    }

    return (
        <div className="py-20 px-4 min-h-screen pt-32">
            <div className="max-w-7xl mx-auto">
                {/* Page header */}
                <div className="text-center mb-12">
                    <h1 className="text-5xl md:text-6xl font-bold mb-4">
                        <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                            CNN Seismic Analysis
                        </span>
                    </h1>
                    <p className="text-foreground/60 max-w-2xl mx-auto text-lg mb-3">
                        Upload audio and let our 1D CNN classify seismic events with 97% accuracy
                    </p>
                    <div className="inline-flex items-center gap-2 px-4 py-2 bg-primary/10 border border-primary/30 rounded-full">
                        <span className="text-2xl">🧠</span>
                        <span className="text-sm font-semibold text-primary">Deep Learning · 97% Accuracy</span>
                    </div>
                </div>

                {/* Main content */}
                <div className="grid lg:grid-cols-3 gap-8">
                    {/* Left column - Upload */}
                    <div className="lg:col-span-1">
                        <Card className="p-6 bg-card/50 border-border/50 backdrop-blur sticky top-32">
                            <h2 className="text-2xl font-bold mb-4">Upload Audio</h2>
                            <CnnFileUpload onFileUpload={handleFileUpload} />

                            {uploadedFile && (
                                <div className="mt-6 p-4 bg-primary/10 border border-primary/30 rounded-lg">
                                    <p className="text-sm text-foreground/70">File selected:</p>
                                    <p className="text-foreground font-semibold truncate">{uploadedFile.name}</p>
                                    <p className="text-xs text-foreground/50 mt-1">
                                        {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB
                                    </p>
                                </div>
                            )}

                            <Button
                                onClick={handleAnalyze}
                                disabled={!uploadedFile || isAnalyzing}
                                className="w-full mt-6 bg-gradient-to-r from-primary to-accent hover:from-primary/80 hover:to-accent/80 text-background font-semibold disabled:opacity-50"
                            >
                                {isAnalyzing ? 'CNN Analyzing... 🧠' : 'Analyze with CNN'}
                            </Button>
                        </Card>
                    </div>

                    {/* Right column - Results */}
                    <div className="lg:col-span-2 space-y-8">
                        {isAnalyzing && (
                            <Card className="p-8 bg-card/50 border-border/50 backdrop-blur text-center">
                                <div className="flex justify-center mb-4">
                                    <div className="w-12 h-12 border-4 border-primary/30 border-t-primary rounded-full animate-spin"></div>
                                </div>
                                <p className="text-xl text-foreground mb-2">CNN is analyzing the seismic data...</p>
                                <p className="text-foreground/60">Deep learning model processing waveform patterns</p>
                            </Card>
                        )}

                        {analysisResult && (
                            <>
                                {/* CNN Description Card */}
                                <Card className="p-6 bg-card/50 border-border/50 backdrop-blur">
                                    <h3 className="text-2xl font-bold mb-4 flex items-center gap-2">
                                        <span className="text-3xl">🧠</span>
                                        CNN Analysis
                                    </h3>
                                    <div className="bg-primary/5 border border-primary/20 rounded-lg p-4">
                                        <p className="text-foreground leading-relaxed">
                                            {analysisResult.description}
                                        </p>
                                    </div>
                                </Card>

                                <CnnDetectionResults result={analysisResult} />
                            </>
                        )}

                        {!isAnalyzing && !analysisResult && (
                            <Card className="p-8 bg-card/50 border-border/50 backdrop-blur text-center">
                                <div className="text-5xl mb-4 opacity-50">🌊</div>
                                <p className="text-xl text-foreground mb-2">Ready to analyze</p>
                                <p className="text-foreground/60">Upload an audio file and click "Analyze with CNN" to get started</p>
                            </Card>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}
