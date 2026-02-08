'use client'

import { Card } from '@/components/ui/card'
import { Activity, AlertTriangle, CheckCircle, Waves } from 'lucide-react'

interface CnnDetectionResultsProps {
    result: {
        detectionType: string
        confidence: number
        modelAccuracy: number
    }
}

export default function CnnDetectionResults({ result }: CnnDetectionResultsProps) {
    const isSeismicEvent = result.detectionType === 'Seismic Event' || result.detectionType === 'Earthquake'

    return (
        <Card className="p-6 bg-card/50 border-border/50 backdrop-blur">
            <h3 className="text-2xl font-bold mb-6 flex items-center gap-2">
                <Activity className="w-6 h-6 text-primary" />
                Detection Results
            </h3>

            <div className="grid md:grid-cols-2 gap-6">
                {/* Detection Type */}
                <div className={`p-6 rounded-lg border-2 ${isSeismicEvent
                    ? 'bg-destructive/10 border-destructive/50'
                    : 'bg-green-500/10 border-green-500/50'
                    }`}>
                    <div className="flex items-center gap-3 mb-2">
                        {isSeismicEvent ? (
                            <AlertTriangle className="w-8 h-8 text-destructive" />
                        ) : (
                            <CheckCircle className="w-8 h-8 text-green-500" />
                        )}
                        <div>
                            <p className="text-sm text-foreground/60">Classification</p>
                            <p className={`text-2xl font-bold ${isSeismicEvent ? 'text-destructive' : 'text-green-500'
                                }`}>
                                {result.detectionType}
                            </p>
                        </div>
                    </div>
                </div>

                {/* Confidence Score */}
                <div className="p-6 rounded-lg border-2 bg-primary/10 border-primary/50">
                    <div className="flex items-center gap-3 mb-2">
                        <Waves className="w-8 h-8 text-primary" />
                        <div>
                            <p className="text-sm text-foreground/60">Confidence</p>
                            <p className="text-2xl font-bold text-primary">
                                {result.confidence.toFixed(1)}%
                            </p>
                        </div>
                    </div>
                    <div className="mt-3 w-full bg-secondary rounded-full h-2 overflow-hidden">
                        <div
                            className="h-full bg-gradient-to-r from-primary to-accent transition-all duration-500"
                            style={{ width: `${result.confidence}%` }}
                        />
                    </div>
                </div>
            </div>

            {/* Model Info */}
            <div className="mt-6 p-4 bg-accent/10 border border-accent/30 rounded-lg">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <span className="text-2xl">🏆</span>
                        <div>
                            <p className="text-sm text-foreground/60">Model Performance</p>
                            <p className="text-lg font-bold text-accent">
                                {result.modelAccuracy}% Test Accuracy
                            </p>
                        </div>
                    </div>
                    <div className="text-right">
                        <p className="text-xs text-foreground/50">Architecture</p>
                        <p className="text-sm font-semibold text-foreground">1D CNN</p>
                    </div>
                </div>
            </div>

            {/* Technical Details */}
            <div className="mt-6 grid grid-cols-3 gap-4">
                <div className="text-center p-3 bg-background/50 rounded-lg">
                    <p className="text-xs text-foreground/50 mb-1">Input</p>
                    <p className="text-sm font-semibold text-foreground">3-Channel</p>
                </div>
                <div className="text-center p-3 bg-background/50 rounded-lg">
                    <p className="text-xs text-foreground/50 mb-1">Window</p>
                    <p className="text-sm font-semibold text-foreground">2000 samples</p>
                </div>
                <div className="text-center p-3 bg-background/50 rounded-lg">
                    <p className="text-xs text-foreground/50 mb-1">Training</p>
                    <p className="text-sm font-semibold text-foreground">STEAD Dataset</p>
                </div>
            </div>
        </Card>
    )
}
