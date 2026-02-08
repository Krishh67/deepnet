'use client'

import { Button } from '@/components/ui/button'

interface DemoTrack {
    trace_index: number
    trace_id: string
}

interface DemoTrackSelectorProps {
    onTrackSelect: (track: DemoTrack) => void
}

export default function DemoTrackSelector({ onTrackSelect }: DemoTrackSelectorProps) {
    const handleSelectDemo = () => {
        // Use fixed trace 0 for simplicity
        const demoTrack = {
            trace_index: 0,
            trace_id: "demo_trace_0"
        }
        onTrackSelect(demoTrack)
    }

    return (
        <div className="space-y-4">
            <Button
                onClick={handleSelectDemo}
                className="w-full bg-gradient-to-r from-primary to-accent hover:from-primary/80 hover:to-accent/80 text-background font-semibold"
            >
                Use Demo Trace
            </Button>

            <div className="p-4 bg-primary/10 border border-primary/30 rounded-lg">
                <p className="text-sm text-foreground/70 mb-2">Demo Data:</p>
                <p className="text-xs text-foreground/60">
                    Using pre-loaded seismic trace from STEAD dataset
                </p>
            </div>
        </div>
    )
}
