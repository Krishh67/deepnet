'use client'

import React from "react"

import { useRef, useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'
import Link from 'next/link'

export default function UploadSection() {
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const files = e.dataTransfer.files
    if (files[0]) {
      setSelectedFile(files[0])
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.currentTarget.files
    if (files?.[0]) {
      setSelectedFile(files[0])
    }
  }

  const handleUpload = () => {
    if (selectedFile) {
      // Store file in session storage for analysis page
      sessionStorage.setItem('uploadedFile', selectedFile.name)
    }
  }

  return (
    <section className="py-20 px-4 relative">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-12">
          <div className="inline-block px-4 py-2 rounded-full bg-accent/10 border border-accent/30 mb-4">
            <span className="text-accent font-medium text-sm">ANALYSIS</span>
          </div>
          <h2 className="text-4xl md:text-5xl font-bold mb-4 text-foreground">Upload Your Own Audio</h2>
          <p className="text-foreground/70 max-w-2xl mx-auto">Analyze hydrophone recordings or custom audio files to detect underwater phenomena in real-time</p>
        </div>

        <Card
          className={`p-12 border-2 border-dashed transition-all cursor-pointer ${
            isDragging
              ? 'border-primary/80 bg-primary/10 scale-105'
              : 'border-primary/30 bg-card/30 hover:bg-card/50'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,.mp3,.flac,.ogg"
            onChange={handleFileSelect}
            className="hidden"
          />

          <div className="text-center">
            {selectedFile ? (
              <>
                <div className="w-16 h-16 bg-primary/20 rounded-full flex items-center justify-center mx-auto mb-4">
                  <svg className="w-8 h-8 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <h3 className="text-xl font-semibold text-foreground mb-2">File Selected</h3>
                <p className="text-primary font-medium mb-2">{selectedFile.name}</p>
                <p className="text-sm text-foreground/60 mb-6">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
                <button
                  onClick={() => setSelectedFile(null)}
                  className="text-sm text-foreground/60 hover:text-foreground transition-colors"
                >
                  Change file
                </button>
              </>
            ) : (
              <>
                <div className="w-16 h-16 bg-primary/20 rounded-full flex items-center justify-center mx-auto mb-4">
                  <svg className="w-8 h-8 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                  </svg>
                </div>
                <h3 className="text-xl font-semibold text-foreground mb-2">Drag and drop your audio</h3>
                <p className="text-foreground/60 mb-4">or click to browse files</p>
                <p className="text-sm text-foreground/50">Supported: WAV, MP3, FLAC, OGG up to 100MB</p>
              </>
            )}
          </div>
        </Card>

        {selectedFile && (
          <div className="flex gap-4 justify-center mt-8">
            <Link href="/analyze">
              <Button
                size="lg"
                className="bg-gradient-to-r from-primary to-accent hover:from-primary/80 hover:to-accent/80 text-background font-semibold px-8"
              >
                Analyze Now
              </Button>
            </Link>
            <button
              onClick={() => setSelectedFile(null)}
              className="px-8 py-3 rounded-lg border border-primary/30 text-primary font-semibold hover:bg-primary/10 transition-colors"
            >
              Cancel
            </button>
          </div>
        )}

        {!selectedFile && (
          <div className="text-center mt-8">
            <p className="text-foreground/60 mb-6">Or try the demo without uploading</p>
            <Link href="/analyze">
              <Button
                size="lg"
                variant="outline"
                className="border-primary/30 hover:bg-primary/5 font-semibold px-8 bg-transparent"
              >
                View Demo Analysis
              </Button>
            </Link>
          </div>
        )}
      </div>
    </section>
  )
}
