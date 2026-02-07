'use client'

import React from "react"

import { useRef, useState } from 'react'
import { Upload } from 'lucide-react'

interface FileUploadProps {
  onFileUpload: (file: File) => void
}

export default function FileUpload({ onFileUpload }: FileUploadProps) {
  const [isDragActive, setIsDragActive] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(e.type === 'dragenter' || e.type === 'dragover')
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragActive(false)

    const file = e.dataTransfer.files?.[0]
    if (file && isValidAudioFile(file)) {
      onFileUpload(file)
    }
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && isValidAudioFile(file)) {
      onFileUpload(file)
    }
  }

  const isValidAudioFile = (file: File) => {
    const validTypes = ['audio/wav', 'audio/mpeg', 'audio/mp4']
    return validTypes.includes(file.type) || file.name.endsWith('.wav') || file.name.endsWith('.mp3')
  }

  return (
    <div
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onClick={() => fileInputRef.current?.click()}
      className={`relative border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all duration-300 ${
        isDragActive
          ? 'border-primary bg-primary/10 scale-105'
          : 'border-border/50 bg-background/50 hover:border-primary/50 hover:bg-primary/5'
      }`}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*"
        onChange={handleChange}
        className="hidden"
      />

      <div className="flex flex-col items-center">
        <Upload size={32} className="text-primary mb-3" />
        <p className="text-foreground font-semibold mb-1">Drop audio file here</p>
        <p className="text-foreground/60 text-sm">or click to browse</p>
        <p className="text-foreground/40 text-xs mt-2">WAV, MP3, or M4A supported</p>
      </div>
    </div>
  )
}
