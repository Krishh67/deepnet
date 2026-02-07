'use client'

import Link from 'next/link'
import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Menu, X } from 'lucide-react'

export default function Navigation() {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <nav className="fixed top-0 w-full z-50 bg-background/80 backdrop-blur-md border-b border-border">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2">
            <div className="w-8 h-8 bg-gradient-to-br from-primary to-accent rounded-full flex items-center justify-center pulse-glow">
              <span className="text-sm font-bold text-foreground">O</span>
            </div>
            <span className="text-xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent hidden sm:inline">
              OceanAI
            </span>
          </Link>

          {/* Desktop Menu */}
          <div className="hidden md:flex items-center gap-8">
            <Link href="/" className="text-foreground/80 hover:text-foreground transition">
              Home
            </Link>
            <Link href="/analyze" className="text-foreground/80 hover:text-foreground transition">
              Analyze
            </Link>
            <Link href="/#how-it-works" className="text-foreground/80 hover:text-foreground transition">
              How It Works
            </Link>
            <Link href="/#use-cases" className="text-foreground/80 hover:text-foreground transition">
              Use Cases
            </Link>
          </div>

          {/* CTA Button */}
          <div className="hidden md:block">
            <Link href="/analyze">
              <Button className="bg-gradient-to-r from-primary to-accent hover:from-primary/80 hover:to-accent/80 text-background font-semibold">
                Start Listening
              </Button>
            </Link>
          </div>

          {/* Mobile Menu Button */}
          <button
            className="md:hidden p-2 hover:bg-secondary rounded-lg transition"
            onClick={() => setIsOpen(!isOpen)}
          >
            {isOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>

        {/* Mobile Menu */}
        {isOpen && (
          <div className="md:hidden pb-4 space-y-3 border-t border-border pt-4">
            <Link href="/" className="block text-foreground/80 hover:text-foreground transition py-2">
              Home
            </Link>
            <Link href="/analyze" className="block text-foreground/80 hover:text-foreground transition py-2">
              Analyze
            </Link>
            <Link href="/#how-it-works" className="block text-foreground/80 hover:text-foreground transition py-2">
              How It Works
            </Link>
            <Link href="/#use-cases" className="block text-foreground/80 hover:text-foreground transition py-2">
              Use Cases
            </Link>
            <Link href="/analyze">
              <Button className="w-full bg-gradient-to-r from-primary to-accent hover:from-primary/80 hover:to-accent/80 text-background font-semibold">
                Start Listening
              </Button>
            </Link>
          </div>
        )}
      </div>
    </nav>
  )
}
