'use client'

import Link from 'next/link'

export default function Footer() {
  return (
    <footer className="border-t border-border/30 bg-background/50 backdrop-blur py-12 px-4 mt-20">
      <div className="max-w-7xl mx-auto">
        <div className="grid md:grid-cols-4 gap-8 mb-8">
          {/* Brand */}
          <div>
            <div className="flex items-center gap-2 mb-4">
              <div className="w-8 h-8 bg-gradient-to-br from-primary to-accent rounded-full flex items-center justify-center">
                <span className="text-sm font-bold text-background">O</span>
              </div>
              <span className="text-lg font-bold text-foreground">OceanAI</span>
            </div>
            <p className="text-foreground/60 text-sm">Listening to the ocean's secrets with AI.</p>
          </div>

          {/* Product */}
          <div>
            <h4 className="font-semibold mb-4 text-foreground">Product</h4>
            <ul className="space-y-2 text-sm text-foreground/60">
              <li>
                <Link href="/analyze" className="hover:text-primary transition">
                  Analyze Audio
                </Link>
              </li>
              <li>
                <Link href="/#features" className="hover:text-primary transition">
                  Features
                </Link>
              </li>
              <li>
                <Link href="/#how-it-works" className="hover:text-primary transition">
                  How It Works
                </Link>
              </li>
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h4 className="font-semibold mb-4 text-foreground">Resources</h4>
            <ul className="space-y-2 text-sm text-foreground/60">
              <li>
                <a href="#" className="hover:text-primary transition">
                  Documentation
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-primary transition">
                  API Reference
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-primary transition">
                  Blog
                </a>
              </li>
            </ul>
          </div>

          {/* Legal */}
          <div>
            <h4 className="font-semibold mb-4 text-foreground">Legal</h4>
            <ul className="space-y-2 text-sm text-foreground/60">
              <li>
                <a href="#" className="hover:text-primary transition">
                  Privacy Policy
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-primary transition">
                  Terms of Service
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-primary transition">
                  Contact
                </a>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom section */}
        <div className="border-t border-border/30 pt-8 flex flex-col md:flex-row justify-between items-center">
          <p className="text-foreground/50 text-sm mb-4 md:mb-0">
            Built for demonstration purposes. © 2026 OceanAI.
          </p>
          <p className="text-foreground/50 text-sm">
            Made with love by oceanographers and AI engineers
          </p>
        </div>
      </div>
    </footer>
  )
}
