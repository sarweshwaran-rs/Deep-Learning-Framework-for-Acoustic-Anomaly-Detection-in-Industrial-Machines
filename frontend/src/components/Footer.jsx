import React from 'react'
import { Activity, Github, Mail, ExternalLink } from 'lucide-react'

const Footer = () => {
  return (
    <footer className="bg-gray-900 text-white py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid md:grid-cols-4 gap-8">
          {/* Brand */}
          <div className="col-span-2">
            <div className="flex items-center mb-4">
              <Activity className="h-8 w-8 text-primary-400" />
              <span className="ml-2 text-xl font-bold">DFCA-Net</span>
            </div>
            <p className="text-gray-400 mb-4 max-w-md">
              Advanced industrial machine anomaly detection using deep learning 
              and audio signal processing for predictive maintenance.
            </p>
            <div className="flex space-x-4">
              <a 
                href="https://www.github.com/sarweshwaran-rs" 
                className="text-gray-400 hover:text-white transition-colors"
                aria-label="GitHub"
              >
                <Github className="h-6 w-6" />
              </a>
              <a 
                href="mailto:sarweshchandran@gmail.com" 
                className="text-gray-400 hover:text-white transition-colors"
                aria-label="Email"
              >
                <Mail className="h-6 w-6" />
              </a>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Quick Links</h3>
            <ul className="space-y-2">
              <li>
                <button 
                  onClick={() => document.getElementById('home')?.scrollIntoView({ behavior: 'smooth' })}
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  Home
                </button>
              </li>
              <li>
                <button 
                  onClick={() => document.getElementById('about')?.scrollIntoView({ behavior: 'smooth' })}
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  About
                </button>
              </li>
              <li>
                <button 
                  onClick={() => document.getElementById('predict')?.scrollIntoView({ behavior: 'smooth' })}
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  Detection
                </button>
              </li>
            </ul>
          </div>

          {/* Resources */}
          <div>
            <h3 className="text-lg font-semibold mb-4">API Resources</h3>
            <ul className="space-y-2">
              <li>
                <a 
                  href="http://localhost:8000/docs" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-gray-400 hover:text-white transition-colors flex items-center"
                >
                  Interactive API Docs
                  <ExternalLink className="h-4 w-4 ml-1" />
                </a>
              </li>
              <li>
                <a 
                  href="http://localhost:8000/redoc" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-gray-400 hover:text-white transition-colors flex items-center"
                >
                  Alternative Docs
                  <ExternalLink className="h-4 w-4 ml-1" />
                </a>
              </li>
              <li>
                <a 
                  href="http://localhost:8000/api-info" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-gray-400 hover:text-white transition-colors flex items-center"
                >
                  API Information
                  <ExternalLink className="h-4 w-4 ml-1" />
                </a>
              </li>
              <li>
                <a 
                  href="http://localhost:8000/openapi.json" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-gray-400 hover:text-white transition-colors flex items-center"
                >
                  OpenAPI Schema
                  <ExternalLink className="h-4 w-4 ml-1" />
                </a>
              </li>
            </ul>
          </div>
        </div>

        <div className="border-t border-gray-800 mt-8 pt-8 text-center">
          <p className="text-gray-400">
            © 2025 DFCA-Net. Built with React and FastAPI for industrial anomaly detection.
          </p>
        </div>
      </div>
    </footer>
  )
}

export default Footer