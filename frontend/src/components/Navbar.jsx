import React, { useState } from 'react'
import { Menu, X, Activity } from 'lucide-react'

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false)

  const scrollToSection = (sectionId) => {
    document.getElementById(sectionId)?.scrollIntoView({ behavior: 'smooth' })
    setIsOpen(false)
  }

  return (
    <nav className="bg-white shadow-lg fixed w-full top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <Activity className="h-8 w-8 text-primary-600" />
            <span className="ml-2 text-xl font-bold text-gray-900">DFCA-Net</span>
          </div>
          
          {/* Desktop Menu */}
          <div className="hidden md:flex items-center space-x-8">
            <button 
              onClick={() => scrollToSection('home')}
              className="text-gray-700 hover:text-primary-600 transition-colors"
            >
              Home
            </button>
            <button 
              onClick={() => scrollToSection('about')}
              className="text-gray-700 hover:text-primary-600 transition-colors"
            >
              About
            </button>
            <button 
              onClick={() => scrollToSection('api-info')}
              className="text-gray-700 hover:text-primary-600 transition-colors"
            >
              API Docs
            </button>
            <button 
              onClick={() => scrollToSection('predict')}
              className="btn-primary"
            >
              Try Detection
            </button>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden flex items-center">
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="text-gray-700 hover:text-primary-600"
            >
              {isOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu */}
      {isOpen && (
        <div className="md:hidden">
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3 bg-white border-t">
            <button 
              onClick={() => scrollToSection('home')}
              className="block px-3 py-2 text-gray-700 hover:text-primary-600 w-full text-left"
            >
              Home
            </button>
            <button 
              onClick={() => scrollToSection('about')}
              className="block px-3 py-2 text-gray-700 hover:text-primary-600 w-full text-left"
            >
              About
            </button>
            <button 
              onClick={() => scrollToSection('api-info')}
              className="block px-3 py-2 text-gray-700 hover:text-primary-600 w-full text-left"
            >
              API Docs
            </button>
            <button 
              onClick={() => scrollToSection('predict')}
              className="block px-3 py-2 text-primary-600 font-medium w-full text-left"
            >
              Try Detection
            </button>
          </div>
        </div>
      )}
    </nav>
  )
}

export default Navbar