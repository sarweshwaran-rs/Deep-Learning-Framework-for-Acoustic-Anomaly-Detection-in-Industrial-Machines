import React from 'react'
import { ArrowRight, Shield, Zap, Brain } from 'lucide-react'

const Hero = () => {
  const scrollToPredict = () => {
    document.getElementById('predict')?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <section id="home" className="pt-16 bg-gradient-to-br from-primary-50 to-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="text-center">
          <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
            Industrial Machine
            <span className="text-primary-600 block">Anomaly Detection</span>
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            Advanced deep learning powered by DFCA-Net to detect anomalies in industrial machinery 
            using audio analysis. Prevent failures before they happen.
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center mb-12">
            <button 
              onClick={scrollToPredict}
              className="btn-primary flex items-center justify-center text-lg px-8 py-3"
            >
              Try Detection Now
              <ArrowRight className="ml-2 h-5 w-5" />
            </button>
            <button 
              onClick={() => document.getElementById('about')?.scrollIntoView({ behavior: 'smooth' })}
              className="btn-secondary text-lg px-8 py-3"
            >
              Learn More
            </button>
          </div>

          {/* Feature highlights */}
          <div className="grid md:grid-cols-3 gap-8 mt-16">
            <div className="text-center">
              <div className="bg-primary-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <Brain className="h-8 w-8 text-primary-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">AI-Powered</h3>
              <p className="text-gray-600">Advanced neural networks for accurate anomaly detection</p>
            </div>
            
            <div className="text-center">
              <div className="bg-success-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <Zap className="h-8 w-8 text-success-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Real-time</h3>
              <p className="text-gray-600">Instant analysis and results within seconds</p>
            </div>
            
            <div className="text-center">
              <div className="bg-orange-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <Shield className="h-8 w-8 text-orange-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Reliable</h3>
              <p className="text-gray-600">High accuracy detection to prevent costly failures</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default Hero