import React from 'react'
import { Cpu, Waves, Target, TrendingUp } from 'lucide-react'

const About = () => {
  return (
    <section id="about" className="py-20 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
            About DFCA-Net
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Our Dual Frequency Cross-Attention Network combines STFT and CQT spectrograms 
            for superior anomaly detection in industrial machinery.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-12 items-center mb-16">
          <div>
            <h3 className="text-2xl font-bold text-gray-900 mb-6">How It Works</h3>
            <div className="space-y-6">
              <div className="flex items-start">
                <div className="bg-primary-100 p-2 rounded-lg mr-4 mt-1">
                  <Waves className="h-6 w-6 text-primary-600" />
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900 mb-2">Audio Processing</h4>
                  <p className="text-gray-600">
                    Converts audio signals into STFT and CQT spectrograms for comprehensive frequency analysis.
                  </p>
                </div>
              </div>
              
              <div className="flex items-start">
                <div className="bg-primary-100 p-2 rounded-lg mr-4 mt-1">
                  <Cpu className="h-6 w-6 text-primary-600" />
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900 mb-2">Deep Learning</h4>
                  <p className="text-gray-600">
                    Uses advanced neural networks with cross-attention mechanisms for dual frequency fusion.
                  </p>
                </div>
              </div>
              
              <div className="flex items-start">
                <div className="bg-primary-100 p-2 rounded-lg mr-4 mt-1">
                  <Target className="h-6 w-6 text-primary-600" />
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900 mb-2">Anomaly Detection</h4>
                  <p className="text-gray-600">
                    Identifies abnormal patterns in machinery sounds with high precision and confidence scores.
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-gray-50 p-8 rounded-xl">
            <h3 className="text-2xl font-bold text-gray-900 mb-6">Key Features</h3>
            <ul className="space-y-4">
              <li className="flex items-center">
                <TrendingUp className="h-5 w-5 text-success-600 mr-3" />
                <span className="text-gray-700">High accuracy anomaly detection</span>
              </li>
              <li className="flex items-center">
                <TrendingUp className="h-5 w-5 text-success-600 mr-3" />
                <span className="text-gray-700">Multi-modal spectrogram analysis</span>
              </li>
              <li className="flex items-center">
                <TrendingUp className="h-5 w-5 text-success-600 mr-3" />
                <span className="text-gray-700">Real-time processing capabilities</span>
              </li>
              <li className="flex items-center">
                <TrendingUp className="h-5 w-5 text-success-600 mr-3" />
                <span className="text-gray-700">Confidence score reporting</span>
              </li>
              <li className="flex items-center">
                <TrendingUp className="h-5 w-5 text-success-600 mr-3" />
                <span className="text-gray-700">Industrial-grade reliability</span>
              </li>
            </ul>
          </div>
        </div>

        {/* API Information Section */}
        <div className="bg-gray-50 rounded-xl p-8 mb-8">
          <h3 className="text-2xl font-bold text-gray-900 mb-6 text-center">
            🔗 API Documentation & Resources
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-gray-900">Interactive Documentation</h4>
              <div className="space-y-3">
                <a 
                  href="http://localhost:8000/docs" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="flex items-center p-3 bg-white rounded-lg border hover:border-primary-300 transition-colors group"
                >
                  <div className="bg-primary-100 p-2 rounded-lg mr-3">
                    <TrendingUp className="h-5 w-5 text-primary-600" />
                  </div>
                  <div>
                    <div className="font-medium text-gray-900 group-hover:text-primary-600">Swagger UI</div>
                    <div className="text-sm text-gray-600">Interactive API testing interface</div>
                  </div>
                </a>
                
                <a 
                  href="http://localhost:8000/redoc" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="flex items-center p-3 bg-white rounded-lg border hover:border-primary-300 transition-colors group"
                >
                  <div className="bg-primary-100 p-2 rounded-lg mr-3">
                    <TrendingUp className="h-5 w-5 text-primary-600" />
                  </div>
                  <div>
                    <div className="font-medium text-gray-900 group-hover:text-primary-600">ReDoc</div>
                    <div className="text-sm text-gray-600">Alternative documentation view</div>
                  </div>
                </a>
              </div>
            </div>
            
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-gray-900">Developer Resources</h4>
              <div className="space-y-3">
                <a 
                  href="http://localhost:8000/api-info" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="flex items-center p-3 bg-white rounded-lg border hover:border-primary-300 transition-colors group"
                >
                  <div className="bg-success-100 p-2 rounded-lg mr-3">
                    <TrendingUp className="h-5 w-5 text-success-600" />
                  </div>
                  <div>
                    <div className="font-medium text-gray-900 group-hover:text-primary-600">API Information</div>
                    <div className="text-sm text-gray-600">Quick start guide and examples</div>
                  </div>
                </a>
                
                <a 
                  href="http://localhost:8000/openapi.json" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="flex items-center p-3 bg-white rounded-lg border hover:border-primary-300 transition-colors group"
                >
                  <div className="bg-orange-100 p-2 rounded-lg mr-3">
                    <TrendingUp className="h-5 w-5 text-orange-600" />
                  </div>
                  <div>
                    <div className="font-medium text-gray-900 group-hover:text-primary-600">OpenAPI Schema</div>
                    <div className="text-sm text-gray-600">JSON schema for integration</div>
                  </div>
                </a>
              </div>
            </div>
          </div>
          
          <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
            <h5 className="font-semibold text-blue-900 mb-2">🚀 API Base URL</h5>
            <code className="text-blue-800 bg-blue-100 px-2 py-1 rounded text-sm">
              http://localhost:8000
            </code>
            <p className="text-blue-700 text-sm mt-2">
              Use this base URL for all API requests. Make sure the backend server is running.
            </p>
          </div>
        </div>

        <div className="bg-primary-50 rounded-xl p-8 text-center">
          <h3 className="text-2xl font-bold text-gray-900 mb-4">
            Ready to Detect Anomalies?
          </h3>
          <p className="text-gray-600 mb-6">
            Upload your industrial machine audio file and get instant anomaly detection results.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button 
              onClick={() => document.getElementById('predict')?.scrollIntoView({ behavior: 'smooth' })}
              className="btn-primary text-lg px-8 py-3"
            >
              Start Detection
            </button>
            <a 
              href="http://localhost:8000/docs" 
              target="_blank" 
              rel="noopener noreferrer"
              className="btn-secondary text-lg px-8 py-3 inline-flex items-center justify-center"
            >
              View API Docs
            </a>
          </div>
        </div>
      </div>
    </section>
  )
}

export default About