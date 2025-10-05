import React from 'react'
import { ExternalLink, Code, Book, FileText, Zap, Shield, Globe } from 'lucide-react'
import ApiStatus from './ApiStatus'

const ApiInfo = () => {
  const endpoints = [
    {
      method: 'GET',
      path: '/',
      description: 'Health check endpoint to verify API status',
      example: 'curl -X GET "http://localhost:8000/"'
    },
    {
      method: 'POST', 
      path: '/predict/',
      description: 'Upload .wav audio file for anomaly detection',
      example: 'curl -X POST "http://localhost:8000/predict/" -F "file=@machine.wav"'
    }
  ]

  const codeExamples = [
    {
      language: 'Python',
      code: `import requests

# Health check
response = requests.get("http://localhost:8000/")
print(response.json())

# Anomaly detection
files = {"file": ("machine.wav", open("machine.wav", "rb"), "audio/wav")}
response = requests.post("http://localhost:8000/predict/", files=files)
result = response.json()
print(f"Prediction: {result['prediction']}")`,
    },
    {
      language: 'JavaScript',
      code: `// Using fetch API
const formData = new FormData();
formData.append('file', audioFile);

const response = await fetch('http://localhost:8000/predict/', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log('Prediction:', result.prediction);`,
    },
    {
      language: 'cURL',
      code: `# Health check
curl -X GET "http://localhost:8000/"

# Anomaly detection
curl -X POST "http://localhost:8000/predict/" \\
  -H "accept: application/json" \\
  -H "Content-Type: multipart/form-data" \\
  -F "file=@machine_sound.wav"`,
    }
  ]

  return (
    <section id="api-info" className="py-20 bg-white">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
            🔗 API Documentation
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Complete API reference and integration guide for the DFCA-Net anomaly detection system
          </p>
        </div>

        {/* Quick Links */}
        <div className="grid md:grid-cols-4 gap-6 mb-12">
          <a 
            href="http://localhost:8000/docs" 
            target="_blank" 
            rel="noopener noreferrer"
            className="card p-6 hover:shadow-xl transition-shadow group"
          >
            <div className="bg-primary-100 w-12 h-12 rounded-lg flex items-center justify-center mb-4 group-hover:bg-primary-200 transition-colors">
              <Book className="h-6 w-6 text-primary-600" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">Swagger UI</h3>
            <p className="text-gray-600 text-sm mb-3">Interactive API documentation with testing interface</p>
            <div className="flex items-center text-primary-600 text-sm font-medium">
              Open Docs <ExternalLink className="h-4 w-4 ml-1" />
            </div>
          </a>

          <a 
            href="http://localhost:8000/redoc" 
            target="_blank" 
            rel="noopener noreferrer"
            className="card p-6 hover:shadow-xl transition-shadow group"
          >
            <div className="bg-success-100 w-12 h-12 rounded-lg flex items-center justify-center mb-4 group-hover:bg-success-200 transition-colors">
              <FileText className="h-6 w-6 text-success-600" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">ReDoc</h3>
            <p className="text-gray-600 text-sm mb-3">Alternative documentation with clean, readable format</p>
            <div className="flex items-center text-success-600 text-sm font-medium">
              View ReDoc <ExternalLink className="h-4 w-4 ml-1" />
            </div>
          </a>

          <a 
            href="http://localhost:8000/api-info" 
            target="_blank" 
            rel="noopener noreferrer"
            className="card p-6 hover:shadow-xl transition-shadow group"
          >
            <div className="bg-orange-100 w-12 h-12 rounded-lg flex items-center justify-center mb-4 group-hover:bg-orange-200 transition-colors">
              <Zap className="h-6 w-6 text-orange-600" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">Quick Start</h3>
            <p className="text-gray-600 text-sm mb-3">Fast introduction and basic usage examples</p>
            <div className="flex items-center text-orange-600 text-sm font-medium">
              Get Started <ExternalLink className="h-4 w-4 ml-1" />
            </div>
          </a>

          <a 
            href="http://localhost:8000/openapi.json" 
            target="_blank" 
            rel="noopener noreferrer"
            className="card p-6 hover:shadow-xl transition-shadow group"
          >
            <div className="bg-purple-100 w-12 h-12 rounded-lg flex items-center justify-center mb-4 group-hover:bg-purple-200 transition-colors">
              <Code className="h-6 w-6 text-purple-600" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">OpenAPI Schema</h3>
            <p className="text-gray-600 text-sm mb-3">JSON schema for code generation and integration</p>
            <div className="flex items-center text-purple-600 text-sm font-medium">
              Download JSON <ExternalLink className="h-4 w-4 ml-1" />
            </div>
          </a>
        </div>

        {/* API Base URL */}
        <div className="bg-blue-50 rounded-xl p-6 mb-12 border border-blue-200">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center">
              <Globe className="h-6 w-6 text-blue-600 mr-3" />
              <h3 className="text-lg font-semibold text-blue-900">API Base URL</h3>
            </div>
            <ApiStatus />
          </div>
          <div className="bg-blue-100 rounded-lg p-4 mb-4">
            <code className="text-blue-800 font-mono text-lg">http://localhost:8000</code>
          </div>
          <p className="text-blue-700">
            All API requests should be made to this base URL. The status indicator above shows if the backend server is currently running.
          </p>
        </div>

        {/* Endpoints */}
        <div className="mb-12">
          <h3 className="text-2xl font-bold text-gray-900 mb-6">Available Endpoints</h3>
          <div className="space-y-6">
            {endpoints.map((endpoint, index) => (
              <div key={index} className="card p-6">
                <div className="flex items-center mb-4">
                  <span className={`px-3 py-1 rounded-full text-sm font-medium mr-4 ${
                    endpoint.method === 'GET' 
                      ? 'bg-green-100 text-green-800' 
                      : 'bg-blue-100 text-blue-800'
                  }`}>
                    {endpoint.method}
                  </span>
                  <code className="text-lg font-mono text-gray-800">{endpoint.path}</code>
                </div>
                <p className="text-gray-600 mb-4">{endpoint.description}</p>
                <div className="bg-gray-50 rounded-lg p-4">
                  <h4 className="text-sm font-semibold text-gray-700 mb-2">Example:</h4>
                  <code className="text-sm text-gray-800 font-mono">{endpoint.example}</code>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Code Examples */}
        <div className="mb-12">
          <h3 className="text-2xl font-bold text-gray-900 mb-6">Integration Examples</h3>
          <div className="grid lg:grid-cols-1 gap-6">
            {codeExamples.map((example, index) => (
              <div key={index} className="card p-6">
                <h4 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                  <Code className="h-5 w-5 mr-2 text-primary-600" />
                  {example.language}
                </h4>
                <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
                  <pre className="text-green-400 text-sm font-mono whitespace-pre-wrap">
                    {example.code}
                  </pre>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Response Format */}
        <div className="mb-12">
          <h3 className="text-2xl font-bold text-gray-900 mb-6">Response Format</h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="card p-6">
              <h4 className="text-lg font-semibold text-success-700 mb-4">✅ Success Response</h4>
              <div className="bg-gray-900 rounded-lg p-4">
                <pre className="text-green-400 text-sm font-mono">
{`{
  "filename": "machine_sound.wav",
  "prediction": "Normal",
  "confidicence": "0.8542"
}`}
                </pre>
              </div>
            </div>
            
            <div className="card p-6">
              <h4 className="text-lg font-semibold text-danger-700 mb-4">❌ Error Response</h4>
              <div className="bg-gray-900 rounded-lg p-4">
                <pre className="text-red-400 text-sm font-mono">
{`{
  "detail": "Invalid file type. Please upload a .wav file."
}`}
                </pre>
              </div>
            </div>
          </div>
        </div>

        {/* Requirements */}
        <div className="bg-gray-50 rounded-xl p-8">
          <h3 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
            <Shield className="h-6 w-6 mr-3 text-primary-600" />
            File Requirements
          </h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold text-gray-900 mb-3">Supported Format</h4>
              <ul className="space-y-2 text-gray-600">
                <li>• <strong>Format:</strong> .wav files only</li>
                <li>• <strong>Duration:</strong> 1-10 seconds recommended</li>
                <li>• <strong>Sample Rate:</strong> Any (auto-resampled to 16kHz)</li>
                <li>• <strong>File Size:</strong> Maximum 50MB</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-gray-900 mb-3">Best Practices</h4>
              <ul className="space-y-2 text-gray-600">
                <li>• Use clear machine audio without background noise</li>
                <li>• Keep files under 10MB for faster processing</li>
                <li>• Ensure audio contains actual machine sounds</li>
                <li>• Test with the interactive documentation first</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default ApiInfo