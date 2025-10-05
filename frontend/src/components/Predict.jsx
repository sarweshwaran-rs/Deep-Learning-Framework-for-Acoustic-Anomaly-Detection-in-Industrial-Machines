import React, { useState, useRef } from 'react'
import { Upload, Play, Pause, Square, AlertCircle, CheckCircle, Loader2, FileAudio } from 'lucide-react'
import axios from 'axios'

const Predict = () => {
  const [file, setFile] = useState(null)
  const [audioUrl, setAudioUrl] = useState(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [prediction, setPrediction] = useState(null)
  const [error, setError] = useState(null)
  const audioRef = useRef(null)
  const fileInputRef = useRef(null)

  const API_BASE_URL = 'http://localhost:8000'

  const handleFileSelect = (event) => {
    const selectedFile = event.target.files[0]
    if (selectedFile) {
      if (!selectedFile.name.toLowerCase().endsWith('.wav')) {
        setError('Please select a .wav audio file')
        return
      }
      
      setFile(selectedFile)
      setError(null)
      setPrediction(null)
      
      // Create audio URL for playback
      const url = URL.createObjectURL(selectedFile)
      setAudioUrl(url)
      
      // Reset audio player
      if (audioRef.current) {
        audioRef.current.pause()
        audioRef.current.currentTime = 0
        setIsPlaying(false)
      }
    }
  }

  const handleDrop = (event) => {
    event.preventDefault()
    const droppedFile = event.dataTransfer.files[0]
    if (droppedFile) {
      const fakeEvent = { target: { files: [droppedFile] } }
      handleFileSelect(fakeEvent)
    }
  }

  const handleDragOver = (event) => {
    event.preventDefault()
  }

  const togglePlayPause = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause()
      } else {
        audioRef.current.play()
      }
      setIsPlaying(!isPlaying)
    }
  }

  const stopAudio = () => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.currentTime = 0
      setIsPlaying(false)
    }
  }

  const removeFile = () => {
    setFile(null)
    setAudioUrl(null)
    setPrediction(null)
    setError(null)
    setIsPlaying(false)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handlePredict = async () => {
    if (!file) {
      setError('Please select an audio file first')
      return
    }

    setIsLoading(true)
    setError(null)
    setPrediction(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await axios.post(`${API_BASE_URL}/predict/`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000, // 30 second timeout
      })

      setPrediction(response.data)
    } catch (err) {
      console.error('Prediction error:', err)
      if (err.code === 'ECONNABORTED') {
        setError('Request timeout. Please try again.')
      } else if (err.response?.data?.detail) {
        setError(err.response.data.detail)
      } else if (err.request) {
        setError('Unable to connect to the server. Please ensure the API is running.')
      } else {
        setError('An unexpected error occurred. Please try again.')
      }
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <section id="predict" className="py-20 bg-gray-50">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
            Anomaly Detection
          </h2>
          <p className="text-xl text-gray-600">
            Upload your industrial machine audio file (.wav) to detect anomalies
          </p>
        </div>

        <div className="card p-8">
          {/* File Upload Area */}
          <div
            className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
              file ? 'border-success-300 bg-success-50' : 'border-gray-300 hover:border-primary-400'
            }`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
          >
            {!file ? (
              <>
                <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  Upload Audio File
                </h3>
                <p className="text-gray-600 mb-4">
                  Drag and drop your .wav file here, or click to browse
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".wav"
                  onChange={handleFileSelect}
                  className="hidden"
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="btn-primary"
                >
                  Choose File
                </button>
              </>
            ) : (
              <div className="space-y-4">
                <FileAudio className="h-12 w-12 text-success-600 mx-auto" />
                <div>
                  <h3 className="text-lg font-medium text-gray-900">{file.name}</h3>
                  <p className="text-gray-600">
                    {(file.size / 1024 / 1024).toFixed(2)} MB
                  </p>
                </div>
                
                {/* Audio Controls */}
                {audioUrl && (
                  <div className="flex justify-center space-x-4 mt-4">
                    <button
                      onClick={togglePlayPause}
                      className="flex items-center space-x-2 bg-primary-600 hover:bg-primary-700 text-white px-4 py-2 rounded-lg transition-colors"
                    >
                      {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                      <span>{isPlaying ? 'Pause' : 'Play'}</span>
                    </button>
                    <button
                      onClick={stopAudio}
                      className="flex items-center space-x-2 bg-gray-600 hover:bg-gray-700 text-white px-4 py-2 rounded-lg transition-colors"
                    >
                      <Square className="h-4 w-4" />
                      <span>Stop</span>
                    </button>
                    <button
                      onClick={removeFile}
                      className="btn-secondary"
                    >
                      Remove
                    </button>
                  </div>
                )}
                
                <audio
                  ref={audioRef}
                  src={audioUrl}
                  onEnded={() => setIsPlaying(false)}
                  className="hidden"
                />
              </div>
            )}
          </div>

          {/* Error Display */}
          {error && (
            <div className="mt-6 p-4 bg-danger-50 border border-danger-200 rounded-lg flex items-center">
              <AlertCircle className="h-5 w-5 text-danger-600 mr-3 flex-shrink-0" />
              <p className="text-danger-700">{error}</p>
            </div>
          )}

          {/* Predict Button */}
          {file && (
            <div className="mt-8 text-center">
              <button
                onClick={handlePredict}
                disabled={isLoading}
                className={`btn-primary text-lg px-8 py-3 ${
                  isLoading ? 'opacity-50 cursor-not-allowed' : ''
                }`}
              >
                {isLoading ? (
                  <>
                    <Loader2 className="animate-spin h-5 w-5 mr-2" />
                    Analyzing...
                  </>
                ) : (
                  'Detect Anomaly'
                )}
              </button>
            </div>
          )}

          {/* Results Display */}
          {prediction && (
            <div className="mt-8">
              <div className={`p-6 rounded-xl border-2 ${
                prediction.prediction === 'Normal' 
                  ? 'bg-success-50 border-success-200' 
                  : 'bg-danger-50 border-danger-200'
              }`}>
                <div className="flex items-center justify-center mb-4">
                  {prediction.prediction === 'Normal' ? (
                    <CheckCircle className="h-8 w-8 text-success-600 mr-3" />
                  ) : (
                    <AlertCircle className="h-8 w-8 text-danger-600 mr-3" />
                  )}
                  <h3 className="text-2xl font-bold">
                    <span className={
                      prediction.prediction === 'Normal' 
                        ? 'text-success-700' 
                        : 'text-danger-700'
                    }>
                      {prediction.prediction}
                    </span>
                  </h3>
                </div>
                
                <div className="text-center">
                  {prediction.prediction === 'Normal' ? (
                    <p className="text-success-700 text-lg">
                      ✅ No anomalies detected. The machine appears to be operating normally.
                    </p>
                  ) : (
                    <p className="text-danger-700 text-lg">
                      ⚠️ Anomaly detected! The machine may require inspection or maintenance.
                    </p>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </section>
  )
}

export default Predict