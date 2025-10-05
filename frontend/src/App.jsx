import React from 'react'
import Navbar from './components/Navbar'
import Hero from './components/Hero'
import About from './components/About'
import ApiInfo from './components/ApiInfo'
import Predict from './components/Predict'
import Footer from './components/Footer'

function App() {
  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      <Hero />
      <About />
      <ApiInfo />
      <Predict />
      <Footer />
    </div>
  )
}

export default App