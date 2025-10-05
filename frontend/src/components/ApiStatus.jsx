import React, { useState, useEffect } from 'react'
import { CheckCircle, XCircle, Loader2 } from 'lucide-react'
import axios from 'axios'

const ApiStatus = () => {
    const [status, setStatus] = useState('checking') // 'checking', 'online', 'offline'
    const [lastChecked, setLastChecked] = useState(null)

    const checkApiStatus = async () => {
        try {
            setStatus('checking')
            const response = await axios.get('http://localhost:8000/', { timeout: 5000 })
            if (response.status === 200) {
                setStatus('online')
                setLastChecked(new Date())
            }
        } catch (error) {
            setStatus('offline')
            setLastChecked(new Date())
        }
    }

    useEffect(() => {
        checkApiStatus()
        // Check status every 30 seconds
        const interval = setInterval(checkApiStatus, 30000)
        return () => clearInterval(interval)
    }, [])

    const getStatusColor = () => {
        switch (status) {
            case 'online': return 'text-success-600 bg-success-50 border-success-200'
            case 'offline': return 'text-danger-600 bg-danger-50 border-danger-200'
            default: return 'text-gray-600 bg-gray-50 border-gray-200'
        }
    }

    const getStatusIcon = () => {
        switch (status) {
            case 'online': return <CheckCircle className="h-4 w-4" />
            case 'offline': return <XCircle className="h-4 w-4" />
            default: return <Loader2 className="h-4 w-4 animate-spin" />
        }
    }

    const getStatusText = () => {
        switch (status) {
            case 'online': return 'API Online'
            case 'offline': return 'API Offline'
            default: return 'Checking...'
        }
    }

    return (
        <div className={`inline-flex items-center px-3 py-1 rounded-full border text-sm font-medium ${getStatusColor()}`}>
            {getStatusIcon()}
            <span className="ml-2">{getStatusText()}</span>
            {lastChecked && (
                <span className="ml-2 text-xs opacity-75">
                    {lastChecked.toLocaleTimeString()}
                </span>
            )}
        </div>
    )
}

export default ApiStatus