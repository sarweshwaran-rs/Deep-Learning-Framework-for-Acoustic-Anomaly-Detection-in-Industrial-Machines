#!/usr/bin/env python3
"""
DFCA-Net API Server Startup Script

This script starts the FastAPI server with enhanced documentation.
"""

import uvicorn
import os
import sys

def main():
    """Start the DFCA-Net API server"""
    
    print("🔧 Starting DFCA-Net Industrial Machine Anomaly Detection API...")
    print("📋 Enhanced documentation will be available at:")
    print("   • Swagger UI: http://localhost:8000/docs")
    print("   • ReDoc: http://localhost:8000/redoc") 
    print("   • API Info: http://localhost:8000/api-info")
    print("   • OpenAPI Schema: http://localhost:8000/openapi.json")
    print("🚀 Starting server...\n")
    
    # Start the server
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["app"],
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()