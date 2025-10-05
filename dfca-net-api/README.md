# DFCA-Net API - Enhanced Documentation

## 🔧 Industrial Machine Anomaly Detection API

A comprehensive FastAPI application for detecting anomalies in industrial machinery using the DFCA-Net deep learning model.

## 🌟 Enhanced Features

### 📋 Professional API Documentation
- **Comprehensive Swagger UI** at `http://localhost:8000/docs`
- **Detailed endpoint descriptions** with examples
- **Response models** with validation
- **Error handling** documentation
- **Interactive testing** interface

### 🎯 Key Enhancements
- ✅ Detailed API metadata and descriptions
- ✅ Request/response examples for all endpoints
- ✅ Comprehensive error handling with examples
- ✅ Professional styling and branding
- ✅ Model architecture documentation
- ✅ Performance specifications
- ✅ Usage guidelines and best practices

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Enhanced API Server
```bash
python run_api.py
```

### 3. Access Documentation
- **Main Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **API Information**: http://localhost:8000/api-info
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 📋 API Endpoints

### Health Check
```http
GET /
```
Returns API status and welcome message.

### Anomaly Detection
```http
POST /predict/
```
Upload a .wav audio file for anomaly detection analysis.

**Parameters:**
- `file` (required): .wav audio file

**Response:**
```json
{
  "filename": "machine_sound.wav",
  "prediction": "Normal" | "Abnormal", 
  "confidicence": "0.8542"
}
```

## 🔬 Model Specifications

| Component | Specification |
|-----------|---------------|
| **STFT FFT Size** | 512 |
| **STFT Hop Length** | 256 |
| **Mel Bands** | 64 |
| **CQT Bins** | 84 |
| **Bins per Octave** | 36 |
| **Dual Frequency Fusion Dimensions** | 256 |
| **Sample Rate** | 16kHz |
| **Threshold** | 0.65 |

## 📊 Documentation Features

### Interactive Examples
- **Normal Machine**: Example with healthy machine audio
- **Abnormal Machine**: Example with faulty machine audio
- **Error Cases**: Examples of various error scenarios

### Detailed Descriptions
- **Endpoint purposes** and use cases
- **Parameter requirements** and validation
- **Response formats** and field descriptions
- **Error handling** and troubleshooting
- **Performance expectations** and optimization tips

### Professional Styling
- **Custom branding** with DFCA-Net theme
- **Organized sections** with clear navigation
- **Code examples** in multiple languages
- **Visual indicators** for different response types

## 🛠️ Development Features

### Enhanced Error Handling
- Detailed error messages with context
- Proper HTTP status codes
- Example error responses
- Troubleshooting guidance

### Validation & Models
- Pydantic models for request/response validation
- Type hints and field descriptions
- Automatic schema generation
- Input validation with helpful error messages

### Performance Monitoring
- Request/response logging
- Processing time tracking
- Error rate monitoring
- Health check endpoints

## 📖 Usage Examples

### cURL
```bash
# Health check
curl -X GET "http://localhost:8000/"

# Anomaly detection
curl -X POST "http://localhost:8000/predict/" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@machine_sound.wav"
```

### Python
```python
import requests

# Health check
response = requests.get("http://localhost:8000/")
print(response.json())

# Anomaly detection
files = {"file": ("machine.wav", open("machine.wav", "rb"), "audio/wav")}
response = requests.post("http://localhost:8000/predict/", files=files)
print(response.json())
```

### JavaScript
```javascript
// Anomaly detection
const formData = new FormData();
formData.append('file', audioFile);

const response = await fetch('http://localhost:8000/predict/', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result);
```

## 🔧 Configuration

### Environment Variables
- `MODEL_PATH`: Path to the trained model weights
- `THRESHOLD`: Anomaly detection threshold (default: 0.65)
- `MAX_FILE_SIZE`: Maximum upload file size
- `LOG_LEVEL`: Logging level (INFO, DEBUG, ERROR)

### CORS Settings
Currently configured for development with React frontend:
- `http://localhost:5173` (React dev server)

## 📈 Performance

### Expected Response Times
- **Small files (<1MB)**: 1-3 seconds
- **Medium files (1-5MB)**: 3-8 seconds
- **Large files (5-10MB)**: 8-15 seconds

### Optimization Tips
- Use .wav format to avoid conversion overhead
- Keep files under 10MB for faster processing
- Ensure clear audio without excessive noise
- Use appropriate sample rates (16kHz recommended)

## 🔍 Troubleshooting

### Common Issues
1. **Model Loading Error**: Check model path and file permissions
2. **CUDA/GPU Issues**: Verify PyTorch CUDA installation
3. **File Format Error**: Ensure .wav format and valid audio
4. **Memory Error**: Reduce file size or increase system memory
5. **CORS Error**: Check allowed origins in middleware settings

### Debug Mode
Enable detailed logging by setting `LOG_LEVEL=DEBUG` in environment variables.

## 📝 API Documentation Structure

The enhanced documentation includes:

1. **Overview**: API purpose and capabilities
2. **Authentication**: Current auth requirements (none)
3. **Endpoints**: Detailed endpoint documentation
4. **Models**: Request/response schemas
5. **Examples**: Real-world usage examples
6. **Errors**: Comprehensive error handling
7. **Performance**: Speed and optimization guidelines
8. **Troubleshooting**: Common issues and solutions

## 🚀 Production Deployment

For production deployment:

1. **Security**: Add authentication and rate limiting
2. **Monitoring**: Implement logging and metrics
3. **Scaling**: Use multiple workers with Gunicorn
4. **HTTPS**: Enable SSL/TLS encryption
5. **CORS**: Update allowed origins for production domains

## 📞 Support

For technical support or questions about the DFCA-Net API:
- Check the interactive documentation at `/docs`
- Review the API information at `/api-info`
- Examine the OpenAPI schema at `/openapi.json`