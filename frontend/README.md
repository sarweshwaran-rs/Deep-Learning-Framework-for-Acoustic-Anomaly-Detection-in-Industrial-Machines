# DFCA-Net Frontend

A modern React frontend for the DFCA-Net Industrial Machine Anomaly Detection system.

## Features

- 🎵 **Audio Upload & Playback**: Upload .wav files with built-in audio controls
- 🤖 **Real-time Detection**: Get instant anomaly detection results
- 📱 **Responsive Design**: Works seamlessly on desktop and mobile devices
- 🎨 **Modern UI**: Clean, professional interface with Tailwind CSS
- ⚡ **Fast Performance**: Optimized with Vite for quick development and builds

## Tech Stack

- **React 18** - Modern React with hooks
- **Vite** - Fast build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **Lucide React** - Beautiful icons
- **Axios** - HTTP client for API calls

## Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- DFCA-Net backend API running on `http://localhost:8000`

## Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

4. Open your browser and visit `http://localhost:5173`

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## API Integration

The frontend connects to the FastAPI backend at `http://localhost:8000`. Make sure your backend is running before using the detection features.

### API Endpoints Used:
- `GET /` - Health check
- `POST /predict/` - Upload audio file for anomaly detection

## Usage

1. **Upload Audio**: Drag and drop or click to select a .wav audio file
2. **Preview**: Use the play/pause/stop controls to listen to your audio
3. **Detect**: Click "Detect Anomaly" to analyze the audio
4. **Results**: View the prediction results with confidence scores

## File Structure

```
frontend/
├── public/
├── src/
│   ├── components/
│   │   ├── Navbar.jsx
│   │   ├── Hero.jsx
│   │   ├── About.jsx
│   │   ├── Predict.jsx
│   │   └── Footer.jsx
│   ├── App.jsx
│   ├── main.jsx
│   └── index.css
├── package.json
└── README.md
```

## Customization

### Styling
The app uses Tailwind CSS for styling. You can customize colors and themes in `tailwind.config.js`.

### API URL
To change the backend API URL, modify the `API_BASE_URL` constant in `src/components/Predict.jsx`.

## Deployment

1. Build the project:
```bash
npm run build
```

2. The `dist/` folder contains the production-ready files that can be deployed to any static hosting service.

## Troubleshooting

### Common Issues:

1. **API Connection Error**: Ensure the backend is running on `http://localhost:8000`
2. **File Upload Issues**: Only .wav files are supported
3. **CORS Errors**: Make sure the backend CORS settings include your frontend URL

### Development Tips:

- Use browser dev tools to debug API calls
- Check the browser console for error messages
- Ensure audio files are in the correct format (.wav)