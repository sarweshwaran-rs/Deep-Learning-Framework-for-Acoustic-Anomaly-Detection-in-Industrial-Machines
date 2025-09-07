import librosa
import numpy as np
import io


# --- Parameters from your preprocessing script ---
SR = 16000
N_FFT_STFT = 512
HOP_LENGTH_STFT = 256
N_MELS_STFT = 64
BINS_PER_OCTAVE_CQT = 36
N_BINS_CQT = 84
# ---------------------------------------------------


def minmax_normalize(spec):
    min_val = spec.min()
    max_val = spec.max()
    return (spec - min_val) / (max_val - min_val + 1e-6)


def process_audio(audio_bytes: bytes):
    """
        Takes raw audio bytes, processes them returns STFT and CQT spectrograms.
        Args:
            audio_bytes (bytes): Raw audio bytes.
        Returns:
            tuple: A tuple containing two Numpy arrays:
    """
    audio_stream = io.BytesIO(audio_bytes)

    y, sr = librosa.load(audio_stream, sr=SR)

    y = librosa.effects.preemphasis(y)

    S_stft = librosa.stft(y=y, n_fft=N_FFT_STFT,hop_length=HOP_LENGTH_STFT) 
    S_mel = librosa.feature.melspectrogram(S=np.abs(S_stft), sr=sr, n_mels=N_MELS_STFT, hop_length=HOP_LENGTH_STFT) 
    log_mel_spectrogram = librosa.power_to_db(S_mel, ref=np.max) 
    log_mel_spectrogram = minmax_normalize(log_mel_spectrogram)
    
    CQT = librosa.cqt(y,sr=sr, hop_length=512, bins_per_octave=BINS_PER_OCTAVE_CQT, n_bins=N_BINS_CQT) 
    cqt_spectrogram = librosa.amplitude_to_db(np.abs(CQT), ref=np.max) 
    
    return log_mel_spectrogram.astype(np.float32), cqt_spectrogram.astype(np.float32)
