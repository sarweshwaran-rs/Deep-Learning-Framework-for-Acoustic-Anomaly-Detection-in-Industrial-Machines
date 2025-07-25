import os
import librosa
import librosa.display
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

BASE_DIR = r'data'
DATA_RAW_DIR = os.path.join(BASE_DIR,'raw')
FEATURES_DIR = os.path.join(BASE_DIR,'features')

SR = 16000
N_FFT_STFT = 1024
HOP_LENGTH_STFT = 512
N_MELS_STFT = 128

BINS_PER_OCTAVE_CQT = 36
N_BINS_CQT = 84

os.makedirs(os.path.join(FEATURES_DIR,'normal','stft'),exist_ok=True)
os.makedirs(os.path.join(FEATURES_DIR,'normal','cqt'),exist_ok=True)
os.makedirs(os.path.join(FEATURES_DIR,'abnormal','stft'),exist_ok=True)
os.makedirs(os.path.join(FEATURES_DIR,'abnormal','cqt'),exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

def process_audio_file(audio_path):

    y, sr = librosa.load(audio_path, sr=SR)

    #New Line
    y = librosa.effects.preemphasis(y)

    S_stft = librosa.stft(y=y, n_fft=N_FFT_STFT,hop_length=HOP_LENGTH_STFT)
    S_mel = librosa.feature.melspectrogram(S=np.abs(S_stft), sr=sr, n_mels=N_MELS_STFT, hop_length=HOP_LENGTH_STFT)
    log_mel_spectrogram = librosa.power_to_db(S_mel, ref=np.max)

    CQT = librosa.cqt(y,sr=sr, hop_length=HOP_LENGTH_STFT, bins_per_octave=BINS_PER_OCTAVE_CQT, n_bins=N_BINS_CQT)
    cqt_spectrogram = librosa.amplitude_to_db(np.abs(CQT), ref=np.max)

    return log_mel_spectrogram.astype(np.float32), cqt_spectrogram.astype(np.float32)

def main():
    for category in ['normal','abnormal']:
        catetory_path = os.path.join(DATA_RAW_DIR, category)
        audio_files = [f for f in os.listdir(catetory_path) if f.endswith('.wav')]
        print(f"\nProcessing {len(audio_files)} {category} files...")

        for audio_file in tqdm(audio_files, desc=f"Converting {category} audio"):
            audio_path = os.path.join(catetory_path, audio_file)
            base_name = os.path.splitext(audio_file)[0]

            lms_spec, cqt_spec = process_audio_file(audio_path)

            np.save(os.path.join(FEATURES_DIR, category, 'stft',f'{base_name}.npy'),lms_spec)
            np.save(os.path.join(FEATURES_DIR,category,'cqt',f'{base_name}.npy'),cqt_spec)

        
        print("\nPreprocessing Complete!")
        print(f"Spectrograms saved to: {FEATURES_DIR}")

    # --- Verifying the saved file and plotting ---
    plot_sample_spectrograms(FEATURES_DIR, SR, HOP_LENGTH_STFT, N_FFT_STFT, BINS_PER_OCTAVE_CQT)


def plot_sample_spectrograms(FEATURES_DIR, sr, hop_length_stft, n_fft_stft, bins_per_octave_cqt):
    try:
        sample_category = 'normal'
        stft_dir = os.path.join(FEATURES_DIR, sample_category, 'stft')
        cqt_dir = os.path.join(FEATURES_DIR, sample_category, 'cqt')

        if not os.path.exists(stft_dir) or not os.path.exists(cqt_dir):
            print(f"Error: Spectrogram directories not found: {stft_dir} or {cqt_dir}")
            return

        stft_files = [f for f in os.listdir(stft_dir) if f.endswith('.npy')]
        cqt_files = [f for f in os.listdir(cqt_dir) if f.endswith('.npy')]

        if not stft_files or not cqt_files:
            print(f"No .npy files found in {stft_dir} or {cqt_dir} for plotting.")
            return

        sample_stft_path = os.path.join(stft_dir, stft_files[0])
        sample_cqt_path = os.path.join(cqt_dir, cqt_files[0])

        loaded_lms = np.load(sample_stft_path)
        loaded_cqt = np.load(sample_cqt_path)

        if loaded_lms.size == 0 or loaded_cqt.size == 0:
            print(f"Warning: Loaded spectrogram for plotting is empty. LMS size: {loaded_lms.size}, CQT size: {loaded_cqt.size}")
            return

        print(f"\nExample loaded LMS shape: {loaded_lms.shape}")
        print(f"Example loaded CQT shape: {loaded_cqt.shape}")

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        librosa.display.specshow(loaded_lms, sr=sr, x_axis='time', y_axis='mel', hop_length=hop_length_stft, vmin=-80, vmax=0, cmap='gray')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Log-Mel Spectrogram (STFT-based)')
        plt.tight_layout()

        plt.subplot(2, 1, 2)
        librosa.display.specshow(loaded_cqt, sr=sr, x_axis='time', y_axis='cqt_hz', bins_per_octave=bins_per_octave_cqt, vmin=-80, vmax=0, cmap='gray')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Constant-Q Transform Spectrogram')
        plt.tight_layout()

        plt.show()

    except Exception as e:
        print(f"Could not load and plot a sample: {e}")

if __name__ == "__main__":
    main()