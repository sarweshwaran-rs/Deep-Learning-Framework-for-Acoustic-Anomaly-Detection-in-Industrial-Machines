import os
import librosa
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchaudio.transforms as T

# Paths
BASE_DIR = r'C:\Users\sarwe\raw'
DATA_RAW_DIR = os.path.join(BASE_DIR, '-6_dB_pump')
FEATURES_DIR = os.path.join(BASE_DIR, '-6_dB_features')
IMAGES_DIR = os.path.join(FEATURES_DIR, 'images')

# Audio Settings
SR = 16000
N_FFT_STFT = 512
HOP_LENGTH_STFT = 256
N_MELS_STFT = 64

BINS_PER_OCTAVE_CQT = 24
N_BINS_CQT = 72

MAX_DURATION = 10
MAX_LENGTH = SR * MAX_DURATION

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")


def minmax_normalize(spec):
    min_val = spec.min()
    max_val = spec.max()
    return (spec - min_val) / (max_val - min_val + 1e-6)


def apply_augmentation(spec_tensor):
    spec_tensor = T.FrequencyMasking(freq_mask_param=12)(spec_tensor)
    spec_tensor = T.TimeMasking(time_mask_param=18)(spec_tensor)
    return spec_tensor


def process_audio_file(audio_path, apply_aug=False):
    y, sr = librosa.load(audio_path, sr=SR)
    y, _ = librosa.effects.trim(y, top_db=30)

    if len(y) < MAX_LENGTH:
        pad_len = MAX_LENGTH - len(y)
        left_pad = np.random.randint(0, pad_len + 1)
        right_pad = pad_len - left_pad
        y = np.pad(y, (left_pad, right_pad))
    else:
        y = y[:MAX_LENGTH]

    y = librosa.effects.preemphasis(y)

    # STFT
    S_stft = librosa.stft(y=y, n_fft=N_FFT_STFT, hop_length=HOP_LENGTH_STFT)
    S_mel = librosa.feature.melspectrogram(S=np.abs(S_stft), sr=sr, n_mels=N_MELS_STFT, hop_length=HOP_LENGTH_STFT)
    log_mel_spectrogram = librosa.power_to_db(S_mel, ref=np.max)
    log_mel_spectrogram = minmax_normalize(log_mel_spectrogram)

    # CQT
    CQT = librosa.cqt(y, sr=sr, hop_length=HOP_LENGTH_STFT, bins_per_octave=BINS_PER_OCTAVE_CQT, n_bins=N_BINS_CQT)
    cqt_spectrogram = librosa.amplitude_to_db(np.abs(CQT), ref=np.max)
    cqt_spectrogram = minmax_normalize(cqt_spectrogram)

    if apply_aug:
        log_mel_spectrogram = apply_augmentation(torch.tensor(log_mel_spectrogram).unsqueeze(0)).squeeze(0).numpy()
        cqt_spectrogram = apply_augmentation(torch.tensor(cqt_spectrogram).unsqueeze(0)).squeeze(0).numpy()

    return log_mel_spectrogram.astype(np.float32), cqt_spectrogram.astype(np.float32)


def save_spectrogram_image(npy_path, output_img_path, cmap='viridis'):
    try:
        spec = np.load(npy_path)
        if spec.ndim == 3:
            spec = spec[0]
        plt.figure(figsize=(4, 3))
        plt.axis('off')
        plt.imshow(spec, aspect='auto', origin='lower', cmap=cmap)
        plt.tight_layout(pad=0)
        plt.savefig(output_img_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        print(f"Error processing {npy_path}: {e}")


def process_and_save():
    for id_folder in os.listdir(DATA_RAW_DIR):
        id_folder_path = os.path.join(DATA_RAW_DIR, id_folder)
        if not os.path.isdir(id_folder_path):
            continue

        print(f"\nProcessing data for {id_folder}...")

        for category in ['normal', 'abnormal']:
            category_path = os.path.join(id_folder_path, category)
            if not os.path.isdir(category_path):
                print(f"Skipping {category_path} as it does not exist.")
                continue

            output_stft_dir = os.path.join(FEATURES_DIR, id_folder, category, 'stft')
            output_cqt_dir = os.path.join(FEATURES_DIR, id_folder, category, 'cqt')
            os.makedirs(output_stft_dir, exist_ok=True)
            os.makedirs(output_cqt_dir, exist_ok=True)

            image_stft_dir = os.path.join(IMAGES_DIR, id_folder, category, 'stft')
            image_cqt_dir = os.path.join(IMAGES_DIR, id_folder, category, 'cqt')
            os.makedirs(image_stft_dir, exist_ok=True)
            os.makedirs(image_cqt_dir, exist_ok=True)

            audio_files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
            print(f"Processing {len(audio_files)} {category} files in {id_folder}...")

            for audio_file in tqdm(audio_files, desc=f"Converting {id_folder}/{category}"):
                audio_path = os.path.join(category_path, audio_file)
                base_name = os.path.splitext(audio_file)[0]
                unique_name = f"{id_folder}_{category}_{base_name}"

                try:
                    lms_spec, cqt_spec = process_audio_file(audio_path, apply_aug=False)

                    # Save .npy
                    stft_path = os.path.join(output_stft_dir, f"{unique_name}.npy")
                    cqt_path = os.path.join(output_cqt_dir, f"{unique_name}.npy")
                    np.save(stft_path, lms_spec)
                    np.save(cqt_path, cqt_spec)

                    # Save images
                    save_spectrogram_image(stft_path, os.path.join(image_stft_dir, f"{unique_name}.png"), cmap='viridis')
                    save_spectrogram_image(cqt_path, os.path.join(image_cqt_dir, f"{unique_name}.png"), cmap='plasma')

                except Exception as e:
                    print(f"[Error] Skipping {audio_file}: {e}")

            print(f"Preprocessing complete for {id_folder}/{category}")


if __name__ == "__main__":
    process_and_save()
