import os
import librosa
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

MAX_DURATION = 10
MAX_LENGTH = SR * MAX_DURATION

os.makedirs(FEATURES_DIR,exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

def normalize(spec):
    return (spec - np.mean(spec)) / (np.std(spec) + 1e-6)

def process_audio_file(audio_path):

    y, sr = librosa.load(audio_path, sr=SR)
    #Pad or truncate audio to fixed length
    if len(y) < MAX_LENGTH:
        y = np.pad(y,(0,MAX_LENGTH - len(y)))
    else:
        y = y[:MAX_LENGTH]
    
    y = librosa.effects.preemphasis(y)

    #STFT -> Mel -> dB
    S_stft = librosa.stft(y=y, n_fft=N_FFT_STFT,hop_length=HOP_LENGTH_STFT)
    S_mel = librosa.feature.melspectrogram(S=np.abs(S_stft), sr=sr, n_mels=N_MELS_STFT, hop_length=HOP_LENGTH_STFT)
    log_mel_spectrogram = librosa.power_to_db(S_mel, ref=np.max)
    log_mel_spectrogram = normalize(log_mel_spectrogram)

    #CQT -> dB
    CQT = librosa.cqt(y,sr=sr, hop_length=HOP_LENGTH_STFT, bins_per_octave=BINS_PER_OCTAVE_CQT, n_bins=N_BINS_CQT)
    cqt_spectrogram = librosa.amplitude_to_db(np.abs(CQT), ref=np.max)
    cqt_spectrogram = normalize(cqt_spectrogram)

    return log_mel_spectrogram.astype(np.float32), cqt_spectrogram.astype(np.float32)

def main():

    #Iterating through the each id_xx folder
    for id_folder in os.listdir(DATA_RAW_DIR):
        id_folder_path = os.path.join(DATA_RAW_DIR, id_folder)

        if not os.path.isdir(id_folder_path):
            continue

        print(f"\nProcessing data for {id_folder}...")

        for category in ['normal','abnormal']:
            catetory_path = os.path.join(id_folder_path, category)

            #Ensure the catefory directory exists within the id_folder
            if not os.path.isdir(catetory_path):
                print(f"Skipping {catetory_path} as it does not exist.")
                continue

            #Create output directories for the current id_folder and category
            output_stft_dir = os.path.join(FEATURES_DIR, id_folder, category, 'stft')
            output_cqt_dir = os.path.join(FEATURES_DIR, id_folder,category, 'cqt')
            os.makedirs(output_stft_dir, exist_ok=True)
            os.makedirs(output_cqt_dir, exist_ok=True)

            audio_files = [f for f in os.listdir(catetory_path) if f.endswith('.wav')]
            print(f"Processing {len(audio_files)} {category} files in {id_folder}...")

            for audio_file in tqdm(audio_files, desc=f"Converting {id_folder}/{category} audio"):
                audio_path = os.path.join(catetory_path, audio_file)
                base_name = os.path.splitext(audio_file)[0]
                
                unique_name = f"{id_folder}_{category}_{base_name}"
                lms_spec, cqt_spec = process_audio_file(audio_path)

                #save the features with the id_folder and category folder
                np.save(os.path.join(output_stft_dir, f"{unique_name}.npy"), lms_spec)
                np.save(os.path.join(output_cqt_dir, f"{unique_name}.npy"),cqt_spec)

                # Visual sanity check
                if np.random.rand() < 0.01:
                    plt.figure(figsize=(10, 4))
                    plt.imshow(lms_spec, aspect='auto', origin='lower')
                    plt.title(f"Mel Spectrogram - {unique_name}")
                    plt.colorbar()
                    plt.tight_layout()
                    plt.show()

            print(f"Preprocessing complete for {id_folder}/{category}")

if __name__ == "__main__":
    main()
