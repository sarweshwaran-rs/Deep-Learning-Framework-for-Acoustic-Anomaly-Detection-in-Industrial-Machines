import os
import librosa
import numpy as np
import torch
from tqdm import tqdm

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
    log_mel_spectrogram = np.clip(log_mel_spectrogram, -40, 5)
    log_mel_spectrogram = normalize(log_mel_spectrogram)

    delta = librosa.feature.delta(log_mel_spectrogram)
    delta2 = librosa.feature.delta(log_mel_spectrogram, order=2)
    lms_stack = np.stack([log_mel_spectrogram, delta, delta2], axis=0)
    
    for i in range(3):
        mean = np.mean(lms_stack[i])
        std = np.std(lms_stack[i])
        lms_stack[i] = (lms_stack[i] - mean) / (std + 1e-6)
    
    #CQT -> dB
    CQT = librosa.cqt(y,sr=sr, hop_length=HOP_LENGTH_STFT, bins_per_octave=BINS_PER_OCTAVE_CQT, n_bins=N_BINS_CQT)
    cqt_spectrogram = librosa.amplitude_to_db(np.abs(CQT), ref=np.max)
    cqt_spectrogram = np.clip(cqt_spectrogram, -40, 5)
    cqt_spectrogram = (cqt_spectrogram - np.mean(cqt_spectrogram)) / (np.std(cqt_spectrogram) + 1e-6)
    cqt_spectrogram = np.expand_dims(cqt_spectrogram.astype(np.float32),axis=0)

    return lms_stack.astype(np.float32), cqt_spectrogram

def main():

    #Iterating through the each id_xx folder
    for id_folder in os.listdir(DATA_RAW_DIR):
        id_folder_path = os.path.join(DATA_RAW_DIR, id_folder)

        if not os.path.isdir(id_folder_path):
            continue

        print(f"\nProcessing data for {id_folder}...")

        for category in ['normal','abnormal']:
            category_path = os.path.join(id_folder_path, category)

            #Ensure the catefory directory exists within the id_folder
            if not os.path.isdir(category_path):
                print(f"Skipping {category_path} as it does not exist.")
                continue

            #Create output directories for the current id_folder and category
            output_stft_dir = os.path.join(FEATURES_DIR, id_folder, category, 'stft')
            output_cqt_dir = os.path.join(FEATURES_DIR, id_folder,category, 'cqt')
            os.makedirs(output_stft_dir, exist_ok=True)
            os.makedirs(output_cqt_dir, exist_ok=True)

            audio_files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
            print(f"Processing {len(audio_files)} {category} files in {id_folder}...")

            for audio_file in tqdm(audio_files, desc=f"Converting {id_folder}/{category} audio"):
                audio_path = os.path.join(category_path, audio_file)
                base_name = os.path.splitext(audio_file)[0]
                
                unique_name = f"{id_folder}_{category}_{base_name}"
                lms_spec, cqt_spec = process_audio_file(audio_path)

                #save the features with the id_folder and category folder
                np.save(os.path.join(output_stft_dir, f"{unique_name}.npy"), lms_spec)
                np.save(os.path.join(output_cqt_dir, f"{unique_name}.npy"),cqt_spec)

            print(f"Preprocessing complete for {id_folder}/{category}")

if __name__ == "__main__":
    main()
