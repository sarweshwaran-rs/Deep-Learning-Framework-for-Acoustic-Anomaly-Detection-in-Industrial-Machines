import os 
import librosa 
import numpy as np 
import torch 
from tqdm import tqdm 
 
BASE_DIR = r'C:\Users\sarwe\raw'
DATA_RAW_DIR = os.path.join(BASE_DIR, '0_dB_valve')
FEATURES_DIR = os.path.join(r'F:\CapStone\DFCA\data\features', '0_dB_valve_features')
print(f"Extracting Feature from: {DATA_RAW_DIR}")
print(f"Storing the Extracting Feature in the  {FEATURES_DIR}")

SR = 16000 
N_FFT_STFT = 512 
HOP_LENGTH_STFT = 256 
N_MELS_STFT = 64 
 
BINS_PER_OCTAVE_CQT = 36 
N_BINS_CQT = 84 
 
os.makedirs(os.path.join(FEATURES_DIR),exist_ok=True) 
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(f"Using Device: {device}") 


def minmax_normalize(spec):
    min_val = spec.min()
    max_val = spec.max()
    return (spec - min_val) / (max_val - min_val + 1e-6)


def process_audio_file(audio_path): 
 
    y, sr = librosa.load(audio_path, sr=SR) 
 
    y = librosa.effects.preemphasis(y) 
 
    S_stft = librosa.stft(y=y, n_fft=N_FFT_STFT,hop_length=HOP_LENGTH_STFT) 
    S_mel = librosa.feature.melspectrogram(S=np.abs(S_stft), sr=sr, n_mels=N_MELS_STFT, hop_length=HOP_LENGTH_STFT) 
    log_mel_spectrogram = librosa.power_to_db(S_mel, ref=np.max) 
    log_mel_spectrogram = minmax_normalize(log_mel_spectrogram)
    
    CQT = librosa.cqt(y,sr=sr, hop_length=512, bins_per_octave=BINS_PER_OCTAVE_CQT, n_bins=N_BINS_CQT) 
    cqt_spectrogram = librosa.amplitude_to_db(np.abs(CQT), ref=np.max) 
 
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
 
            print("\nPreprocessing Complete") 
             
if __name__ == "__main__": 
    main() 
