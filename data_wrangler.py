import os
import torch
import torchaudio
from tqdm import tqdm
from audio_encoder import EncodecWrapper


encodec = EncodecWrapper(bandwidth=3.0) # residual vq with K codebook

def convert_mp3s_to_tensors(folder_paths):
    for data_folder_path in folder_paths:
        # Create the output folder path by appending "_tensors" to the original folder name
        output_folder_path = data_folder_path + "_tensors"
        
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        
        # Loop through all files in the original folder
        print(f"Converting MP3 files in {data_folder_path} to tensors...")
        for filename in tqdm(os.listdir(data_folder_path)):
            # Check if the file is an MP3
            if filename.endswith('.wav'):
                # Construct the full path to the input MP3 file
                wav_path = os.path.join(data_folder_path, filename)
                
                # Load the MP3 file using torchaudio
                waveform, sample_rate = torchaudio.load(wav_path)
                waveform = waveform.mean(dim=0, keepdim=True)

                # Encode the waveform using the EncodecWrapper
                tokens = encodec.tokenize(waveform)
                
                # Construct the full path to the output tensor file
                tensor_filename = filename.replace('.wav', '.pt')
                tensor_path = os.path.join(output_folder_path, tensor_filename)
                
                # Save the tensor to the output path
                print(f"Saving tensor to {tensor_path}")
                torch.save(tokens, tensor_path)

# Example usage:
folder_paths = [
    '/Users/imenkedir/dev/airap/data',
]  # Replace with the paths to your MP3 folders
convert_mp3s_to_tensors(folder_paths)
