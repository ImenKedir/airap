import os
import torch
import torchaudio
from tqdm import tqdm
from audio_encoder import EncodecWrapper


encodec = EncodecWrapper(bandwidth=1.5) # residual vq with K codebook

def convert_mp3s_to_tensors(folder_paths):
    for mp3_folder_path in folder_paths:
        # Create the output folder path by appending "_tensors" to the original folder name
        output_folder_path = mp3_folder_path + "_tensors"
        
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
        
        # Loop through all files in the original folder
        print(f"Converting MP3 files in {mp3_folder_path} to tensors...")
        for filename in tqdm(os.listdir(mp3_folder_path)):
            # Check if the file is an MP3
            if filename.endswith('.mp3'):
                # Construct the full path to the input MP3 file
                mp3_path = os.path.join(mp3_folder_path, filename)
                
                # Load the MP3 file using torchaudio
                waveform, sample_rate = torchaudio.load(mp3_path)
                waveform = waveform.mean(dim=0, keepdim=True) 
                
                # Construct the full path to the output tensor file
                tensor_filename = filename.replace('.mp3', '.pt')
                tensor_path = os.path.join('/Users/imenkedir/dev/airap/data', tensor_filename)
                
                # Save the tensor to the output path
                torch.save(waveform, tensor_path)

# Example usage:
folder_paths = [
    '/Users/imenkedir/Desktop/music/acapellas/Juesswork - Acapella Stash Vol. 1',
    '/Users/imenkedir/Desktop/music/acapellas/Juesswork - Acapella Stash Vol. 2',
    '/Users/imenkedir/Desktop/music/acapellas/Juesswork - Acapella Stash Vol. 3',
]  # Replace with the paths to your MP3 folders
convert_mp3s_to_tensors(folder_paths)
