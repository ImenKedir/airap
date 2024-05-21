import os
import torch
from tqdm import tqdm
from audio_encoder import EncodecWrapper

encodec = EncodecWrapper(bandwidth=6.0)

def process_tensors(source_dir, target_dir):
    # Check if the source directory exists
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist.")
        return
    
    # Create the target directory if it does not exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Iterate over all files in the source directory
    for filename in os.listdir(source_dir):
            print(f"On: {filename}")
            source_path = os.path.join(source_dir, filename)
            
            # Load the tensor
            tensor = torch.load(source_path)
            
            # Process the tensor in smaller chunks if it's too large
            chunk_size = 100000  # Adjust chunk size based on your memory capacity
            num_chunks = (tensor.shape[-1] + chunk_size - 1) // chunk_size
            tokens = []

            for i in tqdm(range(num_chunks)):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, tensor.shape[-1])
                tensor_chunk = tensor[..., start:end]

                # Tokenize the tensor chunk
                tokens.append(encodec.tokenize(tensor_chunk))

                # Save the tokens to the target directory
                chunk_filename = f"{os.path.splitext(filename)[0]}_chunk{i}.pt"
                target_path = os.path.join(target_dir, chunk_filename)
            
            torch.save(torch.concat(tokens, dim=-2), target_path)
            
    print("Processing complete.")

if __name__ == "__main__":
    source_directory = "/home/imenkedir/airap/data"  # Replace with the path to your source directory
    target_directory = "/home/imenkedir/airap/simple_data"  # Replace with the path to your target directory

    process_tensors(source_directory, target_directory)
