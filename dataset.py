import os
import torch
from random import randint
from torch.utils.data import Dataset

class LazyLoadedAndEncodedAudioDataset(Dataset):
    def __init__(self, folder, chunk_size):
        self.folder = folder
        self.files = [filename for filename in os.listdir(folder) if filename.endswith('.pt')]
        self.chunck_size = chunk_size - 8

        tensors = []
        for file in self.files:
            path = os.path.join(self.folder, file)
            tensor = torch.load(path)
            tensors.append(tensor)
        
        self.tensor = torch.cat(tensors, dim=1)
        print(f'full data shape: {self.tensor.shape}')

    def __len__(self):
        return self.tensor[:, :5000, :].shape[1] - self.chunck_size

    def __getitem__(self, idx):
        x = self.tensor[:, idx:idx+self.chunck_size, :]
        y = self.tensor[:, idx+1:idx+self.chunck_size+1, :]

        return x.clone().detach(), y.clone().detach()
