import os
import torch
import torchaudio

from audio_encoder import EncodecWrapper
encodec = EncodecWrapper(bandwidth=6.0) # residual vq with K codebooks

from gpt import RapGPT
config = RapGPT.get_default_config()

config.device = 'cuda'
config.vocab_size = encodec.codebook_size + 1 # this is the number of distinct tokens in the codebook plus the special token
config.speacial_token_id = config.vocab_size - 1 # the special token is the last token in the vocab (0 based indexing)
config.num_codebooks = encodec.num_quantizers
config.block_size = 512 # effectivly the context window of the model
config.n_layer = 8
config.n_head = 12 
config.n_embd = config.n_head * 32

# training related stuff
config.learning_rate = 5e-5
config.batch_size = 1
# dataloder parameters
# config.num_workers = 4

from patterns import DelayedPatternProvider # Codebook interleaving patterns: https://arxiv.org/pdf/2306.05284
config.pattern_provider = DelayedPatternProvider(n_q=encodec.num_quantizers)

model = RapGPT(config)

from dataset import LazyLoadedAndEncodedAudioDataset
dataset = LazyLoadedAndEncodedAudioDataset('/home/imenkedir/airap/simple_data', config.block_size)

from trainer import Trainer
trainer = Trainer(config, model, dataset)

from tracker import LossTracker
tracker = LossTracker()
def on_iteration(self):
    stats = tracker.update(self.iter_num, self.loss)
    print(f"Iteration: {stats['iteration']}, "
          f"Current Loss: {stats['current_loss']:.4f}, "
          f"Lowest Loss: {stats['lowest_loss']:.4f}, "
          f"Avg Loss (last 10000): {stats['avg_loss_10000']:.4f}, "
          f"Avg Loss (last 1000): {stats['avg_loss_1000']:.4f}, "
          f"Avg Loss (last 100): {stats['avg_loss_100']:.4f}")

trainer.add_callback('on_iteration', on_iteration)

def on_checkpoint(self):
    stats = tracker.update(self.iter_num, self.loss)
    avg_loss = stats['avg_loss_10000']
    # Get the current script directory
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # Create the checkpoints directory if it doesn't exist
    checkpoints_dir = os.path.join(current_script_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    model_path = os.path.join(checkpoints_dir, f'Step-{self.iter_num}_AvgLoss-{avg_loss}.pth')
    torch.save(self.model.state_dict(), model_path)

trainer.add_callback('on_checkpoint', on_checkpoint)

trainer.run()
# cond = torch.load("encoded_data/80 BPM - T-Pain - Buy You A Drink (Feat. Young Joc)_chunk101.pt")
# print(f"cond :{cond.shape}")
# cond = cond[:, :500, :]

# generation = model.generate(cond.to('cuda'), max_new_tokens=500)
# generation = generation.to('cpu')
# print(f"Generation shape: {generation.shape}")

# _, T, _ = generation.shape
# # Calculate the number of chunks needed
# max_chunck = 10
# num_chunks = (T + max_chunck - 1) // max_chunck  # Equivalent to ceil(n / max_n)
# chuncks = torch.chunk(generation, num_chunks, dim=1)
# processed_chunks = []
# for chunck in chuncks:
#     decoded = encodec.decode_from_codebook_indices(chunck).squeeze(1)
#     processed_chunks.append(decoded)

# output = torch.concat(processed_chunks, dim=-1)

# torchaudio.save('output.wav', output, 44100)