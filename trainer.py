"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader

class LossTracker:
    def __init__(self, max_len=10000):
        self.lowest_loss = float('inf')
        self.losses = []
        self.max_len = max_len
    
    def update(self, iteration, current_loss):
        # Update the lowest loss
        if current_loss < self.lowest_loss:
            self.lowest_loss = current_loss
        
        # Add the current loss to the list
        self.losses.append(current_loss)
        
        # Ensure we only keep the last max_len losses
        if len(self.losses) > self.max_len:
            self.losses.pop(0)
        
        # Calculate average losses for the last 10,000, 1,000, and 100 iterations
        avg_loss_10000 = self._calculate_average_loss(10000)
        avg_loss_1000 = self._calculate_average_loss(1000)
        avg_loss_100 = self._calculate_average_loss(100)
        
        return {
            'iteration': iteration,
            'current_loss': current_loss,
            'lowest_loss': self.lowest_loss,
            'avg_loss_10000': avg_loss_10000,
            'avg_loss_1000': avg_loss_1000,
            'avg_loss_100': avg_loss_100
        }
    
    def _calculate_average_loss(self, num_iterations):
        if len(self.losses) < num_iterations:
            return sum(self.losses) / len(self.losses)
        else:
            return sum(self.losses[-num_iterations:]) / num_iterations

class Trainer:
    def __init__(self, config, model, train_dataset):
        self.config = config
        
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print(f"Created trainer on device: {self.device}")

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)

        # setup the dataloader
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        model.train()
        # self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        self.iter_num = 0
        while True:
            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            _, self.loss = model(x, y)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            self.optimizer.step()

            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            self.trigger_callbacks('on_iteration')

            if self.iter_num % 10000 == 0:
                self.trigger_callbacks('on_checkpoint')
