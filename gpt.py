"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""
import math

import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from gpt_config import GPTConfig
from patterns import CodebooksPatternProvider
from tqdm import tqdm

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class SelfAttentionBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


class RapGPT(nn.Module):
    """ GPT Language Model """

    @staticmethod
    def get_default_config() -> GPTConfig:
        C = GPTConfig()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 0
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        C.embd_pdrop = 0.1
        C.attn_pdrop = 0.1
        C.resid_pdrop = 0.1

        return C

    def __init__(self, config):
        super().__init__()
        self.config = config
        
        assert config.vocab_size is not None, "vocab_size must be specified"
        assert config.block_size is not None, "block_size must be specified"

        assert isinstance(config.pattern_provider, CodebooksPatternProvider), "pattern_provider must be specified and be an instance of CodebooksPatternProvider"
        self.pattern_provider: CodebooksPatternProvider = config.pattern_provider

        assert isinstance(config.block_size, int), "block_size must be specified and be an integer"
        self.block_size = config.block_size
        
        assert isinstance(config.num_codebooks, int), "num_codebooks must be specified and be an integer"
        self.num_codebooks = config.num_codebooks

        assert isinstance(config.speacial_token_id, int), "speacial_token_id must be specified and be an integer"
        self.special_token_id = config.speacial_token_id

        self.embedding_layers = nn.ModuleList([
            nn.Embedding(config.vocab_size, config.n_embd) for _ in range(config.num_codebooks)
        ])

        self.transformer = nn.ModuleDict(dict(
            positional_embeddings = nn.Embedding(config.block_size, config.n_embd),
            dropout = nn.Dropout(config.embd_pdrop),
            attention = nn.ModuleList([SelfAttentionBlock(config) for _ in range(config.n_layer)]),
            layernorm = nn.LayerNorm(config.n_embd),
        ))

        self.linear_heads = nn.ModuleList(
            [nn.Linear(config.n_embd, config.vocab_size) for _ in range(config.num_codebooks)]
        )

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        transformer_parameters = [p.numel() for p in self.transformer.parameters()]
        n_transformer_params = sum(transformer_parameters)

        embedding_parameters = [p.numel() for p in self.embedding_layers.parameters()]
        n_embedding_params = sum(embedding_parameters)

        linear_head_parameters = [p.numel() for p in self.linear_heads.parameters()]
        n_linear_head_params = sum(linear_head_parameters)

        print(f"number of transformer parameters: {n_transformer_params / 1e6:.2f}M")
        print(f"number of embedding parameters: {n_embedding_params / 1e6:.2f}M")
        print(f"number of linear head parameters: {n_linear_head_params / 1e6:.2f}M")


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, codes, targets=None):
        codes = torch.squeeze(codes, dim=1)
        codes = codes.permute(0, 2, 1) # [B, T, K] -> [B, K, T]
        codes = codes.contiguous()
        B, K, T = codes.shape

        # map codes [B, K, T] into pattern sequence [B, K, S] using special_token_id for masked tokens
        pattern = self.pattern_provider.get_pattern(T)
        sequence_codes, _, _ = pattern.build_pattern_sequence(
            codes,
            self.special_token_id,
            keep_only_valid_steps=True,
        )
        B, K, T = sequence_codes.shape

        token_embeddings = sum(
            [self.embedding_layers[k](sequence_codes[:, k, :]) for k in range(K)]
        ) # [B, K, S] -> [B, S, D] where D is the embedding dimension

        positional_embeddings = self.transformer.positional_embeddings( 
          torch.arange(0, T, dtype=torch.long, device=self.config.device).unsqueeze(0).to(self.config.device) 
        ).repeat(B, 1, 1) # [B, S, D]

        # combine the positional information with the toekn embeddings
        x = self.transformer.dropout(token_embeddings + positional_embeddings)

        for block in self.transformer.attention:
            x = block(x)

        out = self.transformer.layernorm(x)

        logits = torch.stack([self.linear_heads[k](out) for k in range(K)], dim=1)  # [B, S, D] -> [B, K, S, card]

        # map back the logits on pattern sequence to logits on original codes: [B, K, S, card] -> [B, K, T, card]
        # and provide the corresponding mask over invalid positions of tokens
        logits = logits.permute(0, 3, 1, 2)  # [B, card, K, S]
        # note: we use nans as special token to make it obvious if we feed unexpected logits
        logits, _, logits_mask = pattern.revert_pattern_logits(
            logits, 
            float('nan'), 
            keep_only_valid_steps=True
        )
        logits = logits.permute(0, 2, 3, 1)  # [B, card, K, T] -> [B, K, T, card]

        loss = None
        if targets is not None:
            logits_mask = logits_mask[None, :, :].expand(B, -1, -1)  # [K, T] -> [B, K, T]

            flat_logits = logits.reshape(-1, logits.size(-1)) # [B, K, T, card] -> [B*K*T, card]
            flat_logits = flat_logits[logits_mask.reshape(-1)]

            flat_targets = targets.reshape(-1) # [B, K, T] -> [B*K*T]
            flat_targets = flat_targets[logits_mask.reshape(-1)]

            loss = F.cross_entropy(flat_logits, flat_targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, in_seq, max_new_tokens):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        in_seq = in_seq.to(self.config.device)
        for _ in tqdm(range(max_new_tokens)):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = in_seq if in_seq.size(1) <= self.block_size else in_seq[:, -(self.block_size - 1):, :]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            logits = logits.permute(0, 1, 3, 2)  # [B, K, T, card] -> [B, K, card, T]
            logits = logits[..., -1]  # [B, K, card, T] -> [B, K, card]
            # greedly select the next token
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            # append sampled index to the running sequence and continue
            in_seq = torch.cat((in_seq, next_token.permute(0, 2, 1)), dim=1)

        return in_seq

    @classmethod
    def from_pretrained(cls, model_state_dict_path, config):
        model = cls(config)
        state_dict = torch.load(model_state_dict_path, map_location=config.device)
        model.load_state_dict(state_dict)
        model.to(config.device)
        return model

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer