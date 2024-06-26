from functools import reduce
from einops import rearrange, pack, unpack

import torch
from torch import nn

from torchaudio.functional import resample

from vector_quantize_pytorch import ResidualVQ

from encodec import EncodecModel
from encodec.utils import _linear_overlap_add

# helper functions

def exists(val):
    return val is not None

# hacky way to get num quantizers

def get_num_quantizers(model: EncodecModel, audio_length = 512):
    out = model.encode(torch.randn(1, 1, audio_length))
    return out[0][0].shape[1]

class EncodecWrapper(nn.Module):
    """
    Support pretrained 24kHz Encodec by Meta AI, if you want to skip training SoundStream.

    TODO:
    - see if we need to keep the scaled version and somehow persist the scale factors for when we need to decode? Right
        now I'm just setting self.model.normalize = False to sidestep all of that
    - see if we can use the 48kHz model, which is specifically for music. Right now we're using the 24kHz model because
        that's what was used in MusicLM and avoids any resampling issues.
    -

    """
    def __init__(
        self,
        target_sample_hz = 24000,
        strides = (2, 4, 5, 8),
        num_quantizers = 8,
        bandwidth = 6.0
    ):
        super().__init__()
        # Instantiate a pretrained EnCodec model
        self.model = EncodecModel.encodec_model_24khz()
        self.model.normalize = False # this means we don't need to scale codes e.g. when running model.encode(wav)

        # The number of codebooks used will be determined by the bandwidth selected.
        # E.g. for a bandwidth of 6kbps, `n_q = 8` codebooks are used.
        # Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
        # For the 48 kHz model, only 3, 6, 12, and 24 kbps are supported. The number
        # of codebooks for each is half that of the 24 kHz model as the frame rate is twice as much.

        # bandwidth affects num quantizers used: https://github.com/facebookresearch/encodec/pull/41
        self.model.set_target_bandwidth(bandwidth)
        num_quantizers = get_num_quantizers(self.model)

        # Fields that SoundStream has that get used externally. We replicate them here.
        self.target_sample_hz = target_sample_hz
        assert self.target_sample_hz == 24000, "haven't done anything with non-24kHz yet"

        codebook_size = 1024
        codebook_dim = 128
        self.rq_groups = 1
        self.strides = strides # used in seq_len_multiple_of

        # cross entropy loss to indices passed in on l2 distance logits introduced in vector-quantize-pytorch 1.2.2
        self.rq = ResidualVQ(
            dim = codebook_dim,
            codebook_size = codebook_size,
            num_quantizers = num_quantizers
        )

        # copy codebook over to ResidualVQ for cross entropy loss logic from naturalspeech2
        # luckily, it seems Meta AI basically used ResidualVQ code verbatim. makes porting it over easy
        for encodec_rq_layer, rq_layer in zip(self.model.quantizer.vq.layers, self.rq.layers):
            encodec_codebook = dict(encodec_rq_layer._codebook.named_buffers()).get('embed')
            vq_codebook = dict(rq_layer._codebook.named_buffers()).get('embed')

            encodec_codebook = rearrange(encodec_codebook, '... -> 1 ...')
            vq_codebook.copy_(encodec_codebook)
    
    @property
    def num_quantizers(self):
        return self.rq.codebooks.shape[0] # quantizers, size, dim

    @property
    def codebook_size(self):
        return self.rq.codebooks.shape[1] # quantizers, size, dim 
    
    @property
    def codebook_dim(self):
        return self.codebook_dim.shape[2] # quantizers, size, dim

    @property
    def seq_len_multiple_of(self):
        return reduce(lambda x, y: x * y, self.strides)

    @property
    def downsample_factor(self):
        return self.seq_len_multiple_of

    @torch.no_grad()
    def tokenize(self, audio):
        self.eval()
        _, codes, _ = self.forward(audio)
        return codes

    def forward(
        self,
        x,
        input_sample_hz = None,
        return_encoded = False,
        **kwargs
    ):

        x, ps = pack([x], '* n')

        if exists(input_sample_hz):
            x = resample(x, input_sample_hz, self.target_sample_hz)

        # kwargs for stuff like return_encoded=True, which SoundStream uses but Encodec doesn't
        assert not self.model.training, "Encodec is pretrained and should never be called outside eval mode."
        # Unlike in the Encodec sample code in its README, x has already been resampled so we don't need to call
        # convert_audio and unsqueeze. The convert_audio function also doesn't play nicely with batches.

        # b = batch, t = timesteps, 1 channel for the 24kHz model, 2 channels for the 48kHz model
        wav = rearrange(x, f'b t -> b {self.model.channels} t')

        # Extract discrete codes from EnCodec
        with torch.inference_mode():
            encoded_frames = self.model.encode(wav)
        # encoded_frames is a list of (frame, scale) tuples. Scale is a scalar but we don't use it. Frame is a tensor
        # of shape [batch, num_quantizers, num_samples_per_frame]. We want to concatenate the frames to get all the
        # timesteps concatenated.
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [batch, num_quantizers, timesteps]
        # transformer code that uses codec expects codes to be [batch, timesteps, num_quantizers]
        codes = rearrange(codes, 'b q n -> b n q')  # result: [batch, timesteps, num_quantizers]
        # in original soundstream, is x, indices, commit_loss. But we only use indices in eval mode, so just keep that.

        # allow for returning of sum of quantized embeddings

        emb = None

        if return_encoded:
            emb = self.get_emb_from_indices(codes)
            emb, = unpack(emb, ps, '* n c')

        codes, = unpack(codes, ps, '* n q')

        return emb, codes, None

    def decode_from_codebook_indices(self, quantized_indices):
        # Input: batch x num tokens x num quantizers
        # Output: batch x 1 x num samples

        assert self.model.sample_rate == 24000,\
            "if changing to 48kHz, that model segments its audio into lengths of 1.0 second with 1% overlap, whereas " \
            "the 24kHz doesn't segment at all. this means the frame decode logic might change; this is a reminder to " \
            "double check that."
        # Since 24kHz pretrained doesn't do any segmenting, we have all the frames already (1 frame = 1 token in quantized_indices)

        # The following code is hacked in from self.model.decode() (Encodec version 0.1.1) where we skip the part about
        # scaling.
        # Shape: 1 x (num_frames * stride product). 1 because we have 1 frame (because no segmenting)
        frames = self._decode_frame(quantized_indices)
        result = _linear_overlap_add(frames, self.model.segment_stride or 1)
        # TODO: I'm not overly pleased with this because when this function gets called, we just rearrange the result
        #   back to b n anyways, but we'll keep this as a temporary hack just to make things work for now
        return rearrange(result, 'b n -> b 1 n')

    def get_emb_from_indices(self, indices):
        codes = rearrange(indices, 'b t q -> q b t')
        emb = self.model.quantizer.decode(codes)
        return rearrange(emb, 'b c n -> b n c')

    def decode(self, emb):
        emb = rearrange(emb, 'b n c -> b c n')
        return self.model.decoder(emb)

    def _decode_frame(self, quantized_indices):
        # The following code is hacked in from self.model._decode_frame() (Encodec version 0.1.1) where we assume we've
        # already unwrapped the EncodedFrame
        # Input: batch x num tokens x num quantizers
        # Output: batch x new_num_samples, where new_num_samples is num_frames * stride product (may be slightly
        # larger than original num samples as a result, because the last frame might not be "fully filled" with samples
        # if num_samples doesn't divide perfectly).
        # num_frames == the number of acoustic tokens you have, one token per frame
        codes = rearrange(quantized_indices, 'b t q -> q b t')
        emb = self.model.quantizer.decode(codes)
        # emb shape: batch x self.model.quantizer.dimension x T. Note self.model.quantizer.dimension is the embedding dimension
        return self.model.decoder(emb)
