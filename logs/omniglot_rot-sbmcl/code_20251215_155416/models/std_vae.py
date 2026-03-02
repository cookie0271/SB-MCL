import math
import torch
from einops import reduce, rearrange
from torch import nn
import torch.nn.functional as F

from models.components import COMPONENT, Mlp
from models.model import Output
from utils import binarize, reparameterize, kl_div, nll_to_bpd


class StdVae(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        enc_args = config['enc_args']
        enc_args['input_shape'] = config['x_shape']
        self.encoder = COMPONENT[config['encoder']](config, enc_args)
        self.x_dims = math.prod(config['x_shape'])

        dec_args = config['dec_args']
        dec_args['output_shape'] = config['x_shape']
        self.decoder = COMPONENT[config['decoder']](config, dec_args)

        mlp_args = config['mlp_args']
        mlp_args['input_shape'] = self.encoder.output_shape
        mlp_args['output_shape'] = [2, config['latent_dim']]
        self.mlp = Mlp(config, mlp_args)

        self.kl_weight = 0.0

    def forward(
        self,
        train_x,
        train_y,
        test_x=None,
        test_y=None,
        summarize=False,
        meta_split=None,
        split=None,
    ):
        base_split = meta_split or split or 'train'
        split_key = f'meta_{meta_split}' if meta_split else base_split
        x = test_x if test_x is not None else train_x

        if x.ndim == 4:
            x = rearrange(x, 'b c h w -> b 1 c h w')
        x = binarize(x)

        x_enc = self.encoder(x)
        mlp_out = self.mlp(x_enc)
        mean, log_var = torch.unbind(mlp_out, dim=-2)

        self.kl_weight += 1. / self.config['kl_warmup']
        self.kl_weight = min(self.kl_weight, 1.0)
        kl_loss = kl_div(mean, log_var)

        latent_samples = self.config['eval_latent_samples'] if base_split == 'test' else 1
        recon_loss = torch.zeros_like(kl_loss)
        for _ in range(latent_samples):
            latent = reparameterize(mean, log_var)
            logit = self.decoder(latent)
            bce = F.binary_cross_entropy_with_logits(logit, x, reduction='none')
            bce = reduce(bce, 'b 1 c h w -> b 1', 'sum')
            recon_loss = recon_loss + bce
        recon_loss = recon_loss / latent_samples

        kl_weight = self.kl_weight if base_split == 'train' else 1.0
        loss = nll_to_bpd(recon_loss + kl_loss * kl_weight, self.x_dims)
        loss = reduce(loss, 'b l -> b', 'mean')

        output = Output()
        output[f'loss/{split_key}'] = loss
        if not summarize:
            return output

        output[f'loss/kl/{split_key}'] = reduce(kl_loss, 'b l -> b', 'mean')
        output[f'loss/recon/{split_key}'] = reduce(recon_loss, 'b l -> b', 'mean')
        output.add_image_comparison_summary(x, torch.sigmoid(logit), key=f'recon/{split_key}')
        return output