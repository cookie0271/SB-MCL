import torch
from einops import reduce, repeat, rearrange

from models.components import COMPONENT
from models.components.diffusion import Diffusion
from models.model import Output


class StdDDPM(Diffusion):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.set_denoiser(config)

    def set_denoiser(self, config):
        # UNet
        unet_args = config['unet_args']
        unet_args['img_channels'] = config['x_shape'][0]
        self.denoiser = COMPONENT[config['backbone']](**unet_args)

<<<<<<< HEAD
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
        elif x.ndim != 5:
            raise ValueError(f'Expected 4D or 5D input, got shape {x.shape}')

        if base_split == 'test' and x.shape[1] != self.config['eval_t_batch']:
            x = repeat(x, 'b l ... -> b r ...', r=self.config['eval_t_batch'])
        batch, inner_batch = x.shape[:2]

        it = 1 if base_split == 'train' else self.config['eval_t_num'] // self.config['eval_t_batch']
=======
    def forward(self, x, y, summarize, split):
        if split == 'test':
            x = repeat(x, 'b ... -> b l ...', l=self.config['eval_t_batch'])
        else:
            x = rearrange(x, 'b ... -> b 1 ...')
        batch, inner_batch = x.shape[:2]

        it = 1 if split == 'train' else self.config['eval_t_num'] // self.config['eval_t_batch']
>>>>>>> fd9ffc3fef8de5abda2c3d97498dae9c8a145d15
        loss_sum = 0
        for i in range(it):
            # Forward process
            t = torch.randint(low=0, high=self.n_times, size=(batch, inner_batch), device=x.device)
            noisy_x, noise = self.make_noisy(x, t)
            pred_noise = self.denoiser(noisy_x, t)

            loss = self.loss_fn(pred_noise, noise)
            loss = reduce(loss, 'b l c h w -> b', 'mean')
            loss_sum = loss_sum + loss
        loss = loss_sum / it

        output = Output()
<<<<<<< HEAD
        output[f'loss/{split_key}'] = loss
        if not summarize:
            return output

        if base_split == 'train':
            # Evaluation
            generated_images = self.generate(8)
            output.add_image_comparison_summary(generated_images, key=f'generation/{split_key}')

        return output
=======
        output[f'loss/{split}'] = loss
        if not summarize:
            return output

        if split == 'train':
            # Evaluation
            generated_images = self.generate(8)
            output.add_image_comparison_summary(generated_images, key=f'generation/{split}')

        return output
>>>>>>> fd9ffc3fef8de5abda2c3d97498dae9c8a145d15
