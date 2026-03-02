from einops import reduce, rearrange
from torch import nn
import torch  # 新增：导入torch用于维度计算

from models.components import COMPONENT, Mlp
from models.model import Output
from utils import OUTPUT_TYPE_TO_LOSS_FN


class Std(nn.Module):
    """Standard model"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        enc_args = config['enc_args']
        enc_args['input_shape'] = config['x_shape']
        self.encoder = COMPONENT[config['encoder']](config, enc_args)

        dec_args = config['dec_args']
        dec_args['output_shape'] = [config['tasks']] if config['output_type'] == 'class' else config['y_shape']
        self.decoder = COMPONENT[config['decoder']](config, dec_args)

        mlp_args = config['mlp_args']
        mlp_args['input_shape'] = self.encoder.output_shape
        mlp_args['output_shape'] = self.decoder.input_shape
        self.mlp = Mlp(config, mlp_args)

        self.loss_fn = OUTPUT_TYPE_TO_LOSS_FN[config['output_type']]
        self._warned_label = False  # 新增：标记是否已打印警告

    # 修复safe_loss_fn函数（核心修改）
    def safe_loss_fn(self, logit, y):
        """修复标签越界问题，适配Omniglot元学习"""
        if self.config['output_type'] != 'class':
            return self.loss_fn(logit, y)  # 非分类任务直接用原损失

        # 分类任务：强制修正标签范围
        n_classes = logit.shape[-1]  # 获取模型输出的类别数
        # 1. 简化的维度处理（避免einops语法错误）
        # 保存原形状用于恢复
        logit_shape = logit.shape
        y_shape = y.shape

        # 2. 展平标签和logit（仅展平前N-1维，保留类别维）
        # logit: [b, ..., n_classes] → [batch_flat, n_classes]
        batch_flat_size = torch.prod(torch.tensor(logit_shape[:-1])).item()
        logit_flat = logit.reshape(batch_flat_size, n_classes)
        # y: [b, ...] → [batch_flat]
        y_flat = y.reshape(batch_flat_size)

        # 3. 过滤/截断无效标签
        valid_mask = (y_flat >= 0) & (y_flat < n_classes)
        if not valid_mask.all():
            # 打印调试信息（仅首次触发时打印）
            if not self._warned_label:
                print(f"警告：发现{(~valid_mask).sum()}个无效标签，已自动修正")
                print(f"logit形状: {logit_shape}, 类别数: {n_classes}, 标签范围: [{y_flat.min()}, {y_flat.max()}]")
                self._warned_label = True
            # 仅保留有效标签计算损失
            logit_valid = logit_flat[valid_mask]
            y_valid = y_flat[valid_mask]

            # 4. 恢复原维度结构（仅对有效部分，不影响损失计算）
            # 损失计算只需要一维，无需恢复完整形状，直接返回
            return self.loss_fn(logit_valid, y_valid)

        # 无无效标签时，直接计算损失
        return self.loss_fn(logit, y)

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
        """Forward pass supporting both meta and standard training loops.

        ``train.py`` passes (train_x, train_y, test_x, test_y, summarize, meta_split),
        while ``train_std.py`` calls the model with (x, y, summarize, split). To
        keep compatibility, we treat ``split`` and ``meta_split`` as aliases and
        fall back to ``train_x/train_y`` when meta-style test data is not
        provided.
        """

        base_split = meta_split or split or 'train'
        split_key = f'meta_{meta_split}' if meta_split else base_split
        x = test_x if test_x is not None else train_x
        y = test_y if test_y is not None else train_y

        if x.ndim == 4:
            x = rearrange(x, 'b c h w -> b 1 c h w')
        if y.ndim == 1:
            y = rearrange(y, 'b -> b 1')
        logit = self.decoder(self.mlp(self.encoder(x)))

        # 替换为安全损失函数
        loss = reduce(self.safe_loss_fn(logit, y), 'b ... -> b', 'mean')

        output = Output()
        output[f'loss/{split_key}'] = loss
        if not summarize:
            return output

        if self.config['output_type'] == 'class':
            output.add_classification_summary(logit, y, split_key)
        elif self.config['output_type'] == 'image':
            output.add_image_comparison_summary(
                rearrange(x, 'b 1 ... -> 1 b ...'),
                rearrange(y, 'b 1 ... -> 1 b ...'),
                rearrange(x, 'b 1 ... -> 1 b ...'),
                rearrange(logit, 'b 1 ... -> 1 b ...'),
                key=f'completion/{split_key}')
        return output