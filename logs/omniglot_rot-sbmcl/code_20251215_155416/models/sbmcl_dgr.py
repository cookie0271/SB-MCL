import torch
from einops import rearrange

from models.sbmcl import Sbmcl, sequential_bayes
from models.components import Mlp


class SbmclDgr(Sbmcl):
    """
    SB-MCL variant with a learned geometric gating network (AlphaNet).

    AlphaNet inspects the geometric relation between the incoming update
    (mean/log precision) and the accumulated belief, and outputs a trust
    coefficient that scales the write strength before Bayesian aggregation.
    """

    def __init__(self, config):
        super().__init__(config)
        feature_dim = 8 * config['z_dim']
        alpha_args = config['alpha_args']
        alpha_args.setdefault('output_activation', 'none')
        alpha_args['input_shape'] = (feature_dim,)
        alpha_args['output_shape'] = (config['z_dim'],)
        self.alpha_net = Mlp(config, alpha_args)
        self.alpha_eps = config.get('alpha_eps', 1e-4)

    def forward_train(self, train_x, train_y, prior_mean=None, prior_log_pre=None):
        if 'xy_encoder' in self.config and self.config['xy_encoder'] == 'CompXYEncoder':
            mean_log_pre = self.xy_encoder(train_x, train_y)
        else:
            train_x_enc = self.x_encoder(train_x)
            train_y_enc = self.y_encoder(train_y)
            train_x_enc = rearrange(train_x_enc, 'b l ... -> b l (...)')
            train_y_enc = rearrange(train_y_enc, 'b l ... -> b l (...)')
            mean_log_pre = self.xy_encoder(torch.cat([train_x_enc, train_y_enc], dim=-1))

        mean, log_pre = torch.unbind(mean_log_pre, dim=-2)

        if prior_mean is None or prior_log_pre is None:
            return sequential_bayes(mean, log_pre, prior_mean, prior_log_pre)

        alpha = self._alpha(mean, log_pre, prior_mean, prior_log_pre)
        gated_log_pre = torch.log(alpha.clamp_min(self.alpha_eps)) + log_pre
        return sequential_bayes(mean, gated_log_pre, prior_mean, prior_log_pre)

    def _alpha(self, mean, log_pre, prior_mean, prior_log_pre):
        prior_mean = prior_mean.expand_as(mean)
        prior_log_pre = prior_log_pre.expand_as(log_pre)

        precision = log_pre.exp()
        prior_precision = prior_log_pre.exp()
        mean_delta = mean - prior_mean
        precision_delta = precision - prior_precision
        log_precision_delta = log_pre - prior_log_pre
        mahalanobis = mean_delta * (0.5 * prior_log_pre).exp()

        features = torch.cat([
            mean,
            prior_mean,
            mean_delta,
            precision,
            prior_precision,
            precision_delta,
            log_precision_delta,
            mahalanobis,
        ], dim=-1)

        alpha = torch.sigmoid(self.alpha_net(features))
        return alpha