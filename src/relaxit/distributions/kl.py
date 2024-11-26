from torch.distributions import kl_divergence, register_kl, Normal

from .InvertibleGaussian import InvertibleGaussian


@register_kl(InvertibleGaussian, InvertibleGaussian)
def _kl_igr_igr(p, q):
    p_normal = Normal(p.loc, p.scale)
    q_normal = Normal(q.loc, q.scale)
    return kl_divergence(p_normal, q_normal)
