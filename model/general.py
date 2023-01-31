import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, shape: tuple, act, act_out, gain_out=1.0):
        super(MLP, self).__init__()
        s = []
        for i, j in zip(shape[:-1], shape[1:]):
            s.append(nn.Linear(i, j))
            s.append(act())
            nn.init.orthogonal_(s[-2].weight)
            nn.init.constant_(s[-2].bias, 0)
        nn.init.orthogonal_(s[-2].weight, gain=gain_out)
        s[-1] = act_out()
        self.seq = nn.Sequential(*s)

    def forward(self, feature):
        return self.seq(feature)


class Attention(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        将形如 [batch_size, obstacle_num, feature_num] 的装量加权得到 [batch_size, feature_num] 的张量, 权值根据 feature_num 维
        度决定. 
        【已弃用】若某个 (batch_size, obstacle_num) 对应的 feature 含有 NaN, 则对应权值为 0
        【代替】 若 mask (形如 [batch_size, obstacle_num, 1]) 给出 False, 则对应权值为 0
        """
        super(Attention, self).__init__()
        self.seq = MLP(*args, **kwargs)

    def forward(self, s, mask):
        shape = s.shape
        log_weight = self.seq(s)  # (N, 20, 128) -> (N, 20, 1)
        log_weight_ = log_weight.masked_fill(~mask, -torch.inf)
        weight = torch.softmax(log_weight_, dim=-2)  # N, 20, 1
        # 若 mask 中全是 False (即一个东西也没有) , 则 softmax 将得到一组 nan, 这里将 nan 全部设置为 0.
        weight_ = weight.masked_fill(~mask, 0)  # N, 20, 1
        s_ = torch.einsum('Nof,No->Nf', s, weight_.squeeze(-1))
        return s_

    def get_weight(self, s):
        return self.seq(s)
