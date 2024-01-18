import torch
import torch.nn as nn
import math


def make_gaussian(y_idx, x_idx, height, width, sigma=7.0):
    yv, xv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])

    yv = yv.reshape(height*width).unsqueeze(0).float().cuda()
    xv = xv.reshape(height*width).unsqueeze(0).float().cuda()

    y_idx = y_idx.transpose(0, 1).float()
    x_idx = x_idx.transpose(0, 1).float()

    g = torch.exp(- ((yv-y_idx)**2 + (xv-x_idx)**2) / (2*sigma**2) )

    return g


def softmax_w_g_top(x, top=None, gauss=None):
    if top is not None:
        if gauss is not None:
            maxes = torch.max(x, dim=1, keepdim=True)[0]
            x_exp = torch.exp(x - maxes)*gauss
            x_exp, indices = torch.topk(x_exp, k=top, dim=1)
        else:
            values, indices = torch.topk(x, k=top, dim=1)
            x_exp = torch.exp(values - values[:,0])

        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        x_exp /= x_exp_sum
        x.zero_().scatter_(1, indices, x_exp) # B * THW * HW

        output = x
    else:
        maxes = torch.max(x, dim=1, keepdim=True)[0]
        if gauss is not None:
            x_exp = torch.exp(x-maxes)*gauss

        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        x_exp /= x_exp_sum
        output = x_exp

    return output


class SpaceTimeMemoryReader(nn.Module):
    def __init__(self):
        super(SpaceTimeMemoryReader, self).__init__()
        self.top_k = 50
        self.km = 5.6

    def forward(self, m_key, m_value, q_key, q_value):
        B, C_key, H, W = m_key.size()
        _, C_val, _, _ = m_value.size()

        m_key = m_key.view(B, C_key, H * W)  # B, C_key, HW
        m_key = torch.transpose(m_key, 1, 2)  # B, HW, C_key
        q_key = q_key.view(B, C_key, H * W)  # B, C_key, HW

        w = torch.bmm(m_key, q_key) / math.sqrt(C_key)  # B, HW, HW
        argmax_idx = w.max(2)[1]
        y_idx, x_idx = argmax_idx // W, argmax_idx % W
        g = make_gaussian(y_idx, x_idx, H, W, sigma=self.km)
        g = g.view(B, H * W, H * W)

        w = softmax_w_g_top(w, top=self.top_k, gauss=g)  # B, THW, HW

        m_value = m_value.view(B, C_val, H * W)  # B, C_val, HW
        mem_info = torch.bmm(m_value, w)  # (B, C_val, HW) x (B, HW, HW) = (B, C_val, HW)
        mem_info = mem_info.view(B, C_val, H, W)

        y = torch.cat([mem_info, q_value], dim=1)
        return y
