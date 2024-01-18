import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F




class SegmFromTracker(nn.Module):
    """ This is the TransT module that performs single object tracking """
    def __init__(self,settings):
        super().__init__()
        self.nheads = settings.nheads
        self.hidden_dim = settings.hidden_dim
        self.in_channels = settings.in_channels
        # self.template_feat_size = settings.template_feat_size
        # hanning = np.hanning(self.template_feat_size)
        # window = np.outer(hanning, hanning)
        # self.window = window.reshape(-1)


        self.input_proj = nn.Conv2d(self.in_channels[-1], self.hidden_dim, kernel_size=1)
        self.bbox_attention_self = MHAttentionMap(self.hidden_dim, self.hidden_dim, self.nheads, dropout=0)
        self.bbox_attention_cross = MHAttentionMap(self.hidden_dim, self.hidden_dim, self.nheads, dropout=0)
        self.mask_head = MaskHeadSmallConv(self.hidden_dim + 2*self.nheads, self.in_channels[::-1], self.hidden_dim)


    def forward(self, template, search_list, outputs_class):

        template = self.input_proj(template)
        search = self.input_proj(search_list[-1])
        bs,c,h,w = search.shape

        template_vec = template.permute(0,3,1,2).reshape(bs,-1,c)
        search_vec = search.permute(0,3,1,2).reshape(bs,-1,c)

        # print(template_vec.shape,search_vec.shape)

        score = self._convert_score(outputs_class) #
        index_1 = score.argmax(axis=1)
        index_0 = np.arange(0, bs, 1)
        hs_search_choosen = search_vec[index_0, index_1, :].unsqueeze(1)

        hs_temp_choosen = template_vec[:,27:28,:]

        #这里如果把hs[-1]共1024个向量全输入进取占显存太多，暂时取hs_temp中心的一个，并不是非常合理，后边修改
        bbox_mask_self = self.bbox_attention_self(hs_search_choosen, search) #torch.Size([bs, n_q, nhead, h, w]) attention score 8-head 100-query
        bbox_mask_cross = self.bbox_attention_cross(hs_temp_choosen, search) #torch.Size([bs, n_q, nhead, h, w]) attention score 8-head 100-query
        bbox_mask = torch.cat([bbox_mask_self, bbox_mask_cross], dim=2)

        #fpn的选择不一定合理
        seg_masks = self.mask_head(search, bbox_mask, [search_list[2], search_list[1], search_list[0]])
        num_queries_seg = 1
        outputs_seg_masks = seg_masks.view(bs, num_queries_seg, seg_masks.shape[-2], seg_masks.shape[-1])
        #
        return outputs_seg_masks
    def _convert_score(self, score):
        score = score.sigmoid().cpu().numpy()
        return score
        #score.shape (16, 1024)


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 32] # [264, 128, 64, 32, 16, 8]
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1) # 264->264
        self.gn1 = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)  # 264->128
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)   # 128->64
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)   # 64->32
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)   # 32->16
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        #self.lay6 = torch.nn.Conv2d(inter_dims[4], inter_dims[5], 3, padding=1)   # 16->8
        #self.gn6 = torch.nn.GroupNorm(8, inter_dims[5])
        self.out_lay = torch.nn.Conv2d(inter_dims[4], 1, 3, padding=1)   # 8->1

        self.dim = dim #264

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)   # 1024->128
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)   # 512->64
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)   # 256->32
        #self.adapter4 = torch.nn.Conv2d(fpn_dims[3], inter_dims[4], 1)   # 64->16

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, bbox_mask, fpns):
        # x torch.Size([16, 256, 32, 32])
        # bbox_mask torch.Size([16, 1024, 8, 32, 32])
        # fpns[0] torch.Size([16, 1024, 32, 32])
        # fpns[1] torch.Size([16, 512, 32, 32])
        # fpns[2] torch.Size([16, 256, 64, 64])

        def expand(tensor, length):
            return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)

        x = torch.cat([expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)
        # 把attention score和主干网络的特征cat到一起
        # expand(x, bbox_mask.shape[1]).shape
        # torch.Size([bs*1024, 256, 32, 232])
        # bbox_mask.flatten(0, 1).shape
        # torch.Size([bs*1024, 8, 32, 32])
        # x.shape
        # torch.Size([bs*1024, 264, 32, 32])

        x = self.lay1(x) #torch.Size([bs*1024, 264, 32, 32])
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x) #torch.Size([bs*1024, 128, 32, 32])
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter1(fpns[0]) #torch.Size([bs, 1024, 32, 32])-> torch.Size([bs, 128, 32, 32])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0)) #torch.Size([bs*1024, 128, 32, 32])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest") #与浅层的特征相加
        x = self.lay3(x)   # 128->64
        x = self.gn3(x)
        x = F.relu(x) #torch.Size([bs*1024, 64, 32, 32])

        cur_fpn = self.adapter2(fpns[1]) #torch.Size([bs, 512, 32, 32]) -> torch.Size([bs, 64, 32, 32])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0)) #torch.Size([bs*1024, 64, 32, 32])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)   # 64->32
        x = self.gn4(x)
        x = F.relu(x) #torch.Size([bs*1024, 32, 32, 32])

        cur_fpn = self.adapter3(fpns[2]) #torch.Size([16, 256, 64, 64]) -> torch.Size([16, 32, 64, 64])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0)) #torch.Size([bs*1024, 32, 64, 64])
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest") #torch.Size([bs*1024, 32, 64, 64])
        x = self.lay5(x)   # 32->16
        x = self.gn5(x)
        x = F.relu(x)

        #cur_fpn = self.adapter4(fpns[3]) #torch.Size([16, 64, 128, 128]) -> torch.Size([16, 16, 128, 128])
        #if cur_fpn.size(0) != x.size(0):
        #    cur_fpn = expand(cur_fpn, x.size(0) / cur_fpn.size(0)) #torch.Size([bs*1024, 16, 128, 128])
        #x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest") #torch.Size([bs*1024, 16, 128, 128])
        #x = self.lay6(x)   # 16->8
        #x = self.gn6(x)
        #x = F.relu(x)

        x = self.out_lay(x)   # 8->1 #torch.Size([bs*1024, 1, 128, 128])
        return x


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)
        weights = self.dropout(weights)
        return weights





