import torch
from torchvision.ops.boxes import box_area
import numpy as np
from torch.nn import functional as F
from torch import nn

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xywh_to_xyxy(x):
    x1, y1, w, h = x.unbind(-1)
    b = [x1, y1, x1 + w, y1 + h]
    return torch.stack(b, dim=-1)


def box_xyxy_to_xywh(x):
    x1, y1, x2, y2 = x.unbind(-1)
    b = [x1, y1, x2 - x1, y2 - y1]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
'''Note that this function only supports shape (N,4)'''
def bbox_overlaps(bboxes1, bboxes2, eps=1e-6):
    # bbox1 [b,m,4] bbox2 [b,n,4]
    # iou [b,m,n]
    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    lt = torch.max(bboxes1[..., :, None, :2],
                   bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
    rb = torch.min(bboxes1[..., :, None, 2:],
                   bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

    wh = (rb - lt).clamp(min = 0)
    overlap = wh[..., 0] * wh[..., 1]

    union = area1[..., None] + area2[..., None, :] - overlap

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    return ious #[bs,m,n]





def box_iou(boxes1, boxes2):
    """

    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
    area1 = box_area(boxes1) # (N,)
    area2 = box_area(boxes2) # (N,)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # (N,2)
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # (N,2)

    wh = (rb - lt).clamp(min=0)  # (N,2)
    inter = wh[:, 0] * wh[:, 1]  # (N,)

    union = area1 + area2 - inter + 1e-5
    iou = inter / union
    return iou, union

def compute_iou(rec1, rec2):
    """
    computing IoU
    rec1: (x0, y0, w, h)
    rec2: (x0, y0, w, h)
    :return: scala value of IoU
    """
    rec1[2] = rec1[0] + rec1[2]
    rec1[3] = rec1[1] + rec1[3]
    rec2[2] = rec2[0] + rec2[2]
    rec2[3] = rec2[1] + rec2[3]
    # computing area of each rectangle
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect area
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


'''Note that this implementation is different from DETR's'''


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    boxes1: (N, 4)
    boxes2: (N, 4)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # try:
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2) # (N,)

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # (N,2)
    area = wh[:, 0] * wh[:, 1] # (N,)

    return iou - (area - union) / area, iou


def giou_loss(boxes1, boxes2):
    """

    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
    giou, iou = generalized_box_iou(boxes1, boxes2)
    return (1 - giou).mean(), iou

def clip_box(box: list, H, W, margin=0):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W-margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H-margin)
    y2 = min(max(margin, y2), H)
    w = max(margin, x2-x1)
    h = max(margin, y2-y1)
    return [x1, y1, w, h]


class REGLoss(nn.Module):
    def __init__(self, dim=4, loss_type='iou'):
        super(REGLoss, self).__init__()
        self.dim = dim
        if loss_type == 'iou':
            self.loss = IOULoss()
        else:
            raise ValueError("Only support iou loss.")

    def forward(self, output, ind, target, radius=1, norm=1/20.):
        width, height = output.size(-2), output.size(-1)
        output = output.view(-1, self.dim, width, height)
        # mask =  mask.view(-1, 2)
        target = target.view(-1, self.dim)
        ind = ind.view(-1, 1)
        center_w = (ind % width).int().float()
        center_h = (ind / width).int().float()

        # regress for the coordinates in the vicinity of the target center, the default radius is 1.
        if radius is not None:
            loss = []
            for r_w in range(-1 * radius, radius + 1):
                for r_h in range(-1 * radius, radius + 1):
                    target_wl = target[:, 0] + r_w * norm
                    target_wr = target[:, 1] - r_w * norm
                    target_ht = target[:, 2] + r_h * norm
                    target_hb = target[:, 3] - r_h * norm
                    if (target_wl < 0.).any() or (target_wr < 0.).any() or (target_ht < 0.).any() or (target_hb < 0.).any():
                        continue
                    if (center_h + r_h < 0.).any() or (center_h + r_h >= 1.0 * width).any() \
                            or (center_w + r_w < 0.).any() or (center_w + r_w >= 1.0 * width).any():
                        continue

                    target_curr = torch.stack((target_wl, target_wr, target_ht, target_hb), dim=1)  # [num_images * num_sequences, 4]
                    ind_curr = ((center_h + r_h) * width + (center_w + r_w)).long()
                    pred_curr = _tranpose_and_gather_feat(output, ind_curr)
                    loss_curr = self.loss(pred_curr, target_curr)
                    loss.append(loss_curr)
            if len(loss) == 0:
                pred = _tranpose_and_gather_feat(output, ind.long())  # pred shape: [num_images * num_sequences, 4]
                loss = self.loss(pred, target)
                return loss
            loss = torch.stack(loss, dim=0)
            loss = torch.mean(loss, dim=0)
            return loss
        pred = _tranpose_and_gather_feat(output, ind.long())     # pred shape: [num_images * num_sequences, 4]
        loss = self.loss(pred, target)

        return loss

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(dim=1, index=ind)   # [num_images * num_sequences, 1, 2]
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat.view(ind.size(0), dim)


class LBHinge(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """
    def __init__(self, error_metric=nn.MSELoss(), threshold=None, clip=None):
        super().__init__()
        self.error_metric = error_metric
        self.threshold = threshold if threshold is not None else -100
        self.clip = clip

    def forward(self, prediction, label, target_bb=None):
        # print("pred shape: {}, label shape: {}".format(prediction.shape, label.shape))
        negative_mask = (label < self.threshold).float()
        positive_mask = (1.0 - negative_mask)

        prediction = negative_mask * F.relu(prediction) + positive_mask * prediction

        loss = self.error_metric(prediction, positive_mask * label)

        if self.clip is not None:
            loss = torch.min(loss, torch.tensor([self.clip], device=loss.device))
        return loss

class SafeLog(nn.Module):
    r"""
    Safly perform log operation
    """

    def __init__(self):
        super(SafeLog, self).__init__()
        eps = np.finfo(np.float32).tiny
        self.register_buffer("t_eps", torch.tensor(eps, requires_grad=False))

    def forward(self, t):
        return torch.log(torch.max(self.t_eps.to(t.device), t))

class IOULoss(nn.Module):

    # default_hyper_params = dict(
    #     name="iou_loss",
    #     background=0,
    #     ignore_label=-1,
    #     weight=1.0,
    # )

    def __init__(self,
                 background=0,
                 ignore_label=-1,
                 weight=1.0,
                 branch_name=""):
        super().__init__()
        self.safelog = SafeLog()
        self.background = background
        self.ignore_label = ignore_label
        self.weight = weight
        self.iou_key = "iou_{}".format(branch_name)
        self.register_buffer("t_one", torch.tensor(1., requires_grad=False))
        self.register_buffer("t_zero", torch.tensor(0., requires_grad=False))

    def forward(self, pred, gt, cls_gt):
        # pred = pred_data[self.name]
        # gt = target_data["box_gt"]
        # cls_gt = target_data["cls_gt"]
        # print(pred.shape,gt.shape,cls_gt.shape)
        mask = ((~(cls_gt == self.background)) *
                (~(cls_gt == self.ignore_label))).detach()
        # print(mask.shape)
        # mask = mask.type(torch.Tensor).squeeze(2).to(pred.device)
        mask = mask.to(pred.device)

        aog = torch.abs(gt[:, :, 2] - gt[:, :, 0] +
                        1) * torch.abs(gt[:, :, 3] - gt[:, :, 1] + 1)
        aop = torch.abs(pred[:, :, 2] - pred[:, :, 0] +
                        1) * torch.abs(pred[:, :, 3] - pred[:, :, 1] + 1)

        iw = torch.min(pred[:, :, 2], gt[:, :, 2]) - torch.max(
            pred[:, :, 0], gt[:, :, 0]) + 1
        ih = torch.min(pred[:, :, 3], gt[:, :, 3]) - torch.max(
            pred[:, :, 1], gt[:, :, 1]) + 1
        inter = torch.max(iw, self.t_zero.to(pred.device)) * torch.max(ih, self.t_zero.to(pred.device))

        union = aog + aop - inter
        iou = torch.max(inter / union, self.t_zero.to(pred.device))
        loss = -self.safelog(iou)

        loss = (loss * mask).sum() / torch.max(mask.sum(), self.t_one.to(pred.device)) * self.weight
        iou = iou.detach()
        iou = (iou * mask).sum() / torch.max(mask.sum(), self.t_one.to(pred.device))

        return loss, iou


class FocalLoss(nn.Module):
    def __init__(self,
                 background=0,
                 ignore_label=-1,
                 alpha=0.75,
                 gamma=2.0,
                 weight=1.0):
        super().__init__()
        self.background = background
        self.ignore_label = ignore_label
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.register_buffer("t_one", torch.tensor(1., requires_grad=False))

    def forward(self, pred, label):
        r"""
        Focal loss
        :param pred: shape=(B, HW, C), classification logits (BEFORE Sigmoid)
        :param label: shape=(B, HW)
        """
        r"""
        Focal loss
        Arguments
        ---------
        pred: torch.Tensor
            classification logits (BEFORE Sigmoid)
            format: (B, HW)
        label: torch.Tensor
            training label
            format: (B, HW)

        Returns
        -------
        torch.Tensor
            scalar loss
            format: (,)
        """
        # pred = pred_data[self.name]
        # label = target_data["cls_gt"]
        mask = ~(label == self.ignore_label)
        mask = mask.type(torch.Tensor).to(label.device)
        vlabel = label * mask

        zero_mat = torch.zeros(pred.shape[0], pred.shape[1], pred.shape[2] + 1)
        one_mat = torch.ones(pred.shape[0], pred.shape[1], pred.shape[2] + 1)

        index_mat = vlabel.type(torch.LongTensor)
        onehot_ = zero_mat.scatter(2, index_mat, one_mat)
        onehot = onehot_[:, :, 1:].type(torch.Tensor).to(pred.device)
        loss = sigmoid_focal_loss_jit(pred, onehot, self.alpha, self.gamma, "none")
        positive_mask = (label > 0).type(torch.Tensor).to(pred.device)
        loss = (loss.sum(dim=2) * mask.squeeze(2)).sum() / (torch.max(
            positive_mask.sum(), self.t_one.to(pred.device))) * self.weight
        return loss



def sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = -1,
        gamma: float = 2,
        reduction: str = "none"):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs,
                                                 targets,
                                                 reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

sigmoid_focal_loss_jit = torch.jit.script(
    sigmoid_focal_loss)  # type: torch.jit.ScriptModule


# def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
#     """
#     Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
#     Args:
#         inputs: A float tensor of arbitrary shape.
#                 The predictions for each example.
#         targets: A float tensor with the same shape as inputs. Stores the binary
#                  classification label for each element in inputs
#                 (0 for the negative class and 1 for the positive class).
#         alpha: (optional) Weighting factor in range (0,1) to balance
#                 positive vs negative examples. Default = -1 (no weighting).
#         gamma: Exponent of the modulating factor (1 - p_t) to
#                balance easy vs hard examples.
#     Returns:
#         Loss tensor
#     """
#     prob = inputs.sigmoid()
#     ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
#     p_t = prob * targets + (1 - prob) * (1 - targets)
#     loss = ce_loss * ((1 - p_t) ** gamma)

#     if alpha >= 0:
#         alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
#         loss = alpha_t * loss

#     return loss.mean(1).sum() / num_boxes


