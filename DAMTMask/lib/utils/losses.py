import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils import box_ops
import numpy as np

def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes

class ReppointCriterion(nn.Module):
    def __init__(self,pos_iou = 0.5, neg_iou = 0.4,
                    object_cls = None,object_reg = None):
        super(ReppointCriterion, self).__init__()
        self.pos_iou = pos_iou
        self.neg_iou = neg_iou

        self.object_cls = object_cls
        self.object_reg = object_reg

    @torch.no_grad()
    def get_refine_label(self,pred_refine_bboxes,gt_bboxes):
        # pred_refine_bbox [b,m,4]
        # gt_bbox [b,n,4]
        # ious [b,m,n]
        bs,num_queries,_ = pred_refine_bboxes.shape
        bs,num_gt,_ = gt_bboxes.shape

        assert num_gt == 1
        cls_labels = []
        cls_weight = []
        for i in range(bs):
            labels = torch.ones((num_queries),dtype=torch.int64).reshape(-1) * -1
            weight = torch.zeros(num_queries).reshape(-1).float()

            pred_refine_bbox = pred_refine_bboxes[i]
            gt_bbox = gt_bboxes[i]
            ious= box_ops.box_iou(pred_refine_bbox, gt_bbox)[0].reshape(-1)

            pos_idx = ious > self.pos_iou
            neg_idx = ious < self.neg_iou

            labels[pos_idx] = 1
            labels[neg_idx] = 0

            weight[pos_idx] = 1.0

            # print(weight.reshape(20,20).int())

            cls_labels.append(labels)
            cls_weight.append(weight)

        cls_labels = torch.stack(cls_labels,0).to(pred_refine_bboxes.device)
        cls_weight = torch.stack(cls_weight,0).to(pred_refine_bboxes.device)

        return cls_labels, cls_weight

    def loss_labels(self,pred,label):
        # labels [bs, h*w]
        # pred [bs, h*w]
        loss = self.object_cls(pred, label)
        losses = {'reppoint_cls': loss}
        return losses

    def loss_init_boxes(self,pred_bbox,gt_bbox,weight):
        # pred_bbox [bs, h*w, 4]
        # gt_bbox [bs, h*w, 4]
        # weight [bs, h*w]
        loss, iou = self.object_reg(pred_bbox, gt_bbox, weight)

        # iou = box_ops.box_iou(pred_bbox.reshape(-1,4),gt_bbox.reshape(-1,4)) * weight.reshape(-1)
        # if weight.sum() > 0:
        #     iou = iou.sum()/weight.sum()
        # else:
        #     iou = iou.sum()

        losses = {'reppoint_init_bbox': loss, 'reppoint_init_iou': iou}
        return losses

    def loss_refine_boxes(self,pred_bbox,gt_bbox,weight):
        # pred_bbox [bs, h*w]
        # gt_bbox [bs, h*w]
        loss,iou = self.object_reg(pred_bbox, gt_bbox, weight)

        # iou = box_ops.box_iou(pred_bbox.reshape(-1,4),gt_bbox.reshape(-1,4)) * weight.reshape(-1)
        # if weight.sum() > 0:
        #     iou = iou.sum()/weight.sum()
        # else:
        #     iou = iou.sum()

        losses = {'reppoint_refine_bbox': loss, 'reppoint_refine_iou': iou}
        return losses

    def forward(self,outputs, data):

        pred_refine_bboxes = outputs['reppoint_refine_bboxes'] # [num_image*bs_raw,h*w,4]
        pred_init_bboxes = outputs['reppoint_init_bboxes'] # [num_image*bs_raw,h*w,4]
        pred_cls = outputs['reppoint_cls'] # [num_image*bs_raw,h*w]

        bs_cat, num_queries, _ = pred_refine_bboxes.shape# [num_image*bs_raw,h*w, 4]
        num_image, bs_raw, _ = data['search_anno_crop'].shape# [num_image,bs_raw, 4]
        assert bs_cat == bs_raw * num_image

        gt_bboxes = data['search_anno_crop'].clone().reshape(num_image*bs_raw,1,4)
        gt_bboxes = box_ops.box_xywh_to_xyxy(gt_bboxes) # [num_image*bs_raw,1,4]
        cls_labels, reg_weight = self.get_refine_label(pred_init_bboxes,gt_bboxes)
        # [num_image*bs_raw,h*w] # [num_image*bs_raw,h*w]

        reppoint_init_weight = data['reppoint_init_weight'].clone().reshape(num_image*bs_raw,-1) # [bs*num_image,h*w]
        reppoint_refine_weight = reg_weight # [num_image*bs_raw, h*w]

        weight_state = {'reppoint_init_weight':reppoint_init_weight.clone().detach(),'reppoint_refine_weight':reppoint_refine_weight.clone().detach()}

        gt_bboxes = gt_bboxes.repeat((1, num_queries, 1))
        losses = {}
        loss_cls = self.loss_labels(pred_cls.unsqueeze(-1),cls_labels.unsqueeze(-1))
        loss_init_box = self.loss_init_boxes(pred_init_bboxes,gt_bboxes,reppoint_init_weight)
        loss_refine_box = self.loss_refine_boxes(pred_refine_bboxes,gt_bboxes,reppoint_refine_weight)
        losses.update(loss_cls)
        losses.update(loss_init_box)
        losses.update(loss_refine_box)

        return losses,weight_state




class ClsRegCriterion(nn.Module):
    def __init__(self, eos_coef=1.0):
        super(ClsRegCriterion, self).__init__()
        empty_weight = torch.tensor([1.0, eos_coef], dtype=torch.float)
        self.register_buffer("empty_weight", empty_weight)

    @torch.no_grad()
    def matcher(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Always tensor([0]) represents the foreground,
                           since single target tracking has only one foreground category
                 "boxes": Tensor of dim [1, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order),
                  and it is always 0, because single target tracking has only one target per image
            For each batch element, it holds:
                len(index_i) = len(index_j)
        """
        indices = []
        bs, num_queries = outputs["pred_cls"].shape[:2]
        for i in range(bs):
            cx, cy, w, h = targets[i]['label_reg'][0]
            cx = cx.item()
            cy = cy.item()
            w = w.item()
            h = h.item()
            xmin = cx - w / 2
            ymin = cy - h / 2
            xmax = cx + w / 2
            ymax = cy + h / 2
            len_feature = int(np.sqrt(num_queries))
            Xmin = int(np.ceil(xmin * len_feature))
            Ymin = int(np.ceil(ymin * len_feature))
            Xmax = int(np.floor(xmax * len_feature))
            Ymax = int(np.floor(ymax * len_feature))
            if Xmin == Xmax:
                Xmax = Xmax + 1
            if Ymin == Ymax:
                Ymax = Ymax + 1
            a = np.arange(0, num_queries, 1)
            b = a.reshape([len_feature, len_feature])
            c = b[Ymin:Ymax, Xmin:Xmax].flatten()
            d = np.zeros(len(c), dtype=int)
            indice = (c, d)
            indices.append(indice)
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @torch.no_grad()
    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        if target.numel() == 0:
            return [torch.zeros([], device=output.device)]
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_cls' in outputs
        src_logits = outputs['pred_cls']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["label_cls"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 1, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_cls': loss_ce}

        # TODO this should probably be a separate loss, not hacked in this one here
        losses['class_error'] = 100 - self.accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_reg' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_reg'][idx]
        target_boxes = torch.cat([t['label_reg'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_reg_l1'] = loss_bbox.sum() / num_boxes
        giou, iou = box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes))
        # giou = torch.diag(giou)
        # iou = torch.diag(iou)
        loss_giou = 1 - giou
        iou = iou
        losses['loss_reg_giou'] = loss_giou.sum() / num_boxes
        losses['iou'] = iou.sum() / num_boxes
        return losses

    def forward(self, outputs, data):
        targets = []
        # serach_anno has already been normalized to [0, 1].
        # print(data['search_anno'].shape)
        targets_origin = data['search_anno']
        num_image,bs,_ = targets_origin.shape
        targets_origin = targets_origin.reshape(-1,4)


        targets_origin[:, 0] += targets_origin[:, 2] / 2
        targets_origin[:, 1] += targets_origin[:, 3] / 2
        targets_origin = targets_origin.unsqueeze(1)
        for i in range(len(targets_origin)):
            target_origin = targets_origin[i]
            target = {}
            # print('target_origin = {}'.format(target_origin.shape))
            target['label_reg'] = target_origin
            label = np.array([0])
            label = torch.tensor(label, device=data['search_anno'].device)
            target['label_cls'] = label
            targets.append(target)

        # Retrieve the matching between the outputs of the last layer and the target
        indices = self.matcher(outputs, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes_pos = sum(len(t[0]) for t in indices)

        num_boxes_pos = torch.as_tensor([num_boxes_pos], dtype=torch.float,
                                        device=next(iter(outputs.values())).device)

        num_boxes_pos = torch.clamp(num_boxes_pos, min=1).item()

        loss_cls = self.loss_labels(outputs, targets, indices, num_boxes_pos)
        loss_reg = self.loss_boxes(outputs, targets, indices, num_boxes_pos)
        losses = {}
        losses.update(loss_cls)
        losses.update(loss_reg)
        return losses
