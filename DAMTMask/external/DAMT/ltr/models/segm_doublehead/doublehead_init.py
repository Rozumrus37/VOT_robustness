import torch.nn as nn
from ltr import model_constructor

import torch
import torch.nn.functional as F
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor,
                       nested_tensor_from_tensor_2, interpolate,
                       nested_tensor_from_tensor_list, accuracy)

from ltr.models.backbone.double_backbone import build_backbone
from ltr.models.neck.featurefusion_cnn_network import build_featurefusion_seg_cnn_network
from ltr.models.neck.featurefusion_seg_network import build_featurefusion_seg_network
from ltr.models.segm_doublehead.doublehead_seg import ResSegm
from ltr.models.segm_doublehead.doublehead_seg_transtres import TransTFusionSegm




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


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
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
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes



class ResSplit(nn.Module):
    """ This is the TransT module that performs single object tracking """
    def __init__(self, backbone_template, backbone_search, featurefusion_network_cnn, num_classes):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See transt_backbone.py
            featurefusion_network: torch module of the featurefusion_network architecture, a variant of transformer.
                                   See featurefusion_network.py
            num_classes: number of object classes, always 1 for single object tracking
        """
        super().__init__()
        self.featurefusion_network_cnn = featurefusion_network_cnn
        self.backbone_template = backbone_template
        self.backbone_search = backbone_search


class TranstResSplit(nn.Module):
    """ This is the TransT module that performs single object tracking """
    def __init__(self, model, featurefusion_network_transformer, num_classes):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See transt_backbone.py
            featurefusion_network: torch module of the featurefusion_network architecture, a variant of transformer.
                                   See featurefusion_network.py
            num_classes: number of object classes, always 1 for single object tracking
        """
        super().__init__()
        self.backbone_template = model.backbone_template
        self.backbone_search = model.backbone_search
        self.featurefusion_network_cnn = model.featurefusion_network_cnn
        self.cnn_mask = model.cnn_mask

        hidden_dim = featurefusion_network_transformer.d_model
        self.input_proj_transformer = nn.Conv2d(self.backbone_search.num_channels, hidden_dim, kernel_size=1)
        self.featurefusion_network_transformer = featurefusion_network_transformer


class TransTFusion(nn.Module):
    """ This is the TransT module that performs single object tracking """
    def __init__(self, backbone, featurefusion_network, num_classes):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See transt_backbone.py
            featurefusion_network: torch module of the featurefusion_network architecture, a variant of transformer.
                                   See featurefusion_network.py
            num_classes: number of object classes, always 1 for single object tracking
        """
        super().__init__()
        self.featurefusion_network = featurefusion_network
        hidden_dim = featurefusion_network.d_model
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        # self.stage_channels = backbone.stage_channels

class TransTFusionDoubleHead(nn.Module):
    """ This is the TransT module that performs single object tracking """
    def __init__(self, backbone, featurefusion_network_for_transformer,featurefusion_network_for_cnn, num_classes):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See transt_backbone.py
            featurefusion_network: torch module of the featurefusion_network architecture, a variant of transformer.
                                   See featurefusion_network.py
            num_classes: number of object classes, always 1 for single object tracking
        """
        super().__init__()
        self.featurefusion_network_for_transformer = featurefusion_network_for_transformer
        hidden_dim = featurefusion_network_for_transformer.d_model
        self.input_proj_for_transformer = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        self.featurefusion_network_for_cnn = featurefusion_network_for_cnn
        hidden_dim = featurefusion_network_for_cnn.d_model
        self.input_proj_for_cnn = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        self.backbone = backbone
        # self.stage_channels = backbone.stage_channels




class SetCriterion(nn.Module):
    """ This class computes the loss for TransT.
    The process happens in two steps:
        1) we compute assignment between ground truth box and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, always be 1 for single object tracking.
            matcher: module able to compute a matching between target and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.iouhead_loss = nn.MSELoss()

    def loss_masks_double_head(self, outputs, targets):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks_for_transformer" in outputs
        assert "pred_masks_for_cnn" in outputs
        src_masks_transformer = outputs["pred_masks_for_transformer"] # torch.Size([bs, 1, 128, 128])
        src_masks_cnn = outputs["pred_masks_for_cnn"] # torch.Size([bs, 1, 128, 128])

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose() #torch.Size([32, 1, 256, 256])
        target_masks = target_masks.to(src_masks_transformer) #torch.Size([bs, 1, 256, 256])



        # upsample predictions to the target size
        src_masks_transformer = interpolate(src_masks_transformer, size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks_transformer = src_masks_transformer[:, 0].flatten(1) #torch.Size([18, 660969])
        # upsample predictions to the target size
        src_masks_cnn = interpolate(src_masks_cnn, size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks_cnn = src_masks_cnn[:, 0].flatten(1) #torch.Size([18, 660969])



        target_masks = target_masks[:, 0].flatten(1) #torch.Size([18, 660969])
        losses = {
            "loss_mask_transformer": sigmoid_focal_loss(src_masks_transformer, target_masks.clone(), src_masks_transformer.shape[0]),
            "loss_dice_transformer": dice_loss(src_masks_transformer, target_masks.clone(), src_masks_transformer.shape[0]),
            "loss_bce_transformer": F.binary_cross_entropy(src_masks_transformer.sigmoid(), target_masks.clone(), reduction="mean"),

            "loss_mask_cnn": sigmoid_focal_loss(src_masks_cnn, target_masks.clone(), src_masks_cnn.shape[0]),
            "loss_dice_cnn": dice_loss(src_masks_cnn, target_masks.clone(), src_masks_cnn.shape[0]),
            "loss_bce_cnn": F.binary_cross_entropy(src_masks_cnn.sigmoid(), target_masks.clone(), reduction="mean")
        }
        return losses

    def loss_masks(self, outputs, targets):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_masks = outputs["pred_masks"] # torch.Size([bs, 1, 128, 128])

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose() #torch.Size([32, 1, 256, 256])
        target_masks = target_masks.to(src_masks) #torch.Size([bs, 1, 256, 256])

        # upsample predictions to the target size
        src_masks = interpolate(src_masks, size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        #torch.Size([bs, 1, 256, 256])

        src_masks = src_masks[:, 0].flatten(1) #torch.Size([18, 660969])

        target_masks = target_masks[:, 0].flatten(1) #torch.Size([18, 660969])

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, src_masks.shape[0]),
            "loss_dice": dice_loss(src_masks, target_masks, src_masks.shape[0]),
            "loss_bce": F.binary_cross_entropy(src_masks.sigmoid(), target_masks, reduction="mean")
        }
        return losses


    def get_loss(self, loss, outputs, targets):
        loss_map = {
            'masks': self.loss_masks,
            'masks_double_head': self.loss_masks_double_head,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))

        return losses

# @model_constructor
# def transt_resnet50_neckfusion_double_head(settings):
#     num_classes = 1
#     backbone_net = build_backbone(settings, backbone_pretrained=True)
#     featurefusion_network_for_transformer = build_featurefusion_seg_network(settings)
#     featurefusion_network_for_cnn = build_featurefusion_cnn_network(settings)
#     model = TransTFusionDoubleHead(
#         backbone_net,
#         featurefusion_network_for_transformer = featurefusion_network_for_transformer,
#         featurefusion_network_for_cnn = featurefusion_network_for_cnn,
#         num_classes=num_classes
#     )
#     model = TransTFusionSegmDoubleHead(model, freeze_transt=settings.freeze_transt)
#     device = torch.device(settings.device)
#     model.to(device)
#     return model

# @model_constructor
# def transt_resnet50_neckfusiondeep(settings):
#     num_classes = 1
#     backbone_net = build_backbone(settings, backbone_pretrained=True, use_dilation_for_resolution = False)
#     featurefusion_network = build_featurefusion_seg_network(settings)
#     model = TransTFusion(
#         backbone_net,
#         featurefusion_network,
#         num_classes=num_classes
#     )
#     model = TransTFusionSegm(model, freeze_transt=settings.freeze_transt)
#     device = torch.device(settings.device)
#     model.to(device)
#     return model


@model_constructor
def transt_resnet50_backbonesplit(settings):
    num_classes = 1
    backbone_net_template, backbone_net_search = build_backbone(settings, backbone_pretrained=True)
    featurefusion_network_cnn = build_featurefusion_seg_cnn_network(settings)
    model = ResSplit(
        backbone_net_template,
        backbone_net_search,
        featurefusion_network_cnn,
        num_classes=num_classes
    )
    model = ResSegm(model, freeze_transt=settings.freeze_transt)

    if settings.use_transformer:
        featurefusion_network_transformer = build_featurefusion_seg_network(settings)
        model = TranstResSplit(
            model,
            featurefusion_network_transformer,
            num_classes=num_classes
        )
        model = TransTFusionSegm(model, freeze_transt=settings.freeze_transt)

    device = torch.device(settings.device)
    model.to(device)
    return model

model_constructor
def transt_resnet50_backbonesplit_dilated(settings):
    num_classes = 1
    backbone_net_template, backbone_net_search = build_backbone(settings, backbone_pretrained=True,use_dilated = True)
    featurefusion_network_cnn = build_featurefusion_seg_cnn_network(settings)
    model = ResSplit(
        backbone_net_template,
        backbone_net_search,
        featurefusion_network_cnn,
        num_classes=num_classes
    )
    model = ResSegm(model, freeze_transt=settings.freeze_transt,use_dilated = True)

    if settings.use_transformer:
        featurefusion_network_transformer = build_featurefusion_seg_network(settings)
        model = TranstResSplit(
            model,
            featurefusion_network_transformer,
            num_classes=num_classes
        )
        model = TransTFusionSegm(model, freeze_transt=settings.freeze_transt,use_dilated = True)

    device = torch.device(settings.device)
    model.to(device)
    return model

def transt_loss(settings):
    num_classes = 1

    weight_dict = {'loss_mask': settings.loss_mask_weight, 'loss_dice': settings.loss_dice_weight, 'loss_bce':settings.loss_bce_weight}
    losses = ['masks']

    criterion = SetCriterion(num_classes, weight_dict=weight_dict,
                             eos_coef=0.0625, losses=losses)
    device = torch.device(settings.device)
    criterion.to(device)
    return criterion

def transt_loss_double_head(settings):
    num_classes = 1

    weight_dict = {'loss_mask_transformer': settings.loss_mask_weight, 'loss_dice_transformer': settings.loss_dice_weight, 'loss_bce_transformer':settings.loss_bce_weight,
                   'loss_mask_cnn': settings.loss_mask_weight_cnn, 'loss_dice_cnn': settings.loss_dice_weight_cnn, 'loss_bce_cnn':settings.loss_bce_weight_cnn}
    losses = ['masks_double_head']

    criterion = SetCriterion(num_classes, weight_dict=weight_dict,
                             eos_coef=0.0625, losses=losses)
    device = torch.device(settings.device)
    criterion.to(device)
    return criterion
