import torch.nn as nn
from ltr import model_constructor

import torch
import torch.nn.functional as F
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor,
                       nested_tensor_from_tensor_2, interpolate,
                       nested_tensor_from_tensor_list, accuracy)

from ltr.models.backbone.transt_backbone import build_backbone
from ltr.models.backbone.transt_backbone_resize import build_backbone_resize
from ltr.models.loss.matcher import build_matcher
from ltr.models.neck.featurefusion_network import build_featurefusion_network
from ltr.models.segm.transt_seg import (TransTCVTiouhsegm,TransTCVTLargeiouhsegm, dice_loss, sigmoid_focal_loss)



class TransTCVT(nn.Module):
    """ This is the TransT module that performs single object tracking with CVT backbone"""
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
        self.class_embed = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

class TransTCVTLarge(nn.Module):
    """ This is the TransT module that performs single object tracking with CVT backbone"""
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
        self.class_embed = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(1024, hidden_dim, kernel_size=1)
        self.backbone = backbone


class TransT(nn.Module):
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
        self.class_embed = MLP(hidden_dim, hidden_dim, num_classes + 1, 3)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
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

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@model_constructor
def transt_resnet50(settings):
    num_classes = 1
    backbone_net = build_backbone(settings, backbone_pretrained=True, use_cvt=False)
    featurefusion_network = build_featurefusion_network(settings)
    model = TransT(
        backbone_net,
        featurefusion_network,
        num_classes=num_classes
    )
    if settings.iou_head:
        assert settings.masks == False
        model = TransTiouh(model, freeze_transt=settings.freeze_transt)
    elif settings.masks:
        assert settings.iou_head == False
        model = TransTiouh(model, freeze_transt=settings.freeze_transt)
        model = TransTiouhsegm(model, freeze_transt=settings.freeze_transt)
    device = torch.device(settings.device)
    model.to(device)
    return model


@model_constructor
def transt_cvt(settings):
    num_classes = 1
    backbone_net = build_backbone(settings, backbone_pretrained=True, use_cvt_seg=True)
    featurefusion_network = build_featurefusion_network(settings)
    # TODO: TranT fit CVT
    model = TransTCVT(
        backbone_net,
        featurefusion_network,
        num_classes=num_classes
    )

    model = TransTCVTiouhsegm(model, freeze_transt=settings.freeze_transt)
    device = torch.device(settings.device)
    model.to(device)
    return model

@model_constructor
def transt_cvt_resize(settings):
    num_classes = 1
    backbone_net = build_backbone_resize(settings, backbone_pretrained=True)
    featurefusion_network = build_featurefusion_network(settings)
    # TODO: TranT fit CVT
    model = TransTCVT(
        backbone_net,
        featurefusion_network,
        num_classes=num_classes
    )

    model = TransTCVTiouhsegm(model, freeze_transt=settings.freeze_transt)
    device = torch.device(settings.device)
    model.to(device)
    return model

@model_constructor
def transt_cvt_large(settings):
    num_classes = 1
    backbone_net = build_backbone(settings, backbone_pretrained=True, use_cvt_seg=True, use_large_backbone = True)
    featurefusion_network = build_featurefusion_network(settings)
    # TODO: TranT fit CVT
    model = TransTCVTLarge(
        backbone_net,
        featurefusion_network,
        num_classes=num_classes
    )

    model = TransTCVTLargeiouhsegm(model, freeze_transt=settings.freeze_transt)
    device = torch.device(settings.device)
    model.to(device)
    return model

@model_constructor
def transt_cvt_large_resize(settings):
    num_classes = 1
    backbone_net = build_backbone_resize(settings, backbone_pretrained=False, use_large_backbone = True)
    featurefusion_network = build_featurefusion_network(settings)
    # TODO: TranT fit CVT
    model = TransTCVTLarge(
        backbone_net,
        featurefusion_network,
        num_classes=num_classes
    )

    model = TransTCVTLargeiouhsegm(model, freeze_transt=settings.freeze_transt)
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
