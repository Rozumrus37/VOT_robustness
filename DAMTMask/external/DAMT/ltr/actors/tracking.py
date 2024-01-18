import torch
import numpy as np
from pytracking import TensorDict
from ltr.admin import multigpu
import cv2
from ltr.actors.show import show_base,show_temporal
class TranstActor:
    """ Actor for training the TransT"""
    def __init__(self, net, objective, settings):
        """
        args:
            net - The network to train
            objective - The loss function
        """
        self.net = net
        self.objective = objective
        self.settings = settings

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the fields 'search_images', 'template_images', 'search_anno'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # a = data['search_images'][0].permute([1,2,0]).cpu().numpy()
        # cv2.imshow('a',a)
        # b = data['static_template_images'][0].permute([1,2,0]).cpu().numpy()
        # cv2.imshow('b',b)
        # c = data['dynamic_template_images'][0,0].permute([1,2,0]).cpu().numpy()
        # cv2.imshow('c',c)
        # d = data['dynamic_template_images'][0,1].permute([1,2,0]).cpu().numpy()
        # cv2.imshow('d',d)
        # cv2.waitKey(0)
        # b = data['search_anno'][0].cpu().numpy()
        # cv2.rectangle(a, (int(b[0]), int(b[1])), (int(b[0]+b[2]), int(b[1]+b[3])), (0, 1.0, 0), 2)
        outputs = self.net(data['search_images'], data['template_images'])
        # generate labels
        # data['search_masks'].shape: torch.Size([bs, 1, 256, 256])

        targets =[]
        targets_origin = data['search_anno']
        _, _, h, w = data['search_images'].shape
        targets_origin[:, 0] += targets_origin[:, 2] / 2
        targets_origin[:, 0] /= w
        targets_origin[:, 1] += targets_origin[:, 3] / 2
        targets_origin[:, 1] /= h
        targets_origin[:, 2] /= w
        targets_origin[:, 3] /= h
        targets_origin = targets_origin.unsqueeze(1)
        for i in range(len(targets_origin)):
            target_origin = targets_origin[i]
            target = {}
            target['boxes'] = target_origin
            label = np.array([0])
            label = torch.tensor(label, device=data['search_anno'].device)
            target['labels'] = label
            target['masks'] = data['search_masks'][i]
            targets.append(target)


        # Compute loss
        # outputs:(center_x, center_y, width, height)
        loss_dict = self.objective(outputs, targets)
        weight_dict = self.objective.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Return training stats
        stats = {'Loss/total': losses.item(),
                 'Loss/ce': loss_dict['loss_ce'].item(),
                 'Loss/bbox': loss_dict['loss_bbox'].item(),
                 'Loss/giou': loss_dict['loss_giou'].item(),
                 'iou': loss_dict['iou'].item()
                 }
        if self.settings.masks:
            stats['Loss/mask'] = loss_dict['loss_mask'].item()
            stats['Loss/dice'] = loss_dict['loss_dice'].item()
        if self.settings.iou_head:
            stats['Loss/iou_head'] = loss_dict['loss_iouh'].item()
        return losses, stats

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        self.net.to(device)

    def train(self, mode=True):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.net.train(mode)
        self.objective.train(mode)

    def eval(self):
        """ Set network to eval mode"""
        self.train(False)

    '''added by chenxin to fix bn'''
    def fix_bns(self):
        net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
        net.featurefusion_network.apply(self.fix_bn)
        net.class_embed.apply(self.fix_bn)
        net.bbox_embed.apply(self.fix_bn)
        net.input_proj.apply(self.fix_bn)
        net.backbone.apply(self.fix_bn)
        if hasattr(net, 'mask_head'):
            net.iou_embed.apply(self.fix_bn)
    '''added by chenxin to fix bn'''

    def fix_bn(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            # print(classname)
            m.eval()


class TranstActorSegm:
    """ Actor for training the TransT"""
    def __init__(self, net, objective, settings):
        """
        args:
            net - The network to train
            objective - The loss function
        """
        self.net = net
        self.objective = objective
        self.settings = settings

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the fields 'search_images', 'template_images', 'search_anno'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # a = data['search_images'][0].permute([1,2,0]).cpu().numpy()
        # cv2.imshow('a',a)
        # b = data['static_template_images'][0].permute([1,2,0]).cpu().numpy()
        # cv2.imshow('b',b)
        # c = data['dynamic_template_images'][0,0].permute([1,2,0]).cpu().numpy()
        # cv2.imshow('c',c)
        # d = data['dynamic_template_images'][0,1].permute([1,2,0]).cpu().numpy()
        # cv2.imshow('d',d)
        # cv2.waitKey(0)
        # b = data['search_anno'][0].cpu().numpy()
        # cv2.rectangle(a, (int(b[0]), int(b[1])), (int(b[0]+b[2]), int(b[1]+b[3])), (0, 1.0, 0), 2)
        outputs = self.net(data['search_images'], data['template_images'],data['template_masks'])
        # generate labels
        # data['search_masks'].shape: torch.Size([bs, 1, 256, 256])

        targets =[]
        targets_origin = data['search_anno']
        _, _, h, w = data['search_images'].shape
        targets_origin[:, 0] += targets_origin[:, 2] / 2
        targets_origin[:, 0] /= w
        targets_origin[:, 1] += targets_origin[:, 3] / 2
        targets_origin[:, 1] /= h
        targets_origin[:, 2] /= w
        targets_origin[:, 3] /= h
        targets_origin = targets_origin.unsqueeze(1)
        for i in range(len(targets_origin)):
            target_origin = targets_origin[i]
            target = {}
            target['masks'] = data['search_masks'][i]
            targets.append(target)


        # Compute loss
        # outputs:(center_x, center_y, width, height)
        loss_dict = self.objective(outputs, targets)
        weight_dict = self.objective.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Return training stats
        stats = {'Loss/total': losses.item()}

        stats['Loss/mask'] = loss_dict['loss_mask'].item()
        stats['Loss/dice'] = loss_dict['loss_dice'].item()
        stats['Loss/bce'] = loss_dict['loss_bce'].item()
        return losses, stats

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        self.net.to(device)

    def train(self, mode=True):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.net.train(mode)
        self.objective.train(mode)

    def eval(self):
        """ Set network to eval mode"""
        self.train(False)

    '''added by chenxin to fix bn'''
    def fix_bns(self):
        net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
        net.featurefusion_network.apply(self.fix_bn)
        net.input_proj.apply(self.fix_bn)
        net.backbone.apply(self.fix_bn)
    '''added by chenxin to fix bn'''

    def fix_bn(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            # print(classname)
            m.eval()

class TranstActorSegmDoubleHead:
    """ Actor for training the TransT"""
    def __init__(self, net, objective, settings):
        """
        args:
            net - The network to train
            objective - The loss function
        """
        self.net = net
        self.objective = objective
        self.settings = settings

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the fields 'search_images', 'template_images', 'search_anno'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # a = data['search_images'][0].permute([1,2,0]).cpu().numpy()
        # cv2.imshow('a',a)
        # b = data['static_template_images'][0].permute([1,2,0]).cpu().numpy()
        # cv2.imshow('b',b)
        # c = data['dynamic_template_images'][0,0].permute([1,2,0]).cpu().numpy()
        # cv2.imshow('c',c)
        # d = data['dynamic_template_images'][0,1].permute([1,2,0]).cpu().numpy()
        # cv2.imshow('d',d)
        # cv2.waitKey(0)
        # b = data['search_anno'][0].cpu().numpy()
        # cv2.rectangle(a, (int(b[0]), int(b[1])), (int(b[0]+b[2]), int(b[1]+b[3])), (0, 1.0, 0), 2)
        outputs = self.net(data['search_images'], data['template_images'],data['template_masks'])
        # generate labels
        # data['search_masks'].shape: torch.Size([bs, 1, 256, 256])

        targets =[]
        targets_origin = data['search_anno']
        _, _, h, w = data['search_images'].shape
        targets_origin[:, 0] += targets_origin[:, 2] / 2
        targets_origin[:, 0] /= w
        targets_origin[:, 1] += targets_origin[:, 3] / 2
        targets_origin[:, 1] /= h
        targets_origin[:, 2] /= w
        targets_origin[:, 3] /= h
        targets_origin = targets_origin.unsqueeze(1)
        for i in range(len(targets_origin)):
            target_origin = targets_origin[i]
            target = {}
            target['masks'] = data['search_masks'][i]
            targets.append(target)


        # Compute loss
        # outputs:(center_x, center_y, width, height)
        loss_dict = self.objective(outputs, targets)
        weight_dict = self.objective.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Return training stats
        stats = {'Loss/total': losses.item()}

        stats['Loss/mask_tr'] = loss_dict['loss_mask_transformer'].item()
        stats['Loss/dice_tr'] = loss_dict['loss_dice_transformer'].item()
        stats['Loss/bce_tr'] = loss_dict['loss_bce_transformer'].item()

        stats['Loss/mask_cnn'] = loss_dict['loss_mask_cnn'].item()
        stats['Loss/dice_cnn'] = loss_dict['loss_dice_cnn'].item()
        stats['Loss/bce_cnn'] = loss_dict['loss_bce_cnn'].item()
        return losses, stats

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        self.net.to(device)

    def train(self, mode=True):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.net.train(mode)
        self.objective.train(mode)

    def eval(self):
        """ Set network to eval mode"""
        self.train(False)

    '''added by chenxin to fix bn'''
    def fix_bns(self):
        net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
        net.backbone_template.apply(self.fix_bn)
        net.backbone_search.apply(self.fix_bn)
        net.featurefusion_network_cnn.apply(self.fix_bn)
        net.cnn_mask.apply(self.fix_bn)

    '''added by chenxin to fix bn'''

    def fix_bn(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            # print(classname)
            m.eval()


class TranstActorSegmFusion:
    """ Actor for training the TransT"""
    def __init__(self, net, objective, settings):
        """
        args:
            net - The network to train
            objective - The loss function
        """
        self.net = net
        self.objective = objective
        self.settings = settings

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the fields 'search_images', 'template_images', 'search_anno'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # a = data['search_images'][0].permute([1,2,0]).cpu().numpy()
        # cv2.imshow('a',a)
        # b = data['static_template_images'][0].permute([1,2,0]).cpu().numpy()
        # cv2.imshow('b',b)
        # c = data['dynamic_template_images'][0,0].permute([1,2,0]).cpu().numpy()
        # cv2.imshow('c',c)
        # d = data['dynamic_template_images'][0,1].permute([1,2,0]).cpu().numpy()
        # cv2.imshow('d',d)
        # cv2.waitKey(0)
        # b = data['search_anno'][0].cpu().numpy()
        # cv2.rectangle(a, (int(b[0]), int(b[1])), (int(b[0]+b[2]), int(b[1]+b[3])), (0, 1.0, 0), 2)
        outputs = self.net(data['search_images'], data['template_images'],data['template_masks'])
        # generate labels
        # data['search_masks'].shape: torch.Size([bs, 1, 256, 256])

        targets =[]
        targets_origin = data['search_anno']
        _, _, h, w = data['search_images'].shape
        targets_origin[:, 0] += targets_origin[:, 2] / 2
        targets_origin[:, 0] /= w
        targets_origin[:, 1] += targets_origin[:, 3] / 2
        targets_origin[:, 1] /= h
        targets_origin[:, 2] /= w
        targets_origin[:, 3] /= h
        targets_origin = targets_origin.unsqueeze(1)
        for i in range(len(targets_origin)):
            target_origin = targets_origin[i]
            target = {}
            target['masks'] = data['search_masks'][i]
            targets.append(target)


        # Compute loss
        # outputs:(center_x, center_y, width, height)
        loss_dict = self.objective(outputs, targets)
        weight_dict = self.objective.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Return training stats
        stats = {'Loss/total': losses.item()}

        stats['Loss/mask'] = loss_dict['loss_mask'].item()
        stats['Loss/dice'] = loss_dict['loss_dice'].item()
        stats['Loss/bce'] = loss_dict['loss_bce'].item()
        return losses, stats

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        self.net.to(device)

    def train(self, mode=True):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.net.train(mode)
        self.objective.train(mode)

    def eval(self):
        """ Set network to eval mode"""
        self.train(False)

    '''added by chenxin to fix bn'''
    def fix_bns(self):
        net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
        net.backbone.apply(self.fix_bn)
    '''added by chenxin to fix bn'''

    def fix_bn(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            # print(classname)
            m.eval()





class TranstTemporalActor:
    """ Actor for training the TransT"""
    def __init__(self, net, objective, settings):
        """
        args:
            net - The network to train
            objective - The loss function
        """
        self.net = net
        self.objective = objective
        self.settings = settings
        self.count = 0

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the fields 'search_images', 'template_images', 'search_anno'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # a = data['search_images'][0].permute([1,2,0]).cpu().numpy()
        # cv2.imshow('a',a)
        # b = data['static_template_images'][0].permute([1,2,0]).cpu().numpy()
        # cv2.imshow('b',b)
        # c = data['dynamic_template_images'][0,0].permute([1,2,0]).cpu().numpy()
        # cv2.imshow('c',c)
        # d = data['dynamic_template_images'][0,1].permute([1,2,0]).cpu().numpy()
        # cv2.imshow('d',d)
        # cv2.waitKey(0)
        # b = data['search_anno'][0].cpu().numpy()
        # cv2.rectangle(a, (int(b[0]), int(b[1])), (int(b[0]+b[2]), int(b[1]+b[3])), (0, 1.0, 0), 2)
        outputs = self.net(data['search_images'], data['template_images'], data['search_response_anno'], data['search_abspositions'], data['search_paddingmasks'])
        # generate labels
        #data['search_masks'].shape: torch.Size([bs, 1, 256, 256])
        self.count += 1
        try:
            if self.count % 10 == 0:
                show_temporal(image_temp = data['template_images'][0,0],
                    bbox_temp = data['template_anno'][0,0],
                    images_search = data['search_images'][0],
                    bboxes_search = data['search_anno'][0],
                    respones = data['search_response_anno'][0],

                    pred_respones = outputs['pred_responses'][0],
                    pred_scores = outputs['pred_logits'][0],
                    pred_boxes = outputs['pred_boxes'][0],
                    pred_masks = outputs['pred_masks'][0],

                    save_root = '/home/tiger/tracking_code/transt_mask_cvt/ltr/vis_train/transt_temporal')
        except:
            pass

        targets =[]
        targets_origin = data['search_anno']
        _, _, _, h, w = data['search_images'].shape
        targets_origin[:,:, 0] += targets_origin[:,:, 2] / 2
        targets_origin[:,:, 0] /= w
        targets_origin[:,:, 1] += targets_origin[:,:, 3] / 2
        targets_origin[:,:, 1] /= h
        targets_origin[:,:, 2] /= w
        targets_origin[:,:, 3] /= h
        targets_origin = targets_origin.unsqueeze(1)
        for i in range(len(targets_origin)):
            target_origin = targets_origin[i,:,-1]
            target = {}
            target['boxes'] = target_origin
            label = np.array([0])
            label = torch.tensor(label, device=data['search_anno'].device)
            target['labels'] = label
            target['masks'] = data['search_masks'][i]
            target['responses'] = data['search_response_anno'][i,-1]
            targets.append(target)


        # Compute loss
        # outputs:(center_x, center_y, width, height)
        loss_dict = self.objective(outputs, targets)
        weight_dict = self.objective.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Return training stats
        stats = {'Loss/total': losses.item(),
                 'Loss/ce': loss_dict['loss_ce'].item(),
                 'Loss/bbox': loss_dict['loss_bbox'].item(),
                 'Loss/giou': loss_dict['loss_giou'].item(),
                 'iou': loss_dict['iou'].item()
                 }
        if self.settings.masks:
            stats['Loss/mask'] = loss_dict['loss_mask'].item()
            stats['Loss/dice'] = loss_dict['loss_dice'].item()
        if self.settings.iou_head:
            stats['Loss/iou_head'] = loss_dict['loss_iouh'].item()

        if self.settings.responses:
            stats['Loss/response'] = loss_dict['loss_responses'].item()
        return losses, stats

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        self.net.to(device)

    def train(self, mode=True):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.net.train(mode)
        self.objective.train(mode)

    def eval(self):
        """ Set network to eval mode"""
        self.train(False)

    '''added by chenxin to fix bn'''
    def fix_bns(self):
        net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
        net.backbone.apply(self.fix_bn)
        net.input_proj.apply(self.fix_bn)
        net.featurefusion_network.apply(self.fix_bn)
        net.class_embed.apply(self.fix_bn)
        net.bbox_embed.apply(self.fix_bn)
        net.iou_embed.apply(self.fix_bn)

        net.bbox_attention_self.apply(self.fix_bn)
        net.bbox_attention_cross.apply(self.fix_bn)
        net.mask_head.apply(self.fix_bn)

    '''added by chenxin to fix bn'''

    def fix_bn(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            # print(classname)
            m.eval()



