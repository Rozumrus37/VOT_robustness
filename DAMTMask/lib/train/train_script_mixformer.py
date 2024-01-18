import os
# loss function related
from lib.utils.losses import ClsRegCriterion,ReppointCriterion,dice_loss
from lib.utils.box_ops import giou_loss, IOULoss,LBHinge,FocalLoss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.mixformer import (build_mixformer, build_mixformer_online_score,build_mixformer_online_score_dimp,
                                build_mixformer_online_score_dimp_dfcos,build_mixformer_online_score_dimp_dfcos_neck,
                                build_mixformer_online_score_dimp_dfcos_neck2,build_mixformer_online_score_dimp_dfcos_transt,
                                build_mixformer_online_score_dimp_reppoint,build_mixformer_score_dimp_reppoint_mask,
                                build_mixformer_online_score_dimp_reppoint_segm,build_mixformer_score_dimp_reppoint_mask_st,
                                build_mixformer_score_dimp_reppoint_mask_segm_st)
# forward propagation related
from lib.train.actors import (MixFormerActor,MixFormerKLDimpActor,MixFormerKLDimpDFcosActor,
                                MixFormerKLDimpDFcosTranstActor,MixFormerKLDimpReppointActor,
                                MixFormerKLDimpReppointMaskActor,MixFormerKLDimpReppointSegmActor,
                                MixFormerKLDimpReppointMaskSingleActor,MixFormerKLDimpReppointMaskSegmSingleActor)
# for import modules
import importlib


def prepare_input(res):
    res_t, res_s = res
    t = torch.FloatTensor(1, 3, res_t, res_t).cuda()
    s = torch.FloatTensor(1, 3, res_s, res_s).cuda()
    return dict(template=t, search=s)


def run(settings):
    settings.description = 'Training script for Mixformer'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    print(cfg)

    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)
    if 'dimp' in settings.script_name:
        update_dimp_settings(settings, cfg)
    if 'dfcos' in settings.script_name:
        update_dfcos_settings(settings, cfg)
    if 'reppoint' in settings.script_name:
        update_reppoint_settings(settings, cfg)
    if 'segm' in settings.script_name:
        update_segm_settings(settings, cfg)


    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Create network
    if settings.script_name == "mixformer":
        net = build_mixformer(cfg)
        # Build dataloaders
        loader_train, loader_val = build_dataloaders(cfg, settings)
    elif settings.script_name == "mixformer_online":
        net = build_mixformer_online_score(cfg, settings)
        # Build dataloaders
        loader_train, loader_val = build_dataloaders(cfg, settings)


    elif 'mask' in settings.script_name and 'segm' in settings.script_name and 'reppoint' in settings.script_name and 'single' in settings.script_name:
        net = build_mixformer_score_dimp_reppoint_mask_segm_st(cfg, settings)
        # Build dataloaders
        loader_train, loader_val = build_dimp_dataloaders(cfg, settings)

    elif 'segm' in settings.script_name and 'reppoint' in settings.script_name:
        net = build_mixformer_online_score_dimp_reppoint_segm(cfg, settings)
        # Build dataloaders
        loader_train, loader_val = build_dimp_dataloaders(cfg, settings)

    elif 'mask' in settings.script_name and 'reppoint' in settings.script_name and 'single' in settings.script_name:
        net = build_mixformer_score_dimp_reppoint_mask_st(cfg, settings)
        # Build dataloaders
        loader_train, loader_val = build_dimp_dataloaders(cfg, settings)

    elif 'mask' in settings.script_name and 'reppoint' in settings.script_name:
        net = build_mixformer_score_dimp_reppoint_mask(cfg, settings)
        # Build dataloaders
        loader_train, loader_val = build_dimp_dataloaders(cfg, settings)

    elif 'reppoint' in settings.script_name:
        net = build_mixformer_online_score_dimp_reppoint(cfg, settings)
        # Build dataloaders
        loader_train, loader_val = build_dimp_dataloaders(cfg, settings)


    elif 'transt' in settings.script_name:
        net = build_mixformer_online_score_dimp_dfcos_transt(cfg, settings)
        # Build dataloaders
        loader_train, loader_val = build_dimp_dataloaders(cfg, settings)

    elif 'neck2' in settings.script_name:
        net = build_mixformer_online_score_dimp_dfcos_neck2(cfg, settings)
        # Build dataloaders
        loader_train, loader_val = build_dimp_dataloaders(cfg, settings)


    elif 'neck' in settings.script_name:
        net = build_mixformer_online_score_dimp_dfcos_neck(cfg, settings)
        # Build dataloaders
        loader_train, loader_val = build_dimp_dataloaders(cfg, settings)

    elif 'dimp' in settings.script_name and 'dfcos' in settings.script_name:
        net = build_mixformer_online_score_dimp_dfcos(cfg, settings)
        # Build dataloaders
        loader_train, loader_val = build_dimp_dataloaders(cfg, settings)
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    # settings.save_every_epoch = True
    # Loss functions and Actors
    if settings.script_name == 'mixformer':
        objective = {'giou': giou_loss, 'l1': l1_loss}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT}
        actor = MixFormerActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings)
    elif settings.script_name == 'mixformer_online':
        objective = {'giou': giou_loss, 'l1': l1_loss, 'score': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'score': cfg.TRAIN.SCORE_WEIGHT}
        actor = MixFormerActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, run_score_head=True)

    elif 'mask' in settings.script_name and 'segm' in settings.script_name and 'reppoint' in settings.script_name:
        objective = {'giou': giou_loss, 'l1': l1_loss, 'score': BCEWithLogitsLoss(),
                    'test_clf': LBHinge(threshold=settings.dimp_param.hinge_threshold),
                    'reppoint':ReppointCriterion(pos_iou = settings.reppoint_param.pos_iou,
                                                 neg_iou = settings.reppoint_param.neg_iou,
                                                 object_cls = FocalLoss(alpha=0.75,gamma=2.0),
                                                 object_reg = IOULoss()).to(settings.device),
                    'segm_focal':FocalLoss(alpha=0.75,gamma=2.0),
                    'segm_dice':dice_loss
                    }
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'score': cfg.TRAIN.SCORE_WEIGHT,
                        'test_clf': 100, 'test_init_clf': 100, 'test_iter_clf': 400,
                        'reppoint_cls': 1,'reppoint_init_bbox':0.5,'reppoint_refine_bbox': 1,
                        'segm_focal':1,'segm_dice':1
                    }
        if 'single' in settings.script_name:
            actor = MixFormerKLDimpReppointMaskSegmSingleActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, run_score_head=True)
        else:
            actor = MixFormerKLDimpReppointSegmActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, run_score_head=True)

    elif 'mask' in settings.script_name and 'reppoint' in settings.script_name:
        objective = {'giou': giou_loss, 'l1': l1_loss, 'score': BCEWithLogitsLoss(),
                    'test_clf': LBHinge(threshold=settings.dimp_param.hinge_threshold),
                    'reppoint':ReppointCriterion(pos_iou = settings.reppoint_param.pos_iou,
                                                 neg_iou = settings.reppoint_param.neg_iou,
                                                 object_cls = FocalLoss(alpha=0.75,gamma=2.0),
                                                 object_reg = IOULoss()).to(settings.device)
                    }
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'score': cfg.TRAIN.SCORE_WEIGHT,
                        'test_clf': 100, 'test_init_clf': 100, 'test_iter_clf': 400,
                        'reppoint_cls': 1,'reppoint_init_bbox':0.5,'reppoint_refine_bbox': 1
                    }
        if 'single' in settings.script_name:
            actor = MixFormerKLDimpReppointMaskSingleActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, run_score_head=True)
        else:
            actor = MixFormerKLDimpReppointMaskActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, run_score_head=True)

    elif 'reppoint' in settings.script_name:
        objective = {'giou': giou_loss, 'l1': l1_loss, 'score': BCEWithLogitsLoss(),
                    'test_clf': LBHinge(threshold=settings.dimp_param.hinge_threshold),
                    'reppoint':ReppointCriterion(pos_iou = settings.reppoint_param.pos_iou,
                                                 neg_iou = settings.reppoint_param.neg_iou,
                                                 object_cls = FocalLoss(alpha=0.75,gamma=2.0),
                                                 object_reg = IOULoss()).to(settings.device)
                    }
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'score': cfg.TRAIN.SCORE_WEIGHT,
                        'test_clf': 100, 'test_init_clf': 100, 'test_iter_clf': 400,
                        'reppoint_cls': 1,'reppoint_init_bbox':0.5,'reppoint_refine_bbox': 1
                    }
        actor = MixFormerKLDimpReppointActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, run_score_head=True)


    elif 'transt' in settings.script_name:
        objective = {'giou': giou_loss, 'l1': l1_loss, 'score': BCEWithLogitsLoss(),
                    'test_clf': LBHinge(threshold=settings.dimp_param.hinge_threshold),
                    'iou': IOULoss(),'cls':FocalLoss(alpha=0.75,gamma=2.0),
                    'transt':ClsRegCriterion().to(settings.device)
                    }
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'score': cfg.TRAIN.SCORE_WEIGHT,
                        'test_clf': 100, 'test_init_clf': 100, 'test_iter_clf': 400,
                        'dfcos_cls_fc': 1.4, 'dfcos_bbox_fc': 0.6, 'dfcos_cls_conv':0.5, 'dfcos_bbox_conv':2.0,
                        'cls': 1, 'reg_giou': 1, 'reg_l1': 1
                    }
        actor = MixFormerKLDimpDFcosTranstActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, run_score_head=True)

    elif 'dimp' in settings.script_name and 'dfcos' in settings.script_name:
        objective = {'giou': giou_loss, 'l1': l1_loss, 'score': BCEWithLogitsLoss(),
                    'test_clf': LBHinge(threshold=settings.dimp_param.hinge_threshold),
                    'iou': IOULoss(),'cls':FocalLoss(alpha=0.75,gamma=2.0),
                    }
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'score': cfg.TRAIN.SCORE_WEIGHT,
                        'test_clf': 100, 'test_init_clf': 100, 'test_iter_clf': 400,
                        'dfcos_cls_fc': 1.4, 'dfcos_bbox_fc': 0.6, 'dfcos_cls_conv':0.5, 'dfcos_bbox_conv':2.0
                    }
        actor = MixFormerKLDimpDFcosActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, run_score_head=True)
    else:
        raise ValueError("illegal script name")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)
