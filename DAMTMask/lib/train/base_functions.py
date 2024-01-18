import torch
from torch.utils.data.distributed import DistributedSampler
# datasets related
from lib.train.dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet
from lib.train.dataset import Lasot_lmdb, Got10k_lmdb, MSCOCOSeq_lmdb, ImagenetVID_lmdb, TrackingNet_lmdb
from lib.train.dataset.trdataset import Youtube_VOS, Saliency, OVIS, YoutubeVIS2021
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process
import lib.train.admin.settings as ws_settings


def update_segm_settings(settings, cfg):
    settings.segm_param = ws_settings.Settings()
    settings.segm_param.nheads = cfg.MODEL.SEGM.NHEADS
    settings.segm_param.hidden_dim = cfg.MODEL.SEGM.HIDDEN_DIM
    settings.segm_param.in_channels = cfg.MODEL.SEGM.IN_CHANNELS


def update_dfcos_settings(settings, cfg):
    settings.dfcos_param = ws_settings.Settings()

    settings.dfcos_param.merge_layer = cfg.MODEL.DFCOS.MERGW_LAYER
    settings.dfcos_param.in_channels = cfg.MODEL.DFCOS.IN_CHANNELS
    settings.dfcos_param.mid_channels = cfg.MODEL.DFCOS.MID_CHANNELS
    settings.dfcos_param.num_share_convs = cfg.MODEL.DFCOS.NUM_SHARE_CONVS
    settings.dfcos_param.num_convs = cfg.MODEL.DFCOS.NUM_CONVS
    settings.dfcos_param.total_stride = cfg.MODEL.DFCOS.TOTAL_STRIDE
    settings.dfcos_param.input_size_adapt = cfg.MODEL.DFCOS.INPUT_SIZE_ADAPT
    settings.dfcos_param.x_size = cfg.DATA.SEARCH.SIZE
    settings.dfcos_param.score_size = cfg.DATA.SEARCH.SIZE//cfg.MODEL.DFCOS.TOTAL_STRIDE
    settings.dfcos_param.fm_offset = (cfg.DATA.SEARCH.SIZE - (cfg.DATA.SEARCH.SIZE//cfg.MODEL.DFCOS.TOTAL_STRIDE - 1)*cfg.MODEL.DFCOS.TOTAL_STRIDE)//2

def update_reppoint_settings(settings, cfg):
    settings.reppoint_param = ws_settings.Settings()

    settings.reppoint_param.num_points = cfg.MODEL.REPPOINT.NUM_POINTS
    settings.reppoint_param.stacked_convs = cfg.MODEL.REPPOINT.STACKED_CONVS
    settings.reppoint_param.in_channels = cfg.MODEL.REPPOINT.IN_CHANNELS

    settings.reppoint_param.feat_channels = cfg.MODEL.REPPOINT.FEAT_CHANNELS
    settings.reppoint_param.point_feat_channels = cfg.MODEL.REPPOINT.POINT_FEAT_CHANNELS

    settings.reppoint_param.cls_out_channels = cfg.MODEL.REPPOINT.CLS_OUT_CHANNELS
    settings.reppoint_param.transform_method = cfg.MODEL.REPPOINT.TRANSFORM_METHOD

    settings.reppoint_param.transform_method = cfg.MODEL.REPPOINT.TRANSFORM_METHOD
    settings.reppoint_param.moment_mul = cfg.MODEL.REPPOINT.MOMENT_MUL
    settings.reppoint_param.gradient_mul = cfg.MODEL.REPPOINT.GRADIENT_MUL
    settings.reppoint_param.offset = cfg.MODEL.REPPOINT.OFFSET
    settings.reppoint_param.stride = cfg.MODEL.REPPOINT.STRIDE
    settings.reppoint_param.merge_layer = cfg.MODEL.REPPOINT.MERGW_LAYER

    settings.reppoint_param.score_size = cfg.DATA.SEARCH.SIZE//cfg.MODEL.REPPOINT.STRIDE
    settings.reppoint_param.fm_offset = (cfg.DATA.SEARCH.SIZE - (cfg.DATA.SEARCH.SIZE//cfg.MODEL.REPPOINT.STRIDE - 1)*cfg.MODEL.REPPOINT.STRIDE)//2

    settings.reppoint_param.init_pos_num = cfg.MODEL.REPPOINT.INIT_POS_NUM

    settings.reppoint_param.pos_iou = cfg.MODEL.REPPOINT.POS_IOU
    settings.reppoint_param.neg_iou = cfg.MODEL.REPPOINT.NEG_IOU
    settings.reppoint_param.pos_num = 1

def update_dimp_settings(settings, cfg):
    settings.dimp_param = ws_settings.Settings()

    settings.dimp_param.robust_filter = cfg.MODEL.DIMP.ROBUST_FILTER #robust_filter
    settings.dimp_param.drop_prob = cfg.MODEL.DIMP.DROP_PROB #drop_prob

    settings.dimp_param.target_filter_sz = cfg.MODEL.DIMP.TARGET_FILTER_SZ
    settings.dimp_param.output_sigma_factor = cfg.MODEL.DIMP.OUTPUT_SIGMA_FACTOR
    settings.dimp_param.output_sigma = settings.dimp_param.output_sigma_factor / cfg.DATA.SEARCH.FACTOR
    settings.dimp_param.hinge_threshold = cfg.MODEL.DIMP.HINGE_THRESHOLD

    settings.dimp_param.filter_size = cfg.MODEL.DIMP.FILTER_SIZE
    settings.dimp_param.optim_iter = cfg.MODEL.DIMP.OPTIM_ITER
    settings.dimp_param.optim_init_step = cfg.MODEL.DIMP.OPTIM_INIT_STEP
    settings.dimp_param.optim_init_reg = cfg.MODEL.DIMP.OPTIM_INIT_REG
    settings.dimp_param.feat_stride = cfg.MODEL.DIMP.FEAT_STRIDE
    settings.dimp_param.clf_feat_blocks = cfg.MODEL.DIMP.CLF_FEAT_BLOCKS
    settings.dimp_param.clf_feat_norm = cfg.MODEL.DIMP.CLF_FEAT_NORM
    settings.dimp_param.init_filter_norm = cfg.MODEL.DIMP.INIT_FILTER_NORM
    settings.dimp_param.final_conv = cfg.MODEL.DIMP.FINAL_CONV
    settings.dimp_param.out_feature_dim = cfg.MODEL.DIMP.OUT_FEATURE_DIM
    settings.dimp_param.init_gauss_sigma = settings.dimp_param.output_sigma * (cfg.DATA.SEARCH.SIZE//cfg.MODEL.DIMP.FEAT_STRIDE)
    settings.dimp_param.num_dist_bins = cfg.MODEL.DIMP.NUM_DIST_BINS
    settings.dimp_param.bin_displacement = cfg.MODEL.DIMP.BIN_DISPLACEMENT
    settings.dimp_param.mask_init_factor = cfg.MODEL.DIMP.MASK_INIT_FACTOR
    settings.dimp_param.score_act = cfg.MODEL.DIMP.SCORE_ACT
    settings.dimp_param.act_param = cfg.MODEL.DIMP.ACT_PARAM
    settings.dimp_param.target_mask_act = cfg.MODEL.DIMP.TARGET_MASK_ACT
    settings.dimp_param.detach_length = cfg.MODEL.DIMP.DETACH_LENGTH

    settings.dimp_param.backbone_outdim = cfg.MODEL.DIMP.BACKBONE_OUTDIM
    settings.dimp_param.merge_layer = cfg.MODEL.DIMP.MERGW_LAYER
    settings.dimp_param.feature_sz = {'template': cfg.DATA.TEMPLATE.SIZE//cfg.MODEL.DIMP.FEAT_STRIDE,
                                        'search': cfg.DATA.SEARCH.SIZE//cfg.MODEL.DIMP.FEAT_STRIDE}

def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
                                   'search': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
                          'search': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER}
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in ["LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full", "COCO17", "VID", "TRACKINGNET","Youtube_VOS","Saliency","OVIS","YoutubeVIS2021"]
        if name == "LASOT":
            if settings.use_lmdb:
                print("Building lasot dataset from lmdb")
                datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader))
            else:
                datasets.append(Lasot(settings.env.lasot_dir, split='train', image_loader=image_loader))
        if name == "GOT10K_vottrain":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='vottrain', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='vottrain', image_loader=image_loader))
        if name == "GOT10K_train_full":
            if settings.use_lmdb:
                print("Building got10k_train_full from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='train_full', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='train_full', image_loader=image_loader))
        if name == "GOT10K_votval":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='votval', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='votval', image_loader=image_loader))
        if name == "COCO17":
            if settings.use_lmdb:
                print("Building COCO2017 from lmdb")
                datasets.append(MSCOCOSeq_lmdb(settings.env.coco_lmdb_dir, version="2017", image_loader=image_loader))
            else:
                datasets.append(MSCOCOSeq(settings.env.coco_dir, version="2017", image_loader=image_loader))
        if name == "VID":
            if settings.use_lmdb:
                print("Building VID from lmdb")
                datasets.append(ImagenetVID_lmdb(settings.env.imagenet_lmdb_dir, image_loader=image_loader))
            else:
                datasets.append(ImagenetVID(settings.env.imagenet_dir, image_loader=image_loader))
        if name == "TRACKINGNET":
            if settings.use_lmdb:
                print("Building TrackingNet from lmdb")
                datasets.append(TrackingNet_lmdb(settings.env.trackingnet_lmdb_dir, image_loader=image_loader))
            else:
                # raise ValueError("NOW WE CAN ONLY USE TRACKINGNET FROM LMDB")
                datasets.append(TrackingNet(settings.env.trackingnet_dir, image_loader=image_loader))
        if name == "Youtube_VOS":
            datasets.append(Youtube_VOS(settings.env.youtube_vos_dir, image_loader=image_loader))
        if name == "Saliency":
            datasets.append(Saliency(settings.env.saliency_dir, image_loader=image_loader))
        if name == "OVIS":
            datasets.append(OVIS(root = settings.env.ovis_dir, image_loader=image_loader))
        if name == "YoutubeVIS2021":
            datasets.append(YoutubeVIS2021(root = settings.env.youtubevis2021_dir, image_loader=image_loader))
    return datasets


def build_dimp_dataloaders(cfg, settings):
    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    # Train sampler and loader
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_score = getattr(cfg.TRAIN, "TRAIN_SCORE", False)
    print("sampler_mode", sampler_mode)

    # The tracking pairs processing module
    # output_sigma = settings.output_sigma_factor / settings.search_area_factor['search']
    label_params = {'feature_sz': settings.dimp_param.feature_sz['search'], 'sigma_factor': settings.dimp_param.output_sigma, 'kernel_sz': settings.dimp_param.target_filter_sz}
    label_density_params = {'feature_sz': settings.dimp_param.feature_sz['search'], 'sigma_factor': settings.dimp_param.output_sigma, 'kernel_sz': settings.dimp_param.target_filter_sz}

    data_processing_train = processing.MixformerKLDiMPProcessing(search_area_factor=search_area_factor,
                                                           output_sz=output_sz,
                                                           center_jitter_factor=settings.center_jitter_factor,
                                                           scale_jitter_factor=settings.scale_jitter_factor,
                                                           mode='sequence',
                                                           label_function_params=label_params,
                                                           label_density_params=label_density_params,
                                                           transform=transform_train,
                                                           joint_transform=transform_joint,
                                                           settings=settings,
                                                           train_score=train_score)

    data_processing_val = processing.MixformerKLDiMPProcessing(search_area_factor=search_area_factor,
                                                         output_sz=output_sz,
                                                         center_jitter_factor=settings.center_jitter_factor,
                                                         scale_jitter_factor=settings.scale_jitter_factor,
                                                         mode='sequence',
                                                         label_function_params=label_params,
                                                         label_density_params=label_density_params,
                                                         transform=transform_val,
                                                         joint_transform=transform_joint,
                                                         settings=settings,
                                                         train_score=train_score)


    dataset_train = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
                                            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_train,
                                            frame_sample_mode=sampler_mode, train_cls=train_score, pos_prob=0.5)

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    # Validation samplers and loaders
    dataset_val = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader),
                                          p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
                                          samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                          max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                          num_template_frames=settings.num_template, processing=data_processing_val,
                                          frame_sample_mode=sampler_mode, train_cls=train_score, pos_prob=0.5)
    val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                           num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                           epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

    return loader_train, loader_val




def build_dataloaders(cfg, settings):
    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    transform_train = tfm.Transform(tfm.ToTensorAndJitter(0.2),
                                    tfm.RandomHorizontalFlip_Norm(probability=0.5),
                                    tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_val = tfm.Transform(tfm.ToTensor(),
                                  tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    # Train sampler and loader
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_score = getattr(cfg.TRAIN, "TRAIN_SCORE", False)
    print("sampler_mode", sampler_mode)

    data_processing_train = processing.MixformerProcessing(search_area_factor=search_area_factor,
                                                           output_sz=output_sz,
                                                           center_jitter_factor=settings.center_jitter_factor,
                                                           scale_jitter_factor=settings.scale_jitter_factor,
                                                           mode='sequence',
                                                           transform=transform_train,
                                                           joint_transform=transform_joint,
                                                           settings=settings,
                                                           train_score=train_score)

    data_processing_val = processing.MixformerProcessing(search_area_factor=search_area_factor,
                                                         output_sz=output_sz,
                                                         center_jitter_factor=settings.center_jitter_factor,
                                                         scale_jitter_factor=settings.scale_jitter_factor,
                                                         mode='sequence',
                                                         transform=transform_val,
                                                         joint_transform=transform_joint,
                                                         settings=settings,
                                                         train_score=train_score)


    dataset_train = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
                                            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
                                            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
                                            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                            num_template_frames=settings.num_template, processing=data_processing_train,
                                            frame_sample_mode=sampler_mode, train_cls=train_score, pos_prob=0.5)

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    # Validation samplers and loaders
    dataset_val = sampler.TrackingSampler(datasets=names2datasets(cfg.DATA.VAL.DATASETS_NAME, settings, opencv_loader),
                                          p_datasets=cfg.DATA.VAL.DATASETS_RATIO,
                                          samples_per_epoch=cfg.DATA.VAL.SAMPLE_PER_EPOCH,
                                          max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL, num_search_frames=settings.num_search,
                                          num_template_frames=settings.num_template, processing=data_processing_val,
                                          frame_sample_mode=sampler_mode, train_cls=train_score, pos_prob=0.5)
    val_sampler = DistributedSampler(dataset_val) if settings.local_rank != -1 else None
    loader_val = LTRLoader('val', dataset_val, training=False, batch_size=cfg.TRAIN.BATCH_SIZE,
                           num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=val_sampler,
                           epoch_interval=cfg.TRAIN.VAL_EPOCH_INTERVAL)

    return loader_train, loader_val

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1 or classname.find('LayerNorm') != -1:
        # print(classname)
        m.eval()
def get_optimizer_scheduler(net, cfg):
    train_score = getattr(cfg.TRAIN, "TRAIN_SCORE", False)
    freeze_stage0 = getattr(cfg.TRAIN, "FREEZE_STAGE0", False)
    freeze_mixformer = getattr(cfg.TRAIN, "FREEZE_MIXFORMER", False)
    freeze_mixformer_dimp = getattr(cfg.TRAIN, "FREEZE_DIMP", False)
    freeze_mixformer_dimp_transt = getattr(cfg.TRAIN, "FREEZE_DIMP_TRANST", False)
    freeze_mixformer_dimp_for_reppoint = getattr(cfg.TRAIN, "FREEZE_FOR_REPPOINT", False)
    mixformer_train_all = getattr(cfg.TRAIN, "TRAIN_ALL", False)
    freeze_for_segm = getattr(cfg.TRAIN, "FREEZE_FOR_SEGM", False)
    freeze_for_mask = getattr(cfg.TRAIN, "FREEZE_FOR_MASK", False)
    # freeze_stage0 = True
    if train_score:
        print("Only training score_branch. Learnable parameters are shown below.")
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "score" in n and p.requires_grad]}
        ]

        for n, p in net.named_parameters():
            if "score" not in n:
                p.requires_grad = False
            else:
                if is_main_process():
                    print(n)
    elif freeze_stage0:
        print("Freeze Stage0 of MixFormer backbone.")
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if (("stage2" in n or "stage1" in n) and p.requires_grad)],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]

        for n, p in net.named_parameters():
            if "stage2" not in n and "box_head" not in n and "stage1" not in n:
                p.requires_grad = False
            else:
                if is_main_process():
                    print(n)
    elif freeze_mixformer:
        print("Only training dimp-dfcos-neck, Learnable parameters are shown below.")
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "dimp_merge_feature_layer" in n and p.requires_grad],"lr": 1e-3},
            {"params": [p for n, p in net.named_parameters() if "filter_initializer" in n and p.requires_grad],"lr": 5e-5},
            {"params": [p for n, p in net.named_parameters() if "filter_optimizer" in n and p.requires_grad],"lr": 5e-4},
            {"params": [p for n, p in net.named_parameters() if "feature_extractor" in n and p.requires_grad],"lr": 5e-5},
            {"params": [p for n, p in net.named_parameters() if "dfcos" in n and p.requires_grad],"lr": 1e-3},

            {"params": [p for n, p in net.named_parameters() if "input_proj" in n and p.requires_grad]
                       + [p for n, p in net.named_parameters() if "position_encoding" in n and p.requires_grad]
                       + [p for n, p in net.named_parameters() if "featurefusion_network" in n and p.requires_grad],"lr": 1e-4}
        ]
        for n, p in net.named_parameters():
            if ("backbone" in n) or ("box_head" in n) or ("score_branch" in n):
                p.requires_grad = False
            else:
                if is_main_process():
                    print(n)

        net.module.backbone.apply(fix_bn)
        net.module.box_head.apply(fix_bn)
        net.module.score_branch.apply(fix_bn)

    elif freeze_mixformer_dimp:
        print("Only training dfcos and neck, Learnable parameters are shown below.")
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "dfcos" in n and p.requires_grad],"lr": 1e-3},
            {"params": [p for n, p in net.named_parameters() if "input_proj" in n and p.requires_grad]
                       + [p for n, p in net.named_parameters() if "position_encoding" in n and p.requires_grad]
                       + [p for n, p in net.named_parameters() if "featurefusion_network" in n and p.requires_grad],"lr": 1e-4}
        ]
        for n, p in net.named_parameters():
            if ("backbone" in n) or ("box_head" in n) or ("score_branch" in n)  or ("dimp" in n):
                p.requires_grad = False
            else:
                if is_main_process():
                    print(n)

        net.module.backbone.apply(fix_bn)
        net.module.box_head.apply(fix_bn)
        net.module.score_branch.apply(fix_bn)
        net.module.dimp_branch.apply(fix_bn)

    elif freeze_mixformer_dimp_transt:
        print("Only training dfcos and neck, Learnable parameters are shown below.")
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "dfcos" in n and p.requires_grad],"lr": 1e-6},
        ]
        for n, p in net.named_parameters():
            if ("backbone" in n) or ("box_head" in n) or ("score_branch" in n)  or ("dimp" in n) or ("input_proj" in n) or ("position_encoding" in n) or ("featurefusion_network" in n) or ("cls_head" in n) or ("reg_head" in n):
                p.requires_grad = False
            else:
                if is_main_process():
                    print(n)
    elif freeze_mixformer_dimp_for_reppoint:
        print("Only training reppoint, Learnable parameters are shown below.")
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "reppoint" in n and p.requires_grad],"lr": 1e-3},
        ]
        for n, p in net.named_parameters():
            if ("backbone" in n) or ("box_head" in n) or ("score_branch" in n)  or ("dimp" in n):
                p.requires_grad = False
            else:
                if is_main_process():
                    print(n)

        net.module.backbone.apply(fix_bn)
        net.module.box_head.apply(fix_bn)
        net.module.score_branch.apply(fix_bn)
        net.module.dimp_branch.apply(fix_bn)

    elif freeze_for_segm:
        print("Only training segm, Learnable parameters are shown below.")
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "segm" in n and p.requires_grad],"lr": 1e-3},
        ]
        for n, p in net.named_parameters():
            if ("backbone" in n) or ("box_head" in n) or ("score_branch" in n)  or ("dimp" in n) or ("reppoint" in n):
                p.requires_grad = False
            else:
                if is_main_process():
                    print(n)

        net.module.backbone.apply(fix_bn)
        net.module.box_head.apply(fix_bn)
        net.module.score_branch.apply(fix_bn)
        net.module.dimp_branch.apply(fix_bn)
        net.module.reppoint_head.apply(fix_bn)

    elif mixformer_train_all:
        print("training all dimp-corner-reppoint, Learnable parameters are shown below.")
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "dimp_merge_feature_layer" in n and p.requires_grad],"lr": 1e-3},
            {"params": [p for n, p in net.named_parameters() if "filter_initializer" in n and p.requires_grad],"lr": 5e-5},
            {"params": [p for n, p in net.named_parameters() if "filter_optimizer" in n and p.requires_grad],"lr": 5e-4},
            {"params": [p for n, p in net.named_parameters() if "feature_extractor" in n and p.requires_grad],"lr": 5e-5},
            {"params": [p for n, p in net.named_parameters() if "reppoint" in n and p.requires_grad],"lr": 1e-3},

            {"params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad]
                       + [p for n, p in net.named_parameters() if "box_head" in n and p.requires_grad]
                       + [p for n, p in net.named_parameters() if "score_branch" in n and p.requires_grad],"lr": 1e-4}
        ]
        for n, p in net.named_parameters():
            if "score_branch" in n:
                p.requires_grad = False
            else:
                if is_main_process():
                    print(n)

        net.module.score_branch.apply(fix_bn)

    elif freeze_for_mask:
        print("training  dimp-corner-reppoint use_all_data, Learnable parameters are shown below.")
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "dimp_merge_feature_layer" in n and p.requires_grad],"lr": 1e-3},
            {"params": [p for n, p in net.named_parameters() if "filter_initializer" in n and p.requires_grad],"lr": 5e-5},
            {"params": [p for n, p in net.named_parameters() if "filter_optimizer" in n and p.requires_grad],"lr": 5e-4},
            {"params": [p for n, p in net.named_parameters() if "feature_extractor" in n and p.requires_grad],"lr": 5e-5},
            {"params": [p for n, p in net.named_parameters() if "reppoint" in n and p.requires_grad],"lr": 1e-3},

            {"params": [p for n, p in net.named_parameters() if "box_head" in n and p.requires_grad],"lr": 1e-4}
        ]
        for n, p in net.named_parameters():
            if ("backbone" in n) or ("score_branch" in n):
                p.requires_grad = False
            else:
                if is_main_process():
                    print(n)
        net.module.backbone.apply(fix_bn)
        net.module.score_branch.apply(fix_bn)


    else:
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")

    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH, gamma=cfg.TRAIN.SCHEDULER.DECAY_RATE)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler
