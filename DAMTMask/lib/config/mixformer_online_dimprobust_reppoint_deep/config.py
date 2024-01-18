from easydict import EasyDict as edict
import yaml

"""
Add default config for MixFormerOnline.
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.HEAD_TYPE = "CORNER"
cfg.MODEL.HIDDEN_DIM = 384
cfg.MODEL.NUM_OBJECT_QUERIES = 1
cfg.MODEL.POSITION_EMBEDDING = 'sine'  # sine or learned
cfg.MODEL.PREDICT_MASK = False
# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.PRETRAINED = True
cfg.MODEL.BACKBONE.PRETRAINED_PATH = ''
cfg.MODEL.BACKBONE.INIT = 'trunc_norm'
cfg.MODEL.BACKBONE.NUM_STAGES = 3
cfg.MODEL.BACKBONE.PATCH_SIZE = [ 7, 3, 3 ]
cfg.MODEL.BACKBONE.PATCH_STRIDE = [ 4, 2, 2 ]
cfg.MODEL.BACKBONE.PATCH_PADDING = [ 2, 1, 1 ]
cfg.MODEL.BACKBONE.DIM_EMBED = [ 64, 192, 384 ]
cfg.MODEL.BACKBONE.NUM_HEADS = [ 1, 3, 6 ]
cfg.MODEL.BACKBONE.DEPTH = [ 1, 2, 10 ]
cfg.MODEL.BACKBONE.MLP_RATIO = [ 4.0, 4.0, 4.0 ]
cfg.MODEL.BACKBONE.ATTN_DROP_RATE = [ 0.0, 0.0, 0.0 ]
cfg.MODEL.BACKBONE.DROP_RATE = [ 0.0, 0.0, 0.0 ]
cfg.MODEL.BACKBONE.DROP_PATH_RATE = [ 0.0, 0.0, 0.1 ]
cfg.MODEL.BACKBONE.QKV_BIAS = [ True, True, True ]
cfg.MODEL.BACKBONE.CLS_TOKEN = [ False, False, True ]
cfg.MODEL.BACKBONE.POS_EMBED = [ False, False, False ]
cfg.MODEL.BACKBONE.QKV_PROJ_METHOD = [ 'dw_bn', 'dw_bn', 'dw_bn' ]
cfg.MODEL.BACKBONE.KERNEL_QKV = [ 3, 3, 3 ]
cfg.MODEL.BACKBONE.PADDING_KV = [ 1, 1, 1 ]
cfg.MODEL.BACKBONE.STRIDE_KV = [ 2, 2, 2 ]
cfg.MODEL.BACKBONE.PADDING_Q = [ 1, 1, 1 ]
cfg.MODEL.BACKBONE.STRIDE_Q = [ 1, 1, 1 ]
cfg.MODEL.BACKBONE.FREEZE_BN = True
cfg.MODEL.PRETRAINED_STAGE1 = False
cfg.MODEL.PRETRAINED_MIXFORMER = True
cfg.MODEL.NLAYER_HEAD = 3
cfg.MODEL.HEAD_FREEZE_BN = False

# MODEL.DIMP
cfg.MODEL.DIMP = edict()
cfg.MODEL.DIMP.FILTER_SIZE = 4 #filter_size
cfg.MODEL.DIMP.OPTIM_ITER = 5 #optim_iter
cfg.MODEL.DIMP.OPTIM_INIT_STEP = 0.9 #optim_init_step
cfg.MODEL.DIMP.OPTIM_INIT_REG = 0.1 #optim_init_reg
cfg.MODEL.DIMP.FEAT_STRIDE = 16 #feat_stride
cfg.MODEL.DIMP.CLF_FEAT_BLOCKS = 4 #clf_feat_blocks
cfg.MODEL.DIMP.CLF_FEAT_NORM = True #clf_feat_norm
cfg.MODEL.DIMP.INIT_FILTER_NORM = False #init_filter_norm
cfg.MODEL.DIMP.FINAL_CONV = True #final_conv
cfg.MODEL.DIMP.OUT_FEATURE_DIM = 512 #out_feature_dim
cfg.MODEL.DIMP.INIT_GAUSS_SIGMA = 1.0 #init_gauss_sigma
cfg.MODEL.DIMP.NUM_DIST_BINS = 100 #num_dist_bins
cfg.MODEL.DIMP.BIN_DISPLACEMENT = 0.1 #bin_displacement
cfg.MODEL.DIMP.MASK_INIT_FACTOR = 3.0 #mask_init_factor
cfg.MODEL.DIMP.SCORE_ACT = 'relu' #score_act
cfg.MODEL.DIMP.ACT_PARAM = None #act_param
cfg.MODEL.DIMP.TARGET_MASK_ACT = 'sigmoid' #target_mask_act
cfg.MODEL.DIMP.DETACH_LENGTH = float('Inf') #detach_length
cfg.MODEL.DIMP.BACKBONE_OUTDIM = [192, 768, 1024]
cfg.MODEL.DIMP.MERGW_LAYER = [-1]
cfg.MODEL.DIMP.TARGET_FILTER_SZ = 4 #target_filter_sz
cfg.MODEL.DIMP.OUTPUT_SIGMA_FACTOR = 1/4 #output_sigma_factor
cfg.MODEL.DIMP.HINGE_THRESHOLD = 0.05 #hinge_threshold
cfg.MODEL.DIMP.ROBUST_FILTER = True #robust_filter
cfg.MODEL.DIMP.DROP_PROB = 0.3 #drop_prob

#MODEL.DFCOS
cfg.MODEL.DFCOS = edict()
cfg.MODEL.DFCOS.IN_CHANNELS = [192, 768, 1024]
cfg.MODEL.DFCOS.MID_CHANNELS = 1024
cfg.MODEL.DFCOS.NUM_SHARE_CONVS = 5
cfg.MODEL.DFCOS.NUM_CONVS = 7
cfg.MODEL.DFCOS.TOTAL_STRIDE = 16
cfg.MODEL.DFCOS.INPUT_SIZE_ADAPT = False
cfg.MODEL.DFCOS.MERGW_LAYER = [-1]

#MODEL.REPPOINT
cfg.MODEL.REPPOINT = edict()
cfg.MODEL.REPPOINT.NUM_POINTS = 9 #num_points
cfg.MODEL.REPPOINT.STACKED_CONVS = 5 #stacked_convs
cfg.MODEL.REPPOINT.IN_CHANNELS = [192, 768, 1024] #in_channels
cfg.MODEL.REPPOINT.FEAT_CHANNELS = 512 #feat_channels
cfg.MODEL.REPPOINT.POINT_FEAT_CHANNELS = 512 #point_feat_channels
cfg.MODEL.REPPOINT.CLS_OUT_CHANNELS = 1 #cls_out_channels
cfg.MODEL.REPPOINT.TRANSFORM_METHOD = "moment" #transform_method
cfg.MODEL.REPPOINT.MOMENT_MUL = 0.01 #moment_mul
cfg.MODEL.REPPOINT.GRADIENT_MUL = 0.1 #gradient_mul
cfg.MODEL.REPPOINT.OFFSET = 0.5 #offset
cfg.MODEL.REPPOINT.STRIDE = 16 #stride
cfg.MODEL.REPPOINT.MERGW_LAYER = [-1]
cfg.MODEL.REPPOINT.INIT_POS_NUM = 1
cfg.MODEL.REPPOINT.POS_IOU = 0.5
cfg.MODEL.REPPOINT.NEG_IOU = 0.4



# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.SCORE_WEIGHT = 1.0
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.DEEP_SUPERVISION = False
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
cfg.TRAIN.GRAD_CLIP_NORM = 0.1

cfg.TRAIN.TRAIN_SCORE = False
cfg.TRAIN.FREEZE_STAGE0 = False
cfg.TRAIN.FREEZE_MIXFORMER = False
cfg.TRAIN.FREEZE_DIMP = False
cfg.TRAIN.FREEZE_FOR_REPPOINT = True
# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1

# DATA
cfg.DATA = edict()
cfg.DATA.SAMPLER_MODE = 'trident_pro'
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["GOT10K_vottrain"]#["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = ["GOT10K_votval"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 320
cfg.DATA.SEARCH.FACTOR = 5.0
cfg.DATA.SEARCH.NUMBER = 1
cfg.DATA.SEARCH.CENTER_JITTER = 4.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
cfg.TEST.SEARCH_FACTOR = 5.0
cfg.TEST.SEARCH_SIZE = 320
cfg.TEST.EPOCH = 500
cfg.TEST.UPDATE_INTERVALS = edict()
cfg.TEST.UPDATE_INTERVALS.LASOT = [200]
cfg.TEST.UPDATE_INTERVALS.GOT10K_TEST = [200]
cfg.TEST.UPDATE_INTERVALS.TRACKINGNET = [200]
cfg.TEST.UPDATE_INTERVALS.VOT20 = [200]
cfg.TEST.UPDATE_INTERVALS.VOT20LT = [200]
cfg.TEST.UPDATE_INTERVALS.OTB = [200]
cfg.TEST.UPDATE_INTERVALS.UAV = [200]

cfg.TEST.ONLINE_SIZES= edict()
cfg.TEST.ONLINE_SIZES.LASOT = [3]
cfg.TEST.ONLINE_SIZES.GOT10K_TEST = [3]
cfg.TEST.ONLINE_SIZES.TRACKINGNET = [3]
cfg.TEST.ONLINE_SIZES.VOT20 = [3]
cfg.TEST.ONLINE_SIZES.VOT20LT = [3]
cfg.TEST.ONLINE_SIZES.OTB = [3]
cfg.TEST.ONLINE_SIZES.UAV = [3]


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        _update_config(cfg, exp_config)


