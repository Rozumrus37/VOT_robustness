from lib.test.utils import TrackerParams
import os
from easydict import EasyDict as edict
from lib.test.evaluation.environment import env_settings
from lib.config.mixformer_online_dimprobust_reppoint_deep.config import cfg, update_config_from_file,_update_config

import lib.train.admin.settings as ws_settings
from lib.train.base_functions import update_reppoint_settings,update_dimp_settings


def parameters(yaml_name: str, model=None,mask_model = None, search_area_scale=None):
    params = TrackerParams()

    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../../')
    # update default config from yaml file
    yaml_file = os.path.join(local_path, 'experiments/mixformer_online_dimprobust_reppoint_deep/{}.yaml'.format(yaml_name))
    print(yaml_file)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    params.settings = ws_settings.Settings()
    update_dimp_settings(params.settings, params.cfg)
    update_reppoint_settings(params.settings, params.cfg)



    #=======================================================测试参数
    # template and search region
    params.debug = 0
    params.visualization = False
    params.use_gpu = True
    #========================= Learning parameters
    params.sample_memory_size = 50
    params.learning_rate = 0.01
    params.init_samples_minimum_weight = 0.25
    params.train_skipping = 20
    #========================= Learning parameters

    #========================= Net optimization params
    params.update_classifier = True
    params.net_opt_iter = 10
    params.net_opt_update_iter = 2
    params.net_opt_hn_iter = 1
    #========================= Net optimization params

    #===========================localization parameters
    params.advanced_localization = True
    params.target_not_found_threshold = 0.25
    params.distractor_threshold = 0.8
    params.hard_negative_threshold = 0.5
    params.target_neighborhood_scale = 2.2
    params.dispalcement_scale = 0.8
    params.hard_negative_learning_rate = 0.02
    params.update_scale_when_uncertain = True
    #============================localization parameters

    #============================运动策略
    params.penalty_k = 0.04
    params.online_classification_influence = 0.8
    params.speed_influence = 1.5
    params.window_influence = 0.15
    params.window_influence_fast = 0.03
    params.window_influence_medium = 0.15
    params.window_influence_slow = 0.40
    params.speed_last_calc = 3
    params.lr = 0.85
    params.score_inside_width = 11
    params.score_outside_width = 23
    #============================运动策略

    #============================dimp_init
    params.use_augmentation = True
    params.augmentation_expansion_factor = 2
    params.random_shift_factor = 1/3
    #============================dimp_init

    #============================template
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.augmentation_template = {'fliplr': True,
                                   'rotate': [10, -10, 25, -25, 40, -40, 55, -55],
                                   'blur': [(3,1), (1, 3), (2, 2)],
                                   'dropout': (2, 0.2)}
    #============================template


    #============================search
    if search_area_scale is not None:
        params.search_factor = search_area_scale
        print(params.cfg.TEST.SEARCH_SIZE)
        _update_config(params.cfg, edict({'TEST':edict({'SEARCH_FACTOR':params.search_factor})}))
        _update_config(params.cfg, edict({'TEST':edict({'SEARCH_SIZE':params.search_factor * 64})}))
        print(params.cfg.TEST.SEARCH_SIZE)
    else:
        params.search_factor = cfg.TEST.SEARCH_FACTOR

    print("search_area_scale: {}".format(params.search_factor))
    params.search_size = int(params.search_factor * 64)
    params.search_feature_size = int(params.search_size // cfg.MODEL.DIMP.FEAT_STRIDE)
    params.augmentation_search = {'fliplr': True,
                                   'rotate': [10, -10, 45, -45],
                                   'blur': [(3,1), (1, 3), (2, 2)],
                                   'relativeshift': [(0.6, 0.6), (-0.6, 0.6), (0.6, -0.6), (-0.6,-0.6)],
                                   'dropout': (2, 0.2)}
    #============================search
    #=======================================================测试参数

    #=======================================================加载模型
    if model is None:
        raise NotImplementedError("Please set proper model to test.")
    else:
        params.checkpoint = os.path.join(local_path, "models/%s" % model)

    print(params.checkpoint)
    #========================mask
    if mask_model is not None:
        params.mask_model = os.path.join(local_path, "models/%s" % mask_model)
        params.mask_threshold = 0.6
        params.mask_exemplar_size = 256
        params.mask_instance_size = 256
        print(params.mask_model)
    #========================mask
    #=======================================================加载模型

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
