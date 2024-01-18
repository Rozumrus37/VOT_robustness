from lib.test.tracker.basetracker import BaseTracker
import torch
import torch.nn.functional as F
import math
import time
from lib.train.data.processing_utils import sample_target
# for debug
#=========dimp
from lib.test import dcf, TensorList
from lib.test.features.preprocessing import numpy_to_torch
from lib.test.features.preprocessing import sample_patch_multiscale, sample_patch_transformed
from lib.test.features import augmentation
import lib.train.data.bounding_box_utils as bbutils
from lib.models.target_classifier.initializer import FilterInitializerZero
from lib.models.layers import activation
#=========dimp

import cv2
import os
import numpy as np
from lib.models.mixformer import build_mixformer_online_score_dimp_reppoint
from lib.test.tracker.tracker_utils import Preprocessor_wo_mask
from lib.utils.box_ops import clip_box,compute_iou
from lib.test.tracker.tracker_utils import vis_attn_maps
from lib.utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy

def max2d(a):
    max_val_row, argmax_row = torch.max(a, dim=-2)
    max_val, argmax_col = torch.max(max_val_row, dim=-1)
    argmax_row = argmax_row.view(argmax_col.numel(),-1)[torch.arange(argmax_col.numel()), argmax_col.view(-1)]
    argmax_row = argmax_row.reshape(argmax_col.shape)
    argmax = torch.cat((argmax_row.unsqueeze(-1), argmax_col.unsqueeze(-1)), -1)
    return max_val, argmax


class MixFormerOnline(BaseTracker):
    def __init__(self, params, dataset_name =  "VOT20"):
        super(MixFormerOnline, self).__init__(params)

        network = build_mixformer_online_score_dimp_reppoint(params.cfg, params.settings,  train=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.attn_weights = []

        self.network.box_head.update_param(params.search_feature_size)
        self.network.reppoint_head.update_param(params.search_feature_size)


        self.preprocessor = Preprocessor_wo_mask()
        self.state = None
        # for debug
        self.debug = False
        self.frame_id = 0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        # self.z_dict1 = {}

        # Set the update interval
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
            self.online_sizes = self.cfg.TEST.ONLINE_SIZES[DATASET_NAME]
            self.online_size = self.online_sizes[0]
        else:
            self.update_intervals = self.cfg.DATA.MAX_SAMPLE_INTERVAL
            self.online_size = 3
        self.update_interval = self.update_intervals[0]
        if hasattr(params, 'online_sizes'):
            self.online_size = params.online_sizes
        print("Online size is: ", self.online_size)
        if hasattr(params, 'update_interval'):
            self.update_interval = params.update_interval
        print("Update interval is: ", self.update_interval)
        if hasattr(params, 'max_score_decay'):
            self.max_score_decay = params.max_score_decay
        else:
            self.max_score_decay = 1.0
        if not hasattr(params, 'vis_attn'):
            self.params.vis_attn = 0
        print("max score decay = {}".format(self.max_score_decay))

    def initialize_dimp(self,image,info):
        # Convert image
        im = numpy_to_torch(image)

        # Get target position and size
        state = info['init_bbox']
        self.pos = torch.Tensor([state[1] + (state[3] - 1)/2, state[0] + (state[2] - 1)/2])
        self.target_sz = torch.Tensor([state[3], state[2]])
        # Set sizes
        self.image_sz = torch.Tensor([im.shape[2], im.shape[3]])

        #===================motion
        self.init_motion()
        self.score_size = self.params.search_feature_size
        if self.use_motion_strategy:
            self.score_size_up = 41
        else:
            self.score_size_up = self.score_size
        hanning = np.hanning(self.score_size_up)
        window = np.outer(hanning, hanning)
        self.window = window.reshape(-1)
        #===================motion


        #===================search
        sz = self.params.search_size
        sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
        self.img_sample_sz_search = sz
        self.img_support_sz_search = self.img_sample_sz_search
        # Set search area      self.params.search_factor
        search_area = torch.prod(self.target_sz * self.params.search_factor).item()
        self.target_scale_search =  math.sqrt(search_area) / self.img_sample_sz_search.prod().sqrt()
        # Target size in base scale
        self.base_target_sz_search = self.target_sz / self.target_scale_search
        # Setup scale bounds
        self.min_scale_factor = torch.max(10 / self.base_target_sz_search)
        self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz_search)
        #===================search

        #===================template
        sz = self.params.template_size
        sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
        self.img_sample_sz_template = sz
        self.img_support_sz_template = self.img_sample_sz_template
        # Set template area
        template_area = torch.prod(self.target_sz * self.params.template_factor).item()
        self.target_scale_template =  math.sqrt(template_area) / self.img_sample_sz_template.prod().sqrt()
        # Target size in base scale
        self.base_target_sz_template = self.target_sz / self.target_scale_template
        #===================template

        #===================dimp_score
        # Setup scale factors
        self.params.scale_factors = torch.ones(1)
        # Extract and transform sample
        init_backbone_feat = self.generate_init_feat(im)
        # Initialize classifier
        self.init_classifier(init_backbone_feat)
        #===================dimp_score


    def generate_init_feat(self, im: torch.Tensor):
        im_patches_search = self.preprocessor.process(self.generate_init_search_samples(im))
        im_patches_template = self.preprocessor.process(self.generate_init_template_samples(im))
        with torch.no_grad():
            template_backbone_list,online_template_backbone_list, search_backbone_list = self.network.backbone(im_patches_template, im_patches_template, im_patches_search)
        return template_backbone_list,online_template_backbone_list, search_backbone_list

    def generate_init_template_samples(self, im: torch.Tensor) -> TensorList:
        """Perform data augmentation to generate initial training samples."""
        # mode = self.params.get('border_mode', 'replicate')
        self.init_sample_scale_template = self.target_scale_template
        global_shift = torch.zeros(2)
        self.init_sample_pos_template = self.pos.round()

        # Compute augmentation size
        aug_expansion_factor = self.params.get('augmentation_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz_template.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz_template * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz_template.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz_template.long().tolist()

        # Always put identity transformation first, since it is the unaugmented sample that is always used
        self.transforms_template = [augmentation.Identity(aug_output_sz, global_shift.long().tolist())]
        augs = self.params.augmentation_template if self.params.get('use_augmentation', True) else {}
        # Add all augmentations
        if 'fliplr' in augs and augs['fliplr']:
            self.transforms_template.append(augmentation.FlipHorizontal(aug_output_sz, global_shift.long().tolist()))
        if 'blur' in augs:
            self.transforms_template.extend([augmentation.Blur(sigma, aug_output_sz, global_shift.long().tolist()) for sigma in augs['blur']])
        if 'scale' in augs:
            self.transforms_template.extend([augmentation.Scale(scale_factor, aug_output_sz, global_shift.long().tolist()) for scale_factor in augs['scale']])
        if 'rotate' in augs:
            self.transforms_template.extend([augmentation.Rotate(angle, aug_output_sz, global_shift.long().tolist()) for angle in augs['rotate']])

        # Extract augmented image patches
        im_patches_template = sample_patch_transformed(im, self.init_sample_pos_template, self.init_sample_scale_template, aug_expansion_sz, self.transforms_template)

        # Extract initial backbone features
        # with torch.no_grad():
        #     init_backbone_feat = self.network.extract_backbone(im_patches)
        return im_patches_template

    def generate_init_search_samples(self, im: torch.Tensor) -> TensorList:
        """Perform data augmentation to generate initial training samples."""
        # mode = self.params.get('border_mode', 'replicate')
        self.init_sample_scale_search = self.target_scale_search
        global_shift = torch.zeros(2)

        self.init_sample_pos_search = self.pos.round()

        # Compute augmentation size
        aug_expansion_factor = self.params.get('augmentation_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz_search.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz_search * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz_search.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz_search.long().tolist()

        # Random shift for each sample
        get_rand_shift = lambda: None
        random_shift_factor = self.params.get('random_shift_factor', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: ((torch.rand(2) - 0.5) * self.img_sample_sz_search * random_shift_factor + global_shift).long().tolist()

        # Always put identity transformation first, since it is the unaugmented sample that is always used
        self.transforms_search = [augmentation.Identity(aug_output_sz, global_shift.long().tolist())]

        augs = self.params.augmentation_search if self.params.get('use_augmentation', True) else {}

        # Add all augmentations
        if 'shift' in augs:
            self.transforms_search.extend([augmentation.Translation(shift, aug_output_sz, global_shift.long().tolist()) for shift in augs['shift']])
        if 'relativeshift' in augs:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz_search/2).long().tolist()
            self.transforms_search.extend([augmentation.Translation(get_absolute(shift), aug_output_sz, global_shift.long().tolist()) for shift in augs['relativeshift']])
        if 'fliplr' in augs and augs['fliplr']:
            self.transforms_search.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in augs:
            self.transforms_search.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in augs['blur']])
        if 'scale' in augs:
            self.transforms_search.extend([augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in augs['scale']])
        if 'rotate' in augs:
            self.transforms_search.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in augs['rotate']])

        # Extract augmented image patches
        im_patches = sample_patch_transformed(im, self.init_sample_pos_search, self.init_sample_scale_search, aug_expansion_sz, self.transforms_search)

        # Extract initial backbone features
        # with torch.no_grad():
        #     init_backbone_feat = self.network.extract_backbone(im_patches)
        return im_patches

    def classify_target(self, sample_x):
        """Classify target by applying the DiMP filter."""
        with torch.no_grad():
            scores = self.network.dimp_branch.classifier.classify(self.target_filter, sample_x)
        return scores

    def get_classification_features_track(self, init_backbone_feat):
        with torch.no_grad():
            search_backbone_list = init_backbone_feat
            search = self.network.dimp_branch.dimp_merge_feature_layer(search_backbone_list)
            search_clsfeat = self.network.dimp_branch.classifier.extract_classification_feat(search)
        return search_clsfeat

    def get_classification_features_init(self, init_backbone_feat):
        with torch.no_grad():
            template_backbone_list,online_template_backbone_list, search_backbone_list = init_backbone_feat
            templates_clsfeat, searches_clsfeat = self.network.extract_dimp_feat(template_backbone_list, search_backbone_list)
        return templates_clsfeat, searches_clsfeat

    def _overwrite_classifier_params(self, feature_dim):
        # Overwrite some parameters in the classifier. (These are not generally changed)
        pred_module = getattr(self.net.classifier.filter_optimizer, 'score_predictor', self.net.classifier.filter_optimizer)
        if self.params.get('label_threshold', None) is not None:
            self.net.classifier.filter_optimizer.label_threshold = self.params.label_threshold
        if self.params.get('label_shrink', None) is not None:
            self.net.classifier.filter_optimizer.label_shrink = self.params.label_shrink
        if self.params.get('softmax_reg', None) is not None:
            self.net.classifier.filter_optimizer.softmax_reg = self.params.softmax_reg
        if self.params.get('filter_reg', None) is not None:
            pred_module.filter_reg[0] = self.params.filter_reg
            pred_module.min_filter_reg = self.params.filter_reg
        if self.params.get('filter_init_zero', False):
            self.net.classifier.filter_initializer = FilterInitializerZero(self.net.classifier.filter_size, feature_dim)


    def init_classifier(self, init_backbone_feat):
        # Get classification features
        templates_clsfeat, searches_clsfeat = self.get_classification_features_init(init_backbone_feat)


        # Overwrite some parameters in the classifier. (These are not generally changed)
        # self._overwrite_classifier_params(feature_dim=searches_clsfeat.shape[-3])

        # Add the dropout augmentation here, since it requires extraction of the classification features
        if 'dropout' in self.params.augmentation_search and self.params.get('use_augmentation', True):
            num, prob = self.params.augmentation_search['dropout']
            self.transforms_search.extend(self.transforms_search[:1]*num)
            searches_clsfeat = torch.cat([searches_clsfeat, F.dropout2d(searches_clsfeat[0:1,...].expand(num,-1,-1,-1), p=prob, training=True)])

        if 'dropout' in self.params.augmentation_template and self.params.get('use_augmentation', True):
            num, prob = self.params.augmentation_template['dropout']
            self.transforms_template.extend(self.transforms_template[:1]*num)
            templates_clsfeat = torch.cat([templates_clsfeat, F.dropout2d(templates_clsfeat[0:1,...].expand(num,-1,-1,-1), p=prob, training=True)])


        # Set feature size and other related sizes
        self.feature_sz = torch.Tensor(list(searches_clsfeat.shape[-2:]))
        ksz = self.network.dimp_branch.classifier.filter_size
        self.kernel_size = torch.Tensor([ksz, ksz] if isinstance(ksz, (int, float)) else ksz)
        self.output_sz = self.feature_sz + (self.kernel_size + 1)%2

        # Construct output window
        self.output_window = None
        # if self.params.get('window_output', False):
        #     if self.params.get('use_clipped_window', False):
        #         self.output_window = dcf.hann2d_clipped(self.output_sz.long(), (self.output_sz*self.params.effective_search_area / self.params.search_area_scale).long(), centered=True).to(self.params.device)
        #     else:
        #         self.output_window = dcf.hann2d(self.output_sz.long(), centered=True).to(self.params.device)
        #     self.output_window = self.output_window.squeeze(0)

        # Get target boxes for the different augmentations
        target_boxes_search = self.init_target_boxes_search()
        target_boxes_template = self.init_target_boxes_template()

        # Set number of iterations
        plot_loss = self.params.debug > 0
        num_iter = self.params.get('net_opt_iter', None)
        # Get target filter by running the discriminative model prediction module
        with torch.no_grad():
            self.target_filter, _, losses = self.network.dimp_branch.classifier.get_filter_robust(
                                                                    templates_clsfeat, target_boxes_template,
                                                                    searches_clsfeat, target_boxes_search,
                                                                    False,
                                                                    num_iter=num_iter,compute_losses=plot_loss)
            # self.target_filter, _, losses = self.network.classifier.get_filter(x, target_boxes, num_iter=num_iter,
            #                                                                compute_losses=plot_loss)
        # Init memory
        if self.params.get('update_classifier', True):
            self.init_memory(TensorList([searches_clsfeat]))

        if plot_loss:
            if isinstance(losses, dict):
                losses = losses['train']
            self.losses = torch.cat(losses)
            if self.visdom is not None:
                self.visdom.register((self.losses, torch.arange(self.losses.numel())), 'lineplot', 3, 'Training Loss' + self.id_str)
            elif self.params.debug >= 3:
                plot_graph(self.losses, 10, title='Training Loss' + self.id_str)


    def init_target_boxes_search(self):
        """Get the target bounding boxes for the initial augmented samples."""
        self.classifier_target_box_search = self.get_iounet_box_search(self.pos, self.target_sz, self.init_sample_pos_search, self.init_sample_scale_search)
        init_target_boxes_search = TensorList()
        for T in self.transforms_search:
            init_target_boxes_search.append(self.classifier_target_box_search + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))

        init_target_boxes_search = torch.cat(init_target_boxes_search.view(1, 4), 0).to(self.params.device)
        self.target_boxes_search = init_target_boxes_search.new_zeros(self.params.sample_memory_size, 4)
        self.target_boxes_search[:init_target_boxes_search.shape[0],:] = init_target_boxes_search
        return init_target_boxes_search

    def init_target_boxes_template(self):
        """Get the target bounding boxes for the initial augmented samples."""
        self.classifier_target_box_template = self.get_iounet_box_template(self.pos, self.target_sz, self.init_sample_pos_template, self.init_sample_scale_template)
        init_target_boxes_template = TensorList()
        for T in self.transforms_template:
            init_target_boxes_template.append(self.classifier_target_box_template + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        init_target_boxes_template = torch.cat(init_target_boxes_template.view(1, 4), 0).to(self.params.device)
        self.target_boxes_template = init_target_boxes_template.new_zeros(self.params.sample_memory_size, 4)
        self.target_boxes_template[:init_target_boxes_template.shape[0],:] = init_target_boxes_template
        return init_target_boxes_template

    def get_iounet_box_search(self, pos, sz, sample_pos, sample_scale):
        """All inputs in original image coordinates.
        Generates a box in the cropped image sample reference frame, in the format used by the IoUNet."""
        box_center = (pos - sample_pos) / sample_scale + (self.img_sample_sz_search - 1) / 2
        box_sz = sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        return torch.cat([target_ul.flip((0,)), box_sz.flip((0,))])

    def get_iounet_box_template(self, pos, sz, sample_pos, sample_scale):
        """All inputs in original image coordinates.
        Generates a box in the cropped image sample reference frame, in the format used by the IoUNet."""
        box_center = (pos - sample_pos) / sample_scale + (self.img_sample_sz_template - 1) / 2
        box_sz = sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        return torch.cat([target_ul.flip((0,)), box_sz.flip((0,))])

    def update_memory_hard(self, sample_x: TensorList, target_box, learning_rate = None):

        # Update sample and label memory
        self.training_samples_hard = TensorList(
            [x.new_zeros(1, x.shape[1], x.shape[2], x.shape[3]) for x in sample_x])
        for train_samp, x in zip(self.training_samples_hard, sample_x):
            train_samp[0,...] = x

        # Update bb memory
        tmp = torch.ones(1,4).to(self.params.device)
        self.target_boxes_search_hard = tmp.new_zeros(1, 4)
        self.target_boxes_search_hard[0,:] = target_box

        # Update weight
        self.sample_weights_hard = TensorList([x.new_zeros(1) for x in sample_x])
        temp_weights = TensorList([x.new_ones(1) / x.shape[0] for x in sample_x])
        for sw, init_sw in zip(self.sample_weights_hard, temp_weights):
            sw[:1] = init_sw


    def init_memory(self, train_x: TensorList):
        # Initialize first-frame spatial training samples
        self.num_init_samples = train_x.size(0)
        init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])

        # Sample counters and weights for spatial
        self.num_stored_samples = self.num_init_samples.copy()
        self.previous_replace_ind = [None] * len(self.num_stored_samples)
        self.sample_weights = TensorList([x.new_zeros(self.params.sample_memory_size) for x in train_x])
        for sw, init_sw, num in zip(self.sample_weights, init_sample_weights, self.num_init_samples):
            sw[:num] = init_sw

        # Initialize memory
        self.training_samples = TensorList(
            [x.new_zeros(self.params.sample_memory_size, x.shape[1], x.shape[2], x.shape[3]) for x in train_x])

        for ts, x in zip(self.training_samples, train_x):
            ts[:x.shape[0],...] = x


    def update_memory(self, sample_x: TensorList, target_box, learning_rate = None):
        # Update weights and get replace ind
        replace_ind = self.update_sample_weights(self.sample_weights, self.previous_replace_ind, self.num_stored_samples, self.num_init_samples, learning_rate)
        self.previous_replace_ind = replace_ind

        # Update sample and label memory
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind+1,...] = x

        # Update bb memory
        self.target_boxes_search[replace_ind[0],:] = target_box

        self.num_stored_samples += 1

    def update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, learning_rate = None):
        # Update weights and get index to replace
        replace_ind = []
        for sw, prev_ind, num_samp, num_init in zip(sample_weights, previous_replace_ind, num_stored_samples, num_init_samples):
            lr = learning_rate
            if lr is None:
                lr = self.params.learning_rate

            init_samp_weight = self.params.get('init_samples_minimum_weight', None)
            if init_samp_weight == 0:
                init_samp_weight = None
            s_ind = 0 if init_samp_weight is None else num_init

            if num_samp == 0 or lr == 1:
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace
                if num_samp < sw.shape[0]:
                    r_ind = num_samp
                else:
                    _, r_ind = torch.min(sw[s_ind:], 0)
                    r_ind = r_ind.item() + s_ind

                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr
                    sw[r_ind] = lr
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)

            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init].sum() < init_samp_weight:
                sw /= init_samp_weight + sw[num_init:].sum()
                sw[:num_init] = init_samp_weight / num_init

            replace_ind.append(r_ind)

        return replace_ind


    def initialize(self, image, info: dict, use_motion_strategy = True):
        #==============是否使用策略
        self.use_motion_strategy = use_motion_strategy
        #==============是否使用策略

        # forward the template once
        z_patch_arr, _, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        if self.params.vis_attn==1:
            self.z_patch = z_patch_arr
            self.oz_patch = z_patch_arr
        template = self.preprocessor.process(z_patch_arr)
        self.template = template
        self.online_template = template
        if self.online_size > 1:
            with torch.no_grad():
                self.network.set_online(self.template, self.online_template)

        self.online_state = info['init_bbox']

        self.online_image = image
        self.max_pred_score = -1.0
        self.online_max_template = template
        self.online_forget_id = 0

        # save state
        self.state = info['init_bbox']
        self.frame_id = 0

        #===============init_dimp
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'
        self.initialize_dimp(image,info)

        self.use_corner = True
        #===============init_dimp

        #==============判断是否为有语义的目标
        self.run_first_frame(image, info)
        #==============判断是否为有语义的目标




        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def extract_backbone_features(self, image):
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr.copy())
        with torch.no_grad():
            _, search_backbone_list = self.network.backbone.forward_test(search)
        return search_backbone_list, resize_factor, x_patch_arr

    def get_corner_results(self, search_backbone_list, run_score_head=True):
        with torch.no_grad():
            search = search_backbone_list[-1]
            out, outputs_coord_new = self.network.forward_head(search, self.network.backbone.template, run_score_head)
        return out, outputs_coord_new

    def get_reppoint_results(self, search_backbone_list):
        with torch.no_grad():
            search = self.network.reppoint_head.reppoint_merge_feature_layer(search_backbone_list)
            cls_out,pts_out_init,pts_out_refine = self.network.reppoint_head(search)

        out = {'reppoint_cls': cls_out.sigmoid(), # [num_image*bs_raw,h*w]
                'reppoint_init_bboxes': pts_out_init,# [num_image*bs_raw,h*w,4]
                'reppoint_refine_bboxes': pts_out_refine,# [num_image*bs_raw,h*w,4]
                }
        return out

    def get_bbox_penalty(self,bbox):
        def change(r):
            return np.maximum(r, 1. / r)
        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))
        # scale penalty
        s_c = change(sz((bbox[:, 2]-bbox[:, 0]), (bbox[:, 3]-bbox[:, 1])) /
                     (sz(self.state[2]/self.target_scale_search, self.state[3]/self.target_scale_search)))
        # aspect ratio penalty
        r_c = change((self.state[2]/self.state[3]) /
                     ((bbox[:, 2]-bbox[:, 0])/(bbox[:, 3]-bbox[:, 1])))

        penalty = np.exp(-(r_c * s_c - 1) * self.params.penalty_k)
        return penalty

    def configure_speed(self,before = True,bbox = None):
        if before:
            if self.speed_x > self.params.speed_influence * self.state[2] or self.speed_y > self.params.speed_influence * self.state[3] or \
                self.speed > self.params.speed_influence * max(self.state[2:]):
                self.params.window_influence = self.params.window_influence_fast
            elif self.speed_x > self.state[2] or self.speed_y > self.state[3] or self.speed > max(self.state[2:]):
                self.params.window_influence = self.params.window_influence_medium
            else:
                self.params.window_influence = self.params.window_influence_slow
        else:
            self.dist.append(math.sqrt(bbox[0]**2 + bbox[1]**2))
            self.dist_x.append(np.abs(bbox[0]))
            self.dist_y.append(np.abs(bbox[1]))

            if len(self.dist) < self.params.speed_last_calc:
                self.speed = max(self.dist)
                self.speed_x = max(self.dist_x)
                self.speed_y = max(self.dist_y)
            else:
                self.speed = max(self.dist[-self.params.speed_last_calc:])
                self.speed_x = max(self.dist_x[-self.params.speed_last_calc:])
                self.speed_y = max(self.dist_y[-self.params.speed_last_calc:])

    def map_box_back_of_fcos(self,bbox,lr,shape):
        cx = self.state[0] + self.state[2]/2 + bbox[0]
        cy = self.state[1] + self.state[3]/2 + bbox[1]

        if self.semantics:
            width = self.state[2] * (1 - lr) + bbox[2] * lr
            height = self.state[3] * (1 - lr) + bbox[3] * lr
        else:
            lr = 0.001
            width = self.state[2] * (1 - lr) + bbox[2] * lr
            height = self.state[3] * (1 - lr) + bbox[3] * lr

        # width = bbox[2]
        # height = bbox[3]

        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, shape[:2])
        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        return bbox

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def run_first_frame(self, image, info: dict = None):
        H, W, _ = image.shape
        #================================backbone
        search_backbone_list, resize_factor, x_patch_arr = self.extract_backbone_features(image)
        #================================backbone
        # with torch.no_grad():
        #     out_dict, _ = self.network.forward_test(search, run_score_head=True)
        #===============================corner
        out_corner, _ = self.get_corner_results(search_backbone_list, run_score_head=True)
        pred_boxes_corner = out_corner['pred_boxes'].view(-1, 4)
        pred_score_corner = out_corner['pred_scores'].view(1).sigmoid().item()
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box_corner = (pred_boxes_corner.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        pred_box_corner_fin = clip_box(self.map_box_back(pred_box_corner, resize_factor), H, W, margin=10)
        #===============================corner
        gt_bbox = np.array(info['init_bbox'])
        iou = compute_iou(pred_box_corner_fin.copy(), gt_bbox.copy())
        if iou < 0.5:
            self.semantics = False
        else:
            self.semantics = True

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        self.frame_num = self.frame_id

        #================================backbone
        search_backbone_list, resize_factor, x_patch_arr = self.extract_backbone_features(image)
        self.target_scale_search = 1/resize_factor
        #================================backbone
        # with torch.no_grad():
        #     out_dict, _ = self.network.forward_test(search, run_score_head=True)
        #===============================corner
        out_corner, _ = self.get_corner_results(search_backbone_list, run_score_head=True)
        pred_boxes_corner = out_corner['pred_boxes'].view(-1, 4)
        pred_score_corner = out_corner['pred_scores'].view(1).sigmoid().item()
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box_corner = (pred_boxes_corner.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        pred_box_corner_fin = clip_box(self.map_box_back(pred_box_corner, resize_factor), H, W, margin=10)
        #===============================corner

        # #===============================get_multi_peak_proposals
        # proposals,peakmaxsocres, prob_vec_main_tl,prob_vec_main_br,prob_vec_second_tl,prob_vec_second_br = self.get_proposals(resize_factor,H, W)
        # #===============================get_multi_peak_proposals

        #=================================dimp_pos
        sample_pos = self.pos.unsqueeze(0)
        sample_scales = torch.Tensor([self.target_scale_search])
        #=================================dimp_pos

        #===============================dimp & reppoint
        search_clsfeat = self.get_classification_features_track(search_backbone_list)
        scores_raw = self.classify_target(search_clsfeat)
        translation_vec, scale_ind, s, flag = self.localize_advanced(scores_raw, sample_pos, sample_scales)
        out_reppoint = self.get_reppoint_results(search_backbone_list)


        scores_match = cv2.resize(s.squeeze().cpu().detach().numpy(),(self.score_size,self.score_size)).flatten()
        # scores_match = np.clip(scores_match, 0, 0.999)
        scores_siamese = out_reppoint['reppoint_cls'].cpu().detach().numpy().reshape(-1,1).flatten()
        pred_boxes_reppoint = out_reppoint['reppoint_refine_bboxes'].cpu().detach().numpy().reshape(-1,4)
        #===============================dimp & reppoint

        # #===============================score_merge
        # self.score_size_up = self.score_size
        # pscore_final = scores_siamese * (1 - self.params.online_classification_influence) + scores_match * self.params.online_classification_influence
        # box = self.motion_strategy_simple(pscore_final.copy(), pred_boxes_reppoint)
        # bbox = box / resize_factor
        # pred_box_reppoint_fin = clip_box(self.map_box_back(bbox, resize_factor), H, W, margin=10)
        # #===============================score_merge

        #===============================score_merge
        scores_match = cv2.resize(scores_match.reshape(self.score_size,self.score_size),(self.score_size_up,self.score_size_up)).reshape(-1)
        scores_siamese = cv2.resize(scores_siamese.reshape(self.score_size,self.score_size),(self.score_size_up,self.score_size_up)).reshape(-1)

        penalty = self.get_bbox_penalty(pred_boxes_reppoint)
        penalty = cv2.resize(penalty.reshape(self.score_size,self.score_size),(self.score_size_up,self.score_size_up)).reshape(-1)

        pscore = penalty * scores_siamese
        pscore_final = pscore * (1 - self.params.online_classification_influence) + scores_match * self.params.online_classification_influence
        self.configure_speed(before = True)
        if self.use_motion_strategy:
            box, best_score, lr, final_lr_score, best_idx, pscore_final = self.motion_strategy(pscore_final.copy(), pred_boxes_reppoint,center_pos = [self.pos[1],self.pos[0]], scale_z = resize_factor)
        else:
            box, best_score, lr, final_lr_score, best_idx, pscore_final = self.motion_strategy_simple(pscore_final.copy(), pred_boxes_reppoint)
        bbox = box / resize_factor
        self.configure_speed(before = False,bbox = bbox)
        pred_box_reppoint_fin = self.map_box_back_of_fcos(bbox,lr,image.shape)
        #===============================score_merge

        # #===============================vis
        # self.vis_img_mixformer = x_patch_arr.copy()
        # self.vis_pred_box_corner_fin = np.array(pred_box_corner_fin).copy()
        # self.vis_pred_box_reppoint_fin = np.array(pred_box_reppoint_fin).copy()
        # self.vis_center_mixformer = [self.state[0] + 0.5 * self.state[2],self.state[1] + 0.5 * self.state[3]]
        # self.vis_scale_mixformer = resize_factor
        # self.vis_heatmaps_mixformer = [self.network.box_head.vis_score_tl.mean(axis=0),self.network.box_head.vis_score_br.mean(axis=0)]
        # self.vis_scores_mixformer = [scores_siamese.reshape(self.score_size_up,self.score_size_up),scores_match.reshape(self.score_size_up,self.score_size_up),pscore_final.reshape(self.score_size_up,self.score_size_up)]
        # self.vis_peaksocres_mixformer = [prob_vec_main_tl.mean(axis=0),prob_vec_main_br.mean(axis=0),prob_vec_second_tl.mean(axis=0),prob_vec_second_br.mean(axis=0)]
        # self.vis_peakmaxsocres_mixformer = peakmaxsocres
        # self.vis_proposals_mixformer = proposals
        # self.vis_update_mixformer = 'N'
        # self.vis_update_dimp = 'N'
        # self.vis_dimp_flag = flag
        # self.vis_discriminate_score = pred_score_corner
        # self.vis_use_corner = 'N'
        # #===============================vis

        #===============================result decision making
        pred_bbox = pred_box_reppoint_fin
        update_dimp_hard = False
        iou = compute_iou(pred_box_corner_fin.copy(), pred_box_reppoint_fin.copy())
        if iou < 0.001:
            update_dimp_hard = True


        self.state = pred_bbox
        self.pos = torch.Tensor([self.state[1] + (self.state[3] - 1)/2, self.state[0] + (self.state[2] - 1)/2])
        self.target_sz = torch.Tensor([self.state[3], self.state[2]])
        #===============================result decision making
        # self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        #================================update dimp=====================================
        update_flag = flag not in ['not_found', 'uncertain']
        hard_negative = (flag == 'hard_negative')
        learning_rate = self.params.get('hard_negative_learning_rate', None) if hard_negative else None
        if update_flag and self.params.get('update_classifier', False):
            self.vis_update_dimp = 'Y'
            # Get train sample
            train_x = search_clsfeat[scale_ind:scale_ind+1, ...]
            # Create target_box and label for spatial sample
            target_box = self.get_iounet_box_search(self.pos, self.target_sz, sample_pos[scale_ind,:], sample_scales[scale_ind])
            # Update the classifier model
            self.update_classifier(train_x, target_box, learning_rate, s[scale_ind,...])

        if update_dimp_hard or hard_negative:
            # Get train sample
            train_x = search_clsfeat[scale_ind:scale_ind+1, ...]
            # Create target_box and label for spatial sample
            target_box = self.get_iounet_box_search(self.pos, self.target_sz, sample_pos[scale_ind,:], sample_scales[scale_ind])
            # Update the classifier model
            self.update_classifier_hard(train_x, target_box, learning_rate, s[scale_ind,...])
        #================================update dimp=====================================

        # #================================update mixformer template========================
        # self.max_pred_score = self.max_pred_score * self.max_score_decay
        # # update template
        # if pred_score_corner > 0.5 and pred_score_corner > self.max_pred_score:
        #     z_patch_arr, _, z_amask_arr = sample_target(image, self.state,
        #                                                 self.params.template_factor,
        #                                                 output_sz=self.params.template_size)  # (x1, y1, w, h)
        #     self.online_max_template = self.preprocessor.process(z_patch_arr)
        #     if self.params.vis_attn == 1:
        #         self.oz_patch_max = z_patch_arr
        #     self.max_pred_score = pred_score_corner
        #     self.vis_update_mixformer = 'Y'
        # if self.frame_id % self.update_interval == 0:
        #     if self.online_size == 1:
        #         self.online_template = self.online_max_template
        #         if self.params.vis_attn == 1:
        #             self.oz_patch = self.oz_patch_max
        #     elif self.online_template.shape[0] < self.online_size:
        #         self.online_template = torch.cat([self.online_template, self.online_max_template])
        #     else:
        #         self.online_template[self.online_forget_id:self.online_forget_id+1] = self.online_max_template
        #         self.online_forget_id = (self.online_forget_id + 1) % self.online_size

        #     if self.online_size > 1:
        #         with torch.no_grad():
        #             self.network.set_online(self.template, self.online_template)

        #     self.max_pred_score = -1
        #     self.online_max_template = self.template
        # #================================update mixformer template========================

        # for debug
        if self.debug:
            x1, y1, w, h = self.state
            # image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
            cv2.imwrite(save_path, image)
        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}


    def update_classifier_hard(self, train_x, target_box, learning_rate=None, scores=None):
        # Set flags and learning rate
        learning_rate = self.params.learning_rate
        # Update the tracker memory
        self.update_memory_hard(TensorList([train_x]), target_box, learning_rate)
        # Decide the number of iterations to run
        num_iter = 2

        # Get inputs for the DiMP filter optimizer module
        samples = self.training_samples_hard[0][:1,...]
        target_boxes_search = self.target_boxes_search_hard[:1,:].clone()
        sample_weights = self.sample_weights_hard[0][:1]

        # Run the filter optimizer module
        with torch.no_grad():
            self.target_filter, _, losses = self.network.dimp_branch.classifier.filter_optimizer(self.target_filter,
                                                                                 num_iter=num_iter, feat=samples,
                                                                                 bb=target_boxes_search,
                                                                                 sample_weight=sample_weights,
                                                                                 compute_losses=False)

    def update_classifier(self, train_x, target_box, learning_rate=None, scores=None):
        # Set flags and learning rate
        hard_negative_flag = learning_rate is not None
        if learning_rate is None:
            learning_rate = self.params.learning_rate

        # Update the tracker memory
        if hard_negative_flag or self.frame_num % self.params.get('train_sample_interval', 1) == 0:
            self.update_memory(TensorList([train_x]), target_box, learning_rate)

        # Decide the number of iterations to run
        num_iter = 0
        low_score_th = self.params.get('low_score_opt_threshold', None)
        if hard_negative_flag:
            num_iter = self.params.get('net_opt_hn_iter', None)
        elif low_score_th is not None and low_score_th > scores.max().item():
            num_iter = self.params.get('net_opt_low_iter', None)
        elif (self.frame_num - 1) % self.params.train_skipping == 0:
            num_iter = self.params.get('net_opt_update_iter', None)

        plot_loss = self.params.debug > 0

        if num_iter > 0:
            # Get inputs for the DiMP filter optimizer module
            samples = self.training_samples[0][:self.num_stored_samples[0],...]
            target_boxes_search = self.target_boxes_search[:self.num_stored_samples[0],:].clone()
            sample_weights = self.sample_weights[0][:self.num_stored_samples[0]]

            # Run the filter optimizer module
            with torch.no_grad():
                self.target_filter, _, losses = self.network.dimp_branch.classifier.filter_optimizer(self.target_filter,
                                                                                     num_iter=num_iter, feat=samples,
                                                                                     bb=target_boxes_search,
                                                                                     sample_weight=sample_weights,
                                                                                     compute_losses=plot_loss)

            if plot_loss:
                if isinstance(losses, dict):
                    losses = losses['train']
                self.losses = torch.cat((self.losses, torch.cat(losses)))
                if self.visdom is not None:
                    self.visdom.register((self.losses, torch.arange(self.losses.numel())), 'lineplot', 3, 'Training Loss' + self.id_str)
                elif self.params.debug >= 3:
                    plot_graph(self.losses, 10, title='Training Loss' + self.id_str)




    def split_peak(self, scores, resize_factor):
        """Run the target advanced localization (as in ATOM)."""
        min_value = torch.min(scores).item()
        sz = scores.shape[-2:]
        score_sz = torch.Tensor(list(sz))
        output_sz = score_sz
        score_center = (score_sz - 1)/2

        scores_hn = scores.clone()

        target_neigh_sz = 6 #* (np.array([self.state[3],self.state[2]]) * resize_factor) * (output_sz / self.params.search_size)

        # Mask main_peak
        max_score1, max_disp1 = max2d(scores)
        max_score1 = max_score1[0][0].item()
        tneigh_top = max(round(max_disp1[0][0][0].item() - target_neigh_sz / 2), 0)
        tneigh_bottom = min(round(max_disp1[0][0][0].item() + target_neigh_sz / 2 + 1), sz[0])
        tneigh_left = max(round(max_disp1[0][0][1].item() - target_neigh_sz / 2), 0)
        tneigh_right = min(round(max_disp1[0][0][1].item() + target_neigh_sz / 2 + 1), sz[1])
        scores_masked_main_peak = scores_hn.clone()
        scores_masked_main_peak[...,tneigh_top:tneigh_bottom,tneigh_left:tneigh_right] = min_value

        # Mask second_peak
        max_score2, max_disp2 = max2d(scores_masked_main_peak)
        max_score2 = max_score2[0][0].item()
        tneigh_top = max(round(max_disp2[0][0][0].item() - target_neigh_sz / 2), 0)
        tneigh_bottom = min(round(max_disp2[0][0][0].item() + target_neigh_sz / 2 + 1), sz[0])
        tneigh_left = max(round(max_disp2[0][0][1].item() - target_neigh_sz / 2), 0)
        tneigh_right = min(round(max_disp2[0][0][1].item() + target_neigh_sz / 2 + 1), sz[1])
        scores_masked_second_peak = scores_hn.clone()
        scores_masked_second_peak[...,tneigh_top:tneigh_bottom,tneigh_left:tneigh_right] = min_value

        if max_score1 - max_score2 > 5:
            return scores_hn, scores_hn, max_score1, max_score2
        return scores_masked_main_peak, scores_masked_second_peak, max_score1, max_score2

    def get_proposals(self,resize_factor,H, W):
        score_map_tl = self.network.box_head.score_map_tl
        score_map_br = self.network.box_head.score_map_br
        scores_second_peak_tl, scores_main_peak_tl, max_score_main_tl, max_score_second_tl = self.split_peak(score_map_tl,resize_factor)
        scores_second_peak_br, scores_main_peak_br, max_score_main_br, max_score_second_br = self.split_peak(score_map_br,resize_factor)

        peakmaxsocres = [max_score_main_tl,max_score_main_br,max_score_second_tl,max_score_second_br]
        if max_score_main_tl - max_score_second_tl > 5 and max_score_main_br - max_score_second_br > 5:
            self.is_multi_peak = 'N'
        else:
            self.is_multi_peak = 'Y'
        #main
        coorx_main_tl, coory_main_tl, prob_vec_main_tl = self.network.box_head.soft_argmax(scores_main_peak_tl, return_dist=True, softmax=True)
        coorx_main_br, coory_main_br, prob_vec_main_br = self.network.box_head.soft_argmax(scores_main_peak_br, return_dist=True, softmax=True)
        #second
        coorx_second_tl, coory_second_tl, prob_vec_second_tl = self.network.box_head.soft_argmax(scores_second_peak_tl, return_dist=True, softmax=True)
        coorx_second_br, coory_second_br, prob_vec_second_br = self.network.box_head.soft_argmax(scores_second_peak_br, return_dist=True, softmax=True)


        prob_vec_main_tl = prob_vec_main_tl.detach().cpu().numpy().reshape(-1,self.network.box_head.feat_sz,self.network.box_head.feat_sz)
        prob_vec_main_br = prob_vec_main_br.detach().cpu().numpy().reshape(-1,self.network.box_head.feat_sz,self.network.box_head.feat_sz)
        prob_vec_second_tl = prob_vec_second_tl.detach().cpu().numpy().reshape(-1,self.network.box_head.feat_sz,self.network.box_head.feat_sz)
        prob_vec_second_br = prob_vec_second_br.detach().cpu().numpy().reshape(-1,self.network.box_head.feat_sz,self.network.box_head.feat_sz)


        one = box_xyxy_to_cxcywh(torch.stack((coorx_main_tl, coory_main_tl, coorx_main_br, coory_main_br), dim=1) / self.network.box_head.img_sz).detach().cpu().numpy().reshape(-1)
        two = box_xyxy_to_cxcywh(torch.stack((coorx_second_tl, coory_second_tl, coorx_second_br, coory_second_br), dim=1) / self.network.box_head.img_sz).detach().cpu().numpy().reshape(-1)
        three = box_xyxy_to_cxcywh(torch.stack((coorx_second_tl, coory_second_tl, coorx_main_br, coory_main_br), dim=1) / self.network.box_head.img_sz).detach().cpu().numpy().reshape(-1)
        four = box_xyxy_to_cxcywh(torch.stack((coorx_second_tl, coory_second_tl, coorx_main_br, coory_main_br), dim=1) / self.network.box_head.img_sz).detach().cpu().numpy().reshape(-1)

        one = one * self.params.search_size / resize_factor
        two = two * self.params.search_size / resize_factor
        three = three * self.params.search_size / resize_factor
        four = four * self.params.search_size / resize_factor

        one = np.array(clip_box(self.map_box_back(one, resize_factor), H, W, margin=10))
        two = np.array(clip_box(self.map_box_back(two, resize_factor), H, W, margin=10))
        three = np.array(clip_box(self.map_box_back(three, resize_factor), H, W, margin=10))
        four = np.array(clip_box(self.map_box_back(four, resize_factor), H, W, margin=10))


        proposals = np.stack([one,two,three,four],0)
        return proposals,peakmaxsocres, prob_vec_main_tl,prob_vec_main_br,prob_vec_second_tl,prob_vec_second_br


    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


    def init_motion(self):
        self.dist = []
        self.dist_x = []
        self.dist_y = []
        self.speed = 0.0
        self.speed_x = 0.0
        self.speed_y = 0.0
        self.disturbance_close_to_target=False
        self.disturbance_away_from_target=False
        self.disturbance_in_target=False
        self.previous_t_d_distance = 0
        self.inside_target_pos_x=0
        self.inside_target_pos_y=0


    def motion_strategy_simple(self, score, pred_bbox):
        pscore = score * (1 - self.params.window_influence) + self.window * self.params.window_influence

        best_idx = np.argmax(pscore)

        final_score = pscore[best_idx]
        final_lr_score = score[best_idx]
        lr = self.params.lr * final_lr_score
        if final_lr_score > 0.35:
            lr = max(lr, 0.50)

        final_bbox = pred_bbox[best_idx, :]
        bbox = final_bbox
        box = np.array([0.0, 0.0, 0.0, 0.0])
        box[0] = (bbox[0] + bbox[2]) / 2 - self.params.search_size // 2
        box[1] = (bbox[1] + bbox[3]) / 2 - self.params.search_size // 2
        box[2] = (bbox[2] - bbox[0] + 1)
        box[3] = (bbox[3] - bbox[1] + 1)
        return box, final_score, lr, final_lr_score, best_idx, pscore


    def post_processing(self, score, pred_bbox, scale_z):
        pscore = score * (1 - self.params.window_influence) + self.window * self.params.window_influence
        best_idx = np.argmax(pscore)
        final_lr_score = score[best_idx]
        final_score = pscore[best_idx]

        y = best_idx // self.score_size_up
        x = best_idx % self.score_size_up

        pos = np.array([x,y]) * self.score_size / self.score_size_up + 0.5
        pos = pos.astype(np.int32)
        best_idx = pos[1] * self.score_size + pos[0]

        final_bbox = pred_bbox[best_idx, :]

        lr = self.params.lr * final_lr_score
        if final_lr_score > 0.35:
            lr = max(lr, 0.50)
        bbox = final_bbox
        box = np.array([0.0, 0.0, 0.0, 0.0])
        box[0] = (bbox[0] + bbox[2]) / 2 - self.params.search_size // 2
        box[1] = (bbox[1] + bbox[3]) / 2 - self.params.search_size // 2
        box[2] = (bbox[2] - bbox[0] + 1)
        box[3] = (bbox[3] - bbox[1] + 1)

        return box, final_score, lr, final_lr_score, best_idx,pscore

    def motion_strategy(self, score, pred_bbox,center_pos, scale_z):
        pscore = score.copy()
        score = score.reshape(self.score_size_up, self.score_size_up)

        inside_weight = np.zeros_like(score)
        inside_width = self.params.score_inside_width
        inside_right_idx = int(self.score_size_up - (self.score_size_up - inside_width) / 2)
        inside_left_idx = int(self.score_size_up - (self.score_size_up - inside_width) / 2 - inside_width)
        inside_weight[inside_left_idx:inside_right_idx,inside_left_idx:inside_right_idx] = np.ones((inside_width,inside_width))

        outside_width = self.params.score_outside_width
        outside_right_idx = int(self.score_size_up - (self.score_size_up - outside_width) / 2)
        outside_left_idx = int(self.score_size_up - (self.score_size_up - outside_width) / 2 - outside_width)

        outside_weight = np.zeros_like(score)
        outside_weight[outside_left_idx:outside_right_idx,outside_left_idx:outside_right_idx] = np.ones((outside_width,outside_width))
        outside_weight = outside_weight - inside_weight

        inside_score = score * inside_weight
        outside_score = score * outside_weight

        flag = False

        if outside_score.max() > 0.3 and inside_score.max() > 0.4:
            inside_score = inside_score.reshape(-1)
            outside_score = outside_score.reshape(-1)
            inside_box, final_score, lr, final_lr_score, best_idx_inside, pscore = self.post_processing(inside_score, pred_bbox, scale_z)
            inside_pos_x = center_pos[0] + inside_box[0] / scale_z
            inside_pos_y = center_pos[1] + inside_box[1] / scale_z

            outside_box, final_score, lr, final_lr_score, best_idx_outside, pscore = self.post_processing(outside_score, pred_bbox, scale_z)
            outside_pos_x = center_pos[0] + outside_box[0] / scale_z
            outside_pos_y = center_pos[1] + outside_box[1] / scale_z

            target_disturbance_distance = np.sqrt((outside_pos_x - inside_pos_x)**2+(outside_pos_y - inside_pos_y)**2)

            if self.previous_t_d_distance == 0:
                self.previous_t_d_distance = target_disturbance_distance
            else:
                if target_disturbance_distance - self.previous_t_d_distance < 0:
                    self.disturbance_close_to_target = True

                    self.inside_target_pos_x = inside_pos_x
                    self.inside_target_pos_y = inside_pos_y
                    self.t_d_reset_count = 0
                elif target_disturbance_distance - self.previous_t_d_distance > 0 and self.disturbance_in_target is True:
                    self.disturbance_away_from_target = True

            box = target_box = inside_box
            flag = True
        else:
            box, final_score, lr, final_lr_score, best_idx_else, pscore = self.post_processing(pscore, pred_bbox, scale_z)
            if self.disturbance_close_to_target is True:
                self.disturbance_in_target = True
                self.previous_t_d_distance = 0
                inside_box = box
                inside_pos_x = center_pos[0] + inside_box[0] / scale_z
                inside_pos_y = center_pos[1] + inside_box[1] / scale_z
                self.t_d_reset_count = self.t_d_reset_count + 1
                if self.t_d_reset_count == 10:
                    self.disturbance_close_to_target = False
                    self.disturbance_in_target = False
                    self.disturbance_away_from_target = False
            inside_box = box
            outside_box = box

        if flag:
            best_idx = best_idx_inside
        else:
            best_idx = best_idx_else

        if self.disturbance_away_from_target is True:
            inside_pos_x = center_pos[0] + inside_box[0] / scale_z
            inside_pos_y = center_pos[1] + inside_box[1] / scale_z
            target_inside_distance = np.sqrt((self.inside_target_pos_x - inside_pos_x)**2 + (self.inside_target_pos_y - inside_pos_y)**2)
            outside_pos_x = center_pos[0] + outside_box[0] / scale_z
            outside_pos_y = center_pos[1] + outside_box[1] / scale_z
            target_outside_distance = np.sqrt((outside_pos_x - self.inside_target_pos_x)**2 + (outside_pos_y - self.inside_target_pos_y)**2)
            if target_inside_distance > target_outside_distance:
                disturbance_box = inside_box
                target_box = outside_box
                if flag:
                    best_idx = best_idx_outside
                else:
                    best_idx = best_idx_else
            else:
                disturbance_box = outside_box
                target_box = inside_box
                if flag:
                    best_idx = best_idx_inside
                else:
                    best_idx = best_idx_else

            self.disturbance_close_to_target = False
            self.disturbance_in_target = False
            self.disturbance_away_from_target = False
            box = target_box

        return box, final_score, lr, final_lr_score, best_idx, pscore


    def localize_advanced(self, scores, sample_pos, sample_scales):
        """Run the target advanced localization (as in ATOM)."""
        scores = scores.squeeze(1)
        sz = scores.shape[-2:]
        score_sz = torch.Tensor(list(sz))
        output_sz = score_sz - (self.kernel_size + 1) % 2
        score_center = (score_sz - 1)/2

        scores_hn = scores


        max_score1, max_disp1 = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score1, dim=0)
        sample_scale = sample_scales[scale_ind]
        max_score1 = max_score1[scale_ind]
        max_disp1 = max_disp1[scale_ind,...].float().cpu().view(-1)
        target_disp1 = max_disp1 - score_center
        translation_vec1 = target_disp1 * (self.img_support_sz_search / output_sz) * sample_scale

        if max_score1.item() < self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'not_found'
        if max_score1.item() < self.params.get('uncertain_threshold', -float('inf')):
            return translation_vec1, scale_ind, scores_hn, 'uncertain'
        if max_score1.item() < self.params.get('hard_sample_threshold', -float('inf')):
            return translation_vec1, scale_ind, scores_hn, 'hard_negative'

        # Mask out target neighborhood
        target_neigh_sz = self.params.target_neighborhood_scale * (self.target_sz / sample_scale) * (output_sz / self.img_support_sz_search)

        tneigh_top = max(round(max_disp1[0].item() - target_neigh_sz[0].item() / 2), 0)
        tneigh_bottom = min(round(max_disp1[0].item() + target_neigh_sz[0].item() / 2 + 1), sz[0])
        tneigh_left = max(round(max_disp1[1].item() - target_neigh_sz[1].item() / 2), 0)
        tneigh_right = min(round(max_disp1[1].item() + target_neigh_sz[1].item() / 2 + 1), sz[1])
        scores_masked = scores_hn[scale_ind:scale_ind + 1, ...].clone()
        scores_masked[...,tneigh_top:tneigh_bottom,tneigh_left:tneigh_right] = 0

        # Find new maximum
        max_score2, max_disp2 = dcf.max2d(scores_masked)
        max_disp2 = max_disp2.float().cpu().view(-1)
        target_disp2 = max_disp2 - score_center
        translation_vec2 = target_disp2 * (self.img_support_sz_search / output_sz) * sample_scale

        prev_target_vec = (self.pos - sample_pos[scale_ind,:]) / ((self.img_support_sz_search / output_sz) * sample_scale)

        # Handle the different cases
        if max_score2 > self.params.distractor_threshold * max_score1:
            disp_norm1 = torch.sqrt(torch.sum((target_disp1-prev_target_vec)**2))
            disp_norm2 = torch.sqrt(torch.sum((target_disp2-prev_target_vec)**2))
            disp_threshold = self.params.dispalcement_scale * math.sqrt(sz[0] * sz[1]) / 2

            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'hard_negative'
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec2, scale_ind, scores_hn, 'hard_negative'
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'uncertain'

            # If also the distractor is close, return with highest score
            return translation_vec1, scale_ind, scores_hn, 'uncertain'

        if max_score2 > self.params.hard_negative_threshold * max_score1 and max_score2 > self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'hard_negative'

        return translation_vec1, scale_ind, scores_hn, 'normal'


def get_tracker_class():
    return MixFormerOnline
