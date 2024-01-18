import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import lib.train.data.processing_utils as prutils
import torch.nn.functional as F
import random
import numpy as np
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy,box_xyxy_to_cxcywh

def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""
    def __init__(self, transform=transforms.ToTensor(), template_transform=None, search_transform=None, joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if template_transform or
                                search_transform is None.
            template_transform - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            search_transform  - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the template and search images.  For
                                example, it can be used to convert both template and search images to grayscale.
        """
        self.transform = {'template': transform if template_transform is None else template_transform,
                          'search':  transform if search_transform is None else search_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class STARKProcessing(BaseProcessing):
    """ The processing class used for training LittleBoy. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', settings=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
            data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)

        for s in ['template', 'search']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
            if (crop_sz < 1).any():
                data['valid'] = False
                # print("Too small box is found. Replace it with new data.")
                return data

            # Crop image region centered at jittered_anno box and get the attention mask
            crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                              data[s + '_anno'], self.search_area_factor[s],
                                                                              self.output_sz[s], masks=data[s + '_masks'])
            # Apply transforms
            data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
                image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)

            # 2021.1.9 Check whether elements in data[s + '_att'] is all 1
            # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    # print("Values of original attention mask are all one. Replace it with new data.")
                    return data
            # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
            for ele in data[s + '_att']:
                feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    # print("Values of down-sampled attention mask are all one. "
                    #       "Replace it with new data.")
                    return data

        data['valid'] = True
        # if we use copy-and-paste augmentation
        if data["template_masks"] is None or data["search_masks"] is None:
            data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
            data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))
        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data


class MixformerProcessing(BaseProcessing):
    """ The processing class used for training LittleBoy. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair', settings=None, train_score=False, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings
        self.train_score = train_score
        # self.label_function_params = label_function_params
        self.out_feat_sz = 20  ######## the output feature map size

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def _generate_neg_proposals(self, box, min_iou=0.0, max_iou=0.3, sigma=0.5):
        """ Generates proposals by adding noise to the input box
        args:
            box - input box
        returns:
            torch.Tensor - Array of shape (num_proposals, 4) containing proposals
            torch.Tensor - Array of shape (num_proposals,) containing IoU overlap of each proposal with the input box. The
                        IoU is mapped to [-1, 1]
        """
        # Generate proposals
        # num_proposals = self.proposal_params['boxes_per_frame']
        # proposal_method = self.proposal_params.get('proposal_method', 'default')

        # if proposal_method == 'default':
        num_proposals = box.size(0)
        proposals = torch.zeros((num_proposals, 4)).to(box.device)
        gt_iou = torch.zeros(num_proposals)
        for i in range(num_proposals):
            proposals[i, :], gt_iou[i] = prutils.perturb_box(box[i], min_iou=min_iou, max_iou=max_iou,
                                                             sigma_factor=sigma)
        # elif proposal_method == 'gmm':
        #     proposals, _, _ = prutils.sample_box_gmm(box, self.proposal_params['proposal_sigma'],
        #                                                                      num_samples=num_proposals)
        #     gt_iou = prutils.iou(box.view(1,4), proposals.view(-1,4))

        # # Map to [-1, 1]
        # gt_iou = gt_iou * 2 - 1
        return proposals

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
            data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)

        for s in ['template', 'search']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
            if (crop_sz < 1).any():
                data['valid'] = False
                # print("Too small box is found. Replace it with new data.")
                return data

            # Crop image region centered at jittered_anno box and get the attention mask
            crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                              data[s + '_anno'], self.search_area_factor[s],
                                                                              self.output_sz[s], masks=data[s + '_masks'])
            # Apply transforms
            data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
                image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)

            # 2021.1.9 Check whether elements in data[s + '_att'] is all 1
            # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    # print("Values of original attention mask are all one. Replace it with new data.")
                    return data
            # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
            for ele in data[s + '_att']:
                feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    # print("Values of down-sampled attention mask are all one. "
                    #       "Replace it with new data.")
                    return data

        data['valid'] = True
        # if we use copy-and-paste augmentation
        if data["template_masks"] is None or data["search_masks"] is None:
            data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
            data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))
        print(data['template_images'].s)
        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)


        # if self.train_score:
        #     if random.random() < 0.5:
        #         data['label'] = torch.zeros_like(data['label'])
        #         data['search_anno'] = self._generate_neg_proposals(data['search_anno'])

        # search_anno is with normalized coords. (x,y,w,h)
        # search_anno = data['search_anno'].clone()
        # wl = wr = search_anno[:, 2] * 0.5
        # ht = hb = search_anno[:, 3] * 0.5
        # w2h2 = torch.stack((wl, wr, ht, hb), dim=1)  # [num_images, 4]
        #
        # search_anno = (search_anno * self.out_feat_sz).float()
        # center_float = search_anno[:, :2] + search_anno[:, 2:] / 2.
        # center_int = center_float.int().float()
        # ind = center_int[:, 1] * self.out_feat_sz + center_int[:, 0]  # [num_images, 1]
        #
        # data['ind'] = ind.long()
        # data['w2h2'] = w2h2

        ### Generate label functions and regression mask
        # if self.settings.script_name == 'tsp_cls_online':
        #     search_anno = data['search_anno'].clone() * self.output_sz['search']
        #     data['gt_scores'] = self._generate_label_function(search_anno)

            # search_anno = data['search_anno'].clone() * self.out_feat_sz
            # target_center = search_anno[:, :2] + search_anno[:, 2:] * 0.5
            # # add noise
            # target_center[:, 0] = target_center[:, 0] + np.random.randint(0, 2)
            # target_center[:, 1] = target_center[:, 1] + np.random.randint(0, 2)
            # mask_scale_w = self.settings.mask_scale + np.random.uniform(-0.15, 0.15)
            # mask_scale_h = self.settings.mask_scale + np.random.uniform(-0.15, 0.15)
            # mask_w, mask_h = search_anno[:, 2] * mask_scale_w, search_anno[:, 3] * mask_scale_h
            #
            # data['reg_mask'] = self._generate_regression_mask(target_center, mask_w, mask_h, self.out_feat_sz)

        return data

    def _generate_regression_mask(self, target_center, mask_w, mask_h, mask_size=20):
        """
        NHW format
        :return:
        """
        k0 = torch.arange(mask_size, dtype=torch.float32, device=target_center.device).view(1, 1, -1)
        k1 = torch.arange(mask_size, dtype=torch.float32, device=target_center.device).view(1, -1, 1)

        d0 = (k0 - target_center[:, 0].view(-1, 1, 1)).abs()   # w, (b, 1, w)
        d1 = (k1 - target_center[:, 1].view(-1, 1, 1)).abs()   # h, (b, h, 1)
        # dist = d0.abs() + d1.abs()
        mask_w = mask_w.view(-1, 1, 1)
        mask_h = mask_h.view(-1, 1, 1)

        mask0 = torch.where(d0 <= mask_w*0.5, torch.ones_like(d0), torch.zeros_like(d0)) # (b, 1, w)
        mask1 = torch.where(d1 <= mask_h*0.5, torch.ones_like(d1), torch.zeros_like(d1)) # (b, h, 1)

        return mask0 * mask1  # (b, h, w)

class MixformerKLDiMPProcessing(BaseProcessing):
    """ The processing class used for training LittleBoy. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 mode='pair',label_function_params = None,label_density_params = None, settings=None,
                 train_score=False,use_dfcos = True,use_reppoint = True, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings
        self.train_score = train_score
        self.label_function_params = label_function_params
        self.label_density_params = label_density_params
        self.use_dfcos = use_dfcos
        self.use_reppoint = use_reppoint
        self._generate_meshgrid()

    def _generate_label_function(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), self.label_function_params['sigma_factor'],
                                                      self.label_function_params['kernel_sz'],
                                                      self.label_function_params['feature_sz'], self.output_sz['search'],
                                                      end_pad_if_even=self.label_function_params.get('end_pad_if_even', True))

        return gauss_label

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
            data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)

        for s in ['template', 'search']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
            if (crop_sz < 1).any():
                data['valid'] = False
                # print("Too small box is found. Replace it with new data.")
                return data

            # Crop image region centered at jittered_anno box and get the attention mask
            crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                              data[s + '_anno'], self.search_area_factor[s],
                                                                              self.output_sz[s], masks=data[s + '_masks'])
            # Apply transforms
            data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
                image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)

            # 2021.1.9 Check whether elements in data[s + '_att'] is all 1
            # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    # print("Values of original attention mask are all one. Replace it with new data.")
                    return data
            # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
            for ele in data[s + '_att']:
                feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    # print("Values of down-sampled attention mask are all one. "
                    #       "Replace it with new data.")
                    return data

        data['valid'] = True
        # if we use copy-and-paste augmentation
        if data["template_masks"] is None or data["search_masks"] is None:
            data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
            data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))
        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)


        data["template_masks"] = data["template_masks"].unsqueeze(1)
        data["search_masks"] = data["search_masks"].unsqueeze(1)

        # Generate label functions
        data['search_anno_crop'] = data['search_anno'] * self.output_sz['search']
        data['template_anno_crop'] = data['template_anno'] * self.output_sz['template']

        if self.label_function_params is not None:
            # data['template_label'] = self._generate_label_function(data['template_anno'])
            data['search_label'] = self._generate_label_function(data['search_anno_crop'])
        # if self.label_density_params is not None:
        #     # data['template_label_density'] = self._generate_label_density(data['template_anno'])
        #     data['search_label_density'] = self._generate_label_density(data['search_anno'])

        # if self.use_dfcos:
        #     label_cls = self._generate_regression_mask(data['search_anno_crop'].clone())
        #     reg_weight = label_cls.type(torch.float32)

        #     data['search_cls'] = label_cls
        #     data['reg_weight'] = reg_weight

        if self.use_reppoint:
            reppoint_int_weight = self._generate_reppoint_init(data['search_anno_crop'].clone())
            data['reppoint_init_weight'] = reppoint_int_weight

        return data

    def _generate_meshgrid(self):
        fm_height, fm_width = self.settings.reppoint_param.score_size, self.settings.reppoint_param.score_size  # h, w
        fm_offset = self.settings.reppoint_param.fm_offset
        stride = self.settings.reppoint_param.stride
        # coordinate meshgrid on feature map, shape=(h, w)
        x_coords_on_fm = torch.arange(0, fm_width, dtype=torch.int64) * stride + fm_offset  # (w, )
        y_coords_on_fm = torch.arange(0, fm_height, dtype=torch.int64) * stride + fm_offset  # (h, )
        self.y_coords_on_fm, self.x_coords_on_fm = torch.meshgrid(x_coords_on_fm, y_coords_on_fm)  # (h, w)
        # coords_on_fm = torch.stack([x_coords_on_fm,y_coords_on_fm],-1)


    def _generate_reppoint_init(self, gt_boxes):
        """
        NHW format
        :return:
        """
        fm_size = self.settings.reppoint_param.score_size
        boxes_cnt = len(gt_boxes)
        gt_centers_x = gt_boxes[:,0] + gt_boxes[:,2] / 2
        gt_centers_y = gt_boxes[:,1] + gt_boxes[:,3] / 2

        reppoint_int_weight = []

        for i in range(boxes_cnt):
            cx = gt_centers_x[i]
            cy = gt_centers_y[i]

            dist = torch.sqrt((self.x_coords_on_fm - cx)**2 + (self.y_coords_on_fm - cy)**2).reshape(-1)

            min_dist, min_dist_index = torch.topk(dist, self.settings.reppoint_param.pos_num, largest=False)

            zeros = torch.zeros(fm_size,fm_size).reshape(-1).float()
            zeros[min_dist_index] = 1.0
            zeros = zeros.reshape(fm_size,fm_size)
            reppoint_int_weight.append(zeros)
        reppoint_int_weight = torch.stack(reppoint_int_weight,0)
        return reppoint_int_weight


    def _generate_regression_mask(self, gt_boxes):
        """
        NHW format
        :return:
        """
        gt_boxes = box_xywh_to_xyxy(gt_boxes)
        boxes_cnt = len(gt_boxes)

        raw_width = raw_height = self.settings.dfcos_param.x_size
        x_coords = torch.arange(0, raw_width, dtype=torch.int64)  # (W, )
        y_coords = torch.arange(0, raw_height, dtype=torch.int64)  # (H, )
        y_coords, x_coords = torch.meshgrid(x_coords, y_coords)  # (H, W)

        off_l = (x_coords[:, :, np.newaxis, np.newaxis].type(torch.float32) -
                 gt_boxes[np.newaxis, np.newaxis, :, 0, np.newaxis])
        off_t = (y_coords[:, :, np.newaxis, np.newaxis].type(torch.float32) -
                 gt_boxes[np.newaxis, np.newaxis, :, 1, np.newaxis])
        off_r = -(x_coords[:, :, np.newaxis, np.newaxis].type(torch.float32) -
                  gt_boxes[np.newaxis, np.newaxis, :, 2, np.newaxis])
        off_b = -(y_coords[:, :, np.newaxis, np.newaxis].type(torch.float32) -
                  gt_boxes[np.newaxis, np.newaxis, :, 3, np.newaxis])


        offset = torch.cat([off_l, off_t, off_r, off_b], dim=3)

        fm_height, fm_width = self.settings.dfcos_param.score_size, self.settings.dfcos_param.score_size  # h, w
        fm_offset = self.settings.dfcos_param.fm_offset
        stride = self.settings.dfcos_param.total_stride

        # coordinate meshgrid on feature map, shape=(h, w)
        x_coords_on_fm = torch.arange(0, fm_width, dtype=torch.int64)  # (w, )
        y_coords_on_fm = torch.arange(0, fm_height, dtype=torch.int64)  # (h, )
        y_coords_on_fm, x_coords_on_fm = torch.meshgrid(x_coords_on_fm,
                                                        y_coords_on_fm)  # (h, w)
        y_coords_on_fm = y_coords_on_fm.reshape(-1)  # (hxw, ), flattened
        x_coords_on_fm = x_coords_on_fm.reshape(-1)  # (hxw, ), flattened

        # (hxw, #boxes, 4-d_offset_(l/t/r/b), )
        offset_on_fm = offset[fm_offset + y_coords_on_fm * stride, fm_offset +
                              x_coords_on_fm * stride]  # will reduce dim by 1
        # (hxw, #gt_boxes, )
        is_in_boxes = (offset_on_fm > 0).all(dim=2).float()
        # (h, w, #gt_boxes, ), boolean
        #   valid mask
        offset_valid = torch.zeros((fm_height, fm_width, boxes_cnt)).float()
        offset_valid[
            y_coords_on_fm,
            x_coords_on_fm, :] = is_in_boxes  #& is_in_layer  # xy[:, 0], xy[:, 1] reduce dim by 1 to match is_in_boxes.shape & is_in_layer.shape

        offset_valid = offset_valid.permute(2,0,1)

        return offset_valid