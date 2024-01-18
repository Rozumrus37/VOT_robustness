import torch
import torchvision.transforms as transforms
from pytracking import TensorDict
import ltr.data.processing_utils as prutils

from util.misc import NestedTensor
import math
import numpy as np
import torch.nn.functional as F
import random
def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x

class PositionEmbeddingSine():
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None, max_size = 2000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.max_size = max_size

        self.dim_t = self.get_position_embedding_dim()
        self.max_coord_map = self.get_coord_map()

    def get_position_embedding_dim(self):
        dim_t = np.array(list(range(self.num_pos_feats)),dtype=np.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        return dim_t

    def get_coord_map(self):
        if self.max_size is not None:
            h = self.max_size
            w = self.max_size
        not_mask = np.ones((h,w),dtype = np.float32)
        y_embed = not_mask.cumsum(0, dtype=np.float32)
        x_embed = not_mask.cumsum(1, dtype=np.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale
        return np.stack([y_embed, x_embed],axis=-1)

    def get_position_embedding_sine(self,coord_map):
        if isinstance(coord_map,list):
            coord_map = np.stack(coord_map,0)
        else:
            coord_map = coord_map[None,...]

        y_embed = coord_map[...,0]
        x_embed = coord_map[...,1]

        pos_x = x_embed[:, :, :, None] / self.dim_t
        pos_y = y_embed[:, :, :, None] / self.dim_t

        b,h,w,c = pos_x.shape
        pos_x = np.stack((np.sin(pos_x[:, :, :, 0::2]), np.cos(pos_x[:, :, :, 1::2])), axis=4).reshape((b,h,w,-1))
        pos_y = np.stack((np.sin(pos_y[:, :, :, 0::2]), np.cos(pos_y[:, :, :, 1::2])), axis=4).reshape((b,h,w,-1))
        pos = np.concatenate((pos_y, pos_x), axis=3)
        return pos

class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""
    def __init__(self, transform=transforms.ToTensor(), search_transform=None, template_transform=None, joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if search_transform or
                                template_transform is None.
            search_transform - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            template_transform  - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the search and template images.  For
                                example, it can be used to convert both template and search images to grayscale.
        """
        self.transform = {'search': transform if search_transform is None else search_transform,
                          'static_template':  transform if template_transform is None else template_transform,
                          'dynamic_template':  transform if template_transform is None else template_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class TransTProcessing(BaseProcessing):
    """ The processing class used for training TransT. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument search_sz.

    """

    def __init__(self, search_area_factor, template_area_factor, search_sz, temp_sz, center_jitter_factor, scale_jitter_factor,
                 search_response_sz = None,mode='pair',label_function_params = None,use_abposition = False,hidden_dim = 256, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region relative to the target size.
            template_area_factor - The size of the template region relative to the template target size.
            search_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            temp_sz - An integer, denoting the size to which the template region is resized. The search region is always
                      square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.template_area_factor = template_area_factor
        self.search_sz = search_sz
        self.temp_sz = temp_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.label_function_params = label_function_params
        if use_abposition:
            self.abposition_map = PositionEmbeddingSine(num_pos_feats=hidden_dim//2,max_size = 10000)
        else:
            self.abposition_map = None
        self.search_response_sz = search_response_sz


    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'search' or 'template' indicating search or template data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.sum() * 0.5 * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)
        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'search_images', template_images', 'search_anno', 'template_anno'
        returns:
            TensorDict - output data block with following fields:
                'search_images', 'template_images', 'search_anno', 'template_anno'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'] = self.transform['joint'](image=data['template_images'], bbox=data['template_anno'])
            data['search_images'], data['search_anno'] = self.transform['joint'](image=data['search_images'], bbox=data['search_anno'], new_roll=False)

        for s in ['search', 'template']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num search/template frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # Crop image region centered at jittered_anno box
            if s == 'search':
                crops, boxes, _ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                           self.search_area_factor, self.search_sz)
            elif s == 'template':
                crops, boxes, _ = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                           self.template_area_factor, self.temp_sz)
            else:
                raise NotImplementedError

            # Apply transforms
            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)
        data['template_images'] = data['template_images'].squeeze()
        data['search_images'] = data['search_images'].squeeze()
        data['template_anno'] = data['template_anno'].squeeze()
        data['search_anno'] = data['search_anno'].squeeze()
        return data


class TransTMaskProcessing(TransTProcessing):
    """ The processing class used for training TransT. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument search_sz.

    """

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'search_images', template_images', 'search_anno', 'template_anno'
        returns:
            TensorDict - output data block with following fields:
                'search_images', 'template_images', 'search_anno', 'template_anno'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['static_template_images'], data['static_template_anno'] = self.transform['joint'](image=data['static_template_images'],
                                                                                                   bbox=data['static_template_anno'])
            if data['dynamic_template'] == True:
                data['dynamic_template_images'], data['dynamic_template_anno'] = self.transform['joint'](image=data['dynamic_template_images'],
                                                                                                         bbox=data['dynamic_template_anno'],
                                                                                                         new_roll=False)
            data['search_images'], data['search_anno'] = self.transform['joint'](image=data['search_images'],
                                                                                 bbox=data['search_anno'],
                                                                                 new_roll=False)

        s_index = ['search', 'static_template']
        if data['dynamic_template'] == True:
            s_index.append('dynamic_template')
        for s in s_index:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num search/template frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # Crop image region centered at jittered_anno box
            if s == 'search':
                crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                               self.search_area_factor, self.search_sz)
            elif s == 'static_template':
                # crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                #                                                self.template_area_factor, self.temp_sz)
                crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                               self.template_area_factor, self.temp_sz)
            elif s == 'dynamic_template':
                # crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                #                                                self.template_area_factor, self.temp_sz)
                crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                               self.template_area_factor, self.temp_sz)
            else:
                raise NotImplementedError

            # Apply transforms
            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)
            #mask_crops:(256, 256),crops:(256, 256, 3)
            if s == 'search':
                mask_crops, _ = prutils.jittered_center_crop(data[s + '_masks'], jittered_anno, data[s + '_anno'],
                                                               self.search_area_factor, self.search_sz)
            elif s == 'static_template':
                mask_crops, _ = prutils.jittered_center_crop(data[s + '_masks'], jittered_anno, data[s + '_anno'],
                                                               self.template_area_factor, self.temp_sz)
            elif s == 'dynamic_template':
                mask_crops, _ = prutils.jittered_center_crop(data[s + '_masks'], jittered_anno, data[s + '_anno'],
                                                               self.template_area_factor, self.temp_sz)
            else:
                raise NotImplementedError
            data[s + '_masks'] = [self.mask_np2torch(x) for x in mask_crops]

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        data['search_images'] = data['search_images'].squeeze(0) # torch.Size([3, Hs, Ws])
        data['search_masks'] = data['search_masks'].squeeze(0) #torch.Size([1, Hs, Ws])
        data['search_anno'] = data['search_anno'].squeeze(0) #torch.Size([4])

        data['template_images'] = data['static_template_images'] # torch.Size([1, 3, Ht, Wt])
        data['template_masks'] = data['static_template_masks'] #torch.Size([1, 1, Ht, Wt]) #template的mask其实可以去掉，没用到
        data['template_anno'] = data['static_template_anno'] #torch.Size([1, 4])

        if data['dynamic_template'] == True:
            data['template_images'] = torch.cat((data['template_images'], data['dynamic_template_images']), 0)#torch.Size([number, 3, Ht, Wt])
            data['template_masks'] = torch.cat((data['template_masks'], data['dynamic_template_masks']), 0) #torch.Size([number, 1, 128, 128])
            data['template_anno'] = torch.cat((data['template_anno'], data['dynamic_template_anno']), 0) #torch.Size([number, 4])
        else:
            data['template_images'] = data['template_images'].squeeze(0) # torch.Size([3, Hs, Ws])
            data['template_masks'] = data['template_masks'].squeeze(0) #torch.Size([1, Hs, Ws])
            data['template_anno'] = data['template_anno'].squeeze(0) #torch.Size([4])

        return data

    def mask_np2torch(self, mask_np):
        return torch.from_numpy(mask_np.transpose((2, 0, 1))).float()









class TransTMaskTemporalProcessing(TransTProcessing):
    """ The processing class used for training TransT. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument search_sz.

    """
    # def get_absposition(self,bbox,size):
    #     mask =
    def temporal_augm(self,images,bboxes):

        def occlude(image,box):
            # print(box,image.shape)
            shape = image.shape
            box = box.astype(np.int32)
            x1 = max(box[0],0)
            y1 = max(box[1],0)
            x2 = min(x1 + box[2],shape[1])
            y2 = min(y1 + box[3],shape[0])
            w = x2 - x1
            h = y2 - y1

            if random.random() < 0.1:
                idxes = list(range(w*h))
                random.shuffle(idxes)
                image_crop = image[y1:y2,x1:x2].copy()
                image_crop = image_crop.reshape(-1,3)
                image_crop = image_crop[idxes].copy()
                image_crop = image_crop.reshape((h,w,3))
                image[y1:y2,x1:x2] = image_crop
            elif random.random() < 0.2:
                x_offset = int(np.random.uniform(0,shape[1] - w))
                y_offset = int(np.random.uniform(0,shape[0] - h))
                image[y1:y2,x1:x2] = image[y_offset:y_offset+h,x_offset:x_offset+w]
            else:
                w_occ = int(np.random.uniform(w/4,w))
                h_occ = int(np.random.uniform(h/4,h))
                x_occ = int(np.random.uniform(x1,x2 - w_occ))
                y_occ = int(np.random.uniform(y1,y2 - h_occ))

                x_offset = int(np.random.uniform(0,shape[1] - w_occ))
                y_offset = int(np.random.uniform(0,shape[0] - h_occ))
                image[y_occ:y_occ+h_occ,x_occ:x_occ+w_occ] = image[y_offset:y_offset+h_occ,x_offset:x_offset+w_occ]

            return image.copy()
        def distractor(image,box):
            # print(box,image.shape)
            shape = image.shape
            box = box.astype(np.int32)
            x1 = max(box[0],0)
            y1 = max(box[1],0)
            x2 = min(x1 + box[2],shape[1])
            y2 = min(y1 + box[3],shape[0])
            w = x2 - x1
            h = y2 - y1

            target = image[y1:y2,x1:x2].copy()

            x_offset = int(np.random.uniform(0,shape[1] - w))
            y_offset = int(np.random.uniform(0,shape[0] - h))
            image[y_offset:y_offset+h,x_offset:x_offset+w] = image[y1:y2,x1:x2]
            return image.copy()

        outputs = []
        for i in range(len(images) - 1):
            if random.random() < 0.2:
                image = occlude(images[i].copy(),bboxes[i].numpy())
                outputs.append(image)
            elif random.random() < 0.4:
                image = distractor(images[i].copy(),bboxes[i].numpy())
                outputs.append(image)
            else:
                outputs.append(images[i])

        image = distractor(images[-1].copy(),bboxes[-1].numpy())
        outputs.append(image)

        return tuple(outputs)





    def _generate_label_function(self, target_bb):
        """ Generates the gaussian label function centered at target_bb
        args:
            target_bb - target bounding box (num_images, 4)

        returns:
            torch.Tensor - Tensor of shape (num_images, label_sz, label_sz) containing the label for each sample
        """

        gauss_label = prutils.gaussian_label_function(target_bb.view(-1, 4), self.label_function_params['sigma_factor'],
                                                      self.label_function_params['kernel_sz'],
                                                      self.label_function_params['response_sz'], self.search_sz,
                                                      end_pad_if_even=self.label_function_params.get('end_pad_if_even', True))
        return gauss_label

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'search_images', template_images', 'search_anno', 'template_anno'
        returns:
            TensorDict - output data block with following fields:
                'search_images', 'template_images', 'search_anno', 'template_anno'
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['static_template_images'], data['static_template_anno'] = self.transform['joint'](image=data['static_template_images'],
                                                                                                   bbox=data['static_template_anno'])
            if data['dynamic_template'] == True:
                data['dynamic_template_images'], data['dynamic_template_anno'] = self.transform['joint'](image=data['dynamic_template_images'],
                                                                                                         bbox=data['dynamic_template_anno'],
                                                                                                         new_roll=False)
            data['search_images'], data['search_anno'] = self.transform['joint'](image=data['search_images'],
                                                                                 bbox=data['search_anno'],
                                                                                 new_roll=False)

        s_index = ['search', 'static_template']
        if data['dynamic_template'] == True:
            s_index.append('dynamic_template')
        for s in s_index:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num search/template frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # Crop image region centered at jittered_anno box
            if s == 'search':
                # crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                #                                                self.search_area_factor, self.search_sz)
                assert self.abposition_map is not None
                max_coord_map_list = []
                paddingmask_list = []
                for i in range(len(data[s + '_images'])):
                    search_size = data[s + '_images'][i].shape[:2]
                    max_coord_map = self.abposition_map.max_coord_map[:search_size[0],:search_size[1]].copy()
                    paddingmask = np.zeros(search_size,dtype = np.float32)
                    max_coord_map_list.append(max_coord_map)
                    paddingmask_list.append(paddingmask)

                crops, boxes, coordmaps_crop, paddingmasks_crop = prutils.jittered_center_crop_with_position(
                                    data[s + '_images'], max_coord_map_list, paddingmask_list, jittered_anno,
                                    data[s + '_anno'],self.search_area_factor,
                                    output_sz = self.search_sz, feature_sz = self.search_response_sz)

                abspositions_crop = self.abposition_map.get_position_embedding_sine(coordmaps_crop)
                data[s + '_abspositions'] = [self.mask_np2torch(abspositions_crop[i]) for i in range(len(abspositions_crop))]
                # data[s + '_abcoords'] = [self.mask_np2torch(coordmaps_crop[i]) for i in range(len(coordmaps_crop))]
                data[s + '_paddingmasks'] = [torch.from_numpy(x).squeeze().to(torch.bool) for x in paddingmasks_crop]
                crops = self.temporal_augm(crops,boxes)


            elif s == 'static_template':
                # crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                #                                                self.template_area_factor, self.temp_sz)
                crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                               self.template_area_factor, self.temp_sz)
            elif s == 'dynamic_template':
                # crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                #                                                self.template_area_factor, self.temp_sz)
                crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                               self.template_area_factor, self.temp_sz)
            else:
                raise NotImplementedError

            # Apply transforms
            data[s + '_images'], data[s + '_anno'] = self.transform[s](image=crops, bbox=boxes, joint=False)

            #mask_crops:(256, 256),crops:(256, 256, 3)
            if s == 'search':
                mask_crops, _ = prutils.jittered_center_crop(data[s + '_masks'], jittered_anno, data[s + '_anno'],
                                                               self.search_area_factor, self.search_sz)
            elif s == 'static_template':
                mask_crops, _ = prutils.jittered_center_crop(data[s + '_masks'], jittered_anno, data[s + '_anno'],
                                                               self.template_area_factor, self.temp_sz)
            elif s == 'dynamic_template':
                mask_crops, _ = prutils.jittered_center_crop(data[s + '_masks'], jittered_anno, data[s + '_anno'],
                                                               self.template_area_factor, self.temp_sz)
            else:
                raise NotImplementedError
            data[s + '_masks'] = [self.mask_np2torch(x) for x in mask_crops]



        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)


        data['search_images'] = data['search_images'].squeeze(0) # torch.Size([3, Hs, Ws])
        data['search_masks'] = data['search_masks'].squeeze(0) #torch.Size([1, Hs, Ws])
        data['search_anno'] = data['search_anno'].squeeze(0) #torch.Size([4])

        data['template_images'] = data['static_template_images'] # torch.Size([1, 3, Ht, Wt])
        data['template_masks'] = data['static_template_masks'] #torch.Size([1, 1, Ht, Wt]) #template的mask其实可以去掉，没用到
        data['template_anno'] = data['static_template_anno'] #torch.Size([1, 4])

        if data['dynamic_template'] == True:
            data['template_images'] = torch.cat((data['template_images'], data['dynamic_template_images']), 0)#torch.Size([number, 3, Ht, Wt])
            data['template_masks'] = torch.cat((data['template_masks'], data['dynamic_template_masks']), 0) #torch.Size([number, 1, 128, 128])
            data['template_anno'] = torch.cat((data['template_anno'], data['dynamic_template_anno']), 0) #torch.Size([number, 4])

        if self.label_function_params is not None:
            search_response_anno = self._generate_label_function(data['search_anno'])
            search_response_anno = F.interpolate(search_response_anno[None], size=[self.search_response_sz,self.search_response_sz])[0]
            data['search_response_anno'] = search_response_anno

        return data

    def mask_np2torch(self, mask_np):
        return torch.from_numpy(mask_np.transpose((2, 0, 1))).float()






