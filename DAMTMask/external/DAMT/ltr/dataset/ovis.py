import os
from ltr.dataset.trdataset.base_dataset import BaseDataset
from ltr.data.image_loader import default_image_loader
import json
import tqdm
import torch
import random
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils
from collections import OrderedDict
from ltr.admin.environment import env_settings
import numpy as np


class OVIS(BaseDataset):
    """ The COCO dataset. COCO is an image dataset. Thus, we treat each image as a sequence of length 1.

    Publication:
        Microsoft COCO: Common Objects in Context.
        Tsung-Yi Lin, Michael Maire, Serge J. Belongie, Lubomir D. Bourdev, Ross B. Girshick, James Hays, Pietro Perona,
        Deva Ramanan, Piotr Dollar and C. Lawrence Zitnick
        ECCV, 2014
        https://arxiv.org/pdf/1405.0312.pdf

    Download the images along with annotations from http://cocodataset.org/#download. The root folder should be
    organized as follows.
        - coco_root
            - annotations
                - instances_train2014.json
            - images
                - train2014

    Note: You also have to install the coco pythonAPI from https://github.com/cocodataset/cocoapi.
    """

    def __init__(self, split='train', root=None, image_loader=default_image_loader, data_fraction=None):
        """
        args:
            root - path to the coco dataset.
            image_loader (default_image_loader) -  The function to read the images. If installed,
                                                   jpeg4py (https://github.com/ajkxyz/jpeg4py) is used by default. Else,
                                                   opencv's imread is used.
            data_fraction (None) - Fraction of images to be used. The images are selected randomly. If None, all the
                                  images  will be used
        """
        root = env_settings().ovis_dir if root is None else root
        super().__init__(root, image_loader)
        if split == 'train':
            self.img_pth = os.path.join(root, 'train/')
            self.anno_path = os.path.join(root, 'annotations_coco_style/ovis_train.json')
        elif split == 'valid':
            self.img_pth = os.path.join(root, 'valid/')
            self.anno_path = os.path.join(root, 'annotations_coco_style/ovis_valid.json')
        else:
            raise ValueError('split should be train or valid')

        cache_file = os.path.join(root, 'cache_{}.json'.format(split))
        if os.path.exists(cache_file) and os.path.isfile(cache_file):
            # If available, load the pre-processed cache file containing meta-info for each sequence
            with open(cache_file, 'r') as f:
                sequence_list_dict = json.load(f)
            self.sequence_list = sequence_list_dict['sequence_list']
        else:
            # Load the COCO set.
            self.coco_set = COCO(self.anno_path)
            self.sequence_list = self._get_sequence_list()
            sequence_list_dict = {'sequence_list': self.sequence_list}
            with open(cache_file, 'w') as f:
                json.dump(sequence_list_dict, f)
            del self.coco_set

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))

    def _get_sequence_list0(self):
        ann_list = list(self.coco_set.anns.keys())
        seq_list = [a for a in ann_list if self.coco_set.anns[a]['iscrowd'] == 0]
        return seq_list

    def _get_sequence_list(self):
        def _check_validation(bbox, area, img_height, img_width):
            if img_height <= 0 or img_width <= 0:
                return False
            if bbox[2] <= 0 or bbox[3] <= 0:
                return False
            area_ratio = area / (img_height * img_width)
            aspect_ratio = bbox[2] / bbox[3]
            return bbox[2] > 0 and bbox[3] > 0 and area_ratio > 0 and aspect_ratio > 0

        instance_ids = set([v['instance_id'] for k, v in self.coco_set.anns.items()])
        seq_list = []
        # slightly slow, but more readable
        for instance_id in tqdm.tqdm(instance_ids):
            file_names = []
            segmentations = []
            bboxes = []
            areas = []
            valid = []
            anno_ids = []
            categories = []
            for k, v in self.coco_set.anns.items():
                if v['iscrowd'] != 0:
                    continue
                if v['instance_id'] == instance_id:
                    img_info = self.coco_set.imgs[v['image_id']]
                    file_names.append(img_info['file_name'])
                    segmentations.append(v['segmentation'])
                    bboxes.append(v['bbox'])
                    areas.append(v['area'])
                    valid.append(_check_validation(v['bbox'], v['area'], img_info['height'], img_info['width']))
                    assert k == v['id'], "k = {}, v[id] = {}".format(k, v['id'])
                    anno_ids.append(k)
                    categories.append(v['category_id'])

            assert len(file_names) > 0
            assert len(set(categories)) == 1
            seq = {
                "file_names": file_names,
                "segmentations": segmentations,
                "bboxes": bboxes,
                "areas": areas,
                "valid": valid,
                "anno_ids": anno_ids,
                "category": categories[0]
            }
            seq_list.append(seq)

        return seq_list

    def is_video_sequence(self):
        return True

    def has_mask(self):
        return True

    def get_name(self):
        return 'ovis'

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_sequence_info0(self, seq_id):
        anno = self._get_anno(seq_id)

        bbox = torch.Tensor(anno['bbox']).view(1, 4)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def get_sequence_info(self, seq_id):
        seq = self.sequence_list[seq_id]
        bbox = torch.tensor(seq['bboxes'], dtype=torch.float32)
        valid = torch.tensor(seq['valid'], dtype=torch.bool)
        visible = valid.clone().byte()
        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_anno(self, seq_id):
        anno = self.coco_set.anns[self.sequence_list[seq_id]]
        return anno

    def _get_frames(self, seq_id):
        path = self.coco_set.loadImgs([self.coco_set.anns[self.sequence_list[seq_id]]['image_id']])[0]['file_name']
        img = self.image_loader(os.path.join(self.img_pth, path))
        '''add mask'''
        mask = self.coco_set.annToMask(self.coco_set.anns[self.sequence_list[seq_id]])
        mask_img = mask[..., np.newaxis]
        return (img, mask_img)

    def get_meta_info(self, seq_id):
        try:
            cat_dict_current = self.cats[self.coco_set.anns[self.sequence_list[seq_id]]['category_id']]
            object_meta = OrderedDict({'object_class': cat_dict_current['name'],
                                       'motion_class': None,
                                       'major_class': cat_dict_current['supercategory'],
                                       'root_class': None,
                                       'motion_adverb': None})
        except:
            object_meta = OrderedDict({'object_class': None,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})
        return object_meta

    def get_frames0(self, seq_id, frame_ids, anno=None):

        frame_mask_list = [self._get_frames(seq_id) for f in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        # Create anno dict
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[0, ...].clone() for f_id in frame_ids]
        '''return both frame and mask'''
        frame_list = [f for f, m in frame_mask_list]
        mask_list = [m for f, m in frame_mask_list]
        return frame_list, mask_list, anno_frames, None

    def get_frames(self, seq_id, frame_ids, seq_info_dict=None):
        seq = self.sequence_list[seq_id]
        frame_list = []
        mask_list = []
        bbox_list = []
        for f_id in frame_ids:
            img = self.image_loader(os.path.join(self.img_pth, seq['file_names'][f_id]))
            frame_list.append(img)

            segmentation_dict = seq['segmentations'][f_id]
            # compressed_rle = mask_utils.frPyObjects(segmentation_dict,
            #                                         segmentation_dict['size'][0],
            #                                         segmentation_dict['size'][1])
            mask = mask_utils.decode(segmentation_dict)
            mask = mask[..., np.newaxis]
            mask_list.append(mask)

            bbox = seq_info_dict['bbox'][f_id]
            bbox_list.append(bbox)
        bbox_dict = {"bbox": bbox_list}
        return frame_list, mask_list, bbox_dict, None


def main():
    dataset = OVIS("train", root="/home/tiger/ws/tracking_data/OVIS")
    # print(len(dataset))
    # print(dataset.has_mask())
    # for k, v in dataset.coco_set.imgs.items():
    #     v_str = "k = {}, ".format(k)
    #     for k2, v2 in v.items():
    #         v_str += "v[{}] = {}, ".format(k2, v2)
    #     print(v_str)
    #
    # instance_ids = []
    # occ_types = []
    # for k, v in dataset.coco_set.anns.items():
    #     print(k, v.keys())
    #     print(k, v['id'], v['video_id'], v['image_id'], v['instance_id'], v['bbox'], v['area'])
    #     instance_ids.append(v['instance_id'])
    # print(len(set(instance_ids)))

    # cnt = 0
    # for seq_id in range(dataset.get_num_sequences()):
    #     seq_info_dict = dataset.get_sequence_info(seq_id)
    #     visible = seq_info_dict['visible'].tolist()
    #     for i in range(len(visible)):
    #         if visible[i] == 1:
    #             dataset.get_frames(seq_id, [i], seq_info_dict, cnt)
    #             cnt += 1


if __name__ == '__main__':
    main()
