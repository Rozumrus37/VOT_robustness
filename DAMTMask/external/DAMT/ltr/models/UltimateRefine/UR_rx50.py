import torch.nn as nn
import ltr.models.backbone as backbones
# from ltr.models.neck.PtCorr import PtCorr as Corr
from ltr.models.neck.space_time_memory_reader import SpaceTimeMemoryReader
from ltr.models.head import bbox, corner_coarse, mask, decoder
from ltr import model_constructor


class SEcmnet(nn.Module):
    """ Scale Estimation network module with three branches: bbox, coner and mask. """

    def __init__(self, feature_extractor_m, feature_extractor_q, key_value_m, key_value_q,
                 neck_module, head_module, used_layers, extractor_grad=True, unfreeze_layer3=False):
        """
        args:
            feature_extractor - backbone feature extractor
            bb_regressor - IoU prediction module
            bb_regressor_layer - List containing the name of the layers from feature_extractor, which are input to
                                    bb_regressor
            extractor_grad - Bool indicating whether backbone feature extractor requires gradients
        """
        super(SEcmnet, self).__init__()

        self.feature_extractor_m = feature_extractor_m
        self.feature_extractor_q = feature_extractor_q
        self.key_value_m = key_value_m
        self.key_value_q = key_value_q
        self.neck = neck_module
        assert len(head_module) == 2
        self.corner_head, self.decoder_head = head_module
        self.used_layers = used_layers

        # self.feat_adjust = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
        #         nn.BatchNorm2d(64),
        #         nn.ReLU()
        #     ),
        #
        #     nn.Sequential(
        #         nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0),
        #         nn.BatchNorm2d(256),
        #         nn.ReLU()
        #     ),
        #
        #     nn.Sequential(
        #         nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0),
        #         nn.BatchNorm2d(512),
        #         nn.ReLU()
        #     )
        # ])

        if not extractor_grad:
            for p in self.feature_extractor_m.parameters():
                p.requires_grad_(False)
            for p in self.feature_extractor_q.parameters():
                p.requires_grad_(False)
        if unfreeze_layer3:
            for p in self.feature_extractor_m.layer3.parameters():
                p.requires_grad_(True)
            for p in self.feature_extractor_q.layer3.parameters():
                p.requires_grad_(True)

    def forward(self, train_imgs, test_imgs, masks, mode='train'):
        """ Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5, then the batch dimension
        corresponds to the first dimensions. test_imgs is thus of the form [sequence, batch, feature, row, col]
        """
        m_key, m_value = self.forward_ref(train_imgs, masks)
        pred_dict = self.forward_test(test_imgs, m_key, m_value, mode)
        return pred_dict

    def forward_ref(self, train_imgs, masks):
        """ Forward pass of reference branch """
        '''train_feat_dict's dtype is OrderedDict,key is 'layer3' '''
        # Extract backbone features
        train_feat_dict = self.feature_extractor_m(train_imgs.view(-1, *train_imgs.shape[-3:]),
                                                   masks.view(-1, *masks.shape[-3:]),
                                                   self.used_layers)  # `OrderedDict{'layer3':Tensor}`
        train_feat = [feat for feat in train_feat_dict.values()][-1]
        m_key, m_value = self.key_value_m(train_feat)

        return m_key, m_value

    def forward_test(self, test_imgs, m_key, m_value, mode='train', branches=['corner']):
        """ Forward pass of test branch. size of test_imgs is (1, batch, 3, 256, 256)"""
        # for debugging

        output = {}
        # Extract backbone features
        test_feat_dict = self.feature_extractor_q(test_imgs.view(-1, *test_imgs.shape[-3:]),
                                                  ['conv1', 'layer1', 'layer2', 'layer3'])
        # Save low-level feature list
        Lfeat_list = [feat for name, feat in test_feat_dict.items() if name != 'layer3']

        q_key, q_value = self.key_value_q(test_feat_dict['layer3'])
        mem_out = self.neck(m_key, m_value, q_key, q_value)

        # Obtain bbox prediction
        if mode == 'train':
            output['corner'] = self.corner_head(mem_out)
            # Lfeat_list = [self.feat_adjust[idx](feat) for idx, feat in enumerate(Lfeat_list)]
            # output['mask'] = self.mask_head(fusion_feat, Lfeat_list)
            output['mask'] = self.decoder_head(mem_out, Lfeat_list[-1], Lfeat_list[-2], Lfeat_list[-3])

        elif mode == 'test':
            output['feat'] = mem_out
            if 'corner' in branches:
                output['corner'] = self.corner_head(mem_out)
            if 'mask' in branches:
                # Lfeat_list = [self.feat_adjust[idx](feat) for idx, feat in enumerate(Lfeat_list)]
                # output['mask'] = self.mask_head(fusion_feat, Lfeat_list)
                output['mask'] = self.decoder_head(mem_out, Lfeat_list[-1], Lfeat_list[-2], Lfeat_list[-3])
        else:
            raise ValueError("mode should be train or test")
        return output

    def get_output(self, mode):
        raise NotImplementedError("get_output function is called.")
        # if mode == 'corner':
        #     return self.corner_head(self.fusion_feat)
        # elif mode == 'mask':
        #     return self.mask_head(self.fusion_feat, self.Lfeat_list)
        # else:
        #     raise ValueError('mode should be bbox or corner or mask')


@model_constructor
def URcm_resnet50(backbone_pretrained=True, used_layers=('layer3',), output_sz=16, unfreeze_layer3=False):
    # backbone
    backbone_net_m = backbones.EncoderM(depth=50, pretrained=backbone_pretrained)
    backbone_net_q = backbones.EncoderQ(depth=50, pretrained=backbone_pretrained)
    key_value_m = backbones.KeyValue(in_channels=1024, key_channels=256, value_channels=512)
    key_value_q = backbones.KeyValue(in_channels=1024, key_channels=256, value_channels=512)

    # neck
    stm_reader = SpaceTimeMemoryReader()

    # multiple heads
    corner_head = corner_coarse.Corner_Predictor(inplanes=1024, output_sz=output_sz)  # 64
    # mask_head = mask.Mask_Predictor_fine(pool_size)
    decoder_head = decoder.Decoder(256)

    # net
    net = SEcmnet(feature_extractor_m=backbone_net_m,
                  feature_extractor_q=backbone_net_q,
                  key_value_m=key_value_m,
                  key_value_q=key_value_q,
                  neck_module=stm_reader,
                  head_module=(corner_head, decoder_head),
                  used_layers=used_layers, extractor_grad=True,
                  unfreeze_layer3=unfreeze_layer3)
    return net
