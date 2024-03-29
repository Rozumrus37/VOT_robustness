from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from .show import show_dimp_reppoint_segm

class MixFormerKLDimpReppointMaskSegmSingleActor(BaseActor):
    """ Actor for training the TSP_online and TSP_cls_online"""
    def __init__(self, net, objective, loss_weight, settings, run_score_head=False):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.run_score_head = run_score_head

        self.count = 0

    def __call__(self, data, is_train = True):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data, run_score_head=self.run_score_head)
        # compute losses
        loss, status,weight_state = self.compute_losses(out_dict, data)

        # self.count += 1
        # if self.count % 100 == 0:
        # # if not is_train:
        #     try:
        # show_dimp_reppoint_segm(
        #                 temp_mask = data['template_masks'].detach().cpu()[0][0],
        #                 temp_image = data['template_images'].detach().cpu()[0][0],
        #                 temp_bbox = box_xywh_to_xyxy(data['template_anno_crop'].detach().cpu()[0][0]),


        #                 search_mask = data['search_masks'].detach().cpu()[0][0],
        #                 pred_mask = out_dict['segm'].sigmoid().detach().cpu()[0],

        #                 search_image = data['search_images'].detach().cpu()[0][0],
        #                 search_bbox = box_xywh_to_xyxy(data['search_anno_crop'].detach().cpu()[0][0]),
        #                 pred_bbox = box_cxcywh_to_xyxy(out_dict['pred_boxes']).detach().cpu().view(-1, 4)[0],
        #                 corner_map_tl = self.net.module.box_head.vis_score_tl[0],
        #                 corner_map_br = self.net.module.box_head.vis_score_br[0],
        #                 dimp_anno = data['search_label'].detach().cpu()[0][0],
        #                 dimp_maps = [s.detach().cpu()[0][0] for s in out_dict['dimp_scores']],


        #                 reppoint_init_bboxes = out_dict['reppoint_init_bboxes'].detach().cpu()[0],
        #                 reppoint_refine_bboxes = out_dict['reppoint_refine_bboxes'].detach().cpu()[0],
        #                 reppoint_cls = out_dict['reppoint_cls'].detach().cpu().sigmoid()[0],
        #                 reppoint_init_weight = weight_state['reppoint_init_weight'].detach().cpu()[0],
        #                 reppoint_refine_weight = weight_state['reppoint_refine_weight'].detach().cpu()[0],


        #                 save_root = '/home/tiger/tracking_code/MixFormer/vis_dimp_reppoint_mask_segm')

            # except:
            #     pass

        return loss, status

    def forward_pass(self, data, run_score_head):
        num_search = len(data['search_images'])
        num_template = len(data['template_images'])

        template_mask = data['template_masks'].clone()
        template = data['template_images'].clone()
        search = data['search_images'].clone()


        template_bboxes = data['template_anno_crop'].clone() # 没有归一化的bbox
        search_bboxes = data['search_anno_crop'].clone() # 没有归一化的bbox
        # search_bboxes = box_xywh_to_xyxy(data['search_anno'].clone()) # 归一化的bbox

        out_dict, _ = self.net(template, search,
                               template_seg = template_mask,
                               run_score_head=False,
                               template_bboxes=template_bboxes,
                               search_bboxes=search_bboxes, # 没有归一化的bbox
                               ) # 没有归一化的bbox
        # out_dict: (B, N, C), outputs_coord: (1, B, N, C), target_query: (1, B, N, C)
        return out_dict

    def compute_losses(self, pred_dict, data, return_status=True):
        # Get boxes
        loss = 0
        #==============================corner
        pred_boxes_corner = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes_corner).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes_corner.size(1)
        pred_boxes_corner_vec = box_cxcywh_to_xyxy(pred_boxes_corner).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_bbox_corner = data['search_anno'].reshape(-1,4)
        gt_boxes_corner_vec = box_xywh_to_xyxy(gt_bbox_corner)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss_corner, iou_corner = self.objective['giou'](pred_boxes_corner_vec, gt_boxes_corner_vec)  # (BN,4) (BN,4)
        except:
            giou_loss_corner, iou_corner = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        mean_iou_corner = iou_corner.detach().mean()
        # compute l1 loss
        l1_loss_corner = self.objective['l1'](pred_boxes_corner_vec, gt_boxes_corner_vec)  # (BN,4) (BN,4)
        # weighted sum
        loss_bbox_corner = self.loss_weight['giou'] * giou_loss_corner + self.loss_weight['l1'] * l1_loss_corner
        loss += loss_bbox_corner
        #==============================corner

        #==============================dimp
        loss_target_classifier = 0
        loss_test_init_clf = 0
        loss_test_iter_clf = 0
        if 'test_clf' in self.loss_weight.keys():
            # Classification losses for the different optimization iterations
            clf_losses_test = [self.objective['test_clf'](s, data['search_label'], data['search_anno']) for s in pred_dict['dimp_scores']]

            # Loss of the final filter
            clf_loss_test = clf_losses_test[-1]
            loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test

            # Loss for the initial filter iteration
            if 'test_init_clf' in self.loss_weight.keys():
                loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses_test[0]

            # Loss for the intermediate filter iterations
            if 'test_iter_clf' in self.loss_weight.keys():
                test_iter_weights = self.loss_weight['test_iter_clf']
                if isinstance(test_iter_weights, list):
                    loss_test_iter_clf = sum([a * b for a, b in zip(test_iter_weights, clf_losses_test[1:-1])])
                else:
                    loss_test_iter_clf = (test_iter_weights / (len(clf_losses_test) - 2)) * sum(clf_losses_test[1:-1])
        # Total loss
        loss_dimp = loss_target_classifier + loss_test_init_clf + loss_test_iter_clf

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')
        loss += loss_dimp
        #==============================dimp


        #==============================reppoint
        reppoint_out,weight_state = self.objective['reppoint'](pred_dict, data)

        loss_reppoint_cls = reppoint_out['reppoint_cls']
        loss_reppoint_init_bbox = reppoint_out['reppoint_init_bbox']
        loss_reppoint_refine_bbox = reppoint_out['reppoint_refine_bbox']
        loss_reppoint = self.loss_weight['reppoint_cls'] * loss_reppoint_cls + self.loss_weight['reppoint_init_bbox'] * loss_reppoint_init_bbox + self.loss_weight['reppoint_refine_bbox'] * loss_reppoint_refine_bbox
        loss += loss_reppoint

        reppoint_init_iou = reppoint_out['reppoint_init_iou']
        reppoint_refine_iou = reppoint_out['reppoint_refine_iou']

        #==============================reppoint


        #==============================segm
        pred_segm = pred_dict['segm'] # [bs, 1, 96, 96]
        bs_cat,_,h,w = pred_segm.shape

        search_segm = data['search_masks'] #[num_image, bs_raw, 1, 384, 384]
        search_segm = search_segm.clone().reshape([bs_cat]+list(search_segm.shape[-3:]))
        search_segm = search_segm.to(pred_segm)

        pred_segm = torch.nn.functional.interpolate(pred_segm, size=search_segm.shape[-2:],mode="bilinear")

        pred_segm = pred_segm.reshape(bs_cat,-1,1)
        search_segm = search_segm.reshape(bs_cat,-1,1)

        segm_focal = self.objective['segm_focal'](pred_segm,search_segm)
        segm_dice = self.objective['segm_dice'](pred_segm.squeeze(),search_segm.squeeze(),pred_segm.shape[0])

        loss_segm = self.loss_weight['segm_focal'] * segm_focal + self.loss_weight['segm_dice'] * segm_dice
        loss += loss_segm
        #==============================segm




        status = {"Loss/total": loss.item(),
                      # "Loss/scores": score_loss.item(),
                      # "Loss/corner": loss_bbox_corner.item(),
                      # "Loss/dimp": loss_dimp.item(),
                      # "Loss/reppoint": loss_reppoint.item(),
                      # "Loss/segm": loss_segm.item(),
                      "Loss/segm_focal": segm_focal.item(),
                      "Loss/segm_dice": segm_dice.item(),

                      "corner_iou": mean_iou_corner.item(),
                      "init_iou": reppoint_init_iou.item(),
                      "refine_iou": reppoint_refine_iou.item(),
                      }

        if 'test_clf' in self.loss_weight.keys():
            status['Loss/target_clf'] = loss_target_classifier.item()

        return loss, status,weight_state

