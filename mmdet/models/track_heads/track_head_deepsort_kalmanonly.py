# imports from deep sort
from ..registry import HEADS

from .deep_sort.deep.feature_extractor import Extractor
from .deep_sort.sort.nn_matching import NearestNeighborDistanceMetric
from .deep_sort.sort.preprocessing import non_max_suppression
from .deep_sort.sort.detection import Detection
from .deep_sort.sort.tracker import Tracker

# imports from trackhead.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmdet.core import (delta2bbox, multiclass_nms, bbox_target,
                        weighted_cross_entropy, weighted_smoothl1, accuracy)
from ..registry import HEADS

from mmdet.models.track_heads.deep_sort import build_tracker
from mmdet.models.track_heads.deep_sort.utils.parser import get_config

import os

@HEADS.register_module
class TrackHeadDeepSortKalmanOnly(nn.Module):

    def __init__(self,
                    with_avg_pool=False,
                    num_fcs = 2,
                    in_channels=256,
                    roi_feat_size=7,
                    fc_out_channels=1024,
                    match_coeff=None,
                    bbox_dummy_iou=0,
                    dynamic=True,
                    max_dist=0.2,
                    min_confidence=0.3,
                    nms_max_overlap=1.0,
                    max_iou_distance=0.7,
                    max_age=70, n_init=3,
                    nn_budget=100,
                    use_cuda=True,
                    num_classes=41):
        # from deepsort
        super(TrackHeadDeepSortKalmanOnly, self).__init__()

        ######
        # copied from trackhead for compatibility
        self.in_channels = in_channels
        self.with_avg_pool = with_avg_pool
        self.roi_feat_size = roi_feat_size
        self.match_coeff = match_coeff
        self.match_coeff
        self.bbox_dummy_iou = bbox_dummy_iou
        self.num_fcs = num_fcs
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(roi_feat_size)
        else:
            in_channels *= (self.roi_feat_size * self.roi_feat_size)
        self.fcs = nn.ModuleList()
        for i in range(num_fcs):
            in_channels = (in_channels
                          if i == 0 else fc_out_channels)
            fc = nn.Linear(in_channels, fc_out_channels)
            self.fcs.append(fc)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None
        self.dynamic=dynamic
        ######

        ######
        # copied from trackheaddeepsort
        self.deepsort_flag = True
        if self.deepsort_flag:
            self.use_cuda = torch.cuda.is_available()
            cfg = get_config()
            #print(os.getcwd())
            cfg.merge_from_file(os.getcwd()+"/mmdet/models/track_heads/deep_sort/configs/deep_sort.yaml")
            if not use_cuda:
                warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

            self.min_confidence = cfg.DEEPSORT.MIN_CONFIDENCE
            self.nms_max_overlap = cfg.DEEPSORT.NMS_MAX_OVERLAP
            self.num_classes=num_classes
            #self.extractor = Extractor(num_classes=self.num_classes, model_path=None, use_cuda=use_cuda)
            max_cosine_distance = cfg.DEEPSORT.MAX_DIST
            nn_budget = nn_budget=cfg.DEEPSORT.NN_BUDGET
            metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
            self.tracker = Tracker(metric, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE, max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT)
        ######
        #_ = self.tracker.predict()


    def update_and_predict(self, bbox_xyxy, confidences, features, cls_labels, ori_img, ori_img_shape):
        self.height, self.width = ori_img_shape[0], ori_img_shape[1]
        #print("Image dimensions: ", self.height, ", ", self.width)
        # generate detections
        #features = self._get_features(bbox_xywh, ori_img)
        #print("XYXY:")
        bbox_tlwh = self._xyxy_to_tlwh(bbox_xyxy)
        #print("TLWH:")
        #print(bbox_tlwh)
        features = features.cpu().numpy()
        detections = [Detection(bbox_tlwh[i], conf, features[i], cls_labels[i]) for i,conf in enumerate(confidences) if conf>self.min_confidence]

        # run on non-maximum supression
        # boxes = np.array([d.tlwh for d in detections])
        # scores = np.array([d.confidence for d in detections])
        # indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        # detections = [detections[i] for i in indices]


        # update tracker
        matches = self.tracker.update(detections)
        matched_track_ids = [match[0] for match in matches]

        # output bbox identities
        predictions = []
        track_ids = []
        covariances = []
        matches_withpreds = []
        for track in self.tracker.tracks:
            track.predict(self.tracker.kf)
            if not track.is_confirmed() or track.time_since_update > 1 or track.track_id in matched_track_ids:
                continue
            tlbr = track.to_tlbr()
            x_vals = np.clip(np.array([tlbr[0],tlbr[2]]),a_min=0,a_max=self.width)
            y_vals = np.clip(np.array([tlbr[1],tlbr[3]]),a_min=0,a_max=self.height)
            tlbr_cliped = np.array([x_vals[0], y_vals[0], x_vals[1], y_vals[1]])
            predictions.append(np.append(tlbr_cliped,track.det_conf))
            track_ids.append(track.track_id)
            covariances.append(track.covariance)
            for match in matches:
                if match[0] == track.track_id:
                    matches_withpreds.append(match)
                    matches.remove(match)

        # correct formatting?
        predictions = np.array(predictions)
        covariances = np.array(covariances)

        return predictions, covariances, matches_withpreds, track_ids


    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
        bbox_tlwh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
        return bbox_tlwh


    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2

    # convert from
    def _tlwh_to_xyxy_new(self, bbox_tlwh):
        for x,y,w,h in bbox_tlwh:
            x1 = max(int(x),0)
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        bbox_tlwh = []
        for x1,y1,x2,y2 in bbox_xyxy:
            t = x1
            l = y1
            #w = int(x2-x1)
            #h = int(y2-y1)
            w = int(x2-x1)
            h = int(y2-y1)
            bbox_tlwh.append([t,l,w,h])
        return bbox_tlwh
        # x1,y1,x2,y2 = bbox_xyxy
        #
        # t = x1
        # l = y1
        # w = int(x2-x1)
        # h = int(y2-y1)
        # return t,l,w,h

    # new
    def _xyxy_to_xywh(self, bbox_xyxy):
        bbox_xywh = []
        for x1,y1,x2,y2 in bbox_xyxy:
            w = max(int(x2-x1),0)
            h = max(int(y2-y1),0)
            bbox_xywh.append([x1,y1,w,h])
        return bbox_xywh

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1,y1,x2,y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2,x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features

    def init_weights(self):
        for fc in self.fcs:
            nn.init.normal_(fc.weight, 0, 0.01)
            nn.init.constant_(fc.bias, 0)

    def compute_comp_scores(self, match_ll, bbox_scores, bbox_ious, label_delta, add_bbox_dummy=False):
        # compute comprehensive matching score based on matchig likelihood,
        # bbox confidence, and ious
        if add_bbox_dummy:
            bbox_iou_dummy =  torch.ones(bbox_ious.size(0), 1,
                device=torch.cuda.current_device()) * self.bbox_dummy_iou
            bbox_ious = torch.cat((bbox_iou_dummy, bbox_ious), dim=1)
            label_dummy = torch.ones(bbox_ious.size(0), 1,
                device=torch.cuda.current_device())
            label_delta = torch.cat((label_dummy, label_delta),dim=1)
        if self.match_coeff is None:
            return match_ll
        else:
            # match coeff needs to be length of 3
            assert(len(self.match_coeff) == 3)

            # print("match ll: ", match_ll)
            # print("bbox scores: ", torch.log(bbox_scores))
            # print("bbox_IoUs: ", bbox_ious)
            # print("label delta: ", label_delta)
            #print("match_coeffs: ", self.match_coeff)
            return match_ll + self.match_coeff[0] * \
                torch.log(bbox_scores) + self.match_coeff[1] * bbox_ious \
                + self.match_coeff[2] * label_delta

    def forward(self, x, ref_x, x_n, ref_x_n):
        # x and ref_x are the grouped bbox features of current and reference frame
        # x_n are the numbers of proposals in the current images in the mini-batch,
        # ref_x_n are the numbers of ground truth bboxes in the reference images.
        # here we compute a correlation matrix of x and ref_x
        # we also add a all 0 column denote no matching
        assert len(x_n) == len(ref_x_n)
        if self.with_avg_pool:
            x = self.avg_pool(x)
            ref_x = self.avg_pool(ref_x)
        x = x.view(x.size(0), -1)
        ref_x = ref_x.view(ref_x.size(0), -1)
        for idx, fc in enumerate(self.fcs):
            x = fc(x)
            ref_x = fc(ref_x)
            if idx < len(self.fcs) - 1:
                x = self.relu(x)
                ref_x = self.relu(ref_x)
        n = len(x_n)
        x_split = torch.split(x, x_n, dim=0)
        ref_x_split = torch.split(ref_x, ref_x_n, dim=0)
        prods = []
        for i in range(n):

            prod = torch.mm(x_split[i], torch.transpose(ref_x_split[i], 0, 1))
            prods.append(prod)
        if self.dynamic:
            match_score = []
            for prod in prods:
                m = prod.size(0)
                dummy = torch.zeros( m, 1, device=torch.cuda.current_device())

                prod_ext = torch.cat([dummy, prod], dim=1)
                match_score.append(prod_ext)
        else:
            dummy = torch.zeros(n, m, device=torch.cuda.current_device())
            prods_all = torch.cat(prods, dim=0)
            match_score = torch.cat([dummy,prods_all], dim=2)
        return match_score


    def loss(self,
             match_score,
             ids,
             id_weights,
             reduce=True):
        losses = dict()
        if self.dynamic:
            n = len(match_score)
            x_n = [s.size(0) for s in match_score]
            ids = torch.split(ids, x_n, dim=0)
            loss_match = 0.
            match_acc = 0.
            n_total = 0
            batch_size = len(ids)
            for score, cur_ids, cur_weights in zip(match_score, ids, id_weights):
                valid_idx = torch.nonzero(cur_weights).squeeze()
                if len(valid_idx.size()) == 0: continue
                n_valid = valid_idx.size(0)
                n_total += n_valid
                loss_match += weighted_cross_entropy(
                    score, cur_ids, cur_weights, reduce=reduce)
                match_acc += accuracy(torch.index_select(score, 0, valid_idx),
                                      torch.index_select(cur_ids,0, valid_idx)) * n_valid
            losses['loss_match'] = loss_match / n
            if n_total > 0:
                losses['match_acc'] = match_acc / n_total
        else:
          if match_score is not None:
              valid_idx = torch.nonzero(cur_weights).squeeze()
              losses['loss_match'] = weighted_cross_entropy(
                  match_score, ids, id_weights, reduce=reduce)
              losses['match_acc'] = accuracy(torch.index_select(match_score, 0, valid_idx),
                                              torch.index_select(ids, 0, valid_idx))
        return losses


# # Saved from naive KF implementation
#
#     def init_weights(self):
#         # need to initialize the weights in the extractor object
#         if self.fcs is not None:
#             for fc in self.fcs:
#                 nn.init.normal_(fc.weight, 0, 0.01)
#                 nn.init.constant_(fc.bias, 0)
#         return
#
#
#     def forward(self,x):
#         # if self.with_avg_pool:
#         #     x = self.avg_pool(x)
#         # x = x.view(x.size(0), -1)
#
#         # track_cls_score = self.Extractor(x)
#         # return track_cls_score
#         return
#
#     def loss(self,
#              cls_score,
#              labels,
#              label_weights,
#              reduce=True):
#
#         losses = dict()
#         # criterion = torch.nn.CrossEntropyLoss()
#         # if cls_score is not None:
#         #     losses['track_loss_cls'] = criterion(
#         #         cls_score, labels, label_weights, reduce=reduce)
#         #     losses['acc'] = accuracy(cls_score, labels)):
#         # losses = dict()
#         return losses
