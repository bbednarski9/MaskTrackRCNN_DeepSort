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

# from feature extractor.py

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
            print(os.getcwd())
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

    def update(self, bbox_xyxy, confidences, features, cls_labels, ori_img, ori_img_shape):
        self.height, self.width = ori_img_shape[0], ori_img_shape[1]
        print("Image dimensions: ", self.height, ", ", self.width)
        # generate detections
        #features = self._get_features(bbox_xywh, ori_img)
        print("XYXY:")
        print(bbox_xyxy)
        bbox_tlwh = self._xyxy_to_tlwh(bbox_xyxy)
        print("TLWH:")
        print(bbox_tlwh)
        features = features.cpu().numpy()
        detections = [Detection(bbox_tlwh[i], conf, features[i], cls_labels[i]) for i,conf in enumerate(confidences) if conf>self.min_confidence]

        # run on non-maximum supression
        # boxes = np.array([d.tlwh for d in detections])
        # scores = np.array([d.confidence for d in detections])
        # indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        # detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            print("BOX: ", box)
            x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
            print("BOX1: ", box)
            box1 = x1,y1,x2,y2
            print(box1)
            track_id = track.track_id
            track_label = track.cls_label
            outputs.append(np.array([x1,y1,x2,y2,track_id,track_label], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs,axis=0)
        return outputs


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
        # need to initialize the weights in the extractor object
        if self.fcs is not None:
            for fc in self.fcs:
                nn.init.normal_(fc.weight, 0, 0.01)
                nn.init.constant_(fc.bias, 0)
        return


    def forward(self,x):
        # if self.with_avg_pool:
        #     x = self.avg_pool(x)
        # x = x.view(x.size(0), -1)

        # track_cls_score = self.Extractor(x)
        # return track_cls_score
        return

    def loss(self,
             cls_score,
             labels,
             label_weights,
             reduce=True):

        losses = dict()
        # criterion = torch.nn.CrossEntropyLoss()
        # if cls_score is not None:
        #     losses['track_loss_cls'] = criterion(
        #         cls_score, labels, label_weights, reduce=reduce)
        #     losses['acc'] = accuracy(cls_score, labels)):
        # losses = dict()
        return losses

        # remainder from deep_sort_pytorch train.py line 92-107
        # # accumurating
        # training_loss += loss.item()
        # train_loss += loss.item()
        # correct += outputs.max(dim=1)[1].eq(labels).sum().item()
        # total += labels.size(0)
        #
        # # print
        # if (idx+1)%interval == 0:
        #     end = time.time()
        #     print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
        #         100.*(idx+1)/len(trainloader), end-start, training_loss/interval, correct, total, 100.*correct/total
        #     ))
        #     training_loss = 0.
        #     start = time.time()
        #
        # return train_loss/len(trainloader), 1.- correct/total


# @HEADS.register_module
# class TrackHeadDeepSort(nn.Module):
#
#     def __init__(self,
#                     with_avg_pool=False,
#                     num_fcs = 2,
#                     in_channels=256,
#                     roi_feat_size=7,
#                     fc_out_channels=1024,
#                     match_coeff=None,
#                     bbox_dummy_iou=0,
#                     dynamic=True,
#                     max_dist=0.2,
#                     min_confidence=0.3,
#                     nms_max_overlap=1.0,
#                     max_iou_distance=0.7,
#                     max_age=70, n_init=3,
#                     nn_budget=100,
#                     use_cuda=True):
#         # from deepsort
#         super(TrackHeadDeepSort, self).__init__()
#
#         self.min_confidence = min_confidence
#         self.nms_max_overlap = nms_max_overlap
#
#         #self.extractor = Extractor(model_path=None, use_cuda=use_cuda)
#         #max_cosine_distance = max_dist
#         nn_budget = 100
#         metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
#         self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)
#
#         # from trackhead.py
#         #self.fcs = nn.ModuleList() # print and verify layers at some proposal_inputs
#
#
#     def update(self, bbox_xywh, confidences, ori_img):
#         self.height, self.width = ori_img.shape[:2]
#         # generate detections
#         features = self._get_features(bbox_xywh, ori_img)
#         bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
#         detections = [Detection(bbox_tlwh[i], conf, features[i]) for i,conf in enumerate(confidences) if conf>self.min_confidence]
#
#         # run on non-maximum supression
#         boxes = np.array([d.tlwh for d in detections])
#         scores = np.array([d.confidence for d in detections])
#         indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
#         detections = [detections[i] for i in indices]
#
#         # update tracker
#         self.tracker.predict()
#         self.tracker.update(detections)
#
#         # output bbox identities
#         outputs = []
#         for track in self.tracker.tracks:
#             if not track.is_confirmed() or track.time_since_update > 1:
#                 continue
#             box = track.to_tlwh()
#             x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
#             track_id = track.track_id
#             outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
#         if len(outputs) > 0:
#             outputs = np.stack(outputs,axis=0)
#         return outputs
#
#
#     """
#     TODO:
#         Convert bbox from xc_yc_w_h to xtl_ytl_w_h
#     Thanks JieChen91@github.com for reporting this bug!
#     """
#     @staticmethod
#     def _xywh_to_tlwh(bbox_xywh):
#         if isinstance(bbox_xywh, np.ndarray):
#             bbox_tlwh = bbox_xywh.copy()
#         elif isinstance(bbox_xywh, torch.Tensor):
#             bbox_tlwh = bbox_xywh.clone()
#         bbox_tlwh[:,0] = bbox_xywh[:,0] - bbox_xywh[:,2]/2.
#         bbox_tlwh[:,1] = bbox_xywh[:,1] - bbox_xywh[:,3]/2.
#         return bbox_tlwh
#
#
#     def _xywh_to_xyxy(self, bbox_xywh):
#         x,y,w,h = bbox_xywh
#         x1 = max(int(x-w/2),0)
#         x2 = min(int(x+w/2),self.width-1)
#         y1 = max(int(y-h/2),0)
#         y2 = min(int(y+h/2),self.height-1)
#         return x1,y1,x2,y2
#
#     def _tlwh_to_xyxy(self, bbox_tlwh):
#         """
#         TODO:
#             Convert bbox from xtl_ytl_w_h to xc_yc_w_h
#         Thanks JieChen91@github.com for reporting this bug!
#         """
#         x,y,w,h = bbox_tlwh
#         x1 = max(int(x),0)
#         x2 = min(int(x+w),self.width-1)
#         y1 = max(int(y),0)
#         y2 = min(int(y+h),self.height-1)
#         return x1,y1,x2,y2
#
#     def _xyxy_to_tlwh(self, bbox_xyxy):
#         x1,y1,x2,y2 = bbox_xyxy
#
#         t = x1
#         l = y1
#         w = int(x2-x1)
#         h = int(y2-y1)
#         return t,l,w,h
#
#     def _get_features(self, bbox_xywh, ori_img):
#         im_crops = []
#         for box in bbox_xywh:
#             x1,y1,x2,y2 = self._xywh_to_xyxy(box)
#             im = ori_img[y1:y2,x1:x2]
#             im_crops.append(im)
#         if im_crops:
#             features = self.extractor(im_crops)
#         else:
#             features = np.array([])
#         return features
#
#     # def init_weights(self):
#     #     # need to initialize the weights in the extractor object
#     #     for fc in self.fcs:
#     #         nn.init.normal_(fc.weight, 0, 0.01)
#     #         nn.init.constant_(fc.bias, 0)
#     #     return
#
#     def forward(self, x, ref_x, x_n, ref_x_n):
#         # x and ref_x are the grouped bbox features of current and reference frame
#         # x_n are the numbers of proposals in the current images in the mini-batch,
#         # ref_x_n are the numbers of ground truth bboxes in the reference images.
#         # here we compute a correlation matrix of x and ref_x
#         # we also add a all 0 column denote no matching
#         assert len(x_n) == len(ref_x_n)
#
#         return self.Extractor(x)
#
#     def train_forward(self, inputs):
#         inputs = inputs.to(device)
#         outputs = net(inputs)
#         return outputs
#
#     def loss(self):
#         # we will eventually want to use the reparameterized "Cosine_Metric Learning" Loss
#         # for now just use a cross entropy loss
#         criterion = torch.nn.CrossEntropyLoss()
#         optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)
#
#         loss = criterion(outputs, labels)
#
#         # backward
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         return
#
#         # remainder from deep_sort_pytorch train.py line 92-107
#         # # accumurating
#         # training_loss += loss.item()
#         # train_loss += loss.item()
#         # correct += outputs.max(dim=1)[1].eq(labels).sum().item()
#         # total += labels.size(0)
#         #
#         # # print
#         # if (idx+1)%interval == 0:
#         #     end = time.time()
#         #     print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
#         #         100.*(idx+1)/len(trainloader), end-start, training_loss/interval, correct, total, 100.*correct/total
#         #     ))
#         #     training_loss = 0.
#         #     start = time.time()
#         #
#         # return train_loss/len(trainloader), 1.- correct/total
