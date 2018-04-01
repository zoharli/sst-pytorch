#-*- coding: utf-8 -*-

"""
Data provider: to provide train/val/test data to the model
"""
import random
import numpy as np
import os
import h5py
from collections import OrderedDict
import json
from opt import *
import random
import math
import torch

class DataProvision:
    def __init__(self, options,split='train'):
        
        self._options = options
        self.split=split


        self._anchors = list(range(self._options['c3d_resolution'], (self._options['num_anchors'] + 1) * self._options['c3d_resolution'], self._options['c3d_resolution']))  # proposal anchors (in frame number)
        
        print('Data Size:')
        proposal_data = json.load(open(os.path.join(self._options['proposal_data_path'], 'thumos14_temporal_proposal_%s.json'%split), 'r'))
        self._ids = list(proposal_data.keys())
        self._sizes = len(self._ids)
        self._localization = proposal_data

        print('%s-split: %d videos.'%(split, self._sizes))

        # feature dictionary
        print('Loading c3d features ...')
        features = h5py.File(self._options['feature_data_path'], 'r')
        self._features = {video_id:features[video_id]['c3d_features'].value for video_id in self._ids}

        # load label weight data
        print('Loading anchor weight data ...')
        self._proposal_weight = json.load(open(self._options['anchor_weight_path'], 'r'))

        if not self._options['use_weight']:
            self._proposal_weight = np.ones(shape=(self._options['num_anchors'], 2)) / 2.0

        for i in range(len(self._proposal_weight)):
            self._proposal_weight[i][0] /= self._proposal_weight[i][1]
            self._proposal_weight[i][1] = 1.
        
        self._proposal_weight=torch.Tensor(self._proposal_weight)[:,0]

        print('Done %s set loading.'%split)

    def __getitem__(self,index):

        c3d_resolution = self._options['c3d_resolution']
        vid = self._ids[index]

        feature = self._features[vid]
        feature_len = feature.shape[0]

            
            # sampling
        if self.split == 'train':
            sample_len = int(self._options['sample_len'])
        else:
            sample_len = feature_len

        # starting feature id relative to original video
        start_feat_id = random.randint(0, max((feature_len-sample_len), 0))
        end_feat_id = min(start_feat_id+sample_len, feature_len)
        feature = feature[start_feat_id:end_feat_id]
        start_frame_id = start_feat_id * c3d_resolution + c3d_resolution // 2
        end_frame_id = (end_feat_id - 1) * c3d_resolution + c3d_resolution // 2

        # the ground truth proposal and caption should be changed according to the sampled stream
        localization = self._localization[vid]
        framestamps = localization['framestamps']

        n_anchors = self._options['num_anchors']

        # generate proposal groud truth data
        gt_proposal = np.zeros(shape=(sample_len, n_anchors), dtype=np.int32)
        for stamp_id, stamp in enumerate(framestamps):
            start = stamp[0]
            end = stamp[1]

            # only need to check whether proposals that have end point at region of (frame_check_start, frame_check_end) are "correct" proposals
            start_point = max((start + end) // 2, 0)
            end_point = end + (end - start + 1)
            frame_check_start, frame_check_end = self.get_intersection((start_point, end_point + 1), (start_frame_id, end_frame_id+1))
            feat_check_start, feat_check_end = frame_check_start // c3d_resolution, frame_check_end // c3d_resolution

            for feat_id in range(feat_check_start, feat_check_end + 1):
                frame_id = feat_id*c3d_resolution + c3d_resolution/2
                for anchor_id, anchor in enumerate(self._anchors):
                    pred = (frame_id + 1- anchor, frame_id + 1)
                    tiou = self.get_iou(pred, (start, end + 1))
                    
                    if tiou > 0.5:
                        gt_proposal[feat_id-start_feat_id, anchor_id] = 1
           
        feature=torch.Tensor(feature)
        gt_proposal=torch.Tensor(gt_proposal)
        mask=torch.ones(gt_proposal.shape)
        return feature,gt_proposal,feature.shape[0],mask

    def __len__(self):
        return self._sizes

    def get_iou(self, pred, gt):
        start_pred, end_pred = pred
        start, end = gt
        intersection = max(0, min(end, end_pred) - max(start, start_pred))
        union = min(max(end, end_pred) - min(start, start_pred), end-start + end_pred-start_pred)
        iou = float(intersection) / (union + 1e-8)

        return iou

    def get_intersection(self, region1, region2):
        start1, end1 = region1
        start2, end2 = region2
        start = max(start1, start2)
        end = min(end1, end2)

        return (start, end)


#-------------------------dataloader----------------------
def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label,length,mask)
        """
        # find longest sequence
        max_len = max(list(map(lambda x: x[0].shape[self.dim], batch)))
        # pad according to max_len
        # stack all
        xs = torch.stack(list(map(lambda x: pad_tensor(x[0],max_len,self.dim), batch)), dim=1)
        ys = torch.stack(list(map(lambda x: pad_tensor(x[1],max_len,self.dim), batch)), dim=1)
        lengths=[x[2] for x in batch]
        mask=torch.stack(list(map(lambda x: pad_tensor(x[3],max_len,self.dim), batch)), dim=1)
        
        return xs, ys,lengths,mask

    def __call__(self, batch):
        return self.pad_collate(batch)
