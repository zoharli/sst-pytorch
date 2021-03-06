"""
Default hyper parameters
Use this setting can obtain similar or even better performance as original SST paper
"""

from collections import OrderedDict
import numpy as np
import sys
import json
import time
import os

def default_options():

    options = OrderedDict()
    
    #*** DATA ***#
    options['feature_data_path'] =  'dataset/thumos14/features/thumos14_c3d_fc6.hdf5'   # path to feature data
    options['proposal_data_path'] = 'dataset/thumos14/gt_proposals'                     # path to proposal ground truth data
    options['anchor_weight_path'] = 'dataset/thumos14/anchors/weights.json'             # path to anchor weight path (weighting positive and negative classes, obtain weight matrix in shape of (num_anchors, 2) before training your model)
    options['c3d_resolution'] = 16   # resolution in frames (16 frames per feature)

    #*** MODEL CONFIG ***#
    options['video_feat_dim'] = 4096 # dim of video feature
    options['num_rnn_layers'] = 1   # number of RNN layers
    options['rnn_size'] = 128      # hidden neuron size
    options['rnn_type'] = 'mann'      # LSTM or GRU
    options['rnn_drop'] = 0.4        # rnn dropout ratio 
    options['num_anchors'] = 32      # number of anchors
    
    
    #*** MEMORY CONFIG***#
    options['mem_cap']=32
    options['max_tau']=options['mem_cap']-1
    options['step_tau']=2
    options['head_size']=30
    options['time_fac']=10   
    #*** OPTIMIZATION ***#
    options['train_id'] = 1          # train id (useful when you have multiple runs, store checkpoints from diff runs into different folders: "checkpoints/1", "checkpoints/2", ...)
    options['gpu']=''
    options['use_weight'] = True     # whether use pre-calculated weights for positive/negative samples (deal with imbalance class problem)
    options['solver'] = 'adam'       # 'adam','rmsprop','sgd', or 'momentum'
    options['momentum'] =  0.9       # only valid when solver is set to momentum optimizer
    options['batch_size'] = 80     # training batch size
    options['eval_batch_size'] = 40  # evaluation (loss) batch size
    options['test_batch_size'] = 1  # evaluation (loss) batch size
    options['lr'] = 1e-4             # initial learning rate (I fix learning rate to 1e-3 during training phase)
    options['reg'] = 1e-5            # regularization strength (control L2 regularization ratio)
    options['init_scale'] = 0.08     # the init scale for uniform distribution
    options['max_epochs'] = 200   # maximum training epochs to run
    options['init_epoch'] = 0        # initial epoch (useful when you needs to continue from some checkpoints)
    options['n_eval_per_epoch'] = 1 # number of evaluations per epoch
    options['eval_init'] = False     # whether to evaluate the initialized model
    options['shuffle'] = True        # whether do data shuffling for training set
    options['clip_gradient_norm'] = 100      # threshold to clip gradients: avoid gradient exploding problem; set to -1 to remove gradient clipping
    options['log_input_min']  = 1e-20          # minimum input to the log() function
    options['sample_len'] = 2048//16 + 1        # the length ratio of the sampled stream compared to the video 
    options['proposal_tiou_threshold'] = 0.5   # tiou threshold to generate positive samples, when changed, re-calculate class weights for positive/negative class
    options['n_iters_display'] = 1             # display frequency

    #*** INFERENCE ***#
    options['proposal_score_threshold'] = 0.2  # score threshold to select proposals
    options['nms_threshold'] = 0.8             # threshold for non-maximum suppression
    options['tiou_measure'] = list(np.linspace(0.5, 1.0, 11))  # tIoU thresholds for calculating recall
    #options['tiou_measure'] = [0.8]
    

    return options

def later_options(options):
    options['ckpt_prefix'] = 'checkpoints/' + str(options['train_id']) + '/' # folder path to save checkpionts during training 
    options['init_from'] = options['ckpt_prefix']+'best_model.pth'        # initial model path (set it to empty string when using ranom initialization)
    options['out_json_file'] = 'results/%d/predict_proposals.json'%options['train_id'] # output json file to save prediction results
    if not os.path.exists(options['ckpt_prefix']):
        os.mkdir(options['ckpt_prefix'])
    return options
