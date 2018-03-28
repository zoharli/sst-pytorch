"""
Build graph for both training and inference
"""

import torch
from torch import nn

class multi_rnn_cell(nn.Module):


class ProposalModel(nn.Module):

    def __init__(self, options):
        super(ProposalModel,self).__init__()
        self.options = options
      #  self.initializer = tf.random_uniform_initializer(
      #      minval = - self.options['init_scale'],
      #      maxval = self.options['init_scale'])
        if self.options['rnn_drop'] > 0:
            print('using dropout in rnn!')
        if options['rnn_type'] == 'lstm':
            self.rnn=nn.LSTMCell(options['video_feat_dim'],options['rnn_size'],options['num_rnn_layers'],dropout=options['rnn_drop'])
        elif options['rnn_type'] == 'gru':
            self.rnn=nn.GRU(options['video_feat_dim'],options['rnn_size'],options['num_rnn_layers'],dropout=options['rnn_drop'])
        self.fc=nn.Linear(options['rnn_size'],options['num_anchors'])
        self.sig=nn.Sigmoid()
        self._init()

    def forward(self,x,length):
        x,_=self.rnn(x)
        x=x.view(-1,self.options['rnn_size'])#??
        x=self.fc(x)
        #x=self.sig(x)
        x=x.view(-1,self.options['batch_size'],self.options['num_anchors'])
        return x

    def _init(self):
        for module in self.modules():
            for name,param in module.named_parameters():
                if 'weight' in name:
                    nn.init.uniform(param,-self.options['init_scale'],self.options['init_scale'])
                elif 'bias' in name:
                    nn.init.constant(param,0.)
