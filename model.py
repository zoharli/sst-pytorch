"""
Build graph for both training and inference
"""

import torch
import math
from torch import nn
from torch.autograd import Variable


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
            self.rnn=RNN(LSTMCell,options['video_feat_dim'],options['rnn_size'],options['num_rnn_layers'],dropout=options['rnn_drop'])
        elif options['rnn_type'] == 'gru':
            self.rnn=RNN(GRUCell,options['video_feat_dim'],options['rnn_size'],options['num_rnn_layers'],dropout=options['rnn_drop'])
        self.fc=nn.Linear(options['rnn_size'],options['num_anchors'])
        self._init()

    def forward(self,x,length):
        x,_=self.rnn(x,length)
        x=self.fc(x)
        return x

    def _init(self):
        for module in self.modules():
            for name,param in module.named_parameters():
                if 'weight' in name:
                    nn.init.uniform_(param,-self.options['init_scale'],self.options['init_scale'])
                elif 'bias' in name:
                    nn.init.constant_(param,0.)

class LSTMCell(nn.Module):

    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_bias=True):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """

        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 4 * hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        # The bias is just set to zero vectors.
        if self.use_bias:
            nn.init.constant_(self.bias.data, val=0)


    def forward(self, input_, hx):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).

        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        h_0, c_0 = hx
        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0)
                      .expand(batch_size, *self.bias.size()))
        wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        f, i, o, g = torch.split(wh_b + wi,
                                 self.hidden_size, dim=1)
        c_1 = torch.sigmoid(f)*c_0 + torch.sigmoid(i)*torch.tanh(g)
        h_1 = torch.sigmoid(o) * torch.tanh(c_1)
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, use_bias=True):

        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 3 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 3 * hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(6 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        # The bias is just set to zero vectors.
        if self.use_bias:
            nn.init.constant_(self.bias.data, val=0)


    def forward(self, input_, h_0):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            h_0: contains the initial hidden state
                , where the size of states is
                (batch, hidden_size).

        Returns:
            h_1 : Tensors containing the next hidden.
        """

        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0)
                      .expand(batch_size, *self.bias.size()))
        bias_ih,bias_hh = torch.split(bias_batch,self.hidden_size*3,dim=1)
        wi_b = torch.addmm(bias_ih, input_, self.weight_ih)
        wh_b = torch.addmm(bias_hh, h_0, self.weight_hh)
        ri,zi,ni=torch.split(wi_b,self.hidden_size,dim=1)
        rh,zh,nh=torch.split(wh_b,self.hidden_size,dim=1)
        r=torch.sigmoid(ri+rh)
        z=torch.sigmoid(zi+zh)
        n=torch.tanh(ni+r*nh)
        h_1=(1-z)*n+z*h_0
        return h_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class RNN(nn.Module):

    """A module that runs multiple steps of LSTM."""

    def __init__(self, cell_class, input_size, hidden_size, num_layers=1,
                 use_bias=True, batch_first=False, dropout=0, **kwargs):
        super(RNN, self).__init__()
        self.cell_class = cell_class
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.dropout = dropout

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = cell_class(input_size=layer_input_size,
                              hidden_size=hidden_size,
                              use_bias=self.use_bias)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    def reset_parameters(self):
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            cell.reset_parameters()

    @staticmethod
    def _forward_rnn(cell, input_, length, hx):
        max_time = input_.size(0)
        output = []
        for time in range(max_time):
            if isinstance(cell,LSTMCell):
                h_next, c_next = cell(input_[time], hx)
                mask = Variable((time < length).float().unsqueeze(1).expand_as(h_next))
                h_next = h_next*mask + hx[0]*(1 - mask)
                c_next = c_next*mask + hx[1]*(1 - mask)
                hx_next = (h_next, c_next)
                output.append(h_next)
                hx = hx_next
            elif isinstance(cell,GRUCell):
                h_next = cell(input_[time],hx)
                mask = Variable((time < length).float().unsqueeze(1).expand_as(h_next))
                h_next = h_next*mask + hx*(1 - mask)
                output.append(h_next)
                hx = h_next
        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input_, length=None, hx=None):
        if self.batch_first:
            input_ = input_.transpose(0, 1)
        max_time, batch_size, _ = input_.size()
        if length is None:
            length = Variable(torch.LongTensor([max_time] * batch_size))
        elif not isinstance(length,torch.LongTensor):
            length = torch.LongTensor(length)
        if hx is None:
            hx = Variable(input_.data.new(batch_size, self.hidden_size).zero_())
            if self.cell_class == LSTMCell:
                cx = Variable(input_.data.new(batch_size, self.hidden_size).zero_())
                hx = (hx, cx)
        if input_.is_cuda:
            length=length.cuda()
            if self.cell_class == LSTMCell:
                hx=(hx[0].cuda(),hx[1].cuda())
            elif self.cell_class == GRUCell:
                hx=hx.cuda()
        h = []
        layer_output = None
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            input_ = self.dropout_layer(input_)
            layer_output, layer_h_n = RNN._forward_rnn(
                cell=cell, input_=input_, length=length, hx=hx)
            layer_output = self.dropout_layer(layer_output)
            input_=layer_output
            h.append(layer_h_n)
        output = layer_output
        if self.cell_class == LSTMCell:
            h_n = torch.stack(list(map(lambda x:x[0],h)), 0)
            c_n = torch.stack(list(map(lambda x:x[0],h)), 0)
            h_n = (h_n,c_n)
        elif self.cell_class == GRUCell:
            h_n = torch.stack(h,0)
        return output, h_n 
