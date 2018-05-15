"""
Build graph for both training and inference
"""

import torch
import math
from torch import nn
from torch.autograd import Variable
from model import *


class MaSST(nn.Module):

    def __init__(self, options):
        super().__init__()
        self.options = options
        if self.options['rnn_drop'] > 0:
            print('using dropout in rnn!')
        self.fc=nn.Linear(2*options['rnn_size'],options['num_anchors'])
        self._init()
        self.fc1=nn.Linear(options['video_feat_dim'],options['rnn_size'])
        self.rnn=MARNN(options,options['rnn_size'],options['rnn_size'],options['num_rnn_layers'],dropout=options['rnn_drop'])

    def set_tau(self,tau):
        self.rnn.set_tau(tau)

    def forward(self,x,length):
        x=self.fc1(x)
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

class MARNN(nn.Module):

    """A module that runs multiple steps of LSTM."""

    def __init__(self, options, input_size, hidden_size, num_layers=1,
                 use_bias=True, batch_first=False, dropout=0, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.head_size=options['head_size']

        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = MANNCell(options,input_size=layer_input_size,
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

    def set_tau(self,tau):
        for i in range(self.num_layers):
            cell=self.get_cell(i)
            cell.set_tau(tau)

    @staticmethod
    def _forward_rnn(cell, input_, length, hx,aux_hx):
        max_time = input_.size(0)
        output = []
        cell._reset_mem()
        for time in range(max_time):
            h_next,aux_next = cell(input_[time],hx,aux_hx)
            mask = Variable((time < length).float().unsqueeze(1).expand_as(h_next))
            h_next = h_next*mask + hx*(1 - mask)
            print(aux_next.size(),mask.size())
            aux_next = aux_next*mask + aux_hx*(1 - mask)
            output.append(h_next)
            hx = h_next
            aux_hx=aux_next
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
        aux_hx = Variable(input_.data.new(batch_size, 2*(self.head_size+1)).zero_())
        length=length.cuda(input_.get_device())
        hx=hx.cuda(input_.get_device())
        aux_hx=aux_hx.cuda(input_.get_device())
        h = []
        layer_output = None
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            input_ = self.dropout_layer(input_)
            layer_output, layer_h_n = MARNN._forward_rnn(
                cell=cell, input_=input_, length=length, hx=hx,aux_hx=aux_hx)
            layer_output = self.dropout_layer(layer_output)
            input_=layer_output
            h.append(layer_h_n)
        output = layer_output
        h_n = torch.stack(h,0)
        return output, h_n 

class MANNCell(nn.Module):

    def __init__(self,options,input_size, hidden_size, use_bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(torch.FloatTensor(2*input_size, 3 * hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(2*hidden_size, 3 * hidden_size))
        self.bias_ih = nn.Parameter(torch.FloatTensor(1,3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.FloatTensor(1,3 * hidden_size))
        self.memcnt=0
        self.memcap=options['mem_cap']
        self.entry_size=options['entry_size']
        self.head_size=options['head_size']
        mode=options['mode']
        if mode=='train':
            batch_size=options['batch_size']
        elif mode=='val':
            batch_size=options['eval_batch_size']
        elif mode=='test':
            batch_size=options['test_batch_size']
        self.batch_size=batch_size 
        self.auxcell = GRUCell(input_size+hidden_size,2*(self.head_size+1))
        self.tau=1.
        self.i_fc=nn.Sequential(
                nn.Linear(input_size,self.head_size//2),
                nn.ReLU(),
                nn.Linear(self.head_size//2,self.head_size),
                nn.Sigmoid())  
        self.h_fc=nn.Sequential(
                nn.Linear(hidden_size,self.head_size//2),
                nn.ReLU(),
                nn.Linear(self.head_size//2,self.head_size),
                nn.Sigmoid())
        
        self.last_usage=None
        self.mem=None

        self.reset_parameters()
    
    def _reset_mem(self):
        self.memcnt=0
        self.imem=Variable(torch.zeros(self.batch_size,self.memcap,self.input_size),requires_grad=True).cuda()
        self.hmem=Variable(torch.zeros(self.batch_size,self.memcap,self.hidden_size),requires_grad=True).cuda()
        self.i_last_use=Variable(torch.ones(self.batch_size,self.memcap)*-9999999.).cuda()
        self.h_last_use=Variable(torch.ones(self.batch_size,self.memcap)*-9999999.).cuda()

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def set_tau(self,num):
        self.tau=num

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for n,p in self.named_parameters():
            if 'weight' in n:
                nn.init.orthogonal_(p)
            if 'bias' in n:
                nn.init.constant_(p.data, val=0)
        
    def forward(self, input_, h_0, aux_h_0):

        i=input_
        h=h_0.detach()
        read_head=self.auxcell(torch.cat([i,h],dim=1),aux_h_0) 
        i_read_head,h_read_head=torch.split(read_head,self.head_size+1,dim=1)
        i_head_vecs=torch.cat([self.i_fc(self.imem.detach()),torch.sigmoid(self.i_last_use).detach().unsqueeze(2)],dim=2)
        h_head_vecs=torch.cat([self.h_fc(self.hmem.detach()),torch.sigmoid(self.h_last_use).detach().unsqueeze(2)],dim=2)
        i_read_head=(i_read_head.unsqueeze(1)*i_head_vecs).sum(dim=2)
        h_read_head=(h_read_head.unsqueeze(1)*h_head_vecs).sum(dim=2)
        i_entry,i_read_index,h_entry,h_read_index=self.read(i_read_head,h_read_head,self.tau)
        self.i_last_use.add_(-1).add_(-self.i_last_use*i_read_index)
        self.h_last_use.add_(-1).add_(-self.h_last_use*h_read_index)
        
        new_i=torch.cat([input_,i_entry],dim=1)
        new_h0=torch.cat([h_0,h_entry],dim=1)
        wi_b = torch.addmm(self.bias_ih, new_i, self.weight_ih)
        wh_b = torch.addmm(self.bias_hh, new_h0, self.weight_hh)
        ri,zi,ni=torch.split(wi_b,self.hidden_size,dim=1)
        rh,zh,nh=torch.split(wh_b,self.hidden_size,dim=1)
        r=torch.sigmoid(ri+rh)
        z=torch.sigmoid(zi+zh)
        n=torch.tanh(ni+r*nh)
        h_1=(1-z)*n+z*h_0
        
        if self.memcnt<self.memcap:
            h_write_index=i_write_index=Variable(torch.cat([torch.zeros(self.memcnt),torch.ones(1),torch.zeros(self.memcap-1-self.memcnt)]).unsqueeze(0)).cuda()
            self.memcnt+=1
        else:
            h_write_index=h_read_index
            i_write_index=i_read_index
        self.write(input_,i_write_index,h_0,h_write_index)
        
        return h_1,read_head

    def write(self,i,i_index,h,h_index):
        i_ones=i_index.unsqueeze(2)
        h_ones=h_index.unsqueeze(2)
        self.imem=i.unsqueeze(1)*i_ones+self.imem*(1.-i_ones)
        self.hmem=h.unsqueeze(1)*h_ones+self.hmem*(1.-h_ones)

    def read(self,i_read_head,h_read_head,tau):
        i_index,_=self.gumbel_softmax(i_read_head,tau)
        h_index,_=self.gumbel_softmax(h_read_head,tau)
        i_entry=i_index.unsqueeze(2)*self.imem
        h_entry=h_index.unsqueeze(2)*self.hmem
        i_entry=i_entry.sum(dim=1)
        h_entry=h_entry.sum(dim=1)
        return i_entry,i_index,h_entry,h_index

    def gumbel_softmax(self,input, tau):
            gumbel = Variable(-torch.log(1e-20-torch.log(1e-20+torch.rand(*input.shape)))).cuda()
            y=torch.nn.functional.softmax((input+gumbel)*tau,dim=1)
            ymax,pos=y.max(dim=1)
            hard_y=torch.eq(y,ymax.unsqueeze(1)).float()
            y=(hard_y-y).detach()+y
            return y,pos

    def gumbel_sigmoid(self,input, tau):
            gumbel = Variable(-torch.log(1e-20-torch.log(1e-20+torch.rand(*input.shape)))).cuda()
            y=torch.sigmoid((input+gumbel)*tau)
            #hard_y=torch.eq(y,ymax.unsqueeze(1)).float()
            #y=(hard_y-y).detach()+y
            return y

