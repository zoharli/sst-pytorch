"""
Build graph for both training and inference
"""

import torch
import math
from torch import nn
from torch.autograd import Variable


class MaSST(nn.Module):

    def __init__(self, options):
        super().__init__()
        self.options = options
        if self.options['rnn_drop'] > 0:
            print('using dropout in rnn!')
        self.rnn=MARNN(options,options['rnn_size'],options['rnn_size'],options['num_rnn_layers'],dropout=options['rnn_drop'])
        self.fc=nn.Linear(2*options['rnn_size'],options['num_anchors'])
        self.fc1=nn.Linear(options['video_feat_dim'],options['rnn_size'])
        self._init()

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
                    nn.init.uniform(param,-self.options['init_scale'],self.options['init_scale'])
                elif 'bias' in name:
                    nn.init.constant(param,0.)

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
    def _forward_rnn(cell, input_, length, hx):
        max_time = input_.size(0)
        output = []
        cell._reset_mem()
        for time in range(max_time):
            h_next,o = cell(input_[time],hx)
            mask = Variable((time < length).float().unsqueeze(1).expand_as(h_next))
            h_next = h_next*mask + hx*(1 - mask)
            output.append(o)
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
        length=length.cuda(input_.get_device())
        hx=hx.cuda(input_.get_device())
        h = []
        layer_output = None
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            input_ = self.dropout_layer(input_)
            layer_output, layer_h_n = MARNN._forward_rnn(
                cell=cell, input_=input_, length=length, hx=hx)
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
        self.weight_ih = nn.Parameter(torch.FloatTensor(hidden_size, 3 * hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(hidden_size, 3 * hidden_size))
        self.weight_rh = nn.Parameter(torch.FloatTensor(hidden_size, 3 * hidden_size))
        self.weight_s1 = nn.Parameter(torch.FloatTensor(hidden_size,  hidden_size))
        self.weight_s2 = nn.Parameter(torch.FloatTensor(hidden_size,  hidden_size))
        self.weight_s3 = nn.Parameter(torch.FloatTensor(hidden_size,  hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(1,3 * hidden_size))
        else:
            self.register_parameter('bias', None)

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
        self.weight_im=nn.Parameter(torch.FloatTensor(hidden_size,self.head_size+1))
        self.weight_hm=nn.Parameter(torch.FloatTensor(hidden_size,self.head_size+1))
        self.tau=1.
        self.fc1=nn.Linear(self.hidden_size,self.head_size)
        
        self.last_usage=None
        self.mem=None

        self.weight_ic = nn.Parameter(torch.FloatTensor(hidden_size, 4))
        self.weight_hc = nn.Parameter(torch.FloatTensor(hidden_size, 4))
        self.weight_rc = nn.Parameter(torch.FloatTensor(hidden_size, 4))
        if use_bias:
            self.bias_c = nn.Parameter(torch.FloatTensor(1,4))
        else:
            self.register_parameter('bias_c', None)

        self.weight_im1=nn.Parameter(torch.FloatTensor(hidden_size,self.head_size+1))
        self.weight_hm1=nn.Parameter(torch.FloatTensor(hidden_size,self.head_size+1))
        self.weight_mm1=nn.Parameter(torch.FloatTensor(hidden_size,self.head_size+1))
        self.fc1=nn.Linear(self.hidden_size,self.head_size)
        if use_bias:
            self.bias_m1 = nn.Parameter(torch.FloatTensor(1,self.head_size+1))
        else:
            self.register_parameter('bias_m1', None)

        self.bias_1 = nn.Parameter(torch.FloatTensor(1,hidden_size))
        self.bias_2 = nn.Parameter(torch.FloatTensor(1,hidden_size))
        self.bias_3 = nn.Parameter(torch.FloatTensor(1,hidden_size))

        self.reset_parameters()
    
    def _reset_mem(self):
        self.memcnt=0
        self.mem=Variable(torch.zeros(self.batch_size,self.memcap,self.entry_size),requires_grad=True).cuda()
        self.last_usage=Variable(torch.ones(self.batch_size,self.memcap)*-99999.).cuda()

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
                p.data.uniform_(-stdv, stdv)
            if 'bias' in n and p is not None:
                nn.init.constant(p.data, val=0)
        
        
    def forward(self, input_, h_0):
        if self.memcnt<self.memcap:
            write_index=Variable(torch.cat([torch.zeros(self.memcnt),torch.ones(1),torch.zeros(self.memcap-1-self.memcnt)]).unsqueeze(0)).cuda()
            self.memcnt+=1
        else:
            write_index=self.read_index
        self.write(h_0,write_index)
        last_use=torch.sigmoid(self.last_usage).detach()
        read_head=torch.tanh(input_.detach().mm(self.weight_im)+h_0.detach().mm(self.weight_hm))
        head_vecs=torch.cat([self.fc1(self.mem.detach()),last_use.unsqueeze(2)],dim=2)
        
        read_head=(read_head.unsqueeze(1)*head_vecs).sum(dim=2)
        entry,self.read_index=self.read(read_head,self.tau)
        self.last_usage.add_(-1).add_(-self.last_usage*self.read_index)
        w_b=torch.sigmoid(entry.mm(self.weight_rh)+h_0.mm(self.weight_hh)+input_.mm(self.weight_ih)+self.bias)
        r,z,n=torch.split(w_b,self.hidden_size,dim=1)
        h_new=torch.tanh(input_.mm(self.weight_s1)+self.bias_1+r*(h_0.mm(self.weight_s2)+self.bias_2)+z*(self.bias_3+entry.mm(self.weight_s3)))
        head1=torch.tanh(input_.mm(self.weight_im1)+h_0.mm(self.weight_hm1)+entry.mm(self.weight_mm1)+self.bias_m1)
        read_head1=(head1.unsqueeze(1)*head_vecs).sum(dim=2)
        entry_o,_=self.read(read_head1,self.tau)
        h_1=n*h_new+(1.-n)*h_0
        o=torch.cat([h_1,entry_o],dim=1)
        return h_1,o

    def write(self,entry,index):
        ones=index.unsqueeze(2)
        zeros=1.0-ones
        self.mem=entry.unsqueeze(1)*ones+self.mem*zeros

    def read(self,read_head,tau):
        index,_=self.gumbel_softmax(read_head,tau)
        entry=index.unsqueeze(2)*self.mem
        entry=entry.sum(dim=1)
        return entry,index

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

