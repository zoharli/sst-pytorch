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
      #  self.initializer = tf.random_uniform_initializer(
      #      minval = - self.options['init_scale'],
      #      maxval = self.options['init_scale'])
        if self.options['rnn_drop'] > 0:
            print('using dropout in rnn!')
        self.rnn=MARNN(options,options['video_feat_dim'],options['rnn_size'],options['num_rnn_layers'])
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

    @staticmethod
    def _forward_rnn(cell, input_, length, hx):
        max_time = input_.size(0)
        output = []
        cell._reset_mem()
        for time in range(max_time):
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
        if input_.is_cuda:
            length=length.cuda()
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
        h_n = torch.stack(h,0)
        return output, h_n 

class MANNCell(nn.Module):

    def __init__(self,options,input_size, hidden_size, use_bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(torch.FloatTensor(input_size, 3 * hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(hidden_size, 3 * hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(6 * hidden_size))
        else:
            self.register_parameter('bias', None)

        self.memcnt=0
        self.memcap=options['mem_cap']
        self.entry_size=options['entry_size']
        mode=options['mode']
        if mode=='train':
            batch_size=options['batch_size']
        elif mode=='val':
            batch_size=options['eval_batch_size']
        elif mode=='test':
            batch_size=options['test_batch_size']
        self.batch_size=batch_size 
        self.weight_im=nn.Parameter(torch.FloatTensor(input_size,self.entry_size))
        self.weight_hm=nn.Parameter(torch.FloatTensor(hidden_size,self.entry_size))
        self.weight_um=nn.Parameter(torch.FloatTensor(self.memcap,self.entry_size))
        self.fc1=nn.Linear(self.entry_size,self.memcap)
        self.tau=1.
        self.fc2=nn.Linear(self.entry_size+hidden_size,hidden_size)
        self._grad_hook=GradHook.apply

        self.last_usage=None
        self.last_read=None
        self.mem=None

        self.reset_parameters()
    
    def _reset_mem(self):
        self.last_usage=Variable(torch.ones(self.batch_size,self.memcap)*-99999.)
        self.last_read=None
        self.mem=Memory(self.batch_size,self.memcap,self.entry_size)
        if self.weight_im.is_cuda:
            self.mem.cuda()
            self.last_usage=self.last_usage.cuda()

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def cuda(self,device=None):
        super().cuda(device)
        self.mem.cuda(device)

    def cpu(self):
        super().cpu()
        self.mem.cpu()

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
            write_index=[self.memcnt]*self.batch_size
            self.memcnt+=1
        else:
            write_index=self.last_read
        self.write(h_0,write_index)
        last_use=torch.sigmoid(self.last_usage).detach()
        read_head=self.fc1(torch.tanh(input_.detach().mm(self.weight_im)+h_0.detach().mm(self.weight_hm)+last_use.mm(self.weight_um)))
        entry,read_index,self.last_read=read(read_head,self.tau)
        self.last_usage.add_(-1).add_(-self.last_usage*read_index)
        h_new=self.fc2(torch.cat([entry,h_0],dim=1))
        batch_size = h_new.size(0)
        bias_batch = (self.bias.unsqueeze(0)
                      .expand(batch_size, *self.bias.size()))
        bias_ih,bias_hh = torch.split(bias_batch,split_size=self.hidden_size*3,dim=1)
        wi_b = torch.addmm(bias_ih, input_, self.weight_ih)
        wh_b = torch.addmm(bias_hh, h_new, self.weight_hh)
        ri,zi,ni=torch.split(wi_b,self.hidden_size,dim=1)
        rh,zh,nh=torch.split(wh_b,self.hidden_size,dim=1)
        r=torch.sigmoid(ri+rh)
        z=torch.sigmoid(zi+zh)
        n=torch.tanh(ni+r*nh)
        h_1=(1-z)*n+z*h_new
        return h_1

    def write(self,entry,index):
        entry=torch.split(entry,split_size=1)
        for i in range(len(index)):
            self.mem[i][index[i]]=entry[i].squeeze()

    def read(self,read_head,tau):
        index,pos=gumbel_softmax(read_head,tau)
        pos=pos.data.tolist()
        entry=torch.stack([mem[i][pos[i]] for i in range(len(mem))])
        entry=self._grad_hook(entry,index)
        return entry,index,pos

    @staticmethod
    def gumbel_softmax(self,input, tau):
            gumbel = Variable(-torch.log(1e-20-torch.log(1e-20+torch.rand(*input.shape))))
            if input.is_cuda:
                gumbel=gumbel.cuda()
            y=torch.nn.functional.softmax((input+gumbel)*tau.expand_as(input),dim=1)
            ymax,pos=y.max(dim=1)
            hard_y=torch.eq(y,ymax.unsqueeze(1)).float()
            y=(hard_y-y).detach()+y
            return y,pos

        

class GradHook(torch.autograd.Function):
    
    @staticmethod
    #mem:list of memory entry,[batch_size,entry_size]
    #soft_index:samples produced by gumbel-softmax,[batch_size,memcap]
    #output:[batch_size,entry_size]
    def forward(self,entry,soft_index):
        self.save_for_backward(soft_index)
        return entry

    @staticmethod
    def backward(self, grad_output):
        index=self.saved_variables.t()
        g_ones=grad_output.sum(dim=1)
        g_zeros=-g_ones/(index.shape[0]-1.)
        grad_index=index*g_ones+[torch.ones(*hard_index.shape)-index]*g_zeros
        return grad_output,grad_index.t()

        

class Memory(list):
    def __init__(self,batch_size,N,M):
        super().__init__()
        for _ in range(batch_size):
            l=[]
            for _ in range(N):
                l.append(Variable(torch.zeros(M),requires_grad=True))
            self.append(l)

    def cuda(self,device=None):
        for i in range(len(self)):
            for j in range(len(self[i])):
                self[i][j]=self[i][j].cuda()
        return self

    def cpu(self):
        for i in range(len(self)):
            for j in range(len(self[i])):
                self[i][j]=self[i][j].cpu()
        return self
        
