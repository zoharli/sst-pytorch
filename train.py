"""
Train your model
"""
import warnings
warnings.filterwarnings("ignore")
import os
import shutil
import argparse
import numpy as np
import torch
from torch.optim import *
from opt import *
from data_provider import *
from model import * 
from MemModel import *
from utils import *
from torch.autograd import Variable
from torch.utils.data import DataLoader

def BceLoss(output,target,weight,length):
    log_weight=1 + (weight -1)*target
    loss=(1-target)*output+log_weight*(torch.log(1+(torch.exp(-torch.abs(output))))+torch.nn.functional.relu(-output))
    return loss.mean()*max(length)*len(length)/sum(length)

def evaluation(model,weight,options, val_dataloader):

    val_loss_list = []
    model.eval()

    for i,(input,target,length,mask) in enumerate(val_dataloader):
        print('Evaluating batch: #%d'%i)
        with torch.no_grad():
            input=Variable(input).cuda()
            target=Variable(target).cuda()
            mask=Variable(mask).cuda()
            output=model(input,length)+mask
            pos_weight=weight.expand_as(target)
            loss=BceLoss(output,target,pos_weight,length)
        val_loss_list.append(loss.item())
        
    ave_val_loss = sum(val_loss_list) / len(val_loss_list)
    
    return ave_val_loss
    
def save_checkpoint(filename,state,is_best):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.dirname(filename)+'/best_model.pth')

def train(options):
    
    print('Load data ...')
    collate_fn=PadCollate()
    train_data_provision = DataProvision(options,'train')
    train_dataloader=torch.utils.data.DataLoader(train_data_provision,options['batch_size'],True,collate_fn=collate_fn)
    val_data_provision = DataProvision(options,'val')
    val_dataloader=torch.utils.data.DataLoader(val_data_provision,options['eval_batch_size'],False,collate_fn=collate_fn)
    batch_size = options['batch_size']
    max_epochs = options['max_epochs']
    init_epoch = options['init_epoch']
    lr_init = options['lr']
    lr = lr_init
    
    n_iters_per_epoch = train_data_provision.__len__() // batch_size
    eval_in_iters = int(n_iters_per_epoch / float(options['n_eval_per_epoch']))
    
    # build model 
    
    if options['rnn_type']=='mann':
        model_class=MaSST
    else:
        model_class=ProposalModel
    options['mode']='train'
    model = model_class(options).cuda()
    options['mode']='val'
    val_model = model_class(options).cuda()
    print('Build model for training stage ...')
    
    weight=train_data_provision._proposal_weight.contiguous()
    weight=Variable(weight.view(1,1,*weight.shape)).cuda()
    
    if options['solver'] == 'adam':
        optimizer = Adam(model.parameters(),lr)
    elif options['solver'] == 'adadelta':
        optimizer = Adadelta(model.parameters(),lr,weight_decay=options['reg'])
    else:
        optimizer = SGD(model.parameters(),lr,weight_decay=options['reg'])


    t0 = time.time()
    eval_id = 0
    total_iter = 0
    best_loss=100000.0
    tau=1.
    for epoch in range(init_epoch, max_epochs):
        model.train()
        if epoch==options['max_epochs']//2:
            lr=lr_init/10
            for pg in optimizer.param_groups:
                pg['lr']=lr
        
        print('epoch: %d/%d, lr: %.1E (%.1E) tau:%d'%(epoch, max_epochs, lr, lr_init,tau))
        valoss_list=[]
        for iter,(input,target,length,mask) in enumerate(train_dataloader):
            input_var = Variable(input,requires_grad=True).cuda()
            target_var = Variable(target).cuda()
            mask=Variable(mask).cuda()
            optimizer.zero_grad()
            output=model(input_var,length)+mask
            pos_weight=weight.expand_as(target_var)
            loss=BceLoss(output,target_var,pos_weight,length)
            reg_loss=torch.add(loss,0)
            for x in model.parameters():
                if x is not None:
                    reg_loss+= options['reg']*torch.sum(x**2)/2
            reg_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),options['clip_gradient_norm'])
            optimizer.step()

            if iter % options['n_iters_display'] == 0:
                print('iter: %d, epoch: %d/%d, \nlr: %.1E, loss: %.4f, reg_loss: %.4f'%(iter, epoch, max_epochs, lr, loss.item(),reg_loss.item()))
            
            if (total_iter+1) % eval_in_iters == 0:
                is_best=0       
                print('Evaluating model ...')
                val_model.load_state_dict(model.state_dict())
                if options['rnn_type']=='mann':
                    val_model.set_tau(tau)
                val_loss = evaluation(val_model,weight,options,val_dataloader) 
                valoss_list.append(val_loss)
                if val_loss<best_loss:
                    best_loss=val_loss
                    is_best=1
                print('loss: %.4f'%val_loss)  

                checkpoint_path = '%sepoch%02d_%.2f_%02d_lr%f.ckpt' % (options['ckpt_prefix'], epoch, val_loss, eval_id, lr)

                save_checkpoint(checkpoint_path,
                        {'state_dict':model.state_dict()},
                        is_best)

                eval_id = eval_id + 1

            total_iter += 1
        if tau<options['max_tau']:
            tau+=options['step_tau']
            if options['rnn_type']=='mann':
                model.set_tau(tau)
        if options['rnn_type']=='mann' and len(valoss_list)>3 and valoss_list[-1]<1.2 and valoss_list[-3]>valoss_list[-2] and valoss_list[-2]>valoss_list[-1]:
            model.fc1.requires_grad=False
            print('==================>fix input trans parameter!')
              
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    options = default_options()
    for key, value in options.items():
        parser.add_argument('--%s'%key, dest=key, type=type(value), default=None)
    args = parser.parse_args()
    args = vars(args)
    for key, value in args.items():
        if value is not None:
            options[key] = value
    options=later_options(options)
    work_dir = options['ckpt_prefix']
    if not os.path.exists(work_dir) :
        os.makedirs(work_dir)
    find_idle_gpu(options['gpu'])
    train(options)
