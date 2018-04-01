"""
Train your model
"""

import os
import shutil
import argparse
import numpy as np
import torch
from torch.optim import *
from opt import *
from data_provider import *
from model import * 
from utils import *
from torch.autograd import Variable
from torch.utils.data import DataLoader


def evaluation(model,weight,options, val_dataloader):

    val_loss_list = []
    val_count = val_dataloader.dataset.__len__()
    model.eval()

    for i,(input,target,length,mask) in enumerate(val_dataloader):
        print('Evaluating batch: #%d'%i)
        input=Variable(input,volatile=True).cuda()
        target=Variable(target,volatile=True).cuda()
        mask=Variable(mask,volatile=True).cuda()
        output=model(input,length)*mask
        criterion=torch.nn.BCELoss(weight=weight.expand_as(target)).cuda()
        loss=criterion(output,target)
        val_loss_list.append(loss.data[0]*len(length))
        
    ave_val_loss = sum(val_loss_list) / float(val_count)
    
    return ave_val_loss
    
def save_checkpoint(filename,state,is_best):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'checkpoints/best_model.pth')

def train(options):
    
    print('Load data ...')
    collate_fn=PadCollate()
    train_data_provision = DataProvision(options,'train')
    train_dataloader=torch.utils.data.DataLoader(train_data_provision,options['batch_size'],True,collate_fn=collate_fn)
    val_data_provision = DataProvision(options,'val')
    val_dataloader=torch.utils.data.DataLoader(val_data_provision,options['eval_batch_size'],True,collate_fn=collate_fn)
    batch_size = options['batch_size']
    max_epochs = options['max_epochs']
    init_epoch = options['init_epoch']
    lr_init = options['lr']
    lr = lr_init
    
    n_iters_per_epoch = train_data_provision.__len__() // batch_size
    eval_in_iters = int(n_iters_per_epoch / float(options['n_eval_per_epoch']))
    
    # build model 
    model = ProposalModel(options).cuda()
    print('Build model for training stage ...')
    
    weight=train_data_provision._proposal_weight.contiguous()
    weight=weight.view([1,1,weight.shape[0]])
    
    if options['solver'] == 'adam':
        optimizer = Adam(model.parameters(),lr,weight_decay=options['reg'])
    elif options['solver'] == 'adadelta':
        optimizer = Adadelta(model.parameters(),lr,weight_decay=options['reg'])
    else:
        optimizer = SGD(model.parameters(),lr,weight_decay=options['reg'])

    # initialize model from a given checkpoint path
    #if options['init_from']:
    #    print('Init model from %s'%options['init_from'])
    #    saver.restore(sess, options['init_from'])


    #if options['eval_init']:
    #    print('Evaluating the initialized model ...')
    #    val_loss = evaluation(options, data_provision, sess, inputs, t_loss, t_summary)
    #    print('loss: %.4f'%val_loss)
    #        

    t0 = time.time()
    eval_id = 0
    total_iter = 0
    best_loss=100000.0
    for epoch in range(init_epoch, max_epochs):
        model.train()
        print('epoch: %d/%d, lr: %.1E (%.1E)'%(epoch, max_epochs, lr, lr_init))
        for iter,(input,target,length,mask) in enumerate(train_dataloader):
            input_var = Variable(input).cuda()
            target_var = Variable(target).cuda()
            mask=Variable(mask).cuda()
            optimizer.zero_grad()
            output=model(input_var,length)*mask
            criterion=torch.nn.BCELoss(weight=weight.expand_as(target_var)).cuda()
            loss=criterion(output,target_var)
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(),options['clip_gradient_norm'])
            optimizer.step()

            if iter % options['n_iters_display'] == 0:
                print('iter: %d, epoch: %d/%d, \nlr: %.1E, loss: %.4f'%(iter, epoch, max_epochs, lr, loss.data[0]))
            
            if (total_iter+1) % eval_in_iters == 0:
                is_best=0       
                print('Evaluating model ...')
                val_loss = evaluation(model,weight,options,val_dataloader) 
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

    work_dir = options['ckpt_prefix']
    if not os.path.exists(work_dir) :
        os.makedirs(work_dir)
    find_idle_gpu()
    train(options)

