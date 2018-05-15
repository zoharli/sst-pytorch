import sys
import argparse
import collections


hyper_params=collections.OrderedDict({
        'rnn_type':['mann','gru'],
        'rnn_size':[64,128,160,200,256],
        'rnn_drop':[0.4],
        'mem_cap':[40,64],
        'head_size':[16,32,64],
        'time_fac':[20,10,3,1],
        'step_tau':[2,1],
        })


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx',default=0,type=int)
    args=parser.parse_args()
    idx=args.idx
    list_len=[]
    l=1
    for x in hyper_params:
        list_len.append(len(hyper_params[x]))
        l*=len(hyper_params[x])
    if idx>=l:
        sys.exit(1)
    hyper_params_num=len(hyper_params)
    picks=[]
    for i in range(hyper_params_num-1,-1,-1):
        l//=list_len[i]
        picks.insert(0,idx//l)
        idx%=l
    output_params=' '
    for i,x in enumerate(hyper_params):
        output_params+='--'+x+'='+str(hyper_params[x][picks[i]])+' '
    print(output_params)
