import os
import random
import numpy as np

import torch
from torch import nn
import matplotlib.pyplot as plt
import pdb
from copy import deepcopy

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


def print_summary(acc_taw, acc_tag, forg_taw, forg_tag):
    for name, metric in zip(['TAw Acc', 'TAg Acc', 'TAw Forg', 'TAg Forg'], [acc_taw, acc_tag, forg_taw, forg_tag]):
        print('*' * 108)
        print(name)
        for i in range(metric.shape[0]):
            print('\t', end='')
            for j in range(metric.shape[1]):
                print('{:5.1f}% '.format(100 * metric[i, j]), end='')
            if np.trace(metric) == 0.0:
                if i > 0:
                    print('\tAvg.:{:5.1f}% '.format(100 * metric[i, :i].mean()), end='')
            else:
                print('\tAvg.:{:5.1f}% '.format(100 * metric[i, :i + 1].mean()), end='')
            print()
    print('*' * 108)


def print_modules(model):
    '''
    for module in model._modules:
        try:
            print("%s: %s"%(module,str(np.shape(model._modules[module].weight))))
        except:
            print(str(model._modules[module]))
            #pdb.set_trace()
            print_modules(model._modules[module])
    '''
    print(model._modules)
def print_param_lengths(model):
        for idx,named_params in enumerate(model.named_parameters()):
            name=named_params[0]
            params=named_params[1]
            #print("layer %i: %i"%(idx,len(list(params))))
            print("layer %i:(%s) %i"%(idx,name,np.prod(params.size())))
            

def replace_kernels(model,kernel_size=(1,1),select='all'):
    for module in model._modules: #seqmodel = nn.Sequential(*list(init_model.children()))
        
        #print(np.shape(model._modules[module].weight))
        if "conv" in str(module): #type(module) is nn.modules.conv.Conv2d:
            #print(model._modules[module].kernel_size)
            #pdb.set_trace()
            #model._modules[module].kernel_size=kernel_size
            model._modules[module]=nn.Conv2d(in_channels=model._modules[module].in_channels,out_channels=model._modules[module].out_channels,kernel_size=kernel_size, stride=model._modules[module].stride,padding=0,dilation=model._modules[module].dilation,groups=model._modules[module].groups,bias=model._modules[module].bias,padding_mode=model._modules[module].padding_mode)
            #model._modules[module]=nn.Linear(in_features=model._modules[module].in_channels,out_features=model._modules[module].out_channels,bias=model._modules[module].bias)
            #print(np.shape(model._modules[module].weight))
            if select=='first':
                return model
        else:
            model._modules[module]=replace_kernels(model._modules[module],kernel_size,select)
        
    return model

def replace_batchnorm(model, select='all'):
    for module in model._modules: #seqmodel = nn.Sequential(*list(init_model.children()))
        #print(np.shape(model._modules[module].weight))
        if len(model._modules[module]._modules)>0:
            model._modules[module]=replace_batchnorm(model._modules[module], select)
        elif "BatchNorm" in str(model._modules[module]): 
        #elif "bn" in str(module) or "BatchNorm" in str(module):
            model._modules[module]=nn.Sequential()
            if select=='first':
                return model
    return model

def replace_dim(model,target_dim=1,select='first'):
    for idx,(name, layer) in enumerate(model._modules.items()):
        try:
            #model._modules[name].out_channels*=round(np.divide(model._modules[name].in_channels,target_dim))
            #model._modules[name].in_channels=target_dim
            model._modules[name]=nn.Conv2d(in_channels=target_dim,out_channels=model._modules[name].out_channels,kernel_size=model._modules[name].kernel_size, stride=model._modules[name].stride,padding=model._modules[name].padding,dilation=model._modules[name].dilation,groups=model._modules[name].groups,bias=model._modules[name].bias,padding_mode=model._modules[name].padding_mode)
        except:
            model=replace_dim(model._modules[name])
        if select == 'first':
            return model
    return model

from torchviz import make_dot
def visualize_network(model):
    model_aux=model.clone()
    model_aux.add_head(100)
    input_shape=np.shape(list(model_aux.parameters())[0])
    tensor_input = torch.autograd.Variable(torch.zeros(input_shape,requires_grad=True))
    tensor_output=model_aux(tensor_input)
    try:
        graph=make_dot(tensor_output)
    except:    
        try:
            graph=make_dot(tensor_output[0])
        except:
            graph=make_dot(tensor_output,params=dict(model_aux.model.named_parameters()))    
    #with torch.onnx.set_training(model, False):
    #    trace, _ = torch.jit.get_trace_graph(model, args=(tensor_input,)) 
    graph.view()
    return graph

import signal
def TimedInput(prompt='Question:', timeout=30, timeoutmsg = ""):
    def timeout_error(*_):
        raise TimeoutError
    signal.signal(signal.SIGALRM, timeout_error)
    signal.alarm(timeout)
    try:
        answer = input(prompt)
        signal.alarm(0)
        return answer
    except TimeoutError:   
        if timeoutmsg:
            print(timeoutmsg)
        signal.signal(signal.SIGALRM, signal.SIG_IGN)
        return -1

def binary_question(question,counter=2,default_response=True,timeout=30):
    reply = TimedInput(question+' (y/n): ',timeout,"Time is running out")
    if reply is not -1:
        reply=str(reply).lower().strip() #raw_input for python2:
    else:
        return default_response
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        if counter>0:
            print("Invalid response")
            return binary_question(question,counter-1,default_response,timeout)
        else:
            return False
        
def batch2gray(x,transform_type="mean"): #[batch,channel,row,col]
    if transform_type is "mean":
        xp=torch.mean(x,dim=1,keepdim=True)
    elif transform_type is "cumsum":
        xp=torch.div(torch.cumsum(x,dim=1),3)
    elif transform_type is "divsum":
        xp=torch.div(torch.sum(x,dim=1,keepdim=True),3)
    elif transform_type is "weighted":
        xp=torch.sum([torch.mul(x[:,0,:,:],0.2989),torch.mul(x[:,1,:,:],0.5870),torch.mul(x[:,2,:,:],0.1140)],keepdim=True)
    else:
        xp=torch.mean(x,dim=1,keepdim=True)
    if not "keepdim" in str(transform_type):
        xp=xp.repeat(1,3,1,1) #replicate gray to 3 channels
    return xp 

def batch2scramble(x):
    xp=x
    for batch in range(np.shape(x)[0]):
        for channel in range(np.shape(x)[1]):
            M=np.shape(x)[2]
            N=np.shape(x)[3]
            permutation1D=torch.randperm(M*N)
            xp[batch,channel,:,:]=x[batch,channel,:,:].view(M*N)[permutation1D].view(M,N)
    return xp 
    ''' #numpy (cpu)
    xp=x.cpu()
    for batch in range(np.shape(x)[0]):
        for channel in range(np.shape(x)[1]):
            M=np.shape(x)[2]
            N=np.shape(x)[3]
            permutation1D=torch.randperm(M*N)
            #image1D=np.reshape(x[batch,channel,:,:],(M*N))
            #image1Dscrambled=image1D[permutation1D]
            #image2Dscrambled=np.reshape(image1Dscrambled,(M,N))
            xp[batch,channel,:,:]=np.reshape(np.reshape(x[batch,channel,:,:],(M*N))[permutation1D],(M,N))
    return xp.cuda()
    '''

def visualize_batch(x, print_path=None, one_channel=False,normalize_mean=(0.485, 0.456, 0.406),normalize_std=(0.229, 0.224, 0.225)):
    for idx,img in enumerate(x):
        if one_channel:
            img = img.mean(dim=0)
            img = img * normalize_std[0] + normalize_mean[0]
        else:
            img[0,:,:] = img[0,:,:] * normalize_std[0] + normalize_mean[0]     # unnormalize
            img[1,:,:] = img[1,:,:] * normalize_std[1] + normalize_mean[1]     # unnormalize
            img[2,:,:] = img[2,:,:] * normalize_std[2] + normalize_mean[2]     # unnormalize
        #visualizable img (abs)
        np_img = img.numpy()
        np_img[np_img<0]=0.0
        #np_img=np.abs(np_img)
        if one_channel:
            plt.imshow(np_img, cmap="Greys")
        else:
            plt.imshow(np.transpose(np_img, (1, 2, 0)))
        if print_path is None:
            plt.show()
        else:
            plt.savefig(print_path+"/"+str(idx)+".png")


import json
def read_json_rawlog(json_path):
    json_file=open(json_path)
    json_data=json_file.read()
    json_data="["+json_data[:-2]+"]" #cut and add brackets
    parsed_json = (json.loads(json_data))
    return parsed_json


def parse_rawlog(parsed_json,num_tasks=8):
    loss=[{'train':[],'valid':[],'test':[]} for x in range(num_tasks)] #or dict()
    loss_step=[{'train':[],'valid':[],'test':[]} for x in range(num_tasks)]
    acc=[{'train':[],'valid':[],'test':[]} for x in range(num_tasks)]
    for idx,cell in enumerate(parsed_json):
        group=cell['group']
        task=cell['task']
        if cell['name']=='loss_step':
            loss_step[task][group].append(cell['value'])
        elif cell['name']=='loss':
            loss[task][group].append(cell['value'])
        elif cell['name']=='acc':
            acc[task][group].append(cell['value'])
        else: #name='lr' or 'patience'
            next
    return loss,acc,loss_step

def print_model_report(model, verbose=True):
    #calculate params
    count=0
    for p in model.parameters():
        count+=np.prod(p.size())
    if verbose:
        print('-'*100)
        print(model)
        print('Dimensions =',end=' ')
        for p in model.parameters():
            print(p.size(),end=' ')
        print()
        print('Num parameters = %s'%(human_format(count)))
        print('-'*100)
    return count

def human_format(num):
    magnitude=0
    while abs(num)>=1000:
        magnitude+=1
        num/=1000.0
    return '%.1f%s'%(num,['','K','M','G','T','P'][magnitude])

import functools
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    
    ### debugging
    if type(rgetattr(obj, pre) if pre else obj) == type(list()):
        print(rgetattr(obj, pre) if pre else obj)
        pdb.set_trace()
    ###
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def get_layer_filternum(nn_layer):
    nn_type=str(type(nn_layer))
    if "Conv" in nn_type:
        return nn_layer.in_channels,nn_layer.out_channels
    elif "BatchNorm" in nn_type:
        return nn_layer.num_features,nn_layer.num_features
    elif "Linear" in nn_type:
        return nn_layer.in_features,nn_layer.out_features
    else:
        print("Cannot get filternum for"+nn_type)
        return None,None
def set_input_output_filternum(nn_layer,in_filternum,out_filternum):
    nn_type=str(type(nn_layer))
    if "Conv2d" in nn_type:
        if in_filternum <=3:
            new_stride=nn_layer.stride
        else:
            new_stride=(int(out_filternum/in_filternum),int(out_filternum/in_filternum))
        
        return nn.Conv2d(in_channels=in_filternum,out_channels=out_filternum,kernel_size=nn_layer.kernel_size, stride=new_stride,padding=nn_layer.padding,dilation=nn_layer.dilation,groups=nn_layer.groups,bias=nn_layer.bias,padding_mode=nn_layer.padding_mode)
    elif "BatchNorm2d" in nn_type:
        return nn.BatchNorm2d(num_features=out_filternum, eps=nn_layer.eps, momentum=nn_layer.momentum, affine=nn_layer.affine, track_running_stats=nn_layer.track_running_stats)
    elif "Linear" in nn_type:
        #in linear layers, bias is overwritten by its values: https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html
        if str(nn_layer).split('bias=')[1][:-1]=='True': #if nn_layer.extra_repr().split('bias=')[1]=='True':
            bias=True
        else:
            bias=False
        return nn.Linear(in_features=in_filternum, out_features=out_filternum,bias=bias)
    else:
        print("Cannot change filternum for"+nn_type)
        return nn_layer
    

def multiply_network_capacity(model, multiplier):
    new_model=deepcopy(model)
    list_parameters=list(model.named_parameters())
    for idx,parameter in enumerate(list_parameters):
        pname=parameter[0]
        pname_parts=str(pname).split('.')
        if pname_parts[-1] == 'weight': # or pname_parts[-1] == 'bias'
            layer_name=str('.').join(pname_parts[:len(pname_parts)-1])
            layer_params=parameter[1]
            layer_tensor=layer_tensor=rgetattr(model,layer_name)
            in_filternum,out_filternum=get_layer_filternum(layer_tensor)
            if idx==0:
                new_out_filternum=int(multiplier*out_filternum)
                new_layer_tensor=set_input_output_filternum(layer_tensor,in_filternum,new_out_filternum)
            else:
                new_in_filternum=int(multiplier*in_filternum)
                new_out_filternum=int(multiplier*out_filternum)
                new_layer_tensor=set_input_output_filternum(layer_tensor,new_in_filternum,new_out_filternum)
            rsetattr(new_model,layer_name,new_layer_tensor)
    return new_model
    
def calc_desired_output_filternum(nn_layer,in_filternum,desired_layer_params,num_target_params):
    nn_type=str(type(nn_layer))
    if "Conv2d" in nn_type:
        k1=nn_layer.kernel_size[0]
        k2=nn_layer.kernel_size[1]
        num_layer_params=np.prod([nn_layer.in_channels,nn_layer.out_channels,k1,k2])
        multiplier=round(np.divide(desired_layer_params,num_layer_params))
        if in_filternum <= 3: #first layer
            #out_filternum=round(np.divide(desired_layer_params,np.prod([in_filternum,k1,k2])))
            #'''
            out_filternum=int(np.sqrt(nn_layer.out_channels*nn_layer.out_channels*multiplier))
            #'''
            '''
            out_filternum=round(multiplier*nn_layer.out_channels)
            '''
        else:
            #'''
            out_filternum=round(np.divide(desired_layer_params,np.prod([in_filternum,k1,k2])))
            if out_filternum >= multiplier*nn_layer.out_channels:
                out_filternum=round(multiplier*in_filternum)
            elif out_filternum < multiplier*nn_layer.out_channels and in_filternum < out_filternum:
                out_filternum=round(in_filternum)
            #'''
            '''
            out_filternum=round(multiplier*nn_layer.out_channels)
            '''
    elif "BatchNorm2d" in nn_type:
        #in_filternum=nn_layer.num_features
        #out_filternum=round(np.prod([in_filternum,np.divide(desired_layer_params,in_filternum)]))
        out_filternum=in_filternum
    elif "Linear" in nn_type:
        #in_filternum=nn_layer.in_features
        out_filternum=round(np.divide(desired_layer_params,in_filternum))
    else:
        print("do not recognize output filternum for "+nn_type)
        exit()
    
    return int(out_filternum)

#import utils; import importlib; import torch.nn as nn; tvnet = getattr(importlib.import_module(name='torchvision.models'), 'resnet18'); init_model = tvnet(pretrained=False); num_target_params=utils.print_model_report(init_model,False); init_model.fc = nn.Sequential()
#importlib.reload(utils); init_model2=utils.change_network_capacity(init_model,num_target_params);
def change_network_capacity(model, num_target_params): #in bytes
    num_current_params=print_model_report(model,verbose=False)
    new_model=deepcopy(model)
    list_parameters=list(model.named_parameters())
    for idx,parameter in enumerate(list_parameters):
        pname=parameter[0]
        pname_parts=str(pname).split('.')
        if pname_parts[-1] == 'weight': # or pname_parts[-1] == 'bias'
            layer_name=str('.').join(pname_parts[:len(pname_parts)-1])
            layer_params=parameter[1]
            layer_tensor=layer_tensor=rgetattr(model,layer_name)
            num_layer_current_params=np.prod(layer_params.size())
            #num_layer_next_params=np.prod(list_parameters[idx][1].size())
            prop_layer_current_params=np.divide(num_layer_current_params,num_current_params)
            num_layer_new_params=int(prop_layer_current_params*num_target_params)
            in_filternum,out_filternum=get_layer_filternum(layer_tensor)
            if idx==0:
                #new_out_filternum=calc_desired_output_filternum(layer_tensor,in_filternum,num_layer_new_params)
                new_out_filternum=calc_desired_output_filternum(layer_tensor,in_filternum,num_layer_new_params,num_target_params)
                new_layer_tensor=set_input_output_filternum(layer_tensor,in_filternum,new_out_filternum)
            else:
                prop_layer_sampling=np.divide(in_filternum,previous_out_filternum) #check if up/downsamples
                new_in_filternum=int(prop_layer_sampling*previous_new_out_filternum)
                new_out_filternum=calc_desired_output_filternum(layer_tensor,new_in_filternum,num_layer_new_params,num_target_params)
                new_layer_tensor=set_input_output_filternum(layer_tensor,new_in_filternum,new_out_filternum)
            rsetattr(new_model,layer_name,new_layer_tensor)
        previous_new_in_filternum,previous_new_out_filternum=get_layer_filternum(new_layer_tensor)
        previous_in_filternum,previous_out_filternum=get_layer_filternum(layer_tensor)
    return new_model


def get_model_unique_filternum(model):
    list_parameters=list(model.named_parameters())
    filternums=[]
    for idx,parameter in enumerate(list_parameters):
        pname=parameter[0]
        pname_parts=str(pname).split('.')
        if pname_parts[-1] == 'weight': # or pname_parts[-1] == 'bias'
            layer_name=str('.').join(pname_parts[:len(pname_parts)-1])
            layer_params=parameter[1]
            layer_tensor=layer_tensor=rgetattr(model,layer_name)
            in_filternum,out_filternum=get_layer_filternum(layer_tensor)
            if not out_filternum in filternums:
                filternums.append(out_filternum)
            if idx > 0:
                if not in_filternum in filternums:
                    filternums.append(in_filternum)
    return filternums
def substitute_filternum(model,original_filternums, target_filternums):
    new_model=deepcopy(model)
    list_parameters=list(model.named_parameters())
    for idx,parameter in enumerate(list_parameters):
        pname=parameter[0]
        pname_parts=str(pname).split('.')
        if pname_parts[-1] == 'weight': # or pname_parts[-1] == 'bias'
            layer_name=str('.').join(pname_parts[:len(pname_parts)-1])
            layer_params=parameter[1]
            layer_tensor=layer_tensor=rgetattr(model,layer_name)
            in_filternum,out_filternum=get_layer_filternum(layer_tensor)
            changed_in,changed_out=False,False
            new_in_filternum,new_out_filternum=in_filternum,out_filternum
            for fidx,targetf in enumerate(target_filternums):
                if idx==0:
                    new_in_filternum=in_filternum
                    changed_in=True
                    if out_filternum==original_filternums[fidx]:
                        new_out_filternum=target_filternums[fidx]
                        changed_out=True
                elif idx > 0:
                    if in_filternum==original_filternums[fidx]:
                        new_in_filternum=target_filternums[fidx]
                        changed_in=True
                    if out_filternum==original_filternums[fidx]:
                        new_out_filternum=target_filternums[fidx]
                        changed_out=True
                if changed_in and changed_out:
                    #print(pname+"(in,out)=(%i,%i) (newin,newout)=(%i,%i)"%(in_filternum,out_filternum,new_in_filternum,new_out_filternum))
                    #pdb.set_trace()
                    new_layer_tensor=set_input_output_filternum(layer_tensor,new_in_filternum,new_out_filternum)
                    rsetattr(new_model,layer_name,new_layer_tensor)
                    break
                
    return new_model
def enforce_network_capacity(model, num_target_params):
    new_model=deepcopy(model)
    num_current_params=print_model_report(model,False)
    original_filternum=get_model_unique_filternum(model)
    current_filternum=original_filternum
    original_prop_filternum=list(map(int,np.divide(original_filternum,np.min(original_filternum))))
    offset=2
    if num_current_params < num_target_params:
        while(num_current_params < num_target_params):        
            min_filter=int(np.min(original_filternum)+offset)
            current_filternum=[int(min_filter*prop) for prop in original_prop_filternum]
            new_model=substitute_filternum(model,original_filternum, current_filternum)
            num_current_params=print_model_report(new_model,False)
            offset+=2
    elif num_current_params > num_target_params:
        while(num_current_params > num_target_params):
            min_filter=int(np.min(original_filternum)-offset)
            current_filternum=[int(min_filter*prop) for prop in original_prop_filternum]
            new_model=substitute_filternum(model,original_filternum, current_filternum)
            num_current_params=print_model_report(new_model,False)
            offset-=2
    else:
        return model
    return new_model
    
