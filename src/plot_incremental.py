import argparse
import matplotlib.pyplot as plt
import utils
import numpy as np
from numpy import loadtxt
import IPython.display as display #display plots in line
import os #current folder = os.getcwd()
from natsort import natsorted
import pdb
########################ARGUMENTS
parser = argparse.ArgumentParser(description='xxx')
parser.add_argument('--results_path', type=str, default='data/experiments/LLL/', help='(default=%(default)s)')
parser.add_argument('--plot_exp', type=str, default='auto', nargs='+', help='(default=%(default)s)')
parser.add_argument('--noplot', action='store_true', help='(default=%(default)s)')
parser.add_argument('--name_exp', type=str, default='auto', nargs='+', help='(default=%(default)s)')
args, extra_args = parser.parse_known_args()

## default locations
results_path=args.results_path
if args.plot_exp=='auto':
    plot_exp=args.plot_exp
else:
    plot_exp=args.plot_exp[0].split()
if args.name_exp=='auto':
    name_exp=args.name_exp
else:
    name_exp=args.name_exp[0].split()
results_folder='results'
nepoch=200
file_taw_acc,file_tag_acc,file_forg_taw,file_forg_tag,file_capacity='acc_taw.txt','acc_tag.txt','forg_taw.txt','forg_tag.txt','capacity.txt'
plot_size=(6.4,4.8) #width, height (inches)
experiment_name=results_path.split('/')[-1]
plt_colors=['blue','orange','green','red','yellow', 'cyan', 'magenta', 'black']


########################selected experiments 

#list of experiments
if plot_exp!='auto': #select these
    list_experiments=plot_exp
    color_experiments=plt_colors #red, cyan, magenta, black, white
    adapted_plot_size=plot_size
else: #select all in folder
    #read all experiments and discard empty results ones
    list_experiments=os.listdir(results_path)
    list_experiments=[experiment for experiment in list_experiments if os.path.exists(results_path+'/'+experiment+'/'+results_folder+'/'+file_taw_acc) and os.path.exists(results_path+'/'+experiment+'/'+results_folder+'/'+file_tag_acc)]
    list_experiments=natsorted(list_experiments, key=lambda y: y.lower())
    
    #random permutation of colors
    color_experiments=plt_colors #color_experiments=list(np.random.permutation(plt_colors))
    while len(color_experiments)<len(list_experiments):
        color_experiments*=2
        
    #change plot size if there are too much experiments
    if len(list_experiments)>5:
        adapted_plot_size=(len(list_experiments),len(list_experiments)-2) #width, height
    else:
        adapted_plot_size=plot_size


#arg for legend exp names
if name_exp=='auto':
    name_experiments=list_experiments
else:
    name_experiments=name_exp

#########################read files
acc_taw,acc_tag,forg_taw,forg_tag,capacity=[],[],[],[],[]
#erase unexistent results for existing experiment folders
list_experiments_aux=list_experiments.copy()
name_experiments_aux=name_experiments.copy()
for experiment in list_experiments:
    path_experiment=results_path+'/'+experiment+'/'+results_folder
    json_path=results_path+"/"+experiment+"/"+"raw_log.txt"
    if not os.path.exists(path_experiment+'/'+file_taw_acc):
        list_experiments_aux.remove(experiment)
        name_experiments_aux.remove(experiment)
    elif not os.path.exists(json_path):
        list_experiments_aux.remove(experiment)
        name_experiments_aux.remove(experiment)
list_experiments=list_experiments_aux.copy()
name_experiments=name_experiments_aux.copy()
#read and parse
for experiment in list_experiments:
    path_experiment=results_path+'/'+experiment+'/'+results_folder
    print(path_experiment+'/'+file_taw_acc)
    try:
        acc_taw.append(loadtxt(path_experiment+'/'+file_taw_acc,delimiter=" "))
        acc_tag.append(loadtxt(path_experiment+'/'+file_tag_acc,delimiter=" "))
        forg_taw.append(loadtxt(path_experiment+'/'+file_forg_taw,delimiter=" "))
        forg_tag.append(loadtxt(path_experiment+'/'+file_forg_tag,delimiter=" "))
    except:
        acc_taw.append(loadtxt(path_experiment+'/'+file_taw_acc,delimiter='\t'))
        acc_tag.append(loadtxt(path_experiment+'/'+file_tag_acc,delimiter='\t'))
        forg_taw.append(loadtxt(path_experiment+'/'+file_forg_taw,delimiter='\t'))
        forg_tag.append(loadtxt(path_experiment+'/'+file_forg_tag,delimiter='\t'))
    try:
        capacity.append(loadtxt(path_experiment+'/'+file_capacity))
    except:
        capacity.append(None)
        
forg_taw_normalized=np.zeros(np.shape(forg_taw))
forg_tag_normalized=np.zeros(np.shape(forg_tag))
try:
    for idx,experiment in enumerate(list_experiments):
        for t in range(0,len(forg_taw[idx])):
            forg_taw_normalized[idx][t,:]=np.divide(forg_taw[idx][t,:],acc_taw[idx][t,t])
            forg_tag_normalized[idx][t,:]=np.divide(forg_tag[idx][t,:],acc_tag[idx][t,t])
except:
    print("1 task no forgetting")

try:
    acc_limits=[np.min([acc_taw[:][:],acc_tag[:]]),np.max([acc_taw[:],acc_tag[:]])]
    forg_limits=[np.min([forg_taw[:],forg_tag[:]]),np.max([forg_taw[:],forg_tag[:]])]
except:
    acc_limits=[0,1]        
    forg_limits=[0,1]        
    
try:
    num_tasks=len(acc_taw[idx])
except:
    num_tasks=1
        
while len(plt_colors)<num_tasks:
    plt_colors*=2
    
#########################export csv with averages
csv_array=[[[] for j in range(8)] for i in range(len(list_experiments)+1)]
csv_array[0][0]='Model'
csv_array[0][1]='AvgAccTaw'
csv_array[0][2]='AvgAccTag'
csv_array[0][3]='AvgForgTaw'
csv_array[0][4]='AvgForgTag'
csv_array[0][5]='AvgForgTawNm'
csv_array[0][6]='AvgForgTagNm'
csv_array[0][7]='Capacity'
for idx,experiment in enumerate(list_experiments):
    last_task=num_tasks
    if last_task > 1:
        csv_array[idx+1][0]=str(name_experiments[idx])
        csv_array[idx+1][1]=str(np.nanmean(acc_taw[idx][last_task-1,:]))[0:7]
        csv_array[idx+1][2]=str(np.nanmean(acc_tag[idx][last_task-1,:]))[0:7]
        csv_array[idx+1][3]=str(np.nanmean(forg_taw[idx][last_task-1,:]))[0:7]
        csv_array[idx+1][4]=str(np.nanmean(forg_tag[idx][last_task-1,:]))[0:7]
        csv_array[idx+1][5]=str(np.nanmean(forg_taw_normalized[idx][last_task-1,:]))[0:7]
        csv_array[idx+1][6]=str(np.nanmean(forg_tag_normalized[idx][last_task-1,:]))[0:7]    
        csv_array[idx+1][7]=str(capacity[idx])[0:7]
    else:
        csv_array[idx+1][0]=str(name_experiments[idx])
        csv_array[idx+1][1]=str(np.nanmean(acc_taw[idx]))[0:7]
        csv_array[idx+1][2]=str(np.nanmean(acc_tag[idx]))[0:7]
        csv_array[idx+1][3]=str(0)[0:7]
        csv_array[idx+1][4]=str(0)[0:7]
        csv_array[idx+1][5]=str(0)[0:7]
        csv_array[idx+1][6]=str(0)[0:7]
        csv_array[idx+1][7]=str(capacity[idx])[0:7]
print(csv_array)
np.savetxt(results_path+"/"+experiment_name+".csv", np.asarray(csv_array), delimiter=",",fmt='%s')
print(results_path+"/"+experiment_name+".csv")

############################PLOT

if args.noplot==True:
    exit()
    


########################################
import numpy.matlib
plt.style.use('seaborn-whitegrid')
#fig = plt.figure()
fig, axs = plt.subplots(1, num_tasks, sharex=True)
fig.subplots_adjust(hspace=0)
#ax = plt.axes()
num_classes=100 #note: edit last task tick manually
for idx,experiment in enumerate(list_experiments):
    acc_taw[idx][acc_taw[idx]==0]=None
    for t in range(0,num_tasks):
        x_axis = np.arange(num_tasks)+1
        data=acc_taw[idx][:,t]*100
        axs[t].plot(x_axis, data, color=color_experiments[idx], label=name_experiments[idx], linewidth=1.5,marker='o', linestyle='-', markersize=6)
    #x_axis = [(task + 1)*(num_classes / num_tasks) for task in range(num_tasks)]
    #data=acc_taw[idx]
    #ax.plot(x_axis, data, color=color_experiments[idx], label=name_experiments[idx], linewidth=1.5,marker='o', linestyle='-', markersize=6)

# Put ticks
for t in range(1,num_tasks):
    axs[t].set_yticks([])
    axs[t].set_xticks([])
    axs[t].set_ylim(0, 101)
    axs[t].set_title(''.join(['T=',str(t+1)]))
axs[0].set_title(''.join(['T=',str(1)]))
axs[0].set_ylabel("Accuracy (%)")
axs[0].set_ylim(0, 101)
axs[0].grid(False)
axs[num_tasks-5].set_zorder(1)
axs[num_tasks-5].legend(name_experiments ,bbox_to_anchor=(0,0, 5.5, 0), fancybox=True, loc='lower left',mode="expand", ncol=len(name_experiments)) #, 
fig.set_size_inches(15,3)
fig.subplots_adjust(hspace=0)

# Format plot and save figure
plt.tight_layout()
fig.savefig(results_path+"/"+"acc_taw.png")
########################################

########################PARSE TRAINING LOSSES AND ACCS
all_loss_step=dict()
all_loss=dict()
all_acc=dict()    
#separation between tasks
xtrain=np.zeros(num_tasks)
xtrain_step=np.zeros(num_tasks)
xtest_step=np.zeros(num_tasks)
xvalid=np.zeros(num_tasks)
for experiment in list_experiments:
    json_path=results_path+"/"+experiment+"/"+"raw_log.txt"
    parsed_json=utils.read_json_rawlog(json_path)
    loss,acc,loss_step=utils.parse_rawlog(parsed_json,num_tasks)
    
    '''  #careful, also cuts x values
    #cut data to avoid overplotting
    for t in range(num_tasks):
        loss_step[t]['train']=loss_step[t]['train'][::16]
        loss[t]['train']=loss[t]['train'][::4]
        #loss[t]['test']=loss[t]['test'][::4]
        loss[t]['valid']=loss[t]['valid'][::4]
        acc[t]['train']=acc[t]['train'][::4]
        #acc[t]['test']=acc[t]['test'][::4]
        acc[t]['valid']=acc[t]['valid'][::4]
    '''
    
    #arrange in dicts
    all_loss_step[experiment]={'train':[],'test':[],'valid':[]}
    all_loss[experiment]={'train':[],'test':[],'valid':[]}
    all_acc[experiment]={'train':[],'test':[],'valid':[]}
    all_loss_step[experiment]['train']=[loss_step[t]['train'] for t in range(num_tasks)]
    all_loss_step[experiment]['test']=[loss_step[t]['test'] for t in range(num_tasks)]
    all_loss_step[experiment]['valid']=[loss_step[t]['valid'] for t in range(num_tasks)]
    all_loss[experiment]['train']=[loss[t]['train'] for t in range(num_tasks)]
    all_loss[experiment]['test']=[loss[t]['test'] for t in range(num_tasks)]
    all_loss[experiment]['valid']=[loss[t]['valid'] for t in range(num_tasks)]
    all_acc[experiment]['train']=[acc[t]['train'] for t in range(num_tasks)]
    all_acc[experiment]['test']=[acc[t]['test'] for t in range(num_tasks)]
    all_acc[experiment]['valid']=[acc[t]['valid'] for t in range(num_tasks)]
    #add zeros to lrmin discarded epochs
    nepoch_tasks_train=[len(all_loss[experiment]['train'][t]) for t in range(num_tasks)]
    nepoch_tasks_test=[len(all_loss[experiment]['test'][t]) for t in range(num_tasks)]
    nepoch_tasks_valid=[len(all_loss[experiment]['valid'][t]) for t in range(num_tasks)]
    for t in range(num_tasks):
        try:
            all_loss[experiment]['train'][t][nepoch_tasks_train[t]:nepoch]=np.NaN*np.ones(nepoch-nepoch_tasks_train[t])
            #all_loss[experiment]['test'][t][nepoch_tasks_test[t]:nepoch]=np.NaN*np.ones(nepoch-nepoch_tasks_test[t])
            all_loss[experiment]['valid'][t][nepoch_tasks_valid[t]:nepoch]=np.NaN*np.ones(nepoch-nepoch_tasks_valid[t])
            all_acc[experiment]['train'][t][nepoch_tasks_train[t]:nepoch]=np.NaN*np.ones(nepoch-nepoch_tasks_train[t])
            #all_acc[experiment]['test'][t][nepoch_tasks_test[t]:nepoch]=np.NaN*np.ones(nepoch-nepoch_tasks_test[t])
            all_acc[experiment]['valid'][t][nepoch_tasks_valid[t]:nepoch]=np.NaN*np.ones(nepoch-nepoch_tasks_valid[t])
        except:
            print(experiment)
    #concat lists
    all_loss_step[experiment]['train'] = [val for sublist in all_loss_step[experiment]['train'] for val in sublist]
    all_loss[experiment]['train'] = [val for sublist in all_loss[experiment]['train'] for val in sublist]
    all_loss[experiment]['test'] = [val for sublist in all_loss[experiment]['test'] for val in sublist]
    all_loss[experiment]['valid'] = [val for sublist in all_loss[experiment]['valid'] for val in sublist]
    all_acc[experiment]['train'] = [val for sublist in all_acc[experiment]['train'] for val in sublist]
    all_acc[experiment]['valid'] = [val for sublist in all_acc[experiment]['valid'] for val in sublist]

    #plots
    
    current_task_iter=0
    prev_task_iter=0
    for t in range(num_tasks):
        current_task_iter=len(loss_step[t]['train'])
        plt.plot(range(prev_task_iter,current_task_iter+prev_task_iter),loss_step[t]['train'],color=plt_colors[t])
        prev_task_iter+=current_task_iter
        xtrain_step[t]=prev_task_iter
    plt.ylabel('Loss Step (train)')
    plt.xlabel('step [batch]')
    plt.ylim((0,10))
    plt.legend(['T'+ str(x) for x in range(1,num_tasks+1)])
    plt.rcParams["figure.figsize"]=plot_size
    plt.savefig(results_path+"/"+experiment+"/"+results_folder+"/"+"iter_loss_step_train.png")
    plt.close()
    
    current_task_iter=0
    prev_task_iter=0
    for t in range(num_tasks):
        current_task_iter=len(loss[t]['train'])
        plt.plot(range(prev_task_iter,current_task_iter+prev_task_iter),loss[t]['train'],color=plt_colors[t])
        prev_task_iter+=current_task_iter
        xtrain[t]=prev_task_iter
    plt.ylabel('Loss (train)')
    plt.xlabel('iter [epoch]')
    plt.ylim((0,10))
    plt.legend(['T'+ str(x) for x in range(1,num_tasks+1)])
    plt.rcParams["figure.figsize"]=plot_size
    plt.savefig(results_path+"/"+experiment+"/"+results_folder+"/"+"iter_loss_train.png")
    plt.close()
    
    current_task_iter=0
    prev_task_iter=0
    for t in range(num_tasks):
        current_task_iter=len(acc[t]['train'])
        plt.plot(range(prev_task_iter,current_task_iter+prev_task_iter),acc[t]['train'],color=plt_colors[t])
        prev_task_iter+=current_task_iter
    plt.ylabel('Accuracy (train)')
    plt.xlabel('iter [epoch]')
    plt.legend(['T'+ str(x) for x in range(1,num_tasks+1)])
    plt.rcParams["figure.figsize"]=plot_size
    plt.savefig(results_path+"/"+experiment+"/"+results_folder+"/"+"iter_acc_train.png")
    plt.close()
    '''
    current_task_iter=0
    prev_task_iter=0
    for t in range(num_tasks):
        current_task_iter=len(loss_step[t]['test'])
        plt.plot(range(prev_task_iter,current_task_iter+prev_task_iter),loss_step[t]['test'],color=plt_colors[t])
        prev_task_iter+=current_task_iter
    plt.ylabel('Loss Step (test)')
    plt.xlabel('step [batch]')
    plt.legend(['T'+ str(x) for x in range(1,num_tasks+1)])
    plt.rcParams["figure.figsize"]=plot_size
    plt.savefig(results_path+"/"+experiment+"/"+results_folder+"/"+"iter_loss_step_test.png")
    plt.close()
    '''
    current_task_iter=0
    prev_task_iter=0
    for t in range(num_tasks):
        current_task_iter=len(loss[t]['test'])
        plt.plot(range(prev_task_iter,current_task_iter+prev_task_iter),loss[t]['test'],color=plt_colors[t])
        prev_task_iter+=current_task_iter
        xtest_step[t]=prev_task_iter
    plt.ylabel('Loss (test)')
    plt.xlabel('step [batch]')
    plt.ylim((0,10))
    plt.legend(['T'+ str(x) for x in range(1,num_tasks+1)])
    plt.rcParams["figure.figsize"]=plot_size
    plt.savefig(results_path+"/"+experiment+"/"+results_folder+"/"+"iter_loss_test.png")
    plt.close()
    '''
    current_task_iter=0
    prev_task_iter=0
    for t in range(num_tasks):
        current_task_iter=len(acc[t]['test'])
        plt.plot(range(prev_task_iter,current_task_iter+prev_task_iter),acc[t]['test'],color=plt_colors[t])
        prev_task_iter+=current_task_iter
    plt.ylabel('Accuracy (test)')
    plt.xlabel('iter [epoch]')
    plt.legend(['T'+ str(x) for x in range(1,num_tasks+1)])
    plt.rcParams["figure.figsize"]=plot_size
    plt.savefig(results_path+"/"+experiment+"/"+results_folder+"/"+"iter_acc_test.png")
    plt.close()
    '''
    '''
    current_task_iter=0
    prev_task_iter=0
    for t in range(num_tasks):
        current_task_iter=len(loss_step[t]['valid'])
        plt.plot(range(prev_task_iter,current_task_iter+prev_task_iter),loss_step[t]['valid'],color=plt_colors[t])
        prev_task_iter+=current_task_iter
    plt.ylabel('Loss Step (valid)')
    plt.xlabel('step [batch]')
    plt.ylim((0,10))
    plt.legend(['T'+ str(x) for x in range(1,num_tasks+1)])
    plt.rcParams["figure.figsize"]=plot_size
    plt.savefig(results_path+"/"+experiment+"/"+results_folder+"/"+"iter_loss_step_valid.png")
    plt.close()
    '''
    current_task_iter=0
    prev_task_iter=0
    for t in range(num_tasks):
        current_task_iter=len(loss[t]['valid'])
        plt.plot(range(prev_task_iter,current_task_iter+prev_task_iter),loss[t]['valid'],color=plt_colors[t])
        prev_task_iter+=current_task_iter
        xvalid[t]=prev_task_iter
    plt.ylabel('Loss (valid)')
    plt.xlabel('iter [epoch]')
    plt.ylim((0,10))
    plt.legend(['T'+ str(x) for x in range(1,num_tasks+1)])
    plt.rcParams["figure.figsize"]=plot_size
    plt.savefig(results_path+"/"+experiment+"/"+results_folder+"/"+"iter_loss_valid.png")
    plt.close()
    
    current_task_iter=0
    prev_task_iter=0
    for t in range(num_tasks):
        current_task_iter=len(acc[t]['valid'])
        plt.plot(range(prev_task_iter,current_task_iter+prev_task_iter),acc[t]['valid'],color=plt_colors[t])
        prev_task_iter+=current_task_iter
    plt.ylabel('Accuracy (valid)')
    plt.xlabel('iter [epoch]')
    plt.legend(['T'+ str(x) for x in range(1,num_tasks+1)])
    plt.rcParams["figure.figsize"]=plot_size
    plt.savefig(results_path+"/"+experiment+"/"+results_folder+"/"+"iter_acc_valid.png")
    plt.close()

    
'''    
#plot joined losses and accs (comparison between models)
for idx,experiment in enumerate(list_experiments):
    plt.plot(all_loss_step[experiment]['train'],color=color_experiments[idx],markevery=100)
plt.ylabel('Loss Step (train)')
plt.xlabel('step [batch]')
plt.ylim((0,10))
plt.legend(name_experiments)
plt.rcParams["figure.figsize"]=adapted_plot_size
plt.savefig(results_path+"/"+"all_iter_loss_step_train.png")
plt.close()
'''
for idx,experiment in enumerate(list_experiments):
    plt.plot(all_loss[experiment]['train'],color=color_experiments[idx],markevery=100)
plt.ylabel('Loss (train)')
plt.xlabel('iter [epoch]')
plt.ylim((0,10))
plt.legend(name_experiments)
plt.rcParams["figure.figsize"]=adapted_plot_size
plt.savefig(results_path+"/"+"all_iter_loss_train.png")
plt.close()

for idx,experiment in enumerate(list_experiments):
    plt.plot(all_loss[experiment]['test'],color=color_experiments[idx])
for t in range(num_tasks):
    plt.axvline(x=xtest_step[t],color='black')
plt.ylabel('Loss (test)')
#xtcks=[range(1,i+1) for i in range(1,num_tasks+1)]
#xtcks_concat=[val for sublist in xtcks for val in sublist]
#plt.xticks(range(len(xtcks_concat)),(xtcks_concat))
#plt.xticks(range(num_tasks),["1.."+str(i) for i in range(1,num_tasks+1)])
plt.xticks([])
plt.xlabel('task (1..T)')
plt.ylim((0,10))
plt.legend(name_experiments)
plt.rcParams["figure.figsize"]=adapted_plot_size
plt.savefig(results_path+"/"+"all_iter_loss_test.png")
plt.close()

for idx,experiment in enumerate(list_experiments):
    plt.plot(all_loss[experiment]['valid'],color=color_experiments[idx],markevery=100)
plt.ylabel('Loss (valid)')
plt.xlabel('iter [epoch]')
plt.ylim((0,10))
plt.legend(name_experiments)
plt.rcParams["figure.figsize"]=adapted_plot_size
plt.savefig(results_path+"/"+"all_iter_loss_valid.png")
plt.close()

for idx,experiment in enumerate(list_experiments):
    plt.plot(all_acc[experiment]['train'],color=color_experiments[idx],markevery=100)
plt.ylabel('Accuracy (train)')
plt.xlabel('iter [epoch]')
plt.legend(name_experiments)
plt.rcParams["figure.figsize"]=adapted_plot_size
plt.savefig(results_path+"/"+"all_iter_acc_train.png")
plt.close()

for idx,experiment in enumerate(list_experiments):
    plt.plot(all_acc[experiment]['valid'],color=color_experiments[idx],markevery=100)
plt.ylabel('Accuracy (valid)')
plt.xlabel('iter [epoch]')
plt.legend(name_experiments)
plt.rcParams["figure.figsize"]=adapted_plot_size
plt.savefig(results_path+"/"+"all_iter_acc_valid.png")
plt.close()


###############plot accuracies and forgetting for all tasks
for idx,experiment in enumerate(list_experiments):
    plt.suptitle("Accuracy (Task Aware)")
    plt.title(name_experiments[idx])
    for t in range(0,num_tasks):
        #print(np.mean(acc_taw[idx][t,:]))
        #print(acc_taw[idx][t,:])
        acc_taw[idx][acc_taw[idx]==0]=None #make all zeros to none
        plt.plot(range(1,num_tasks+1),acc_taw[idx][:,t])
        plt.hlines(y=np.nanmean(acc_taw[idx][t-1,:]),xmin=t,xmax=num_tasks, color=plt_colors[t], linestyle='--',label=name_experiments[idx])
        plt.ylim(acc_limits)
    plt.legend(['T'+ str(x) for x in range(1,num_tasks+1)])
    plt.ylabel('Accuracy')
    plt.xlabel('Task')
    plt.xticks(range(1,num_tasks+1))
    plt.rcParams["figure.figsize"]=plot_size
    plt.savefig(results_path+"/"+experiment+"/"+results_folder+"/"+"acc_taw.png")
    plt.close()
    #plt.pause(1)
    #plt.close()
    #plt.show()



for idx,experiment in enumerate(list_experiments):
    plt.suptitle("Accuracy (Task Agnostic)")
    plt.title(name_experiments[idx])
    for t in range(0,num_tasks):
        #print(np.mean(acc_tag[idx][t,:]))
        #print(acc_tag[idx][t,:])
        acc_tag[idx][acc_tag[idx]==0]=None #make all zeros to none
        plt.plot(range(1,num_tasks+1),acc_tag[idx][:,t])
        plt.hlines(y=np.nanmean(acc_tag[idx][t-1,:]),xmin=t,xmax=num_tasks, color=plt_colors[t], linestyle='--',label=name_experiments[idx])
        plt.ylim(acc_limits)
    plt.legend(['T'+ str(x) for x in range(1,num_tasks+1)])
    plt.ylabel('Accuracy')
    plt.xlabel('Task')
    plt.xticks(range(1,num_tasks+1))
    plt.rcParams["figure.figsize"]=plot_size
    plt.savefig(results_path+"/"+experiment+"/"+results_folder+"/"+"acc_tag.png")
    plt.close()
for idx,experiment in enumerate(list_experiments):
    plt.suptitle("Forgetting (Task Aware)")
    plt.title(name_experiments[idx])
    for t in range(0,len(forg_taw[idx])):
        forg_taw[idx][forg_taw[idx]==0]=None #make all zeros to none
        plt.plot(range(1,len(forg_taw[idx])+1),forg_taw[idx][:,t])
        plt.hlines(y=np.nanmean(forg_taw[idx][t-1,:]),xmin=t,xmax=len(forg_taw[idx]), color=plt_colors[t], linestyle='--',label=name_experiments[idx])
        plt.ylim(forg_limits)
    plt.legend(['T'+ str(x) for x in range(1,len(forg_taw[idx])+1)])
    plt.ylabel('Forgetting (Accuracy)')
    plt.xlabel('Task')
    plt.xticks(range(1,num_tasks+1))
    plt.rcParams["figure.figsize"]=plot_size
    plt.savefig(results_path+"/"+experiment+"/"+results_folder+"/"+"forg_taw.png")
    plt.close()
for idx,experiment in enumerate(list_experiments):
    plt.suptitle("Forgetting (Task Agnostic)")
    plt.title(name_experiments[idx])
    for t in range(0,len(forg_tag[idx])):
        forg_tag[idx][forg_tag[idx]==0]=None #make all zeros to none
        plt.plot(range(1,len(forg_tag[idx])+1),forg_tag[idx][:,t])
        plt.hlines(y=np.nanmean(forg_tag[idx][t-1,:]),xmin=t,xmax=len(forg_tag[idx]), color=plt_colors[t], linestyle='--',label=name_experiments[idx])
        plt.ylim(forg_limits)
        plt.rcParams["figure.figsize"]=plot_size
    plt.legend(['T'+ str(x) for x in range(1,len(forg_tag[idx])+1)])
    plt.ylabel('Forgetting (Accuracy)')
    plt.xlabel('Task')
    plt.xticks(range(1,num_tasks+1))
    plt.savefig(results_path+"/"+experiment+"/"+results_folder+"/"+"forg_tag.png")
    plt.close()

#############################plot accuracies and forgetting after last task
'''
for idx,experiment in enumerate(list_experiments):
    plt.suptitle("Accuracy after Last Task (Task Aware)")
    last_task=num_tasks
    #print(name_experiments[idx]+" -> avg Acc_tAw:"+str(np.mean(acc_taw[idx][last_task-1,:])))
    #print(acc_taw[idx][last_task-1,:])
    plt.plot(range(1,num_tasks+1),acc_taw[idx][last_task-1,:],color=color_experiments[idx],label=name_experiments[idx])
    plt.hlines(y=np.nanmean(acc_taw[idx][last_task-1,:]),xmin=1,xmax=last_task, color=color_experiments[idx], linestyle='--',label=name_experiments[idx])
plt.ylim(acc_limits)
plt.legend(name_experiments)
plt.ylabel('Accuracy')
plt.xlabel('Task')
plt.xticks(range(1,num_tasks+1))
plt.rcParams["figure.figsize"]=adapted_plot_size
plt.savefig(results_path+"/"+"last_acc_taw.png")
plt.close()    
for idx,experiment in enumerate(list_experiments):
    plt.suptitle("Accuracy after Last Task (Task Agnostic)")
    last_task=num_tasks
    #print(name_experiments[idx]+" -> avg Acc_tAg:"+str(np.mean(acc_tag[idx][last_task-1,:])))
    #print(acc_tag[idx][last_task-1,:])
    plt.plot(range(1,num_tasks+1),acc_tag[idx][last_task-1,:],color=color_experiments[idx],label=name_experiments[idx])
    plt.hlines(y=np.nanmean(acc_tag[idx][last_task-1,:]),xmin=1,xmax=last_task, color=color_experiments[idx], linestyle='--',label=name_experiments[idx])
plt.ylim(acc_limits)
plt.legend(name_experiments)
plt.ylabel('Accuracy')
plt.xlabel('Task')
plt.xticks(range(1,num_tasks+1))
plt.rcParams["figure.figsize"]=adapted_plot_size
plt.savefig(results_path+"/"+"last_acc_tag.png")
plt.close()
for idx,experiment in enumerate(list_experiments):
    plt.suptitle("Forgetting after Last Task (Task Aware)")
    last_task=len(forg_taw[idx])
    #print(name_experiments[idx]+" -> avg Forg_tAw:"+str(np.mean(forg_taw[idx][last_task-1,:])))
    #print(forg_taw[idx][last_task-1,:])
    plt.plot(range(1,num_tasks+1),forg_taw[idx][last_task-1,:],color=color_experiments[idx],label=name_experiments[idx])
    plt.hlines(y=np.nanmean(forg_taw[idx][last_task-1,:]),xmin=1,xmax=last_task, color=color_experiments[idx], linestyle='--',label=name_experiments[idx])
plt.ylim(forg_limits)
plt.legend(name_experiments)
plt.ylabel('Forgetting (Accuracy)')
plt.xlabel('Task')
plt.xticks(range(1,num_tasks+1))
plt.rcParams["figure.figsize"]=adapted_plot_size
plt.savefig(results_path+"/"+"last_forg_taw.png")
plt.close()    
for idx,experiment in enumerate(list_experiments):
    plt.suptitle("Forgetting after Last Task (Task Agnostic)")
    last_task=len(forg_tag[idx])
    #print(name_experiments[idx]+" -> avg Forg_tAg:"+str(np.mean(forg_tag[idx][last_task-1,:])))
    #print(forg_tag[idx][last_task-1,:])
    plt.plot(range(1,num_tasks+1),forg_tag[idx][last_task-1,:],color=color_experiments[idx],label=name_experiments[idx])
    plt.hlines(y=np.nanmean(forg_tag[idx][last_task-1,:]),xmin=1,xmax=last_task, color=color_experiments[idx], linestyle='--',label=name_experiments[idx])
plt.ylim(forg_limits)
plt.legend(name_experiments)
plt.ylabel('Forgetting (Accuracy)')
plt.xlabel('Task')
plt.xticks(range(1,num_tasks+1))
plt.rcParams["figure.figsize"]=adapted_plot_size
plt.savefig(results_path+"/"+"last_forg_tag.png")
plt.close()
'''

X=np.arange(num_tasks)
fig=plt.figure()
ax = fig.add_subplot(111)
barwid=0.15
for idx,experiment in enumerate(list_experiments):
    last_task=num_tasks
    ax.bar(X+1 + idx*barwid, acc_taw[idx][last_task-1,:]*100, color = color_experiments[idx], width = barwid, label=name_experiments[idx], align='center')
ax.set_ylabel('Accuracy (%)')
ax.set_xlabel('Task')
ax.set_ylim(0,101)
ax.set_title('Accuracy after last task (Task Aware)')
ax.set_xticks(X+1)
ax.legend(labels=name_experiments)
plt.rcParams["figure.figsize"]=(6.4,4.8)
plt.savefig(results_path+"/"+"last_acc_taw.png")
plt.close()    

X=np.arange(num_tasks)
fig=plt.figure()
ax = fig.add_subplot(111)
barwid=0.15
for idx,experiment in enumerate(list_experiments):
    last_task=num_tasks
    ax.bar(X+1 + idx*barwid, acc_tag[idx][last_task-1,:]*100, color = color_experiments[idx], width = barwid, label=name_experiments[idx], align='center')
ax.set_ylabel('Accuracy (%)')
ax.set_xlabel('Task')
ax.set_ylim(0,101)
ax.set_title('Accuracy after last task (Task Agnostic)')
ax.set_xticks(X+1)
ax.legend(labels=name_experiments,loc='upper left')
plt.rcParams["figure.figsize"]=(6.4,4.8)
plt.savefig(results_path+"/"+"last_acc_tag.png")
plt.close()    


X=np.arange(num_tasks-1)
fig=plt.figure()
ax = fig.add_subplot(111)
barwid=0.15
for idx,experiment in enumerate(list_experiments):
    last_task=num_tasks
    ax.bar(X+1 + idx*barwid, -(forg_taw[idx][last_task-1,0:last_task-1]*100), color = color_experiments[idx], width = barwid, label=name_experiments[idx], align='center')
ax.set_ylabel('Forgetting (%)')
ax.set_xlabel('Task')
ax.set_ylim(-101,0)
ax.set_title('Forgetting after last task (Task Aware)')
ax.set_xticks(X+1)
ax.legend(labels=name_experiments)
plt.rcParams["figure.figsize"]=(6.4,4.8)
plt.savefig(results_path+"/"+"last_forg_taw.png")
plt.close()    

X=np.arange(num_tasks-1)
fig=plt.figure()
ax = fig.add_subplot(111)
barwid=0.15
for idx,experiment in enumerate(list_experiments):
    last_task=num_tasks
    ax.bar(X+1 + idx*barwid, -(forg_tag[idx][last_task-1,0:last_task-1]*100), color = color_experiments[idx], width = barwid, label=name_experiments[idx], align='center')
ax.set_ylabel('Forgetting (%)')
ax.set_xlabel('Task')
ax.set_ylim(-101,0)
ax.set_title('Forgetting after last task (Task Agnostic)')
ax.set_xticks(X+1)
ax.legend(labels=name_experiments)
plt.rcParams["figure.figsize"]=(6.4,4.8)
plt.savefig(results_path+"/"+"last_forg_tag.png")
plt.close()    






