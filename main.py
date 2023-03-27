from utils import *
from models import *
from imgaug import augmenters as iaa
# os.environ['CUDA_VISIBLE_DEVICES']='1'
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from tqdm import tqdm
from copy import deepcopy

def main(run, params,k,fold):
    assert fold<k, 'current fold should be smaller than k'
    ## print params
    print('run', run)
    print(params)
    
    ## read params
    model=params['model']
    attn_stru=params['attn_stru']
    c=params['c']
    reuse_weight=params['reuse_weight']
    block_backprop=params['block_backprop']
    epochs=params['epochs']
    learning_rate=params['learning_rate']
    weight_decay=params['weight_decay']
    full_label_perc=params['full_label_perc']
    
    ## dataloaders
    # https://imgaug.readthedocs.io/en/latest/source/examples_basics.html
    transform = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(
            rotate=(-20, 20),
        )
    ], random_order=True)
    dataset_train=ColonCancerDataset('newdata',transform=transform,train=True,k=k,fold=fold,full_label_perc=full_label_perc)
    dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True,num_workers=12)
    dataset_test=ColonCancerDataset('newdata',transform=None,train=False,k=k,fold=fold)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    ## init model,criterion and optimizer
    if model == 'Vanilla_MIL':
        net = Vanilla_MIL(attn_stru)
    elif model == 'Instance_MIL':
        net = Instance_MIL(attn_stru, reuse_weight, block_backprop,c)
    net = net.cuda()
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.5)
    
    ## start training
    shutil.rmtree(f'checkpoints/run{run}', ignore_errors=True);os.mkdir(f'checkpoints/run{run}');np.save(f'checkpoints/run{run}/params.npy',params)

    loss_history = {'bag':{'train': [], 'val': []}, 'inst':{'train': [], 'val': []}}
    error_history = {'bag':{'train': [], 'val': []}, 'inst':{'train': [], 'val': []}}
    lowest = 1e8

    for epoch in tqdm(range(epochs)):
        # train
        LOSS, confusion=train(net,dataloader_train,criterion,optimizer)
        ERR,_=calc_metrics(confusion)
        loss_history['bag']['train'].append(LOSS['bag'])
        loss_history['inst']['train'].append(LOSS['inst'])
        error_history['bag']['train'].append(ERR['bag'])
        error_history['inst']['train'].append(ERR['inst'])

        # validation
        LOSS,confusion,_,_,_ = val_test(net, dataloader_test, criterion)
        ERR,_=calc_metrics(confusion)
        loss_history['bag']['val'].append(LOSS['bag'])
        loss_history['inst']['val'].append(LOSS['inst'])
        error_history['bag']['val'].append(ERR['bag'])
        error_history['inst']['val'].append(ERR['inst'])

        # save
        lowest=save(run,lowest,epoch,net,optimizer,loss_history,error_history,confusion)
        
        # scheduler
        # scheduler.step()
    
    ## test
    load_model=glob.glob(f'checkpoints/run{run}/checkpoint*')[0]
    net.load_state_dict(torch.load(load_model))
    print(f'{load_model} is loaded!')
    
    LOSS, confusion,LAB,RAW, weights = val_test(net, dataloader_test, criterion,auc=True)
    ERR,AUC=calc_metrics(confusion,True,LAB,RAW)
    print(f'test error: {ERR}')
    print(f'test AUC: {AUC}')
    print(f'test confusion: {confusion}')


##
if __name__ == "__main__":
    
    params={
        # basics
        'epochs' : 400,

        # optimizer
        'learning_rate' : 1e-4/2, # 1e-4
        'weight_decay' : 5e-4, # 5e-4
 
        # model
        'model':'Instance_MIL', # Instance_MIL,Vanilla_MIL
        'attn_stru':'tanh', # tanh, sigmoid, none
        'c':1,

        # only applies to Instance_MIL
        ###################################################
        # 'reuse_weight':True,
        'reuse_weight':False,
        ###################################################
        # 'block_backprop':True,
        'block_backprop':False,

        'full_label_perc':0.
    }

    test1=deepcopy(params)
    test2=deepcopy(params);test2['full_label_perc']=0.2
    test3=deepcopy(params);test3['full_label_perc']=0.5
    # test4=deepcopy(params);test4['reuse_weight']=True;test4['block_backprop']=True
    
    ################################# change it accordingly
    param_list=[test1]
    # ,test2,test3]
    #################################
    for p in param_list:
        print(p)

    k=10 # k fold cross val
    for run,params in enumerate(param_list,start=15):
        for f in range(k):
            main(f'{str(run).zfill(2)}_{f}',params,k,f)
    
