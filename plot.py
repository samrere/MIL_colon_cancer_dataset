from utils import *
from models import *
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
def calc_mean_stats(run):
    roots=glob.glob(f'checkpoints/run{run}*')
    ERR_avg={'bag': 0, 'inst': 0}
    AUC_avg={'bag': 0, 'inst': 0}
    confusion_avg={'bag': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}, 'inst': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}}
    print_params=False
    for fn in roots:
        params=np.load(f'{fn}/params.npy',allow_pickle=True).item()
        if not print_params:
            print(f'parameters: {params}')
            print_params=True
        model=params['model']
        attn_stru=params['attn_stru']
        c=params['c']
        reuse_weight=params['reuse_weight']
        block_backprop=params['block_backprop']
        epochs=params['epochs']
        learning_rate=params['learning_rate']
        weight_decay=params['weight_decay']

        if model == 'Vanilla_MIL':
            net = Vanilla_MIL(attn_stru).cuda()
        elif model == 'Instance_MIL':
            net = Instance_MIL(attn_stru, reuse_weight, block_backprop,c).cuda()
        load_model=glob.glob(f'{fn}/checkpoint*')[0]
        net.load_state_dict(torch.load(load_model))
        print(f'{load_model} is loaded!')

        dataset_test=ColonCancerDataset('newdata',transform=None,train=False)
        dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
        criterion = nn.BCEWithLogitsLoss(reduction='none')

        LOSS, confusion,LAB,RAW, weights = val_test(net, dataloader_test, criterion,auc=True)
        ERR,AUC=calc_metrics(confusion,True,LAB,RAW)

        ERR_avg['bag']+=ERR['bag']
        ERR_avg['inst']+=ERR['inst']

        AUC_avg['bag']+=AUC['bag']
        AUC_avg['inst']+=AUC['inst']

        for D in ['bag','inst']:
            confusion_avg[D]['TP']+=confusion[D]['TP']
            confusion_avg[D]['FP']+=confusion[D]['FP']
            confusion_avg[D]['TN']+=confusion[D]['TN']
            confusion_avg[D]['FN']+=confusion[D]['FN']
    
    ERR_avg['bag']=round(ERR_avg['bag']/len(roots),4)
    ERR_avg['inst']=round(ERR_avg['inst']/len(roots),4)

    AUC_avg['bag']=round(AUC_avg['bag']/len(roots),4)
    AUC_avg['inst']=round(AUC_avg['inst']/len(roots),4)

    for D in ['bag','inst']:
        confusion_avg[D]['TP']=round(confusion_avg[D]['TP']/len(roots),2)
        confusion_avg[D]['FP']=round(confusion_avg[D]['FP']/len(roots),2)
        confusion_avg[D]['TN']=round(confusion_avg[D]['TN']/len(roots),2)
        confusion_avg[D]['FN']=round(confusion_avg[D]['FN']/len(roots),2)
    
    
    print(f'mean test error: {ERR_avg}')
    print(f'mean test AUC: {AUC_avg}')
    print(f'mean test confusion: {confusion_avg}')

    
def load_ref_model(run, ref):
    fold=run.split('_')[-1]
    run=f'{ref}_{fold}'
    params=np.load(f'checkpoints/run{run}/params.npy',allow_pickle=True).item()
    model=params['model']
    attn_stru=params['attn_stru']
    c=params['c']
    reuse_weight=params['reuse_weight']
    block_backprop=params['block_backprop']
    epochs=params['epochs']
    learning_rate=params['learning_rate']
    weight_decay=params['weight_decay']
    if model == 'Vanilla_MIL':
        net = Vanilla_MIL(attn_stru).cuda()
    elif model == 'Instance_MIL':
        net = Instance_MIL(attn_stru, reuse_weight, block_backprop,c).cuda()
    load_model=glob.glob(f'checkpoints/run{run}/checkpoint*')[0]
    net.load_state_dict(torch.load(load_model))
    print(f'reference model {load_model} is loaded!\n')
    return net
    
    
    
def plot(run, plot_attn, k, ref=None):
    
    ref=load_ref_model(run,ref)

    params=np.load(f'checkpoints/run{run}/params.npy',allow_pickle=True).item()
    print(f'run{run}\n')
    for i,v in params.items():
        print(f'{i}={v}')

    model=params['model']
    attn_stru=params['attn_stru']
    c=params['c']
    reuse_weight=params['reuse_weight']
    block_backprop=params['block_backprop']

    if model == 'Vanilla_MIL':
        net = Vanilla_MIL(attn_stru).cuda()
    elif model == 'Instance_MIL':
        net = Instance_MIL(attn_stru, reuse_weight, block_backprop,c).cuda()
    load_model=glob.glob(f'checkpoints/run{run}/checkpoint*')[0]
    net.load_state_dict(torch.load(load_model))
    print(f'\n{load_model} is loaded!\n')

    dataset_test=ColonCancerDataset('newdata',transform=None,train=False,k=k,fold=int(run.split('_')[-1]))
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    LOSS, confusion,LAB,RAW, weights = val_test(net, dataloader_test, criterion,auc=True)
    ERR,AUC=calc_metrics(confusion,True,LAB,RAW)
    print(f'test error: {ERR}')
    print(f'test AUC: {AUC}')
    print(f'test confusion: {confusion}')
    
    
    
    loss_history = torch.load(f'checkpoints/run{run}/loss_history.pt')
    loss_bag_train=np.array(loss_history['bag']['train'])
    loss_bag_val=np.array(loss_history['bag']['val'])
    loss_inst_train=np.array(loss_history['inst']['train'])
    loss_inst_val=np.array(loss_history['inst']['val'])
    plt.plot(loss_bag_train+loss_inst_train, 'r', label='train')
    plt.plot(loss_bag_val+loss_inst_val, 'b', label='val_total')
    plt.plot(loss_bag_val, 'g', label='val_bag_loss')
    plt.plot(loss_inst_val, 'y', label='val_inst_loss')
    plt.ylim([0,1.6])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss history')
    plt.legend()
    plt.show()
    
    error_history = torch.load(f'checkpoints/run{run}/error_history.pt')
    plt.plot(np.array(error_history['bag']['train'])*100, 'r', label='train')
    plt.plot(np.array(error_history['bag']['val'])*100, 'b', label='val')
    plt.ylim([0,40])
    plt.xlabel('epoch')
    plt.ylabel('error %')
    plt.title('bag error history')
    plt.legend()
    plt.show()
    
    error_history = torch.load(f'checkpoints/run{run}/error_history.pt')
    plt.plot(np.array(error_history['inst']['train'])*100, 'r', label='train')
    plt.plot(np.array(error_history['inst']['val'])*100, 'b', label='val')
    plt.ylim([0,45])
    plt.xlabel('epoch')
    plt.ylabel('error %')
    plt.title('instance error history')
    plt.legend()
    plt.show()
    
    if plot_attn:
        with torch.no_grad():
            net.eval()
            for inputs,labels,full_labels, ns, _ in dataloader_test:
                print(f'ns: {ns.item()}')
                inpt = inputs.cuda()
                lab = labels.float().cuda()

                # forward
                outputs, (prob,raw,A), _ = net(inpt)
                _,(prob_ref,_,_),_ = ref(inpt)
                prob_ref = prob_ref[0].cpu().numpy()

                # prediction
                pred = (outputs>0).float()
                print('Output Raw:',outputs.item())
                print('Predicted Label:',pred.item())
                print('True Label:',lab.item())
                # print('Raw:\n',raw)

                plt.figure(figsize=(10,5))
                prob = prob[0].cpu().numpy()
                plt.plot(range(len(prob)), prob,'-o',label='instance probability')
                plt.plot(full_labels[0].numpy(), label='true instance label')
                if A is not None:
                    plt.plot(A[0].cpu().numpy(), label='attention weight')
                plt.xlim([0,len(prob)-1])
                plt.ylim([-0.01,1.01])
                plt.legend()
                plt.show()

                # A=A[0].cpu().numpy() # rescaled attention as prob, when using vanilla
                # if A.max()==A.min():
                #     prob=A/A.max()
                # else:
                #     prob=(A-A.min())/(A.max()-A.min())
                viz('newdata',ns.item(),prob,prob_ref)
