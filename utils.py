import numpy as np
import imageio
import matplotlib.pyplot as plt
import os
import shutil
from torch.utils.data import Dataset, DataLoader
import torch
import sklearn.metrics as metrics
import glob

def calculate_loss(net, batch, criterion, LOSS, confusion, auc=False,LAB=None, RAW=None):
 
    inputs, labels, full_labels, num, is_fully_labelled = batch

    # to cuda
    inpt = inputs.cuda()
    lab = labels.float().cuda()
    full_labs=full_labels.float().cuda()

    # forward
    outputs, weights, (L_bag,L_inst) = net(inpt, lab, criterion, is_fully_labelled, full_labs)
    loss = L_bag + L_inst

    # record losses
    LOSS['inst']+=L_inst.item()
    LOSS['bag']+=L_bag.item()
    
    # bag confusion matrix
    predicted=torch.sigmoid(outputs)>0.5
    confusion['bag']['TP']+=((predicted==1) & (lab==1)).sum().item()
    confusion['bag']['FP']+=((predicted==1) & (lab==0)).sum().item()
    confusion['bag']['TN']+=((predicted==0) & (lab==0)).sum().item()
    confusion['bag']['FN']+=((predicted==0) & (lab==1)).sum().item()
    
    # instance confusion matrix
    probs=weights[0] # 1,D
    predicted=probs>0.5
    confusion['inst']['TP']+=((predicted==1) & (full_labs==1)).sum().item()
    confusion['inst']['FP']+=((predicted==1) & (full_labs==0)).sum().item()
    confusion['inst']['TN']+=((predicted==0) & (full_labs==0)).sum().item()
    confusion['inst']['FN']+=((predicted==0) & (full_labs==1)).sum().item()
    
    # record labels and probs if requires AUC
    if auc:
        LAB['bag'].append(lab.item())
        LAB['inst'].append(full_labels.flatten())
        RAW['bag'].append(outputs.item())
        RAW['inst'].append(weights[1].flatten())
   
    return loss,weights,LOSS,confusion,LAB,RAW



def train(net,train_loader,criterion,optimizer):
    net.train()  # train
    confusion={'bag':{'TP':0,'FP':0,'TN':0,'FN':0}, 'inst':{'TP':0,'FP':0,'TN':0,'FN':0}}
    LOSS={'bag':0, 'inst':0}
    for i, batch in enumerate(train_loader, 1):
        
        loss,_,LOSS,confusion,_,_=calculate_loss(net, batch, criterion, LOSS, confusion)
        
        # backward and update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # calc mean loss
    LOSS['bag']/=i
    LOSS['inst']/=i
    
    return LOSS, confusion

@torch.no_grad()
def val_test(net,dataloader,criterion, auc=False):
    net.eval()
    confusion={'bag':{'TP':0,'FP':0,'TN':0,'FN':0}, 'inst':{'TP':0,'FP':0,'TN':0,'FN':0}}
    LAB={'bag':[],'inst':[]}
    RAW={'bag':[],'inst':[]}
    LOSS={'bag':0, 'inst':0}
    for i, batch in enumerate(dataloader, 1):
        loss,weights,LOSS,confusion,LAB,RAW=calculate_loss(net, batch, criterion, LOSS, confusion,auc,LAB,RAW)
    # calc mean loss
    LOSS['bag']/=i
    LOSS['inst']/=i
    return LOSS, confusion,LAB,RAW, weights

def save(run,lowest,epoch,net,optimizer,loss_history,error_history, confusion):
    # save checkpoint at lowest validation error + loss
    target=error_history['bag']['val'][-1]+loss_history['bag']['val'][-1]
    if target <= lowest:
        fn=f'checkpoints/run{run}/checkpoint_{str(epoch).zfill(3)}.pt'
        torch.save(net.state_dict(), fn)
        lowest = target
        for f in glob.glob(f'checkpoints/run{run}/checkpoint*'): # remove the previous checkpoint
            if f!=fn:
                os.remove(f)

    # save loss and error history
    torch.save(loss_history, f'checkpoints/run{run}/loss_history.pt')
    torch.save(error_history, f'checkpoints/run{run}/error_history.pt')
    return lowest


def get_patch_coord(xy):
    x, y = xy
    x_id = int(x)
    y_id = int(y)
    if x_id - 14 < 0:
        xmin = 0
        xmax = xmin + 27
    elif x_id + 13 > 499:
        xmax = 499
        xmin = xmax - 27
    else:
        xmin = x_id - 14
        xmax = x_id + 13

    if y_id - 14 < 0:
        ymin = 0
        ymax = ymin + 27
    elif y_id + 13 > 499:
        ymax = 499
        ymin = ymax - 27
    else:
        ymin = y_id - 14
        ymax = y_id + 13
    return ymin, ymax, xmin, xmax

class ColonCancerDataset(Dataset):
    def __init__(self, root, transform=None,train=True,k=5,fold=0, full_label_perc=0):
        self.transform=transform
        self.num=[]
        self.labels_inst=[]
        self.labels_bag=[]
        self.patches=[]
        self.full_labelled_id=set()
        ## train/test split
        np.random.seed(1)
        shuffled_ids = np.random.permutation(100)
        all_folds=np.array_split(shuffled_ids, k)
        popped=all_folds.pop(fold)
        if train:
            ids=np.concatenate(all_folds)
        else:
            ids=popped
        for num in ids:
            self.num.append(num)
            if np.random.rand()<full_label_perc:
                self.full_labelled_id.add(num)
            path=os.path.join(root, f'img{num}')
            self.labels_inst.append(np.load(os.path.join(path,'labels.npy')))
            self.labels_bag.append(int(self.labels_inst[-1].any()))
            patches=np.load(os.path.join(path,'patches.npy')).astype(np.float32)
            patches[..., 0] -= 123.68
            patches[..., 1] -= 116.779
            patches[..., 2] -= 103.939
            patches /= 255
            self.patches.append(patches)
    def __len__(self):
        return len(self.num)
    def __getitem__(self, ind):
        data = self.patches[ind]
        if self.transform:
            data=self.transform(images=data)
        data=torch.moveaxis(torch.tensor(data), -1, -3).contiguous() #ToTensor
        labels_bag=self.labels_bag[ind]
        labels_inst=self.labels_inst[ind]
        num=self.num[ind]
        is_fully_labelled=num in self.full_labelled_id
        return data,labels_bag,labels_inst,num,is_fully_labelled

def viz(root, num, prob=None, prob_ref=None):
    img = np.asarray(imageio.imread(os.path.join(root, f'img{num}', f'img{num}.bmp')), dtype=np.uint8)
    labels = np.load(os.path.join(root, f'img{num}', 'labels.npy'))
    xys = np.load(os.path.join(root, f'img{num}', 'xys.npy'))

    all_nuclei = np.zeros_like(img, dtype=np.uint8)
    ind = labels.argsort()
    for i in ind:
        ymin, ymax, xmin, xmax = get_patch_coord(xys[i])
        all_nuclei[ymin:ymax, xmin:xmax, :] = img[ymin:ymax, xmin:xmax, :]

    positives = np.zeros_like(img, dtype=np.uint8)
    for i in range(labels.sum()):
        ymin, ymax, xmin, xmax = get_patch_coord(xys[i])
        positives[ymin:ymax, xmin:xmax, :] = img[ymin:ymax, xmin:xmax, :]

    if prob is not None:
        heatmap = np.zeros_like(img, dtype=np.uint8)
        ind = prob.argsort()
        for i in ind:
            ymin, ymax, xmin, xmax = get_patch_coord(xys[i])
            heatmap[ymin:ymax, xmin:xmax, :] = img[ymin:ymax, xmin:xmax, :] * prob[i]
        
        heatmap_ref = np.zeros_like(img, dtype=np.uint8)
        ind_ref = prob_ref.argsort()
        for i in ind_ref:
            ymin, ymax, xmin, xmax = get_patch_coord(xys[i])
            heatmap_ref[ymin:ymax, xmin:xmax, :] = img[ymin:ymax, xmin:xmax, :] * prob_ref[i]

    if prob is not None:
        figure, ax = plt.subplots(1, 5, figsize=(16, 8))
    else:
        figure, ax = plt.subplots(1, 3, figsize=(16, 8))
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[0].set_title('Image')
    ax[1].imshow(all_nuclei)
    ax[1].axis('off')
    ax[1].set_title('All nuclei')
    ax[2].imshow(positives)
    ax[2].axis('off')
    ax[2].set_title('Positive instances')
    if prob is not None:
        ax[3].imshow(heatmap)
        ax[3].axis('off')
        ax[3].set_title('prediction')
        ax[4].imshow(heatmap_ref)
        ax[4].axis('off')
        ax[4].set_title('reference')
    plt.show()

def calc_metrics(confusion,auc=False,LAB=None,RAW=None):
    ERR=dict()
    AUC=dict()

    # errors
    conf_bag=confusion['bag']
    TP, FP, TN, FN = conf_bag['TP'], conf_bag['FP'], conf_bag['TN'], conf_bag['FN']
    ERR['bag']=(FN+FP)/(TN+FP+TP+FN)

    conf_inst=confusion['inst']
    TP, FP, TN, FN = conf_inst['TP'], conf_inst['FP'], conf_inst['TN'], conf_inst['FN']
    ERR['inst']=(FN+FP)/(TN+FP+TP+FN)

    if auc:
        LAB['inst']=torch.cat(LAB['inst']).numpy()
        RAW['inst'] = torch.cat(RAW['inst']).cpu().numpy()

        AUC['bag'] = metrics.roc_auc_score(LAB['bag'], RAW['bag'])
        AUC['inst'] = metrics.roc_auc_score(LAB['inst'], RAW['inst'])
    return ERR, AUC

if __name__=='__main__':
    root='newdata'
    num=79

    pred=np.load(os.path.join(root,f'img{num}','labels.npy'))*0.8
    viz(root, num, pred)
