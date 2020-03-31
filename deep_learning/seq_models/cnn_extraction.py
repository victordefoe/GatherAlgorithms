
# this file is fo CNN training 
# which extract the feature of images from different sources
# And also save all the features
import torch
import torchvision
from torch.nn.modules import LSTM
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import os, json
from os.path import join as opj
import pandas as pd 
import numpy as np 
import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image, ImageFile
import random
import copy


class GE_mini(Dataset):
    def __init__(self, sourceA, sourceB, stage, pure_positive=False):
        ### Acquire the GoogleEarth Dataset ###
        # stage: 'train','val','test'
        # info : a list, info[0]:train_gt.json, info[1]:val_gt.json
        info={}
        info['train'] = opj('deep_learning/seq_models/seq_dis','train_gt.json')
        info['val'] = opj('deep_learning/seq_models/seq_dis','val_gt.json')
        info['test'] = opj('deep_learning/seq_models/seq_dis','test_gt.json')
        gts = json.load(open(info[stage]))
        
        self.A_imgs = [opj(sourceA, i) for i in gts.keys() if i[-3:]=='bmp']
        self.B_imgs = [opj(sourceB, i) for i in gts.values() if i[-3:]=='jpg']

        self.stage = stage
        self.pure_positive = pure_positive
        
        
    def __len__(self):
        return len(self.A_imgs)
    def __getitem__(self, item):
        prepro = []
        prepro.append(transforms.Resize(size=150))
        prepro.append(transforms.ToTensor())
        trans = transforms.Compose(prepro)

        # generate positive and negative img-pairs id
        if self.pure_positive or (self.stage in ['test']):
            pn = 1
        else:
            pn = random.randint(1,2)
        # positive condition
        if pn == 1:
            imgA = Image.open(self.A_imgs[item])
            imgAname = os.path.split(self.A_imgs[item])[1]

            imgB = Image.open(self.B_imgs[item])
            imgBname = os.path.split(self.B_imgs[item])[1]
            label = 1
        # negative condition
        else:
            imgA = Image.open(self.A_imgs[item])
            imgAname = os.path.split(self.A_imgs[item])[1]
            while True:
                choice = random.randint(0, len(self.B_imgs)-1)
                if choice != item:
                    break
            imgB = Image.open(self.B_imgs[choice])
            imgBname = os.path.split(self.B_imgs[choice])[1]
            label = 0

        img_A = trans(imgA)
        img_B = trans(imgB)
        label = torch.Tensor([label])
        imgs = torch.stack([img_A, img_B], dim=0)


        
        return imgs, label, (imgAname, imgBname)
        

        

## This is to handle the data
def data_handler(sourceA, sourceB, info, bs=1, pure_positive=False):
    # input: pathA, pathB, pathInfo(path of train and val set gt), batch_size
    # for data read in
    # A_imgs = [i for i in os.listdir(sourceA) if i[-3:]=='bmp']
    # B_imgs = [i for i in os.listdir(sourceB) if i[-3:]=='jpg']
    # print(B_imgs[-9:]) 
    
    if not os.path.exists('deep_learning/seq_models/seq_dis'):
        os.mkdir('deep_learning/seq_models/seq_dis')
        print('sss')
    # read the groundtruth into json if not initialized
    if not os.path.exists(opj('deep_learning/seq_models/seq_dis', 'train_gt.json')):
        print('Creating initial gts and save in seq_dis dir' )
        for stage in ['train', 'val']:
            gt = os.listdir(opj(info, stage))
            t_dict = dict()
            for each in tqdm.tqdm(gt):
                path = opj(opj(info, stage), each)
                temp = pd.read_csv(path, index_col=0)
                A_imgname = each[:-3] + 'bmp'
                temp.sort_values(by=['dists'], inplace=True)
                temp.reset_index(drop=True, inplace=True)
                # print(temp.head(10))
                B_imgname = temp.loc[0]['imgfname']
                # print(A_imgname)
                t_dict[A_imgname] = B_imgname
            with open(opj('deep_learning/seq_models/seq_dis', 
                        '{}_gt.json'.format(stage)), 'w') as otf:
                json.dump(t_dict, otf)
    # handle the test set
    if not os.path.exists(opj('deep_learning/seq_models/seq_dis', 'test_gt.json')):
        testpath = r'J:\research\datasets\GoogleEarth\collection_1\dists_forms'
        print('Creating initial test gts and save in seq_dis dir' )
        
        t_dict = dict()
        for stage in ['train', 'val']:
            gt = os.listdir(opj(testpath, stage))
            for each in tqdm.tqdm(gt):
                path = opj(opj(testpath, stage), each)
                temp = pd.read_csv(path, index_col=0)
                A_imgname = each[:-3] + 'bmp'
                temp.sort_values(by=['dists'], inplace=True)
                temp.reset_index(drop=True, inplace=True)
                # print(temp.head(10))
                B_imgname = temp.loc[0]['imgfname']
                # print(A_imgname)
                t_dict[A_imgname] = B_imgname
        with open(opj('deep_learning/seq_models/seq_dis', 
                        'test_gt.json'), 'w') as otf:
            json.dump(t_dict, otf)


    test_sourceA = r'J:\research\datasets\GoogleEarth\collection_1\patch'
    test_sourceB = r'J:\research\datasets\GoogleEarth\collection_1\seps\18'

    train_loader = DataLoader(dataset=GE_mini(sourceA, sourceB, 'train', pure_positive),
                            batch_size=bs,
                            shuffle=True,
                            num_workers=2)

    val_loader = DataLoader(dataset=GE_mini(sourceA, sourceB, 'val', pure_positive),
                            batch_size=bs,
                            shuffle=True,
                            num_workers=2)

    test_loader = DataLoader(dataset=GE_mini(test_sourceA, test_sourceB, 'test'),
                            batch_size=bs,
                            shuffle=True,
                            num_workers=2)


    dataloaders = {'train':train_loader, 'val':val_loader, 'test':test_loader}
    return dataloaders

    

class CNN_model(nn.Module):
    def __init__(self, pretrain_cnn):
        super(CNN_model, self).__init__()
        # use torchvision CNN as base CNN net
        base = torchvision.models.resnet18(pretrained=False)
        base.load_state_dict(torch.load(pretrain_cnn))
        cnn = {}
        exclude = ['avgpool', 'fc']
        for name, m in base.named_children():
            if name not in exclude:
                cnn[name] = copy.deepcopy(m)  # copy weights...
        cnn_list = []
        for each in cnn:
            cnn_list.append(cnn[each])
        self.cnn = nn.Sequential(*cnn_list)

        self.fc = nn.Sequential(nn.Linear(512 * 5 * 5, 256),
                                nn.PReLU(),
                                nn.Linear(256, 100)
                                )

    def fonce(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out
    
    def forward(self, x):
        # first dim is batch size
        # second dimension is to split the imgA and imgB
        oA = self.fonce(x[:,0])
        oB = self.fonce(x[:,1])

        # dim normalization
        oA /= oA.detach().pow(2).sum(1, keepdim=True).sqrt()
        oB /= oB.detach().pow(2).sum(1, keepdim=True).sqrt()
        
        # outs = torch.stack([oA, oB], dim=0)
        # outs = torch.cat((oA, oB),dim=1)
        # outs = outs.view(outs.size()[0], -1)
        
        # outs = self.fc(outs)
        # outs = torch.relu(outs)
        

        return oA, oB


class Myloss(nn.Module):

    def __init__(self):
        super(Myloss, self).__init__()
    
    def forward(self, predA, predB, label):
        eps = 1e-9
        
        distances = (predA - predB).pow(2).sum(1).t()
        margin = 0.5 #gap, negative sample punishment strength
        label = label.float().squeeze()
        cost = 0.5 * (label * distances +
            (1 + -1 * label) * F.relu(margin - (distances + eps).sqrt()).pow(2))

        return cost.mean()


def save_weights(model, save_path):
    torch.save(model.state_dict(), save_path)

def distingulish_ability_evaluation(model,data, num):
    # num: how many pairs you want to apply for this evaluation
    TN,TP,FN,FP = 0,0,0,0
    for index, item in enumerate(data):
        pA, pB = model(item[0])
        diss = (pA - pB).pow(2).sum(1).t()
        
        label = item[1].float().squeeze()
        thed = 0.5
        TN += ((diss > thed) * (1-label)).sum() # 实际为负，预测为负
        TP += ((diss < thed) * (label)).sum()  # 实际为正，预测为正
        FN += ((diss > thed) * (label)).sum() # 实际为正，预测为负
        FP += ((diss < thed) * (1-label)).sum() # 实际为负，预测为正


        if index > num:
            break
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    F1 = (2 * precision * recall) / (precision + recall)
    return round(F1.item(), 4)



if __name__ == '__main__':
    sourceA = r'J:\research\datasets\GoogleEarth\collection_10000\patch'
    sourceB = r'J:\research\datasets\GoogleEarth\collection_10000\seps\18'
    info = r'J:\research\datasets\GoogleEarth\collection_10000\dists_forms'
    save_path = opj('deep_learning/seq_models/seq_dis', 'weight.pth')
    pretrain_cnn = opj('deep_learning/seq_models/seq_dis', 'resnet18-5c106cde.pth')
    

    model = CNN_model(pretrain_cnn)
    # torchsummary.summary(model, (2,3,150,150), batch_size=4)
    # count the true whole paramters numer
    totalnum_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The true trainable parmeter number used is: {} '.format(totalnum_p))

    dataloaders = data_handler(sourceA, sourceB, info, bs=4)
    myloss = Myloss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001, weight_decay=0.01)
    
    ## train process
    if False:
        bestacc = 0
        epoches = 5
        try:
            model.load_state_dict(torch.load(save_path))
        except:
            print('cannot use saved model weights')
        for epoch in range(epoches):
            for index, item in enumerate(dataloaders['train']):
                # print(item[0][:,0].shape)
                predA, predB = model(item[0])
                label = item[1]
                
                loss = myloss(predA, predB, label)
                loss.backward()
                optimizer.step()
                print(round(loss.item(),3), end=' ')
                # if index % 10 == 0:
                #     print('loss:{}'.format(loss.item()))
                    # print(torch.cat((pred, label), dim=1).detach())
                
                if (index + 1) % 101 == 0:
                    

                    # evaluation
                    # for i, e_item in enumerate(dataloaders['val']):
                    #     e_predA, epredB = model(e_item[0])
                    #     e_label =e_item[1]
                    #     print(torch.cat((e_pred, e_label),dim=1).detach())
                        
                    #     break
                    # TODO: Change the juadgement
                    acc = distingulish_ability_evaluation(model, dataloaders['val'], 36)
                    print('\n epoch:{}, iter:{}, acc rate(F score) ---- {}'.format(epoch, index, acc))
                    if bestacc <= acc:
                        bestacc = acc
                        print('epoch:{}, iter:{}, acc(F):{}, best_acc:{}, saving...'.format(epoch,index,acc, bestacc))
                        save_weights(model, save_path)


    # use the settle-weighted network for feature extraction of test set
    
    # apply on test set process
    if True:
        A_path = opj('deep_learning/seq_models/seq_dis', 'featAs.pt')
        B_path = opj('deep_learning/seq_models/seq_dis', 'featBs.pt')
        model.eval()
        model.load_state_dict(torch.load(save_path))

        featA_bin, featB_bin = {}, {}
        # where to save the features extracted
        
        for index, item in tqdm.tqdm(enumerate(dataloaders['test'])):
            predA, predB = model(item[0])
            label = item[1]
            A_names, B_names = item[2][0], item[2][1]
            
            # save the features extracted
            for i in range(len(A_names)):
                featA_bin[A_names[i]] = predA[i].data # use .data to prevent stack overflow
                featB_bin[B_names[i]] = predB[i].data
            
        torch.save(featA_bin, A_path)
        torch.save(featB_bin, B_path)
    # keep the training set
    if False:
        A_path = opj('deep_learning/seq_models/seq_dis', 'train_featAs.pt')
        B_path = opj('deep_learning/seq_models/seq_dis', 'train_featBs.pt')
        model.eval()
        model.load_state_dict(torch.load(save_path))
        dloaders = data_handler(sourceA, sourceB, info, bs=4, pure_positive=True)
        featA_bin, featB_bin = {}, {}
        # where to save the features extracted
        for dset in [dloaders['train'], dloaders['val']]:
            for index, item in tqdm.tqdm(enumerate(dset)):
                predA, predB = model(item[0])
                label = item[1]
                A_names, B_names = item[2][0], item[2][1]
                
                # save the features extracted
                for i in range(len(A_names)):
                    featA_bin[A_names[i]] = predA[i].data # use .data to prevent stack overflow
                    featB_bin[B_names[i]] = predB[i].data
                
        torch.save(featA_bin, A_path)
        torch.save(featB_bin, B_path)



    

    
    
        
 
            




        
        
