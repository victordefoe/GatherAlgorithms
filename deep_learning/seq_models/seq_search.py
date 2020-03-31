

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
from collections import OrderedDict

import cnn_extraction as Extraction

class CNN_Layer(nn.Module):
    def __init__(self, inc, outc, 
                kernel, stride=1, padding=0):
        super(CNN_Layer, self).__init__()
        cnn = []
        cnn.append(nn.Conv2d(inc, outc, kernel, stride, padding))
        cnn.append(nn.BatchNorm2d(outc))
        cnn.append(nn.ReLU())
        CNN_BN_ReLu = nn.Sequential(*cnn)

        self.cnn_layer = CNN_BN_ReLu

    def forward(self, x):
        return self.cnn_layer(x)

base_root = r'deep_learning/seq_models/seq_dis'
datasource = r'J:\research\datasets\GoogleEarth\collection_1'
train_datasource = r'J:\research\datasets\GoogleEarth\collection_10000'

def nnb_similarity(vecA, matB):
    # vector A search the nearest neighbor in matB (Tensor)
    # It can be remodeled by machine learning methods
    if len(matB.shape)==1:
        matB = matB.unsqueeze(0)
    eu_distances = (vecA - matB).pow(2).sum(1)
    minimum = torch.min(eu_distances)
    argmin = torch.argmin(eu_distances)
    return argmin, minimum

def orginal_search_evaluation(dataset='850'):
    if dataset == '10000':
        A_feats = OrderedDict(torch.load(opj(base_root, 'train_featAs.pt')))
        B_feats = OrderedDict(torch.load(opj(base_root, 'train_featBs.pt')))
        gts = json.load(open(opj(base_root,'train_gt.json')))
        gts.update(json.load(open(opj(base_root,'val_gt.json'))))
    else:
        A_feats = OrderedDict(torch.load(opj(base_root, 'featAs.pt')))
        B_feats = OrderedDict(torch.load(opj(base_root, 'featBs.pt')))
        gts = json.load(open(opj(base_root,'test_gt.json')))
    
    B_values = torch.stack(list(B_feats.values()))
    B_keys = list(B_feats.keys())
    acc = 0
    for A_key in A_feats:
        nearest_index, nearest_d = nnb_similarity(A_feats[A_key], B_values)
        result = B_keys[nearest_index]
        _, true_eu_d = nnb_similarity(A_feats[A_key], B_feats[gts[A_key]])
        
        if result == gts[A_key]:
            acc += 1
            # print(result)

    print('Correct:{}, Total:{}, acc:{}%'.format(acc,
                    len(A_feats), round(100 * acc/len(A_feats), 2)))
    
class SearchNet(nn.Module):
    def __init__(self):
        super(SearchNet, self).__init__()
        
        self.cnn_lib_1 = CNN_Layer(100, 100, 1, 1, 0)
        self.cnn_lib_2 = CNN_Layer(100, 50, 1, 1, 0)
        self.cnn_q_1 = CNN_Layer(100, 50, 1)

        self.merge = CNN_Layer(50,1, 1)

    def forward(self, query, library):
        library = self.cnn_lib_1(library)
        library = self.cnn_lib_2(library)

        query = self.cnn_q_1(query)
        merge = query + library
        out = self.merge(merge)

        return out

class SearchData(Dataset):
    def __init__(self, stage='test'):
        self.stage = stage
        
        # 需要注意的是，在gts中，存在多个A对应同一个B的情况存在，不存在一个A对应多个B的情况
        if stage in ['train', 'val']:
            pathA = opj(base_root, 'train_featAs.pt')
            pathB = opj(base_root, 'train_featBs.pt')
            gts = json.load(open(opj(base_root,'train_gt.json')))
            gts.update(json.load(open(opj(base_root,'val_gt.json'))))
        elif stage in ['test']:
            pathA = opj(base_root, 'featAs.pt')
            pathB = opj(base_root, 'featBs.pt')
            gts = json.load(open(opj(base_root,'test_gt.json')))
        
        A_feats = OrderedDict(torch.load(pathA))
        A_keys = list(A_feats.keys())
        self.A_feats = A_feats
        self.A_keys = list(A_keys)
        B_feats = OrderedDict(torch.load(pathB))
        self.B_feats = B_feats
        self.gts = gts
        
        # if stage in ['test']
        B_values = torch.stack(list(self.B_feats.values()))
        B_keys = list(B_feats.keys())
        self.B_values = B_values
        self.B_keys = B_keys
        
        self.data_label_train = opj(base_root, 'data_labels', 'train_dist_forms')
        self.region_range = 300 # meters

    def __len__(self):
        return len(self.A_feats)

    def __getitem__(self, item):
        if self.stage in ['test']:
            B_feat = self.B_values
            length = B_feat.shape[0]
            B_feat = B_feat.t().unsqueeze(1)
            
            Akeyname = self.A_keys[item]
            A_feat = self.A_feats[Akeyname].unsqueeze(0)
            A_feat = A_feat.t().unsqueeze(1)
            
            index = self.B_keys.index(self.gts[Akeyname])
            label = index
            # label = torch.zeros(length)
            # label[index] = 1.0
            # label = label.unsqueeze(0).unsqueeze(1)
            gt_name = (Akeyname, self.gts[Akeyname])
            
        elif self.stage in ['train', 'val']:
            Akeyname = self.A_keys[item]
            A_feat = self.A_feats[Akeyname].unsqueeze(0)
            A_feat = A_feat.t().unsqueeze(1)
            
            # find out the proper region features
            A_form_name = os.path.splitext(Akeyname)[0] + '.csv'
            A_form_name = opj(self.data_label_train, A_form_name)
            temp = pd.read_csv(A_form_name, index_col=0)
            temp.sort_values(by=['dists'], inplace=True)
            temp.reset_index(drop=True, inplace=True)
            B_names = list(temp[:100]['imgfname'])
            random.shuffle(B_names)
            B_feats = []
            for B_name in B_names:
                B_feats.append(self.B_feats[B_name])
            B_feat = torch.stack(B_feats)
            B_feat = B_feat.t().unsqueeze(1)
            
            index = B_names.index(self.gts[Akeyname])
            label = index
            
            gt_name = (Akeyname, self.gts[Akeyname])
            # print(A_feat.shape, B_feat.shape, index)

        return A_feat, B_feat, label, gt_name

def dataset_creation(bs=4):
    # dist_forms = r'J:\research\datasets\GoogleEarth\collection_1\dists_forms'
    # forms = []
    # for source in [opj(dist_forms, 'train'), opj(dist_forms, 'val')]:
    #     for each in os.listdir(source):
    #         forms.append(opj(source, each))
    
    test_loader = DataLoader(dataset=SearchData('test'),
                            batch_size=bs,
                            shuffle=True,
                            num_workers=2 )
    train_loader = DataLoader(dataset=SearchData('train'),
                            batch_size=bs,
                            shuffle=True,
                            num_workers=2 )
    dataloaders = {'test':test_loader, 'train':train_loader}

    return dataloaders

class SearchLoss(nn.Module):
    def __init__(self):
        super(SearchLoss, self).__init__()

    def forward(self, x, label):
        # label shape [batch_size, 1]
        x = x.squeeze()
        
        cost = F.cross_entropy(x, label, weight=None,
                               ignore_index=-100, reduction='mean')
        return cost

if __name__ == '__main__':

    model = SearchNet()
    lossfunc = SearchLoss()

    dataloaders = dataset_creation(4)
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01, weight_decay=0.01)
    for index, item in enumerate(dataloaders['train']):
        A_feat, Lib_feats, label = item[0], item[1], item[2]
        gt_names = item[3] 
        pred = model(A_feat, Lib_feats)
        
        loss = lossfunc(pred, label)
        loss.backward()
        optimizer.step()
        # print(pred.shape)
        if index % 10  == 0:
            print(round(loss.item(),3), end=' ')
        
        # if index % 101 ==0:
        #     for e_index, test_s in enumerate(dataloaders['test']):
        #         test_pred = model(test_s[0], test_s[1])
        #         test_label = test_s[2]
        #         pred = torch.argmax(test_pred.squeeze())
                
                
                
                
        
    
