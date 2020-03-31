
import os
from os.path import join as opj
import pandas as pd

source_path = './competition/eleme/'

def changetxt2csv(data_path):
    for root, dirs, files in os.walk(data_path, topdown=False):
        for name in files:
            if name[-3:] == 'txt':
                os.rename(opj(root, name), opj(root, name[:-3]+'csv'))

def datapathes():
    source = opj(source_path, 'data')
    base = {'train':opj(source, 'r1_train'), 
            'testA':opj(source, 'r1_test_A'), 
            'submit':opj(source, 'r1_train')}
    pathes = {}
    for each in ['train','testA']:
        name_dict = {}
        for name in os.listdir(base['train']):
            name_dict[name] = opj(base[each], name)
        pathes[each] = name_dict
    pathes['submit'] = base['submit']
    return pathes
    

dps = datapathes()

actionpath = [opj(dps['train']['action'], x) for x in os.listdir(dps['train']['action'])]
df = pd.read_csv(actionpath[0])
print(df.columns)    
# f = pd.read_csv(p)
# changetxt2csv(data_path)