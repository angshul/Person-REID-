# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
from model import DenseNet121, DenseNet121_classifier
from ResModel import Model
from sys import platform
from torch.utils.data.dataset import Dataset
from preactResnet import PreActResNet50
from ResNetmid import resnet50mid
######################################################################
# Options
######################################################################
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--test_dir',default='/scratch/group/atlas_prid/valset',type=str, help='specify test directory')
parser.add_argument('--name', default='denset121', type=str, help='save model path')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--model_type', type=str, default='densenet121', choices=['resnet50', 'densenet121','preActResnet50', 'resnet50mid'])
parser.add_argument('--workers', default=8, type=int, help='No of num_workers for dataloaders')
parser.add_argument('--last_conv_stride', default=1, type=int, help='stride of last conv layer')
parser.add_argument('--preLoad', action='store_true', help='fetch external weigths')
parser.add_argument('--external_test',action='store_true', help='Use different Val/test set')
parser.add_argument('--add_softmax_loss', default=True, type=bool, help='To use classifier block')
opt = parser.parse_args()

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])

########################################
# Base class to load test/validation
#
########################################
class MyCustomDataset(Dataset):
    def __init__(self, dataset, Class):
        self.class_label = dataset.class_to_idx[Class]
        dataloader = torch.utils.data.DataLoader(dataset,shuffle=False)
        self.class_list = [img[0] for img,label in dataloader if label == self.class_label ]
        self.len = len(self.class_list)

    def __getitem__(self, index):
        return (self.class_list[index], self.class_label)

    def __len__(self):
        return self.len

######################################################################
# Load Data
# ---------
# We will use torchvision and torch.utils.data packages for loading the
# data.
######################################################################
data_transforms = transforms.Compose([
        transforms.Resize((288,144), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = test_dir
if opt.external_test:
    total_set =  datasets.ImageFolder(data_dir,data_transforms)
    image_datasets = {}
    image_datasets['gallery'] = MyCustomDataset(total_set, 'gallery')
    image_datasets['query'] = MyCustomDataset(total_set, 'query')
else:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize, shuffle=False, num_workers=opt.workers) for x in ['gallery','query']}

#class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()

######################################################################
# Load model
######################################################################
def load_network(network):
    save_path = os.path.join('./model',name,'ckpt.pth')
    state_dict = torch.load(save_path)
    if opt.preLoad:
        for key in list(state_dict.keys()):
            list_key = list(key)
            list_key[:4] = list('model')
            state_dict[''.join(list_key)] = state_dict[key]
            del state_dict[key]
    network.load_state_dict(state_dict)
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
######################################################################
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        #print(count)
        if opt.model_type == 'densenet121':
            ff = torch.FloatTensor(n,1024).zero_()
        elif opt.model_type == 'resnet50':
            ff = torch.FloatTensor(n,2048).zero_()
        elif opt.model_type == 'resnet50mid':
            ff = torch.FloatTensor(n,3072).zero_()

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs = model(input_img)

            f = outputs.data.cpu()
            ff = ff+f
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff), 0)
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

def get_names(gallery_label, query_label, img_paths):
    gallery_names,query_names=[],[]

    if platform == 'win32' or platform == 'win64':
        # windows
        separator = '\\'
    else:
        #linux
        separator = '/'

    for path, v in img_paths:
        file_name = path.split(separator)[-1].replace('.png','')
        if v == gallery_label:
            gallery_names.append(file_name)
        elif v == query_label:
            query_names.append(file_name)
    return gallery_names, query_names

if opt.external_test:
    gallery_names, query_names = get_names(total_set.class_to_idx['gallery'],total_set.class_to_idx['query'], total_set.imgs)
else:
    gallery_path = image_datasets['gallery'].imgs
    query_path = image_datasets['query'].imgs
    gallery_cam,gallery_label = get_id(gallery_path)
    query_cam,query_label = get_id(query_path)


######################################################################
# Load Collected data Trained model
print('-------test-----------')

###########
# Models  #
###########
if opt.add_softmax_loss:
    model_structure = DenseNet121_classifier(751)
else:
    if opt.model_type == 'resnet50':
        model_structure = Model(last_conv_stride=opt.last_conv_stride)
    elif opt.model_type == 'densenet121':
        model_structure = DenseNet121()
    elif opt.model_type == 'preActResnet50':
        model_structure = PreActResNet50()
    elif opt.model_type == 'resnet50mid':
        model_structure = resnet50mid()

print('Model selected: ', opt.model_type)

model = load_network(model_structure)

# Remove the final fc layer and classifier layer
if not opt.add_softmax_loss:
    model.model.fc = nn.Sequential()
    model.classifier = nn.Sequential()

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
gallery_feature = extract_feature(model,dataloaders['gallery'])
query_feature = extract_feature(model,dataloaders['query'])


print('-----------------------------------Saving the extracted features-----------------------------------------------')
# Save to Matlab for check
if opt.external_test:
    gallery = {'features': gallery_feature.numpy(), 'names': gallery_names}
    query = {'features': query_feature.numpy(), 'names': query_names}
    feature_dir = os.path.join('./model',name)
    scipy.io.savemat(os.path.join('./model',name,'feature_val_gallery.mat'),gallery)
    scipy.io.savemat(os.path.join('./model',name,'feature_val_query.mat'),query)
    print('Features are stored at: ', str(feature_dir))
else:
    result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
    scipy.io.savemat('pytorch_result.mat',result)
