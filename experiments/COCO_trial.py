import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

import time
import copy
import numpy as np

import matplotlib.pyplot as plt

# ======================================================================
# == Check GPU is connected
# ======================================================================

print("======================")
print("Check GPU is info")
print("======================")
print("How many GPUs are there? Answer:",torch.cuda.device_count())
print("The Current GPU:",torch.cuda.current_device())
print("The Name Of The Current GPU",torch.cuda.get_device_name(torch.cuda.current_device()))
# Is PyTorch using a GPU?
print("Is Pytorch using GPU? Answer:",torch.cuda.is_available())
print("======================")

# switch to False to use CPU
use_cuda = True

use_cuda = use_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu");

# =====================================================
# == Set random seeds
# =====================================================
def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

# =====================================================
# == Load and normalize CIFAR100
# =====================================================
def prepare_dataloader(num_workers=8, train_batch_size=128, eval_batch_size=256):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_set = torchvision.datasets.CIFAR100(root="../datasets/CIFAR100", train=True, download=True, transform=train_transform) 
    # We will use test set for validation and test in this project.
    # Do not use test set for validation in practice!
    test_set = torchvision.datasets.CIFAR100(root="../datasets/CIFAR100", train=False, download=True, transform=test_transform)

    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=train_batch_size,
        sampler=train_sampler, num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=eval_batch_size,
        sampler=test_sampler, num_workers=num_workers)


    return train_set, test_set, train_loader, test_loader




# =====================================================
# == Main
# =====================================================
random_seed = 0
num_classes = 10
cuda_device = torch.device("cuda:0")
cpu_device = torch.device("cpu:0")

model_dir = "models/trained_models/CIFAR10"
model_filename = "vgg11_cifar10.pt"
model_filepath = os.path.join(model_dir, model_filename)

set_random_seeds(random_seed=random_seed)

train_set, test_set, train_loader, test_loader = prepare_dataloader(num_workers=8, train_batch_size=128, eval_batch_size=256)
    
# for image,label,super_label in test_set:
#     print("Image shape: ",image.shape)
#     print("Image tensor: ", image)
#     print("Label: ", label)
#     print("Super Label: ", super_label)

#     image = image / 2 + 0.5     # unnormalize
#     npimg = image.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.savefig("foo.png")

#     # plt.plot(image)
#     # plt.savefig('foo.png')

#     # break
#     input("Press enter for next one ...")


# class Dictlist(dict):
#     def __setitem__(self, key, value):
#         try:
#             self[key]
#         except KeyError:
#             super(Dictlist, self).__setitem__(key, [])
#         self[key].append(value)

# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict
# x=unpickle('../datasets/CIFAR100/cifar-100-python/test')
# fine_to_coarse=Dictlist()

# for i in range(0,len(x[b'coarse_labels'])):
#     fine_to_coarse[x[b'coarse_labels'][i]]=x[ b'fine_labels'][i]
# d=dict(fine_to_coarse)
# for i in d.keys():
#     d[i]=list(dict.fromkeys(d[i]))

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, 'rb') as fo:
        myDict = pickle.load(fo, encoding='latin1')
    return myDict

trainData = unpickle('../datasets/CIFAR100/cifar-100-python/train')
testData = unpickle('../datasets/CIFAR100/cifar-100-python/test')
metaData = unpickle('../datasets/CIFAR100/cifar-100-python/meta')

#type of items in each file
for item in trainData:
    print(item, type(trainData[item]))

print(np.unique(trainData['fine_labels']))
print(np.unique(trainData['coarse_labels']))

#metaData
print("Fine labels:", metaData['fine_label_names'], "\n")
print("Coarse labels:", metaData['coarse_label_names'])


#storing coarse labels along with its number code in a dataframe
category = pd.DataFrame(metaData['coarse_label_names'], columns=['SuperClass'])
#storing fine labels along with its number code in a dataframe
subCategory = pd.DataFrame(metaData['fine_label_names'], columns=['SubClass'])
print(category)
print(subCategory)


# Loop over test data for cifar 100
for image,label in test_set:
    print("====")
    # Display label numerical
    print(label)

    # Display label categorical
    print(subCategory.iloc[label,0])

    # Use the line below to locate the numerical label of subclass labels you want
    # print(subCategory.loc[subCategory['SubClass'] == 'baby'].index[0])
    # print(subCategory.loc[subCategory['SubClass'] == 'boy'].index[0])
    # print(subCategory.loc[subCategory['SubClass'] == 'girl'].index[0])
    # print(subCategory.loc[subCategory['SubClass'] == 'man'].index[0])
    # print(subCategory.loc[subCategory['SubClass'] == 'woman'].index[0])

    # Check if it is for a human
    if any([label == 2, label == 11, label == 35, label == 46, label == 98]):
        print(label)
        # Display image
        image = image / 2 + 0.5     
        # unnormalize
        npimg = image.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.savefig("foo.png")
        input("Found an image of a human. Press enter to proceed for next image.")
    
    
    # break


   
