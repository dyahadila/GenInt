import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from tqdm import tqdm

import utils

img_shape = (3, 32, 32)

# Parameters
image_size = 32
label_dim = 10

learning_rate = 0.00002
betas = (0.5, 0.999)
batch_size = 2048
num_epochs = 20

BATCH_SIZE = 64

cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1,1024 )
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
 

def fit(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters())#,lr=0.001, betas=(0.9,0.999))
    error = nn.CrossEntropyLoss()
    EPOCHS = 5
    model.train()
    for epoch in tqdm(range(EPOCHS)):
        correct = 0
        for batch_idx, (imgs, labels) in enumerate(train_loader):
    
            var_X_batch = Variable(imgs.type(FloatTensor))
            var_y_batch = Variable(labels.type(LongTensor))
            optimizer.zero_grad()
            output = F.log_softmax(model(var_X_batch), dim=1)
            loss = error(output, var_y_batch)
            loss.backward()
            optimizer.step()

            # Total correct predictions
            predicted = torch.max(output.data, 1)[1] 
            correct += (predicted == var_y_batch).sum()
            #print(correct)
            if batch_idx % 20 == 0:
                print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                            epoch, 
                            batch_idx*len(imgs), 
                            len(train_loader.dataset), 
                            100.*batch_idx / len(train_loader), 
                            loss.item(), 
                            float(correct*100) / float(BATCH_SIZE*(batch_idx+1))
                        )
                     )
                
def fit_augment(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters())#,lr=0.001, betas=(0.9,0.999))
    error = nn.CrossEntropyLoss()
    EPOCHS = 5
    model.train()
    for epoch in range(EPOCHS):
        correct = 0
        for batch_idx, (data1, data2) in enumerate(train_loader):
            
            imgs = torch.cat((data1[0],data2[0]))
            labels = torch.cat((data1[1],data2[1]))            
            
            var_X_batch = Variable(imgs.type(FloatTensor))
            var_y_batch = Variable(labels.type(LongTensor))
            optimizer.zero_grad()
            output = model(var_X_batch)
            loss = error(output, var_y_batch)
            loss.backward()
            optimizer.step()

            # Total correct predictions
            predicted = torch.max(output.data, 1)[1] 
            correct += (predicted == var_y_batch).sum()
            if batch_idx % 200 == 0:
                print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                            epoch, 
                            batch_idx*len(imgs)/2, 
                            len(train_loader.dataset), 
                            100.*batch_idx / len(train_loader), 
                            loss.item(), 
                            float(correct*100) / float(BATCH_SIZE*(batch_idx+1))
                        )
                     )
                
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res
                
def evaluate(model, test_loader):
    model.eval()
    correct = 0 
    top1 = 0
    # top5 = 0
    outputs = []
    with torch.no_grad():
        for batch_idx, (test_imgs, test_labels) in enumerate(test_loader):
            test_imgs = Variable(test_imgs.type(FloatTensor))
            test_labels = Variable(test_labels.type(LongTensor))
            output = model(test_imgs)
            softmax_output = F.log_softmax(output, dim=1)
            outputs.extend(torch.max(output, dim=1)[0].cpu().numpy())
            acc1 = accuracy(softmax_output, test_labels)
            top1 += acc1[0]
    print("In distribution test accuracy top1:{:.3f}% ".format( float(top1*100) / (len(test_loader)*BATCH_SIZE)))
    return np.asarray(outputs), test_labels

def evaluate_ood(model, test_loader):
    model.eval()
    correct = 0
    top1 = 0
    threshold = -math.inf
    outputs = []
    with torch.no_grad():
        for batch_idx, (test_imgs, test_labels) in enumerate(test_loader):
            test_imgs = Variable(test_imgs.type(FloatTensor))
            test_labels = Variable(test_labels.type(LongTensor))
            output = model(test_imgs)
            softmax_output = F.log_softmax(output, dim=1)
            outputs.extend(torch.max(output, dim=1)[0].cpu().numpy())
            # out_list = output.tolist()
            # labels_list = test_labels.tolist()
            # output_filtered = []
            # labels_filtered = []
            # count = 0
            # for x in out_list:
            #     if x[0] > threshold and x[1] > threshold:
            #         output_filtered.append(x)
            #         labels_filtered.append(labels_list[count])
            #     count += 1
            # #print(output_filtered)
            # #print(labels_filtered)
            # if len(output_filtered) == 0:
            #     continue
            # out_filtered_tensor = torch.FloatTensor(output_filtered)
            # labels_filtered_tensor = torch.FloatTensor(labels_filtered)
    #         predicted = torch.max(output,1)[1]
    #         correct += (predicted == test_labels).sum()
            acc1 = accuracy(softmax_output, test_labels)
            # print(acc1)
            top1 += acc1[0]
    # print("Test accuracy top1:{:.3f}% ".format( float(top1*100) / (len(test_loader)*BATCH_SIZE)))
    return np.asarray(outputs),test_labels

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class RationDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets, ratio=0.5):
        # self.datasets = datasets
        self.ratio = ratio #
        self.original_dataset = datasets[0]
        self.intervened_dataset = datasets[1]

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)
        



color_mnist_test_indist = '/nobackup/dyah_roopa/VAE_ColorMNIST_original/color_MNIST_1/test_0.25/in_dist/'
color_mnist_test_ood = '/nobackup/dyah_roopa/VAE_ColorMNIST_original/color_MNIST_1/test_0.25/ood/'
color_mnist_test_spurious_ood = '/nobackup/dyah_roopa/temp/Spurious_OOD/datasets/ood_datasets/partial_color_mnist_0&1/'

# color_mnist_confound_test = '/nobackup/dyah_roopa/color_MNIST/confound_test/'
color_mnist_train = '/nobackup/dyah_roopa/VAE_ColorMNIST_original/color_MNIST_1/train_0.25/'
color_mnist_train_intervened = '/nobackup/dyah_roopa/VAE_ColorMNIST_original/color_MNIST_1/intervened_train_0.25/'


composed_transforms = transforms.Compose([
                        transforms.Resize(32), 
                        transforms.ToTensor(), 
                    ])

test_set_indist = datasets.ImageFolder(color_mnist_test_indist, composed_transforms)
test_set_ood = datasets.ImageFolder(color_mnist_test_ood, composed_transforms)
test_set_spurious_ood = datasets.ImageFolder(color_mnist_test_spurious_ood, composed_transforms)
color_mnist_train_set = datasets.ImageFolder(color_mnist_train, composed_transforms)
color_mnist_train_intervened_set = datasets.ImageFolder(color_mnist_train_intervened, composed_transforms)

print(color_mnist_train_set)
print(color_mnist_train_set[:100])
exit()

color_mnist_combined_set = RationDataset(color_mnist_train_set, color_mnist_train_intervened_set)

testloader_indist = torch.utils.data.DataLoader(test_set_indist, batch_size=BATCH_SIZE, shuffle=True)
testloader_ood = torch.utils.data.DataLoader(test_set_ood, batch_size=BATCH_SIZE, shuffle=True)
testloader_spurious_ood = torch.utils.data.DataLoader(test_set_spurious_ood, batch_size=BATCH_SIZE, shuffle=True)

color_mnist_trainloader = torch.utils.data.DataLoader(
                                                        color_mnist_train_set, 
                                                        batch_size=BATCH_SIZE, shuffle=True)

intervened_trainloader = torch.utils.data.DataLoader(
                                                        color_mnist_train_intervened_set, 
                                                        batch_size=BATCH_SIZE, shuffle=True)
combined_trainloader = torch.utils.data.DataLoader(
                                                        color_mnist_combined_set, 
                                                        batch_size=BATCH_SIZE, shuffle=True)

print('train baseline')
cnn_baseline = CNN()
cnn_baseline = cnn_baseline.cuda()
fit(cnn_baseline, color_mnist_trainloader)

print('train intervened')
cnn_intervened = CNN()
cnn_intervened = cnn_intervened.cuda()
fit(cnn_intervened, intervened_trainloader)

print('train augment')
cnn_augment = CNN()
cnn_augment = cnn_augment.cuda()
fit_augment(cnn_augment, combined_trainloader)

print("baseline")
in_pred, _ = evaluate(cnn_baseline, testloader_indist)
print("OOD")
out_pred, _ = evaluate_ood(cnn_baseline, testloader_ood)
utils.get_and_print_results(in_pred,out_pred,"dummy_ood","dummy_method")
print("SPURIOUS OOD")
sp_out_pred, _ = evaluate_ood(cnn_baseline, testloader_spurious_ood)
utils.get_and_print_results(in_pred,sp_out_pred,"dummy_ood","dummy_method")

print("intervened")
in_pred, in_actual = evaluate(cnn_intervened, testloader_indist)
print("OOD")
out_pred, out_actual = evaluate_ood(cnn_intervened, testloader_ood)
utils.get_and_print_results(in_pred,out_pred,"dummy_ood","dummy_method")
print("SPURIOUS OOD")
sp_out_pred, _ = evaluate_ood(cnn_intervened, testloader_spurious_ood)
utils.get_and_print_results(in_pred,sp_out_pred,"dummy_ood","dummy_method")

print("augment")
in_pred, in_actual = evaluate(cnn_augment, testloader_indist)
print("OOD")
out_pred, out_actual = evaluate_ood(cnn_augment, testloader_ood)
utils.get_and_print_results(in_pred,out_pred,"dummy_ood","dummy_method")
print("SPURIOUS OOD")
sp_out_pred, _ = evaluate_ood(cnn_augment, testloader_spurious_ood)
utils.get_and_print_results(in_pred,sp_out_pred,"dummy_ood","dummy_method")
