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
import random

import matplotlib.pyplot as plt
import constants

from torchvision import datasets, models, transforms

import torch.optim as optim
from torch.optim import lr_scheduler


# Parameters
learning_rate = 0.0002
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

class ResNet():
    def create_model(self):
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model_ft.fc = nn.Linear(num_ftrs, 2)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_ft = model_ft.to(device)

        return model_ft

def fit(model, train_loader):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    error = nn.CrossEntropyLoss()
    EPOCHS = 10
    model.train()
    for epoch in tqdm(range(EPOCHS)):
        correct = 0
        for batch_idx, (imgs, labels) in enumerate(train_loader):
            var_X_batch = Variable(imgs.type(FloatTensor))
            var_y_batch = Variable(labels.type(LongTensor))
            optimizer.zero_grad()
            try:
                output = model(var_X_batch)
            except ValueError as e:
                continue

            loss = error(output, var_y_batch)
            loss.backward()
            optimizer.step()

            # Total correct predictions
            predicted = torch.max(output.data, 1)[1] 
            correct += (predicted == var_y_batch).sum()
    
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
    return np.asarray(outputs), float(top1*100) / (len(test_loader)*BATCH_SIZE)

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



def plot_in_dist(in_dist):
    plt.plot(ORIG_INTERV_RATIOs,in_dist)
    plt.xlabel("original : intervened data ratio")
    plt.ylabel("in-distribution accuracy")
    plt.title("In-distribution performance")
    plt.savefig(os.path.join(results_dir,"in_dist.png"))
    plt.close()

def plot_ood_results(ood,spurious,metric):
    metric_ood = [obj[metric] for obj in ood]
    metric_spurious = [obj[metric] for obj in spurious]
    plt.plot(ORIG_INTERV_RATIOs,metric_ood, label="OOD")
    plt.plot(ORIG_INTERV_RATIOs,metric_spurious, label="Spurious OOD")
    plt.xlabel("original : intervened data ratio")
    plt.ylabel(metric)
    plt.legend()
    plt.title(f"{metric} comparison")
    plt.savefig(os.path.join(results_dir,f"{metric}.png"))
    plt.close()

def plot_results(in_dist, ood, spurious):
    plot_in_dist(in_dist)
    plot_ood_results(ood, spurious, 'auroc')
    plot_ood_results(ood, spurious, 'aupr')
    plot_ood_results(ood, spurious, 'fpr')


r = constants.r
results_dir = f"results/r_{r}"
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)

test_indist = f'/nobackup/dyah_roopa/CelebA/test_{r}/in_dist/'
test_ood = '/nobackup/dyah_roopa/temp/Spurious_OOD/datasets/ood_datasets/LSUN_resize/'
spurious_ood = f'/nobackup/dyah_roopa/CelebA/test_{r}/spurious_ood/'

train = f'/nobackup/dyah_roopa/CelebA/train_{r}/'
train_intervened = f'/nobackup/dyah_roopa/CelebA/intervened_train_{r}/'


composed_transforms = transforms.Compose([
                        transforms.Resize(32), 
                        transforms.ToTensor(), 
                    ])

test_set_indist = datasets.ImageFolder(test_indist, composed_transforms)
test_set_ood = datasets.ImageFolder(test_ood, composed_transforms)
test_set_spurious_ood = datasets.ImageFolder(spurious_ood, composed_transforms)
color_mnist_train_set = datasets.ImageFolder(train, composed_transforms)
color_mnist_train_intervened_set = datasets.ImageFolder(train_intervened, composed_transforms)



ORIG_INTERV_RATIOs = np.linspace(0,1,10) #how much original data : intervened data
ablation_loaders = []
for ratio in ORIG_INTERV_RATIOs:
    split_n = int(ratio * len(color_mnist_train_set))
    orig_random_indices = np.random.choice(np.linspace(0, len(color_mnist_train_set), len(color_mnist_train_set)-1, endpoint=False, dtype=int), split_n)
    interv_random_indices = np.random.choice(np.linspace(0, len(color_mnist_train_intervened_set), len(color_mnist_train_intervened_set)-1, endpoint=False, dtype=int), len(color_mnist_train_set)-split_n)
    orig_subset = torch.utils.data.Subset(color_mnist_train_set, orig_random_indices)
    intervened_subset = torch.utils.data.Subset(color_mnist_train_intervened_set, interv_random_indices)
    color_mnist_combined_set = torch.utils.data.ConcatDataset([orig_subset, intervened_subset])
    combined_trainloader = torch.utils.data.DataLoader(
                                                        color_mnist_combined_set, 
                                                        batch_size=BATCH_SIZE, shuffle=True)
    ablation_loaders.append(combined_trainloader)

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

# print('train baseline')
# cnn_baseline = ResNet()
# cnn_baseline = cnn_baseline.create_model().cuda()
# fit(cnn_baseline, color_mnist_trainloader)

# print('train intervened')
# cnn_intervened = ResNet()
# cnn_intervened = cnn_intervened.create_model().cuda()
# fit(cnn_intervened, intervened_trainloader)

print("training ablation models")
ablation_models = []
for loader in ablation_loaders:
    cnn_augment = ResNet()
    cnn_augment = cnn_augment.create_model().cuda()
    fit(cnn_augment, loader)
    ablation_models.append(cnn_augment) 

# print("BASELINE")
# in_pred, _ = evaluate(cnn_baseline, testloader_indist)
# # print("OOD")
# # out_pred, _ = evaluate_ood(cnn_baseline, testloader_ood)
# # utils.get_and_print_results(in_pred,out_pred,"dummy_ood","dummy_method")
# print("------------------------")
# print("SPURIOUS OOD")
# sp_out_pred, _ = evaluate_ood(cnn_baseline, testloader_spurious_ood)
# utils.get_and_print_results(in_pred,sp_out_pred,"dummy_ood","dummy_method")

# print("######################")
# print("intervened")
# in_pred, in_actual = evaluate(cnn_intervened, testloader_indist)
# # print("OOD")
# # # out_pred, out_actual = evaluate_ood(cnn_intervened, testloader_ood)
# # utils.get_and_print_results(in_pred,out_pred,"dummy_ood","dummy_method")
# print("------------------------")
# print("SPURIOUS OOD")
# sp_out_pred, _ = evaluate_ood(cnn_intervened, testloader_spurious_ood)
# utils.get_and_print_results(in_pred,sp_out_pred,"dummy_ood","dummy_method")

print("######################")
print("ABLATION")
ood_results = []
spurious_ood_results = []
in_dist_results = []
for i, model in enumerate(ablation_models):
    print(f"orig:intervene ratio: {ORIG_INTERV_RATIOs[i]}")
    in_pred, in_accuracy = evaluate(model, testloader_indist)
    in_dist_results.append(in_accuracy)
    print("OOD")
    out_pred, out_actual = evaluate_ood(model, testloader_ood)
    ood_auroc, ood_aupr, ood_fpr = utils.get_and_print_results(in_pred,out_pred,"dummy_ood","dummy_method")
    ood_results.append({
        'auroc': ood_auroc,
        'aupr': ood_aupr,
        'fpr': ood_fpr
    })
    print("------------------------")
    print("SPURIOUS OOD")
    sp_out_pred, _ = evaluate_ood(model, testloader_spurious_ood)
    spur_auroc, spur_aupr, spur_fpr = utils.get_and_print_results(in_pred,sp_out_pred,"dummy_ood","dummy_method")
    spurious_ood_results.append({
        'auroc': spur_auroc,
        'aupr': spur_aupr,
        'fpr': spur_fpr
    })

plot_results(in_dist_results, ood_results, spurious_ood_results)
