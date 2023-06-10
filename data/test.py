# -*- coding: utf-8 -*-
"""
Created on Sat May 27 10:06:48 2023

@author: techv
"""

####################################################
#                Import Libraries
####################################################

import matplotlib.pyplot as plt
from pandas.core.common import flatten
import copy
import numpy as np
import random

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

import glob
from tqdm import tqdm

####################################################
#                Define parameters
####################################################

params = {
    "batch_size"  :    1,
    "image_height":  350,
    "image_width" :  350,
    "num_classes" :    6,
    "nepoch"      :    10
}

#######################################################
#               Define Transforms
#######################################################

train_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=350),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, p=0.5),
        A.RandomCrop(height=params["image_height"], width=params["image_width"]),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.MultiplicativeNoise(multiplier=[0.5,2], per_channel=True, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
        ToTensorV2(),
    ]
)

test_transforms = A.Compose(
    [
        A.SmallestMaxSize(max_size=350),
        A.CenterCrop(height=params["image_height"], width=params["image_width"]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

####################################################
#       Create Train, Valid and Test sets
####################################################

train_data_path = 'C:/Users/techv/Projects/Smart Bin/data/dataset/dataset-resized/train'
test_data_path  = 'C:/Users/techv/Projects/Smart Bin/data/dataset/dataset-resized/test'


train_image_paths = [] #to store image paths in list
classes = [] #to store class values

for data_path in glob.glob(train_data_path + '/*'):
    cls_ = (data_path.split('/')[-1]).split('\\')[-1]
    classes.append(cls_)
    train_image_paths.append(glob.glob(data_path + '/*')) 
train_image_paths = list(flatten(train_image_paths))
random.shuffle(train_image_paths)

print('train_image_path example: ', train_image_paths[0])
print('class example: ', classes[0])


#split train valid from train paths (80,20)
train_image_paths, valid_image_paths = train_image_paths[:int(0.8*len(train_image_paths))], train_image_paths[int(0.8*len(train_image_paths)):] 


test_image_paths = []
for data_path in glob.glob(test_data_path + '/*'):
    test_image_paths.append(glob.glob(data_path + '/*'))

test_image_paths = list(flatten(test_image_paths))

print("\nTrain size: {}\nValid size: {}\nTest size: {}".format(len(train_image_paths), len(valid_image_paths), len(test_image_paths)))

#######################################################
#      Create dictionary for class indexes
#######################################################

idx_to_class = {i:j for i, j in enumerate(classes)}
class_to_idx = {value:key for key,value in idx_to_class.items()}
print(idx_to_class)
print(class_to_idx)
print('Length of train: ', len(train_image_paths))
print('Length of test: ', len(test_image_paths))

#######################################################
#               Define Dataset Class
#######################################################

class LandmarkDataset(Dataset):
    def __init__(self, image_paths, transform=False):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        # print(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(image_filepath)
        label = (image_filepath.split('/')[-1]).split('\\')
        # label[0] = 'Micanopy'
        # label = label[0] + "\\" + label[1]
        label = label[1]
        # print(label)
        label = class_to_idx[label]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        
        return image, label
    
#######################################################
#                  Create Dataset
#######################################################

train_dataset = LandmarkDataset(train_image_paths,train_transforms)
valid_dataset = LandmarkDataset(valid_image_paths,test_transforms) #test transforms are applied
test_dataset  = LandmarkDataset(test_image_paths,test_transforms)

print('The shape of tensor for 50th image in train dataset: ',train_dataset[49][0].shape)
print('The label for 50th image in train dataset: ',train_dataset[0][1])


#######################################################
#                  Define Dataloaders
#######################################################

train_loader = DataLoader(train_dataset, batch_size=params["batch_size"],  shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=params["batch_size"],  shuffle=True)
test_loader  = DataLoader( test_dataset, batch_size=params["batch_size"],  shuffle=True)

loaders = {'train': train_loader, 'valid': valid_loader, 'test': test_loader}

print(next(iter(train_loader))[1].shape)

#######################################################
#                  Visualize Dataset
#         Images are plotted after augmentation
#######################################################

def visualize_augmentations(dataset, idx=0, samples=10, cols=5, random_img = False):
    
    dataset = copy.deepcopy(dataset)
    #we remove the normalize and tensor conversion from our augmentation pipeline
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
    
        
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
    for i in range(samples):
        if random_img:
            idx = np.random.randint(1,len(train_image_paths))
        image, lab = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
        ax.ravel()[i].set_title(idx_to_class[lab])
    plt.tight_layout(pad=1)
    plt.show()    

visualize_augmentations(train_dataset,np.random.randint(1,len(train_image_paths)), random_img = True)


#------------------------------------------------------------------------------
#--------------------------Train the model ------------------------------------
#------------------------------------------------------------------------------

from torch.autograd import Variable

# Function to save the model
def saveModel():
    path = "./bin_resnet50_model.pth"
    torch.save(model.state_dict(), path)
    print("Model Saved")

# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy():
    model.eval()
    accuracy = 0.0
    total = 0.0
    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    with torch.no_grad():
        # print(1)
        for data in test_loader:
            # print(2)
            images, labels = data
            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
            # run the model on the test set to predict labels
            # print(3)
            outputs = model(images)
            # the label with the highest energy will be our prediction
            # print(4)
            _, predicted = torch.max(outputs.data, 1)
            # print(5)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
            # print(6)
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return(accuracy)


# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs):
    
    best_accuracy = 0.0

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0
        for i, (images, labels) in enumerate(train_loader, 0):
            # get the inputs
            # print(labels)
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # predict classes using images from the training set
            outputs = model(images)
            
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            
            # backpropagate the loss
            loss.backward()
            
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 images
            running_loss += loss.item()     # extract the loss value
            # print(running_loss,running_acc,i,i%10)
            # print(i)
            if i % 10 == 5:    
                # print every 1000 (twice per epoch) 
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 1000))
                # zero the loss
                running_loss = 0.0

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy = testAccuracy()
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy),best_accuracy)
        
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy

##defining the model
from torchvision.models import resnet50
import torchvision
            
model = resnet50(weights=None)
path = "bin_resnet50_model.pth"
model.load_state_dict(torch.load(path))

from torch.optim import Adam
 
# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

import numpy as np

# Function to show the images
def imageshow(img):
    img_ = img
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


from prettytable import PrettyTable

# Function to test the model with a batch of images and show the labels predictions
def testBatch(images,labels):

    # show all images as one image grid
    imageshow(torchvision.utils.make_grid(images))
   
    # Show the real labels on the screen 
    # print('Real labels: ', ' '.join('%5s' % classes[labels[j]] 
    #                            for j in range(params['batch_size'])))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    # get the inputs
    images = Variable(images.to(device))
    labels = Variable(labels.to(device))
        
    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)
    
    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)
    # Let's show the predicted labels on the screen to compare with the real ones
    # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] 
    #                           for j in range(params['batch_size'])))
    myTable = PrettyTable(["Actual","Predicted"])
    for i in range(len(labels)):
        a,b = classes[labels[i]], classes[predicted[i]]
        myTable.add_row([str(a),str(b)])
        
    print(myTable)
    
# import time
if __name__ == "__main__":
    
    # Let's build our model
    train(params["nepoch"])
    print('Finished Training')

    # Test which classes performed well
    # testModelAccuracy()
    
    # Let's load the model we just created and test the accuracy per label
    sus = iter(test_loader)
    for i in range(1):
        # Test with batch of images
        # get batch of images from the test DataLoader  
        images, labels = next(sus)
        testBatch(images,labels)