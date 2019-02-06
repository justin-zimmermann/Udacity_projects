#import argparse module and fill arguments
import argparse
parser = argparse.ArgumentParser(description='Training a neural network to categorize images of flowers.')
parser.add_argument('data_dir', help='give the directory for the training, validation and testing sets')
parser.add_argument('--save_dir', help='give the directory for the model checkpoint (default: checkpoint.pth)', default='checkpoint.pth')
parser.add_argument('--arch', help='choose which neural network architecture to use: choice between vgg13 (default) and alexnet', choices=['vgg13','alexnet'], default='vgg13')
parser.add_argument('--learning_rate', help='choose the learning rate of the model (default=0.001)',type=int, default=0.001)
parser.add_argument('--epochs', help='choose the epoch (default=1)',type=int, default=1)
parser.add_argument('--hidden_units', help='choose the hidden unit (default=512)',type=int, default=512)
parser.add_argument('--gpu', help='force GPU (uses CPU if not specified)', action="store_true")
args = parser.parse_args()

#import all necessary
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
import copy
from workspace_utils import active_session, keep_awake
from collections import OrderedDict
from PIL import Image

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(40),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor()])
valid_transforms = transforms.Compose([transforms.Resize(225),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(225),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


# TODO: Load the datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)

#create the classifier part of the model to fit the number of categories (102):
#freeze parameters for feature part of model:
hu = args.hidden_units
if args.arch == 'vgg13':
    model = models.vgg13(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
                                            ('dropout', nn.Dropout(p=0.2)),
                                            ('fc1', nn.Linear(25088, hu)),
                                            ('relu', nn.ReLU()),
                                            ('fc2', nn.Linear(hu, 102)),
                                            ('output', nn.LogSoftmax(dim=1))
                                            ]))
model.classifier = classifier
elif args.arch == 'alexnet':
    model = models.alexnet(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
                                            ('dropout', nn.Dropout(p=0.2)),
                                            ('fc1', nn.Linear(9216, hu)),
                                            ('relu', nn.ReLU()),
                                            ('fc2', nn.Linear(hu, 102)),
                                            ('output', nn.LogSoftmax(dim=1))
                                            ]))
model.classifier = classifier

#give the model a criterion and optimizer:
criterion = nn.NLLLoss()
lr = args.learning_rate
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

def train_model(model, criterion, optimizer, epoch):
    '''train the model, then keep the model with the best accuracy with the copy module'''
    with active_session(): #for computations lasting more than 30 minutes, put code indented into this to keep session open
        if args.gpu:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if (torch.cuda.is_available() == False):
                print ("CUDA not available, proceeding with CPU instead")
        else:
            device = torch.device("cpu")
        epochs = epoch
        steps = 0
        running_loss = 0
        print_every = 5
        model.to(device);
        top_model = copy.deepcopy(model.state_dict())
        top_accuracy = 0.
        for epoch in range(epochs): #train the model with training set over x epochs
            for inputs, labels in trainloader:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if steps % print_every == 0: #every x steps, use validation set to check that model is learning
                    valid_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for vinputs, vlabels in validloader:
                            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                            vlogps = model.forward(vinputs)
                            batch_loss = criterion(vlogps, vlabels)
                            
                            valid_loss += batch_loss.item()
                            
                            # Calculate accuracy
                            ps = torch.exp(vlogps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == vlabels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                            if accuracy > top_accuracy: #if new best model: copy it
                                top_accuracy = accuracy
                                top_model = copy.deepcopy(model.state_dict())
                
                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                          f"Validation accuracy: {accuracy/len(validloader):.3f}")
                          running_loss = 0
                          model.train()
        model.load_state_dict(top_model)
        model.to('cpu')
    return model

#train here:
epoch = args.epochs
model = train_model(model, criterion, optimizer, epoch=epoch)

#Save the checkpoint, don't forget to put it back to the cpu
def save_checkpoint(model, filepath):
    model.to('cpu')
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {'model_state_dict': model.state_dict(),
        'class_to_idx': train_dataset.class_to_idx,
            'arch': args.arch,
                'hu': args.hidden_units,
                    'lr': args.learning_rate}
    torch.save(checkpoint, filepath)

save_dir = args.save_dir
save_checkpoint(model, filepath=save_dir)
