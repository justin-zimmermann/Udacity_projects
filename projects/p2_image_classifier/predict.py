#import argparse module and fill arguments
import argparse
parser = argparse.ArgumentParser(description='Using a trained neural network to predict the type of flower in a given image.')
parser.add_argument('input', help='give the directory for the image (ex: "flowers/train/1/image_06734.jpg")')
parser.add_argument('checkpoint', help='give the directory for the model checkpoint (ex: "checkpoint.pth")')
parser.add_argument('--category_names', help='include a .json file about all category names (default: cat_to_name.json)', default='cat_to_name.json')
parser.add_argument('--top_k', help='print the top K most likely classes (default K=1)',type=int, default=1)
parser.add_argument('--gpu', help='force GPU (uses CPU if not specified)', action="store_true")
args = parser.parse_args()

# Imports here
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

#label mapping
filepath = args.category_names
with open(filepath, 'r') as f:
    cat_to_name = json.load(f)

def load_checkpoint(filepath):
    '''function that loads a checkpoint and rebuilds the model'''
    checkpoint = torch.load(filepath)
    hu = checkpoint['hu']
    if checkpoint['arch'] == 'vgg13':
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
elif checkpoint['arch'] == 'alexnet':
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
    model.to('cpu')
    criterion = nn.NLLLoss()
    lr = checkpoint['lr']
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model, optimizer, criterion

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
        '''
    im = Image.open(image)
    size = round(max(im.size)/min(im.size)*256.)
    im.thumbnail((size, size))
    width, height = im.size
    left = (width - 224)/2
    upper = (height - 224)/2
    right = (width + 224)/2
    lower = (height + 224)/2
    np_im = np.array(im.crop((left,upper,right,lower))) /255
    mean = np.array([0.485, 0.456, 0.406])
    sd = np.array([0.229, 0.224, 0.225])
    np_im = (np_im - mean) / sd
    np_im = np_im.transpose((2,0,1))
    return np_im

def predict(image_path, checkpoint, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
    model, optimizer, criterion = load_checkpoint(checkpoint)
    model.eval()
    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if (torch.cuda.is_available() == False):
            print ("CUDA not available, proceeding with CPU instead")
    else:
        device = torch.device("cpu")
    with torch.no_grad():
        image = process_image(image_path)
        image = torch.from_numpy(image)
        image = image.type(torch.FloatTensor)
        image.unsqueeze_(0) #adds batch size of 1 (extra dimension to tensor at position 0)
        image = image.to(device)
        model = model.to(device)
        logps = model.forward(image)
        ps = torch.exp(logps)
        ps = ps.cpu()
        top_p, top_idx = ps.topk(topk, dim=1)
    top_p = np.squeeze(top_p.numpy())
    top_idx = np.squeeze(top_idx.numpy())
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}
    top_class = [idx_to_class[idx] for idx in top_idx]
    top_names = [cat_to_name[cat] for cat in top_class]

return top_p, top_names

image_dir = args.input
checkpoint_dir = args.checkpoint
top_k = args.top_k
probs, classes = predict(image_dir, checkpoint_dir, top_k)
for i in range(top_k):
    print(classes[i], " with probability ", probs[i])
