# Imports here
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2
import time
from PIL import Image

#Inputs
import platform
import webbrowser
import keyboard #toinstall
import json


test_transforms = transforms.Compose([transforms.Resize(225),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
idx_to_class={ 0:'A', 1:'B', 2:'nothing'}

#Pretrained model densenet161
model = models.densenet161(pretrained=True)

#Creating the classifier part
#Freezing the parameters
for param in model.parameters():
    param.requires_grad = False

class Network(nn.Module):
  def __init__(self,n_hidden1=1024,n_hidden2=512):
    super().__init__()

    self.fc1=nn.Linear(2208,n_hidden1)
    self.relu1=nn.ReLU()
    self.fc2=nn.Linear(n_hidden1,n_hidden2)
    self.relu2=nn.ReLU()
    #self.fc3=nn.Linear(n_hidden2,n_hidden3)
    #self.relu3=nn.ReLU()
    self.output=nn.Linear(n_hidden2,3)
    self.dropout=nn.Dropout(p=0.25)

  def forward(self,x):

    x=self.relu1(self.fc1(x))
    x=self.dropout(x)
    x=self.relu2(self.fc2(x))
    x=self.dropout(x)
    #x=self.relu3(self.fc3(x))
    #x=self.dropout(x)
    x=self.output(x)
    #x=F.log_softmax(x,dim=1)

    return x

model.classifier=Network()
model=model.cuda()

#Saving and loading models
# TODO: Train your network
def save_model(model,minimum_loss,filename):
  st=model.state_dict()
  torch.save({"minloss":minimum_loss,"state":st},filename)

def load_model(filename):
  return torch.load(filename)

#Importing the trained model
chpt=load_model("net.pt")
model.load_state_dict(chpt["state"])

#Get the program running
# Camera 0 is the integrated web cam on my netbook
camera_port = 0

#Number of frames to throw away while the camera adjusts to light levels
ramp_frames = 30

# Now we can initialize the camera capture object with the cv2.VideoCapture class.
# All it needs is the index to a camera port.
camera = cv2.VideoCapture(camera_port)

# Captures a single image from the camera and returns it in PIL format
def get_image():
 # read is the easiest way to get a full image out of a VideoCapture object.
 retval, im = camera.read()
 return im

# Ramp the camera - these frames will be discarded and are only used to allow v4l2
# to adjust light levels, if necessary
nb=0
model.eval()
sys=platform.system()
while True:
    nb+=1
    for i in range(ramp_frames):
     temp = get_image()

    camera_capture=get_image()
    #cv2.imwrite("{}.png".format(nb),camera_capture)
    im = Image.fromarray(camera_capture)
    processed=test_transforms(im)
    #print(processed)
    camera_capture = processed.unsqueeze(0).type(torch.cuda.FloatTensor)

    #Process using pytorch and get type of gesture
    with torch.no_grad():
        prob,pred=F.softmax(model(camera_capture),dim=1).topk(1,dim=1)

    #If we're certain of that gesture
    if(prob.item()>0.8):
        print(idx_to_class[pred.item()], " with probability : ",prob.item())
        if(idx_to_class[pred.item()]!="nothing"):
            with open('settings.json', 'r') as f:
                data = json.load(f)
                for d in data:
                    if(idx_to_class[pred.item()]==d['gesture']):
                        if(d['type']=='url'):
                            url = d['value']
                            if(sys=="Windows"):  # Windows
                                chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
                            elif(sys=="Darwin"): # MacOS
                                chrome_path = 'open -a /Applications/Google\ Chrome.app %s'
                            elif(sys=="Linux"):# Linux
                                chrome_path = '/usr/bin/google-chrome %s'
                            #webbrowser.get(chrome_path).open(url)
                            webbrowser.open(url,new=0,autoraise=True)
                        elif(d['type']=='key'):
                            keyboard.press_and_release(d['value'])
                        elif(d['type']=="phrase"):
                            keyboard.write(d['value'])

                        break

del(camera)
