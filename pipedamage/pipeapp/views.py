from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.http import FileResponse
from django.urls import reverse

from reportlab.pdfgen import canvas
from reportlab.lib import utils
from reportlab.lib.pagesizes import A4
import io 
import pipeapp
from pipeapp.models import PicUpload 
from pipeapp.forms import ImageForm
#from django.db import models
from django.core.files import File


# Create your views here.


def index(request):
    image_path = ''

    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)

        if form.is_valid():
            newdoc = PicUpload(imageFile=request.FILES['imageFile'])
            newdoc.save()

            return HttpResponseRedirect(reverse('index'))

    else:
        form = ImageForm()

    documents = PicUpload.objects.all()
    for document in documents:
        image_path = document.imageFile.name 
        image_path = '/' + image_path
        document.delete()

    request.session['image_path'] = image_path

    return render(request, 'index.html',
    {'documents':documents, 'image_path': image_path, 'form':form}
    )
####################################################################################################

def index_plus(request):
    image_path = ''

    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)

        if form.is_valid():
            newdoc = PicUpload(imageFile=request.FILES['imageFile'])
            newdoc.save()

            return HttpResponseRedirect(reverse('index_plus'))

    else:
        form = ImageForm()

    documents = PicUpload.objects.all()
    for document in documents:
        image_path = document.imageFile.name 
        image_path = '/' + image_path
        document.delete()

    request.session['image_path'] = image_path

    ### So could change results_dict value from list to dict, need 
    results_dict = {'fault_files':[]}
    MyPipe = request.session['image_path']
    img_path = MyPipe
    image_path_name = img_path.strip('/pic_upload/')
    image_path_name = image_path_name.strip('.mp4')
    image_path_name = image_path_name.strip('R-')
    
    results_dict['path_name'] = str(image_path_name)
    
    request.session.pop('image_path',None)
    request.session.modified = True
    img_path = img_path.strip('/')

    #extract the individual frames
    frames,data = get_num_frames(img_path)
    #for each frame predict normal or fault
    for i in range(int(frames-1)):
        # Read in each frame using CV2
        success, frame = data.read()
        if i % 10 == 0:
            x_img = frame
             #Correct the colouring - could move this into the image processing function
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
             #Preprocess the frame
            img = process_image_colour(frame)
            pred = make_prediction_myModel("pipeapp/static/myModel23000.pt",img)
        

        #If there is a fault present
        #1. Save the image 
        #2. Add filename to results_dict

            if (pred == 'Fault Present'):
            #save the image to faults_file
                fault_img_file = 'faults_pics/'
                fault_img_file = 'faults_pics/'+str(i)+'.png'
                cv2.imwrite(fault_img_file,x_img)
                part1 = '/'+fault_img_file
                part2 = []
                isRoots = make_prediction_roots("pipeapp/static/myModelRootsModel1.pt",img)
                isBroken = make_prediction_broken("pipeapp/static/myModelBrokenModel1.pt",img)
                isSettledDeposits = make_prediction_settled_deposits("pipeapp/static/myModelSettledModel1.pt",img)
                isAttachedDeposits = make_prediction_attached_deposits("pipeapp/static/myModelAttachedModel2.pt",img)
                if isRoots[0] != 'Other Fault':
                    part2.append(isRoots)
                    # segmentation model predict
                    '''root_seg_model = torch.load("pipeapp/static/roots_seg_weights.pt",weights_only=False,map_location=torch.device('cpu'))
                    root_seg_model.eval()
                    img_w = x_img.shape[0] 
                    img_h = x_img.shape[1]
                    new_img = frame.transpose(2,0,1).reshape(1,3,img_w,img_h)
                    with torch.no_grad():
                        a = root_seg_model(torch.from_numpy(new_img).type(torch.FloatTensor)/255) 
                    # write the mask
                    predicted_output = a['out'].cpu().detach().numpy()[0][0]>0.30

                    # convert bools to 255
                    x_output = np.where(predicted_output == True, 255,0)
                    y_output = x_output.reshape(img_w,img_h,1)
                    y_output = y_output.astype(np.uint8)
                    # put the mask over x_img
                    contours, heiracy= cv2.findContours(y_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    seg_img = cv2.drawContours(x_img,contours,-1,(0,255,0),1)
                    # save out the updated img
                    cv2.imwrite(fault_img_file,seg_img)'''

                if isBroken[0] != 'Other Fault':
                    part2.append(isBroken)
                if isSettledDeposits[0] != 'Other Fault':
                    part2.append(isSettledDeposits)
                    '''settled_seg_model = torch.load("pipeapp/static/settled_deposit_seg_weights.pt")
                    settled_seg_model.eval()
                    img_w = x_img.shape[0] 
                    img_h = x_img.shape[1]
                    new_img = frame.transpose(2,0,1).reshape(1,3,img_w,img_h)
                    with torch.no_grad():
                        a = settled_seg_model(torch.from_numpy(new_img).type(torch.FloatTensor)/255) 
                    # write the mask
                    predicted_output = a['out'].cpu().detach().numpy()[0][0]>0.30

                    # convert bools to 255
                    x_output = np.where(predicted_output == True, 255,0)
                    y_output = x_output.reshape(img_w,img_h,1)
                    y_output = y_output.astype(np.uint8)
                    # put the mask over x_img
                    contours, heiracy= cv2.findContours(y_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    seg_img = cv2.drawContours(x_img,contours,-1,(0,0,255),1)
                    # save out the updated img
                    cv2.imwrite(fault_img_file,seg_img)
                if isAttachedDeposits[0] != 'Other Fault':
                    part2.append(isAttachedDeposits)
                if len(part2) == 0:
                    part2.append('Other Fault')'''
                results_dict['fault_files'].append((part1,part2))

                #if model predicts roots, or deposits,
                #apply segmentation or yolo model to the image (?make a copy)
                #save that image and return that image instead?
                #what if there are roots and deposits predicted?
                #will probably need to do this after isRoots,broken etc?
               

        #print(results_dict)
    
    return render(request, 'results.html', context=results_dict)

    



    ################# ML Model Section  ##################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2



class Net(nn.Module):
  def __init__(self, num_classes):
    super().__init__()
    self.feature_extractor = nn.Sequential(
        nn.Conv2d(3,32,kernel_size=3, padding=1),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Dropout(0.3),
        nn.Conv2d(32,64,kernel_size=3, padding=1),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Flatten(),

    )

    self.classifier = nn.Sequential(
        nn.Linear(64*64*64,2),

    )

  def forward(self,x):
    x = self.feature_extractor(x)
    x = self.classifier(x)
    return x

def get_num_frames(video):
    data = cv2.VideoCapture(video)
    # count the number of frames
    frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Number of frames:",frames)
    return frames, data


def process_image_myModel(img_path):
    preprocess = transforms.Compose([transforms.Resize((256,256))])
    img_file = img_path
    img = Image.open(img_file)
    img = preprocess(img)
    img= np.array(img)
    img = img.astype('float32')
    img = img/255.0
    img = np.array(img)
    img = np.rollaxis(img,2,0)
    img = torch.tensor(img)
    img = img.unsqueeze(0)
    return img

def process_image_colour(img):
    transform4 = transforms.Compose([transforms.ToTensor(),transforms.Resize((256,256)),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #transform4 = transforms.Compose([transforms.ToTensor(),transforms.Resize((256,256))])
    img = Image.fromarray(img)
    img = transform4(img)
    img = img.unsqueeze(0)
    return img



def make_prediction_myModel(weights_path,img):
    class_dict = {0:'Normal Pipe', 1:'Fault Present'}
    model = Net(num_classes=2)
    model.load_state_dict(torch.load(str(weights_path),map_location=torch.device('cpu')))
    model.eval()
    outputs = model(img)
    _,preds = torch.max(outputs, 1)
    pred = preds.item()
    pred = class_dict[pred]
    return pred

def make_prediction_roots(weights_path,img):
    class_dict = {0:'Other Fault', 1:'Roots'}
    model = Net(num_classes=2)
    model.load_state_dict(torch.load(str(weights_path),map_location=torch.device('cpu')))
    model.eval()
    outputs = model(img)
    _,preds = torch.max(outputs, 1)
    probs = torch.softmax(outputs,1)
    pred = preds.item()
    prob = probs[0][1].item()
    prob = round(prob,2)
    pred = class_dict[pred]
    return (pred,prob)

def make_prediction_broken(weights_path,img):
    class_dict = {0:'Other Fault', 1:'Cracked / Broken'}
    model = Net(num_classes=2)
    model.load_state_dict(torch.load(str(weights_path),map_location=torch.device('cpu')))
    model.eval()
    outputs = model(img)
    _,preds = torch.max(outputs, 1)
    probs = torch.softmax(outputs,1)
    pred = preds.item()
    prob = probs[0][1].item()
    prob = round(prob,2)
    pred = class_dict[pred]
    return (pred,prob)

def make_prediction_settled_deposits(weights_path,img):
    class_dict = {0:'Other Fault', 1:'Settled Deposits'}
    model = Net(num_classes=2)
    model.load_state_dict(torch.load(str(weights_path),map_location=torch.device('cpu')))
    model.eval()
    outputs = model(img)
    _,preds = torch.max(outputs, 1)
    probs = torch.softmax(outputs,1)
    pred = preds.item()
    prob = probs[0][1].item()
    prob = round(prob,2)
    pred = class_dict[pred]
    return (pred,prob)

def make_prediction_attached_deposits(weights_path,img):
    class_dict = {0:'Other Fault', 1:'Attached Deposits'}
    model = Net(num_classes=2)
    model.load_state_dict(torch.load(str(weights_path),map_location=torch.device('cpu')))
    model.eval()
    outputs = model(img)
    _,preds = torch.max(outputs, 1)
    probs = torch.softmax(outputs,1)
    pred = preds.item()
    prob = probs[0][1].item()
    prob = round(prob,2)
    pred = class_dict[pred]
    return (pred,prob)

def root_seg_model_prediction(weights_path,x_img,fault_img_file):
    root_seg_model = torch.load("pipeapp/static/roots_seg_weights.pt")
    root_seg_model.eval()
    img_w = x_img.shape[0] 
    img_h = x_img.shape[1]
    new_img = frame.transpose(2,0,1).reshape(1,3,img_w,img_h)
    with torch.no_grad():
        a = root_seg_model(torch.from_numpy(new_img).type(torch.FloatTensor)/255) 
            # write the mask
    predicted_output = a['out'].cpu().detach().numpy()[0][0]>0.30

    # convert bools to 255
    x_output = np.where(predicted_output == True, 255,0)
    y_output = x_output.reshape(img_w,img_h,1)
    y_output = y_output.astype(np.uint8)
    # put the mask over x_img
    contours, heiracy= cv2.findContours(y_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    seg_img = cv2.drawContours(x_img,contours,-1,(0,255,0),1)
    # save out the updated img
    cv2.imwrite(fault_img_file,seg_img)

def roots_object_detection():
    pass
      


################################################## Uploading Video #####################################################


#Set up a dictionary for the results of each video
def engine(request):

    ### So could change results_dict value from list to dict, need 
    results_dict = {'fault_files':[]}
    MyPipe = request.session['image_path']
    img_path = MyPipe
    image_path_name = img_path.strip('/pic_upload/')
    image_path_name = image_path_name.strip('.mp4')
    image_path_name = image_path_name.strip('R-')
    
    results_dict['path'] = image_path_name
    print(image_path_name)
    request.session.pop('image_path',None)
    request.session.modified = True
    img_path = img_path.strip('/')

    #extract the individual frames
    frames,data = get_num_frames(img_path)
    #for each frame predict normal or fault
    for i in range(int(frames-1)):
        # Read in each frame using CV2
        success, frame = data.read()
        if i % 10 == 0:
            x_img = frame
             #Correct the colouring - could move this into the image processing function
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
             #Preprocess the frame
            img = process_image_colour(frame)
            pred = make_prediction_myModel("pipeapp/static/myModel23000.pt",img)
        

        #If there is a fault present
        #1. Save the image 
        #2. Add frame number and filename to results_dict

            if (pred == 'Fault Present'):
            #save the image to faults_file
                fault_img_file = 'faults_pics/'
                fault_img_file = 'faults_pics/'+str(i)+'.png'
                cv2.imwrite(fault_img_file,x_img)
                part1 = '/'+fault_img_file
                part2 = []
                isRoots = make_prediction_roots("pipeapp/static/myModelRootsModel1.pt",img)
                isBroken = make_prediction_broken("pipeapp/static/myModelBrokenModel1.pt",img)
                isSettledDeposits = make_prediction_settled_deposits("pipeapp/static/myModelSettledModel1.pt",img)
                isAttachedDeposits = make_prediction_attached_deposits("pipeapp/static/myModelAttachedModel2.pt",img)
                if isRoots[0] != 'Other Fault':
                    part2.append(isRoots)
                    # segmentation model predict
                    root_seg_model = torch.load("pipeapp/static/roots_seg_weights.pt")
                    root_seg_model.eval()
                    img_w = x_img.shape[0] 
                    img_h = x_img.shape[1]
                    new_img = frame.transpose(2,0,1).reshape(1,3,img_w,img_h)
                    with torch.no_grad():
                        a = root_seg_model(torch.from_numpy(new_img).type(torch.FloatTensor)/255) 
                    # write the mask
                    predicted_output = a['out'].cpu().detach().numpy()[0][0]>0.30

                    # convert bools to 255
                    x_output = np.where(predicted_output == True, 255,0)
                    y_output = x_output.reshape(img_w,img_h,1)
                    y_output = y_output.astype(np.uint8)
                    # put the mask over x_img
                    contours, heiracy= cv2.findContours(y_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    seg_img = cv2.drawContours(x_img,contours,-1,(0,255,0),1)
                    # save out the updated img
                    cv2.imwrite(fault_img_file,seg_img)

                if isBroken[0] != 'Other Fault':
                    part2.append(isBroken)
                if isSettledDeposits[0] != 'Other Fault':
                    part2.append(isSettledDeposits)
                    settled_seg_model = torch.load("pipeapp/static/settled_deposit_seg_weights.pt")
                    settled_seg_model.eval()
                    img_w = x_img.shape[0] 
                    img_h = x_img.shape[1]
                    new_img = frame.transpose(2,0,1).reshape(1,3,img_w,img_h)
                    with torch.no_grad():
                        a = settled_seg_model(torch.from_numpy(new_img).type(torch.FloatTensor)/255) 
                    # write the mask
                    predicted_output = a['out'].cpu().detach().numpy()[0][0]>0.30

                    # convert bools to 255
                    x_output = np.where(predicted_output == True, 255,0)
                    y_output = x_output.reshape(img_w,img_h,1)
                    y_output = y_output.astype(np.uint8)
                    # put the mask over x_img
                    contours, heiracy= cv2.findContours(y_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    seg_img = cv2.drawContours(x_img,contours,-1,(0,0,255),1)
                    # save out the updated img
                    cv2.imwrite(fault_img_file,seg_img)
                if isAttachedDeposits[0] != 'Other Fault':
                    part2.append(isAttachedDeposits)
                if len(part2) == 0:
                    part2.append('Other Fault')
                results_dict['fault_files'].append((part1,part2))

                #if model predicts roots, or deposits,
                #apply segmentation or yolo model to the image (?make a copy)
                #save that image and return that image instead?
                #what if there are roots and deposits predicted?
                #will probably need to do this after isRoots,broken etc?
               

    
    
    return render(request, 'results.html', context=results_dict)

###################################################################################