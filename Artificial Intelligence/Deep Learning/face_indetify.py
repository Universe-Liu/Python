from __future__ import print_function, division
import os
import sys
import cv2
import numpy as np

import os
import sys
import cv2
import numpy as np
import dlib
import matplotlib.pyplot as plt
'''
def listfiles(rootDir):
    list_dirs = os.walk(rootDir) 
    for root, dirs, files in list_dirs:  # 遍历文件夹下的图片
        for d in dirs:
            print((os.path.join(root, d)))
        for f in files:
            fileid = f.split('.')[0]  # 获得图片的名字，不含后缀
            filepath = os.path.join(root, f) 
            print(filepath)
            try:
                src = cv2.imread(filepath, 1)  # 读取原始图片，数据会加载到内存中
                print("src=", filepath, src.shape)
                os.remove(filepath) # 移除原来的图片
                cv2.imwrite(os.path.join(root, fileid + ".jpg"), src)  # 保存经过格式转换的图片
            except:
                os.remove(filepath)
                continue

path = "./Emotion_Recognition_File/img_type_test/"  # 输入图片路径即可，可以在这个文件夹下放置各种后缀名的图片，代码会将所有图片统一成 jpg 格式
#listfiles(path)
cascade_path='./Emotion_Recognition_File/face_detect_model/haarcascade_frontalface_default.xml'
cascade=cv2.CascadeClassifier(cascade_path)
img_path="./Emotion_Recognition_File/face_det_img/" 
images=os.listdir(img_path)
for image in images:
    im = cv2.imread(os.path.join(img_path, image), 1) # 读取图片
    rects = cascade.detectMultiScale(im, 1.3, 5)  # 人脸检测函数
    print("检测到人脸的数量", len(rects))
    if len(rects) == 0:  # len(rects) 是检测人脸的数量，如果没有检测到人脸的话，会显示出图片，适合本地调试使用，在服务器上可能不会显示
        cv2.namedWindow('Result', 0)
        cv2.imshow('Result', im)
        print("没有检测到人脸")
        pass
    plt.imshow(im[:, :, ::-1])  # 显示
    plt.show()
    os.remove(os.path.join(img_path, image)) # 
    k = cv2.waitKey(0)
    if k == ord('q'): # 在英文状态下，按下按键 q 会关闭显示窗口    
        break
    print()
cv2.destroyAllWindows()   

# 配置 Dlib 关键点检测路径
# 文件可以从 http://dlib.net/files/ 下载
PREDICTOR_PATH = "./Emotion_Recognition_File/face_detect_model/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
# 配置人脸检测器路径
cascade_path = './Emotion_Recognition_File/face_detect_model/haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(cascade_path)
def get_landmarks(im):
    rects = cascade.detectMultiScale(im, 1.3, 5) # 人脸检测
    x, y, w, h = rects[0]  # 获取人脸的四个属性值，左上角坐标 x,y 、高宽 w、h
#     print(x, y, w, h)
    rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h)) 
    return np.matrix([[p.x, p.y] for p in predictor(im, rect).parts()])

def annotate_landmarks(im,landmarks):
    im=im.copy()
    for idx,point in enumerate(landmarks):
        pos=(point[0,0],point[0,1])
        cv2.putText(im,str(idx),pos,fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.4,color=(0,0,255))
        cv2.circle(im,pos,5,color=(0,255,255))
    return im

def getlipfromimage(im,landmarks):
    xmin = 10000
    xmax = 0
    ymin = 10000
    ymax = 0
    for i in range(48,67):
        x=landmarks[i,0]
        y=landmarks[i,1]
        if x < xmin:
            xmin = x
        if x > xmax:
            xmax = x
        if y < ymin:
            ymin = y
        if y > ymax:
            ymax = y
    print("xmax:%d xmin:%d ymax:%d ymin:%d" % (xmax,xmin,ymax,ymin))
    roiwidth=xmax-xmin
    roiheight=ymax-ymin
    roi=im[ymin:ymax,xmin:xmax,0:3]
    if roiwidth>roiheight:
        dstlen=1.5*roiwidth
    else:
        dstlen=1.5*roiheight
    diff_xlen=dstlen-roiwidth
    diff_ylen=dstlen-roiheight
    newx=xmin
    newy=ymin
    imagerows, imagecols, channel = im.shape
    if newx >= diff_xlen / 2 and newx + roiwidth + diff_xlen / 2 < imagecols:
        newx = newx - diff_xlen / 2
    elif newx < diff_xlen / 2:
        newx = 0
    else:
        newx = imagecols - dstlen

    if newy >= diff_ylen / 2 and newy + roiheight + diff_ylen / 2 < imagerows:
        newy = newy - diff_ylen / 2
    elif newy < diff_ylen / 2:
        newy = 0
    else:
        newy = imagerows - dstlen

    roi = im[int(newy):int(newy + dstlen), int(newx):int(newx + dstlen), 0:3]
    return roi

def listfiles(rootDir):
    list_dirs = os.walk(rootDir)
    for root, dirs, files in list_dirs:
        for d in dirs:
            print(os.path.join(root, d))
        for f in files:
            fileid = f.split('.')[0]

            filepath = os.path.join(root, f)
            try:
                im = cv2.imread(filepath, 1)
                landmarks = get_landmarks(im)
                roi = getlipfromimage(im, landmarks)
                roipath = filepath.replace('.jpg', '_mouth.png')
#                 cv2.imwrite(roipath, roi)
                plt.imshow(roi[:, :, ::-1])
                plt.show()
            except:
#                 print("error")
                continue


listfiles("./Emotion_Recognition_File/mouth_det_img/")
'''


import torch.nn as nn
import torch.nn.functional as F

class simpleconv3(nn.Module):
    def __init__(self):
        super(simpleconv3,self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 3, 2)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 24, 3, 2)
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 48, 3, 2)
        self.bn3 = nn.BatchNorm2d(48)
        self.fc1 = nn.Linear(48 * 5 * 5 , 1200)
        self.fc2 = nn.Linear(1200 , 128)
        self.fc3 = nn.Linear(128 , 4)

    def forward(self , x):
        x = F.relu(self.bn1(self.conv1(x)))
        #print "bn1 shape",x.shape
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(-1 , 48 * 5 * 5) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) 
        return x



import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import numpy as np

import warnings

warnings.filterwarnings('ignore')

writer = SummaryWriter()

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            for data in dataloders[phase]:
                inputs, labels = data
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data.item()
                running_corrects += torch.sum(preds == labels).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
           
            if phase == 'train':
                writer.add_scalar('data/trainloss', epoch_loss, epoch)
                writer.add_scalar('data/trainacc', epoch_acc, epoch)
            else:
                writer.add_scalar('data/valloss', epoch_loss, epoch)
                writer.add_scalar('data/valacc', epoch_acc, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
    return model

if __name__ == '__main__':

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomSizedCrop(48),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ]),
        'val': transforms.Compose([
            transforms.Scale(64),
            transforms.CenterCrop(48),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ]),
    }

    data_dir = './Emotion_Recognition_File/train_val_data/' # 数据集所在的位置
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x]) for x in ['train', 'val']}
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=64,
                                                 shuffle=True if x=="train" else False,
                                                 num_workers=8) for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    use_gpu = torch.cuda.is_available()
    print("是否使用 GPU", use_gpu)
    modelclc = simpleconv3()
    print(modelclc)
    if use_gpu:
        modelclc = modelclc.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(modelclc.parameters(), lr=0.1, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)

    modelclc = train_model(model=modelclc,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epochs=10)  # 这里可以调节训练的轮次
    if not os.path.exists("models"):
        os.mkdir('models')
    torch.save(modelclc.state_dict(),'models/model.ckpt')