import os
from torch.utils.data import DataLoader
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage.segmentation import slic
from lime import lime_image
from pdb import set_trace
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
class FoodDataset(Dataset):
    def __init__(self, paths, labels, mode):
        # mode: 'train' or 'eval'
        #mode是train就用train的transform
        
        
        #paths是每一个图片的名字
        #labels是每一个图片对应的label        
        self.paths = paths
        self.labels = labels
        
        trainTransform = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
        evalTransform = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.ToTensor(),
        ])        
        self.transform = trainTransform if mode == 'train' else evalTransform
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        X = Image.open(self.paths[index])
        X = self.transform(X)
        Y = self.labels[index]
        return X, Y

#这个函数是为了方便的取出指定index的图片
    def getbatch(self, indices):
        images = []
        labels = []
        for index in indices:
          image, label = self.__getitem__(index)
          images.append(image)
          labels.append(label)
        return torch.stack(images), torch.tensor(labels)

    
# 给一个文件夹的名字，可以返回他下面图片的名字和labels的名字
def get_paths_labels(path):
    imgnames = os.listdir(path)
    imgnames.sort()
    imgpaths = []
    labels = []
    for name in imgnames:
        imgpaths.append(os.path.join(path, name))
        labels.append(int(name.split('_')[0]))
    return imgpaths, labels
    
train_paths, train_labels = get_paths_labels('./food-11/training')
train_set = FoodDataset(train_paths, train_labels, mode='train')

val_paths, val_labels = get_paths_labels('./food-11/validation')
val_set = FoodDataset(val_paths, val_labels, mode='eval')

batch_size = 128

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


model=torch.load('ckpt2.model')
model.eval()
for i, data in enumerate(train_loader):
	pred = model(data[0])
	val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
	print(val_acc/train_set.__len__())
def normalize(image):
  return (image - image.min()) / (image.max() - image.min())


model.eval()

img_indices=[1,2,3,4]
x, y= train_set.getbatch(img_indices)

x.requires_grad_()
y_pred=model(x)
loss_func = torch.nn.CrossEntropyLoss()
loss=loss_func(y_pred,y)
loss.backward()

saliencies = x.grad.abs().detach().cpu()
saliencies = torch.stack([normalize(item) for item in saliencies])

x=x.detach()
fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
for row, target in enumerate([x, saliencies]):
  for column, img in enumerate(target):
    axs[row][column].imshow(img.permute(1, 2, 0).numpy())
#matplolib 的最后一维是图片的三个通道，但是pytorch中我们的X数据第一维是，所以转换一下维度才能打印正常
plt.show()
plt.close()


model=torch.load('ckpt_best.model')
model.eval()

cnnid=0
filterid=0

img_indices=[0,1,2,3]
x, y=train_set.getbatch(img_indices)
model.eval()
def hook(model,input,output):
    global layer_activations
    layer_activations=output
hook_handle = model.cnn[cnnid].register_forward_hook(hook)
model(x)
x=x.detach()
filter_activations=layer_activations[:, filterid, :, :].detach()
hook_handle.remove()

fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
for i, img in enumerate(images):
  axs[0][i].imshow(img.permute(1, 2, 0))
for i, img in enumerate(filter_activations):
  axs[1][i].imshow(normalize(img))
plt.show()
plt.close()



model=torch.load('ckpt_best.model')
model.eval()

cnnid=3
filterid=0

img_indices=[0,1,2,3]
x, y=train_set.getbatch(img_indices)
model.eval()
def hook(model,input,output):
    global layer_activations
    layer_activations=output
hook_handle = model.cnn[cnnid].register_forward_hook(hook)
model(x)
x=x.detach()
filter_activations=layer_activations[:, filterid, :, :].detach()
hook_handle.remove()

fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
for i, img in enumerate(images):
  axs[0][i].imshow(img.permute(1, 2, 0))
for i, img in enumerate(filter_activations):
  axs[1][i].imshow(normalize(img))
plt.show()
plt.close()

model=torch.load('ckpt_best.model')
model.eval()

cnnid=4
filterid=0

img_indices=[0,1,2,3]
x, y=train_set.getbatch(img_indices)
model.eval()
def hook(model,input,output):
    global layer_activations
    layer_activations=output
hook_handle = model.cnn[cnnid].register_forward_hook(hook)
model(x)
x=x.detach()
filter_activations=layer_activations[:, filterid, :, :].detach()
hook_handle.remove()

fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
for i, img in enumerate(images):
  axs[0][i].imshow(img.permute(1, 2, 0))
for i, img in enumerate(filter_activations):
  axs[1][i].imshow(normalize(img))
plt.show()
plt.close()

model=torch.load('ckpt_best.model')
model.eval()

cnnid=8
filterid=0

img_indices=[0,1,2,3]
x, y=train_set.getbatch(img_indices)
model.eval()
def hook(model,input,output):
    global layer_activations
    layer_activations=output
hook_handle = model.cnn[cnnid].register_forward_hook(hook)
model(x)
x=x.detach()
filter_activations=layer_activations[:, filterid, :, :].detach()
hook_handle.remove()

fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
for i, img in enumerate(images):
  axs[0][i].imshow(img.permute(1, 2, 0))
for i, img in enumerate(filter_activations):
  axs[1][i].imshow(normalize(img))
plt.show()
plt.close()

model=torch.load('ckpt_best.model')
model.eval()

cnnid=12
filterid=0

img_indices=[0,1,2,3]
x, y=train_set.getbatch(img_indices)
model.eval()
def hook(model,input,output):
    global layer_activations
    layer_activations=output
hook_handle = model.cnn[cnnid].register_forward_hook(hook)
model(x)
x=x.detach()
filter_activations=layer_activations[:, filterid, :, :].detach()
hook_handle.remove()

fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
for i, img in enumerate(images):
  axs[0][i].imshow(img.permute(1, 2, 0))
for i, img in enumerate(filter_activations):
  axs[1][i].imshow(normalize(img))
plt.show()
plt.close()

model=torch.load('ckpt_best.model')
model.eval()

cnnid=16
filterid=0

img_indices=[0,1,2,3]
x, y=train_set.getbatch(img_indices)
model.eval()
def hook(model,input,output):
    global layer_activations
    layer_activations=output
hook_handle = model.cnn[cnnid].register_forward_hook(hook)
model(x)
x=x.detach()
filter_activations=layer_activations[:, filterid, :, :].detach()
hook_handle.remove()

fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
for i, img in enumerate(images):
  axs[0][i].imshow(img.permute(1, 2, 0))
for i, img in enumerate(filter_activations):
  axs[1][i].imshow(normalize(img))
plt.show()
plt.close()

model=torch.load('ckpt_best.model')
model.eval()

cnnid=0
filterid=0

img_indices=[0,1,2,3]
x, y=train_set.getbatch(img_indices)
model.eval()
def hook(model,input,output):
    global layer_activations
    layer_activations=output
hook_handle = model.cnn[cnnid].register_forward_hook(hook)
model(x)
x=x.detach()
filter_activations=layer_activations[:, filterid, :, :].detach()
#torch.Size([4, 128, 128]))

x.requires_grad_()
optimizer = Adam([x], lr=1)
for iter in range(100):
    optimizer.zero_grad()
    model(x)
    objective = -layer_activations[:, filterid, :, :].sum()
    objective.backward()
    optimizer.step()
filter_visualization = x.detach().cpu().squeeze()[0]
#torch.Size([4, 3, 128, 128])只取第一张

hook_handle.remove()

plt.imshow(normalize(filter_visualization.permute(1, 2, 0)))
plt.show()
plt.close()



def predict(input):
    # input: numpy array, (batches, height, width, channels)                                                                                                                                                     
    
    model.eval()                                                                                                                                                             
    input = torch.FloatTensor(input).permute(0, 3, 1, 2)                                                                                                            
    # 需要先將 input 轉成 pytorch tensor，且符合 pytorch 習慣的 dimension 定義
    # 也就是 (batches, channels, height, width)

    output = model(input)                                                                                                                                             
    return output.detach().cpu().numpy()                                                                                                                              
                                                                                                                                                                             
def segmentation(input):
    # 利用 skimage 提供的 segmentation 將圖片分成 100 塊                                                                                                                                      
    return slic(input, n_segments=100, compactness=1, sigma=1)                                                                                                              
                                                                                                                                                                             
img_indices = [0,1,2,3]
images, labels = train_set.getbatch(img_indices)
fig, axs = plt.subplots(1, 4, figsize=(15, 8))                                                                                                                                                                 
np.random.seed(16)                                                                                                                                                       
# 讓實驗 reproducible
for idx, (image, label) in enumerate(zip(images.permute(0, 2, 3, 1).numpy(), labels)):                                                                                                                                             
    x = image.astype(np.double)
    # lime 這個套件要吃 numpy array

    explainer = lime_image.LimeImageExplainer()                                                                                                                              
    explaination = explainer.explain_instance(image=x, classifier_fn=predict, segmentation_fn=segmentation)
    # 基本上只要提供給 lime explainer 兩個關鍵的 function，事情就結束了
    # classifier_fn 定義圖片如何經過 model 得到 prediction
    # segmentation_fn 定義如何把圖片做 segmentation
    # doc: https://lime-ml.readthedocs.io/en/latest/lime.html?highlight=explain_instance#lime.lime_image.LimeImageExplainer.explain_instance

    lime_img, mask = explaination.get_image_and_mask(                                                                                                                         
                                label=label.item(),                                                                                                                           
                                positive_only=False,                                                                                                                         
                                hide_rest=False,                                                                                                                             
                                num_features=11,                                                                                                                              
                                min_weight=0.05                                                                                                                              
                            )
    # 把 explainer 解釋的結果轉成圖片
    # doc: https://lime-ml.readthedocs.io/en/latest/lime.html?highlight=get_image_and_mask#lime.lime_image.ImageExplanation.get_image_and_mask
    
    axs[idx].imshow(lime_img)

plt.show()
plt.close()


