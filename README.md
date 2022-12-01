# CSE4334_FinalProject
This is an experiment to see which algorithm is the best for image classification. It turned out to be CNN.
## Deployment instructions
1. Run the **cse4334-mainproject.ipynb** notebook. It will a little over 5 hours to train. An .pt binary file should be generated as output (this step should be done if you do not have the .pt file)
2. To deploy my model copy and paste the following code into a notebook. Save the path to your image and the .pt file
```python
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import ToTensor
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image 
import PIL 

trans = torchvision.transforms.Compose([
    torchvision.transforms.RandAugment(),
    torchvision.transforms.AutoAugment(),
#     torchvision.transforms.RandomHorizontalFlip(p=0.5)
#     torchvision.transforms.RandomEqualize(p=0.5),
    torchvision.transforms.RandomVerticalFlip(p=0.5),
#     torchvision.transforms.RandomInvert(p=0.5),
    torchvision.transforms.Resize(size =(300,300)),
    ToTensor()
])
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        #Feature extraction
        self.conv= nn.Conv2d(3,64,9)
        self.conv2 = nn.Conv2d(64,128,5)
        self.conv3 = nn.Conv2d(128,256,5)
        self.conv4 = nn.Conv2d(256,512,5)
        self.conv5 = nn.Conv2d(512,1024,5)
        self.bat2d1 = nn.BatchNorm2d(64)
        self.bat2d2 = nn.BatchNorm2d(128)
        self.bat2d3 = nn.BatchNorm2d(256)
        self.bat2d4 = nn.BatchNorm2d(512)
        self.bat2d5 = nn.BatchNorm2d(1024)
        self.bat1d = nn.BatchNorm1d(196)
        self.bat1d2 = nn.BatchNorm1d(256)
#         self.bat1d3 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(p= 0.8)
        self.dropout2 = nn.Dropout(p=0.5)

        
        #Classification layer
        self.lin = nn.Linear(1024*5*5,196)
        self.lin3 = nn.Linear(196,256)
#         self.lin4 = nn.Linear(256,512)
        self.lin2 = nn.Linear(256,6)
        
    def forward(self,x):
        #First two lines are referenced
        x = F.max_pool2d(F.relu(self.conv(x)),(2,2))
        x=self.bat2d1(x)
        self.dropout(x)
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x=self.bat2d2(x)
        self.dropout(x)
        x = F.max_pool2d(F.relu(self.conv3(x)),(2,2))
        x=self.bat2d3(x)
        self.dropout(x)
        x = F.max_pool2d(F.relu(self.conv4(x)),(2,2))
        x=self.bat2d4(x)
        self.dropout(x)
        x = F.max_pool2d(F.relu(self.conv5(x)),(2,2))
        x=self.bat2d5(x)
        self.dropout(x)
#         print(x.shape)
        #Flattening tensors
        x = torch.flatten(x,1)
#         print(x.shape)
        x = F.relu(self.lin(x))
        self.bat1d(x)
        x= F.relu(self.lin3(x))
        self.bat1d2(x)
        self.dropout2(x)
#         x= F.relu(self.lin4(x))
#         self.bat1d3(x)
#         self.dropout2(x)
        x = F.softmax(self.lin2(x),dim = 1)
        return x


#https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html#save-on-gpu-load-on-cpu
PATH = '{path to state_dict.pt}'
# device = torch.device('cpu')
model = Model()
parallel_model = torch.nn.DataParallel(model)
parallel_model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))
parallel_model.eval()

#https://matplotlib.org/stable/tutorials/introductory/images.html (Plotting image using matplot lib)
#https://www.geeksforgeeks.org/python-pil-image-save-method/ (saving image using PIL)
imagep = mpimg.imread('{path to your image}')
image = Image.open(r"{path to your image}") 
imageplot = plt.imshow(imagep)
plt.show()

Labels = {0: 'cheetah', 1: 'fox', 2: 'hyena', 3: 'lion', 4: 'tiger',5:'wolf'}
parallel_model.eval()
itest = trans(image)
itest = itest.unsqueeze(0)
output = parallel_model(itest)
maxele,maxindx = torch.max(output,1)
#Predicted label

print(Labels[maxindx.item()])
```
That should be it! Have fun classifying animals
