import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

class image_dataset(torch.utils.data.Dataset):
    def __init__(self,image_folder):
        self.image_folder = image_folder
        self.images = os.listdir(image_folder)
    
    def __getitem__(self, idx):
        image_file = self.images[idx]

        image = Image.open((self.image_folder + image_file))
        image = np.array(image)
        image = np.float32(image)
        image = image/255
        image = torch.from_numpy(image)
        label=float(image_file.split("_")[0])
        target = torch.tensor(label)
        return image, target
    
    def __len__(self):
        return len(self.images)
        
dataset = image_dataset("FCNN\dataset")
test_size = 0.15
test_amount = int(dataset.__len__() * test_size)
train_set, test_set = torch.utils.data.random_split(dataset, [dataset.__len__() - test_amount, test_amount])


train_dataloader = torch.utils.data.DataLoader(train_set, batch_size = 1, shuffle = True)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size = 1, shuffle = True)


class FCNN(torch.nn.Module):

    def __init__(self):
        super(FCNN,self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels=8, kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.pool = nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels=16, kernel_size = (3,3), stride = (1,1), padding = (1,1))
        self.ad_pool = nn.AdaptiveAvgPool2d((8,8))
        self.fc1 = nn.Linear(64*16,16)
        self.fc2 = nn.Linear(16,1)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.ad_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = torch.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    

model = FCNN()
PATH = "FCNN\model_params\model_state_dict"
loss_fn = nn.BCEWithLogitsLoss()
if torch.cuda.is_available():
    model = model.cuda()
    loss_fn = loss_fn.cuda()

optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.0000001)

if __name__ == "__main__":
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_save = checkpoint['epoch']
    loss = checkpoint['loss']

    print(f"Loading modeL at epoch: {epoch_save} and avg_loss = {loss} ")

    # epoch_save=0

    epoch_step = 500
    count = 0
    model.train()

    avg_loss = 0
    for epoch in range(epoch_save+1,epoch_save + epoch_step):
        for images, targets in train_dataloader:
            images = images.to('cuda')
            targets = targets.to("cuda")

            output = model(images)

            loss = loss_fn(output, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        avg_loss = avg_loss/len(train_dataloader)
        print(f"Epoch: {epoch}, AVG Loss: {avg_loss}")
        if(epoch%10==0):
            print(f"|------------Saving modeL at epoch: {epoch} and avg loss = {avg_loss}------------| ")
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        }, PATH)




