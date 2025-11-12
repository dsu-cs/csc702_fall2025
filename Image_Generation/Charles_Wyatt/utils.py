import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class PairedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, input_transform=None, target_transform=None):
        self.root_dir = root_dir
        self.image_files = sorted(os.listdir(root_dir))
        self.transform = transform
        self.input_transform = input_transform or transform
        self.target_transform = target_transform or transform

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_files[idx])
        img = Image.open(image_path).convert("RGB")

        input_image = self.generate_input_variant(img)
        target_image = img

        if self.input_transform:
            input_image = self.input_transform(input_image)
        if self.target_transform:
            target_image = self.target_transform(target_image)

        return input_image, target_image

    def __len__(self):
        return len(self.image_files)
    
    # override this depending on project choice
    def generate_input_variant(self, img):
        arr = np.array(img)
        # simple corruption example of gaussian noise
        # we should look at updating this if we use it
        noise = np.random.normal(0, 25, arr.shape).astype(np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    
'''
# if we choose corruption project
from utils import PairedImageDataset
import torchvision.transforms.functional as TF
import cv2
import numpy as np

class CorruptionDataset(PairedImageDataset):
'''

'''
# if we choose coloring sketches project
from utils import PairedImageDataset
import cv2
import numpy as np

class SketchDataset(PairedImageDataset):
    def generate_input_variant(self, img):
        arr = np.array(img)
        edges = cv2.Canny(arr, 80, 150)
        edges_rgb = np.stack([edges]*3, axis=-1)
        return Image.fromarray(edges_rgb)
'''



# simple unet model architecture 
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)

        # down sampling
        for feature in features:
            self.downs.append(self._block(in_channels, feature))
            in_channels = feature

        # bottleneck
        self.bottelneck = self._block(features[-1], features[-1]*2)

        # upsampling
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(self._block(feature*2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottelneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](x)

        return self.final_conv(x)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    

# training loop
def train_model(model, dataloader, device, epochs=5, lr=1e-4):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()

    for epoch in range(epochs):
        for i, (x,y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print(f'epoch [{epoch+1}/{epochs}] batch [{i}/{len(dataloader)}] loss: {loss.item():.4f}')
        save_image(preds[:4], f'outputs/epoch_{epoch+1}.png')



# visualization & evaluation
def show_batch(inputs, outputs, targets):
    grid = torch.cat([inputs, outputs, targets])
    grid = make_grid(grid, nrow=3, normalize=True)
    plt.imshow(grid.permute(1,2,0).cpu())
    plt.axis('off')
    plt.show()