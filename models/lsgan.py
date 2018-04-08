import torch
import torch.nn as nn
import torch.nn.parallel

class LSGAN_D(nn.Module):
    def __init__(self):
        super(LSGAN_D, self).__init__()
        layers = [
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2,bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2,bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        ]
        self.main = nn.Sequential(*layers)
        self.linear = nn.Linear(512*6*6, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(output.size(0), -1)
        output = self.linear(output)
        return output

class LSGAN_G(nn.Module):
    def __init__(self):
        super(LSGAN_G, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, bias=False),
        ]
        self.main = nn.Sequential(*layers)

        self.linear = nn.Sequential(
            nn.Linear(1024, 7*7*256),
            nn.BatchNorm2d(7*7*256),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        output = self.linear(input)
        output = output.view(-1,256,7,7)
        output = self.main(output)
        return output 
