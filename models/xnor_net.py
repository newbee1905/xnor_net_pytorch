import torch.nn as nn
import torch
import torch.nn.functional as F

from components.bin import XNORConv2d
from components.block import Bottleneck

class XNORNet(nn.Module):
	def __init__(self, num_classes=10):
		super().__init__()
		self.seq = nn.Sequential(
			XNORConv2d(3, 192, kernel_size=5, stride=1, padding=2, bias=False),
			nn.BatchNorm2d(192),
			nn.ReLU(inplace=True),

			Bottleneck(192, 192, dilation=2, conv2d=XNORConv2d),
			XNORConv2d(192, 160, kernel_size=1, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(160),
			nn.ReLU(inplace=True),

			Bottleneck(160, 96, conv2d=XNORConv2d),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

			Bottleneck(96, 192, dilation=2, conv2d=XNORConv2d),
			XNORConv2d(192, 192, kernel_size=1, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(192),
			nn.ReLU(inplace=True),

			Bottleneck(192, 192, conv2d=XNORConv2d),
			nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

			XNORConv2d(192, 96, kernel_size=1, stride=1, padding=0, bias=False),
			nn.BatchNorm2d(96),
			nn.ReLU(inplace=True),
			nn.AdaptiveAvgPool2d(output_size=(1, 1)),
		)
		self.fc = nn.Linear(96, num_classes)

	def forward(self, x):
		x = self.seq(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x

