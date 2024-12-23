import torch.nn as nn

class Bottleneck(nn.Module):
	def __init__(self, c_in, c_out, dilation=1, expansion=4, conv2d = nn.Conv2d):
		super().__init__()
		c_mid = c_out // expansion

		self.reduce = conv2d(c_in, c_mid, kernel_size=1, stride=1, padding=0, bias=False)
		self.reduce_bn = nn.BatchNorm2d(c_mid)

		self.conv = conv2d(c_mid, c_mid, kernel_size=3, stride=1, padding=dilation, dilation=dilation, padding_mode='zeros', bias=False)
		self.conv_bn = nn.BatchNorm2d(c_mid)

		self.expand = conv2d(c_mid, c_out, kernel_size=1, stride=1, padding=0, bias=False)
		self.expand_bn = nn.BatchNorm2d(c_out)

		self.relu = nn.ReLU(inplace=True)

		self.downsample = None
		if c_in != c_out:
			self.downsample = nn.Sequential(
				nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, bias=False),
				nn.BatchNorm2d(c_out)
			)

	def forward(self, x):
		identity = x
		
		x = self.reduce(x)
		x = self.reduce_bn(x)
		x = self.relu(x)

		x = self.conv(x)
		x = self.conv_bn(x)
		x = self.relu(x)

		x = self.expand(x)
		x = self.expand_bn(x)

		if self.downsample:
			identity = self.downsample(identity)

		x += identity
		return self.relu(x)
