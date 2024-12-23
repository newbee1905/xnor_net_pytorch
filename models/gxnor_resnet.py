import torch
from torch import autograd, nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torch.ao.quantization import QuantStub, DeQuantStub

from components.bin import GXNORConv2d

class GXNORResNet18(nn.Module):
	def __init__(self, num_classes=10, model=None, quantize=False):
		super().__init__()
		if model is None:
			original_model = resnet18(num_classes=num_classes)
		else:
			original_model = model

		self.model = resnet18(num_classes=num_classes)
		self.bin_conv2d = GXNORConv2d 
		self._binarize_layers(original_model)

		self.quantize = quantize
		if self.quantize:
			self._prepare_for_quantisation()

	def _binarize_layers(self, original_model):
		conv_layers = []
		for name, module in original_model.named_modules():
			if isinstance(module, nn.Conv2d):
				conv_layers.append((name, module))

		for name, module in conv_layers:
			name_parts = name.split('.')
			current_module = self.model
			for part in name_parts[:-1]:
				current_module = getattr(current_module, part)

			new_module = self.bin_conv2d(
				module.in_channels,
				module.out_channels,
				module.kernel_size,
				stride=module.stride,
				padding=module.padding,
				bias=module.bias is not None,
			)

			new_module.weight.data = module.weight.data.clone()
			if module.bias is not None:
				new_module.bias.data = module.bias.data.clone()

			setattr(current_module, name_parts[-1], new_module)

	def forward(self, x):
		if self.quantize:
			x = self.quant(x)
			x = self.model(x)
			x = self.dequant(x)
		else:
			x = self.model(x)
		return x

	def _prepare_for_quantisation(self):
		self.quant = quant.QuantStub()
		self.dequant = quant.DeQuantStub()

		for name, module in self.model.named_modules():
			if isinstance(module, nn.ReLU):
				setattr(self.model, name, nn.ReLU6(inplace=True))

	def fuse_model(self):
		for name, module in self.model.named_modules():
			if isinstance(module, nn.Sequential):
				quant.fuse_modules(module, ['0', '1', '2'], inplace=True)
