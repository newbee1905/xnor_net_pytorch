import torch
from torch import autograd, nn
import torch.nn.functional as F
from torchvision.models import resnet18

class BinActive(autograd.Function):
	@staticmethod
	def forward(self, input):
		self.save_for_backward(input)
		mean = input.abs().mean()
		input = input.sign()
		return input, mean

	@staticmethod
	def backward(self, grad_output, mean):
		input,  = self.saved_tensors
		grad_input = grad_output.clone()

		grad_input[input.ge(1)] = 0
		grad_input[input.le(-1)] = 0

		return grad_input

class XNORConv2d(nn.Conv2d):
	def forward(self, x):
		w_bin, w_mean = BinActive.apply(self.weight)
		x_bin, _ = BinActive.apply(x)

		res = F.conv2d(x_bin, w_bin, self.bias, self.stride, self.padding, self.dilation, self.groups)

		return res

class XNORSConv2d(nn.Conv2d):
	def forward(self, x):
		w_bin, w_mean = BinActive.apply(self.weight)
		x_bin, _ = BinActive.apply(x)

		res = F.conv2d(x_bin, w_bin, self.bias, self.stride, self.padding, self.dilation, self.groups)

		return res * w_mean

class GXNORActive(autograd.Function):
	@staticmethod
	def forward(ctx, x, r, a):
		ctx.save_for_backward(x, r, a)
		output = torch.where(
			x > r, 1.0, torch.where(x < -r, -1.0, 0.0)
		)
		return output

	@staticmethod
	def backward(ctx, grad_output):
		x, r, a = ctx.saved_tensors
		
		# Gradient w.r.t. input
		grad_input = torch.where(
			(r - a <= torch.abs(x)) & (torch.abs(x) <= r + a),
			grad_output / (2 * a),
			torch.zeros_like(grad_output)
		)
		
		# Gradient w.r.t. r
		grad_r = torch.where(
			(-r < x) & (x < r),
			-grad_output,
			torch.zeros_like(grad_output)
		).sum()

		# Gradient w.r.t. a
		grad_a = torch.where(
			(r - a <= torch.abs(x)) & (torch.abs(x) <= r + a),
			-grad_output / (2 * a**2),
			torch.zeros_like(grad_output)
		).sum()

		return grad_input, grad_r, grad_a


class GXNORConv2d(nn.Conv2d):
	def __init__(self, *args, r=0.5, a=0.1, **kwargs):
		super().__init__(*args, **kwargs)
		self.r = nn.Parameter(torch.tensor(r, requires_grad=True))
		self.a = nn.Parameter(torch.tensor(a, requires_grad=True))

	def forward(self, x):
		w_bin = GXNORActive.apply(self.weight, self.r, self.a)
		x_bin = GXNORActive.apply(x, self.r, self.a)

		res = F.conv2d(x_bin, w_bin, self.bias, self.stride, self.padding, self.dilation, self.groups)

		scaling_factor = torch.mean(torch.abs(self.weight))

		return res * scaling_factor

class TestBinActive(autograd.Function):
	@staticmethod
	def forward(self, x, r):
		self.save_for_backward(x, r)
		mean = x.abs().mean()
		x = (x - r).sign()
		return x, mean

	@staticmethod
	def backward(self, grad_x, _):
		x, r = self.saved_tensors

		grad_input = grad_x.clone()

		grad_input[x.ge(r + 1)] = 0
		grad_input[x.le(r - 1)] = 0

		grad_r = -grad_x.clone()

		grad_r[x.ge(r + 1)] = 0
		grad_r[x.le(r - 1)] = 0

		return grad_input, grad_r

class TestXNORConv2d(nn.Conv2d):
	def __init__(self, *args, r=0.5, **kwargs):
		super().__init__(*args, **kwargs)
		self.r = nn.Parameter(torch.tensor(r, requires_grad=True))

	def forward(self, x):
		w_bin, w_mean = TestBinActive.apply(self.weight, self.r)
		x_bin, x_mean = TestBinActive.apply(x, self.r)

		res = F.conv2d(x_bin, w_bin, self.bias, self.stride, self.padding, self.dilation, self.groups)

		return res * w_mean

class Test2BinActive(autograd.Function):
	@staticmethod
	def forward(self, x, r1, r2):
		self.save_for_backward(x, r1, r2)

		x_int = x.floor()
		x_float = x - x_int

		mean = x_float.abs().mean()
		x_int = (x_int - r1).sign()
		x_float = (x_float - r2).sign()

		return x_int, x_float, mean

	@staticmethod
	def backward(self, grad_x_int, grad_x_float, _):
		x, r = self.saved_tensors

		grad_input = grad_x_int.clone()

		grad_input[x.ge(r1 + 1)] = 0
		grad_input[x.le(r1 - 1)] = 0

		grad_r1 = -grad_x_int.clone()

		grad_r1[x.ge(r1 + 1)] = 0
		grad_r1[x.le(r1 - 1)] = 0

		grad_r2 = -grad_x_float.clone()

		grad_r2[x.ge(r2 + 1)] = 0
		grad_r2[x.le(r2 - 1)] = 0

		return grad_input, grad_r1, grad_r2

class Test2XNORConv2d(nn.Conv2d):
	def __init__(self, *args, r1=0.0, r2=0.0, **kwargs):
		super().__init__(*args, **kwargs)
		self.r1 = nn.Parameter(torch.tensor(r1, requires_grad=True))
		self.r2 = nn.Parameter(torch.tensor(r2, requires_grad=True))

	def forward(self, x):
		x_bin, w_int_bin, w_float_bin, w_float_mean = Test2XNORActive.apply(self.weight, self.r1, self.r2)
		w_bin, x_int_bin, x_float_bin, x_float_mean = Test2XNORActive.apply(x, self.r1, self.r2)

		res_int = F.conv2d(x_int_bin, w_int_bin, self.bias, self.stride, self.padding, self.dilation, self.groups)
		res_float = F.conv2d(x_float_bin, w_float_bin, self.bias, self.stride, self.padding, self.dilation, self.groups)

		return res_int + w_float_mean * res_float
