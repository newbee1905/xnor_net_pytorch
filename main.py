import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import v2
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.ao.quantization as quantization
from thop import profile, clever_format
import torch.quantization as quant
import torch.nn.functional as F

import pytorch_lightning as L
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from models.xnor_net import XNORNet
from models.xnor_resnet import XNORResNet18
from models.gxnor_resnet import GXNORResNet18
from models.gxnor_net import GXNORNet
from models.xnor_mobilenetv3 import XNORMobileNetV3
from models.basic_net import Net
from trainer import TrainerLightningModule

from torchvision.datasets import CIFAR10

from pathlib import Path

EPOCHS=50

train_transform = v2.Compose([
	v2.RandomCrop(size=(32, 32), padding=4),
	v2.RandomHorizontalFlip(p=0.5),
	transforms.ToTensor(),
	v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


val_transform = v2.Compose([
	v2.Resize(size=(32, 32)),
	transforms.ToTensor(),
	v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_ds = CIFAR10(root='./data', train=True, download=True, transform=train_transform)
val_ds = CIFAR10(root='./data', train=False, download=True, transform=val_transform)
top_k = min(len(val_ds.classes), 5)

train_dl = DataLoader(dataset=train_ds, batch_size=32, shuffle=True, num_workers=10, pin_memory=True)
val_dl = DataLoader(dataset=val_ds, batch_size=32, shuffle=True, num_workers=10, pin_memory=True)

logger = TensorBoardLogger(".", version="test_xnor_resnet_1")
xnor_resnet_trainer = L.Trainer(
	logger=logger,
	max_epochs=EPOCHS,
	callbacks=[
		EarlyStopping(monitor="val/accuracy", min_delta=0.001, patience=20, mode="max"),
	],
	devices=1, accelerator="gpu"
)
xnor_resnet = XNORResNet18(num_classes=10)

xnor_resnet_module = TrainerLightningModule(xnor_resnet, num_classes=10)
xnor_resnet_trainer.fit(xnor_resnet_module, train_dl, val_dl)

# logger = TensorBoardLogger(".", version="basic_net_1")
# net_trainer = L.Trainer(
# 	logger=logger,
# 	max_epochs=EPOCHS,
# 	callbacks=[
# 		EarlyStopping(monitor="val/accuracy", min_delta=0.001, patience=20, mode="max"),
# 	],
# 	devices=1, accelerator="gpu"
# )
# net = Net(num_classes=10)
#
# net_module = TrainerLightningModule(net, num_classes=10)
# net_trainer.fit(net_module, train_dl, val_dl)

# logger = TensorBoardLogger(".", version="gxnor_basic_net_1")
# gxnor_net_trainer = L.Trainer(
# 	logger=logger,
# 	max_epochs=EPOCHS,
# 	callbacks=[
# 		EarlyStopping(monitor="val/accuracy", min_delta=0.001, patience=20, mode="max"),
# 	],
# 	devices=1, accelerator="gpu"
# )
# gxnor_net = GXNORNet(num_classes=10)
#
# gxnor_net_module = TrainerLightningModule(gxnor_net, num_classes=10)
# gxnor_net_trainer.fit(gxnor_net_module, train_dl, val_dl)
#
# logger = TensorBoardLogger(".", version="gxnor_basic_net_1")

# logger = TensorBoardLogger(".", version="xnor_basic_net_1")
# xnor_net_trainer = L.Trainer(
# 	logger=logger,
# 	max_epochs=EPOCHS,
# 	callbacks=[
# 		EarlyStopping(monitor="val/accuracy", min_delta=0.001, patience=20, mode="max"),
# 	],
# 	devices=1, accelerator="gpu"
# )
# xnor_net = XNORNet(num_classes=10)
# 
# xnor_net_module = TrainerLightningModule(xnor_net, num_classes=10)
# xnor_net_trainer.fit(xnor_net_module, train_dl, val_dl)
