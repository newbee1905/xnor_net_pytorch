import pytorch_lightning as L

import torchmetrics
from torch import optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch

class TrainerLightningModule(L.LightningModule):
	def __init__(self, model, lr=1e-3, t_max=200, weight_decay=1e-4, num_classes=10, top_k=5):
		super().__init__()
		self.model = model
		self.criterion = nn.CrossEntropyLoss()
		self.lr = lr
		self.t_max = t_max
		self.weight_decay = weight_decay
		self.top_k = top_k

		# Metrics
		self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
		self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
		self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="macro")
		self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="macro")
		self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")
		self.val_topk_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=top_k)

	def forward(self, x):
		assert isinstance(x, torch.Tensor), f"Inputs expected to be torch.Tensor, got {type(x)}"
		return self.model(x)

	def training_step(self, batch, batch_idx):
		inputs, labels = batch
		outputs = self.model(inputs)
		loss = self.criterion(outputs, labels)

		# Update metrics
		acc = self.train_acc(outputs, labels)
		self.log('train/loss', loss, on_epoch=True, prog_bar=True)
		self.log('train/accuracy', acc, on_epoch=True, prog_bar=True)

		return loss

	def validation_step(self, batch, batch_idx):
		inputs, labels = batch
		outputs = self.model(inputs)
		loss = self.criterion(outputs, labels)
		
		# Update metrics
		acc = self.val_acc(outputs, labels)
		acc_top_k = self.val_topk_acc(outputs, labels)
		precision = self.val_precision(outputs, labels)
		recall = self.val_recall(outputs, labels)
		f1 = self.val_f1(outputs, labels)

		# Log metrics
		self.log('val/loss', loss, on_epoch=True, prog_bar=True)
		self.log('val/accuracy', acc, on_epoch=True, prog_bar=True)
		self.log(f'val/accuracy_top_{self.top_k}', acc_top_k, on_epoch=True, prog_bar=True)
		self.log('val/precision', precision, on_epoch=True)
		self.log('val/recall', recall, on_epoch=True)
		self.log('val/f1', f1, on_epoch=True)

	def test_step(self, batch, batch_idx):
		inputs, labels = batch
		outputs = self.model(inputs)
		loss = self.criterion(outputs, labels)

		acc = self.val_acc(outputs, labels)
		acc_top_k = self.val_topk_acc(outputs, labels)
		self.log('test/loss', loss, on_epoch=True)
		self.log('test/accuracy', acc, on_epoch=True)
		self.log(f'test/accuracy_top_{self.top_k}', acc_top_k, on_epoch=True)

	def configure_optimizers(self):
		optimizer = optim.AdamW(
			self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
		)
		scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.t_max)
		return [optimizer], [scheduler]
