from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim


class BaseModel(nn.Module):
    def __init__(self, lr, weight_decay, optimizer_type, momentum, device):
        super(BaseModel, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.device = device

    def set_optimizer(self):

        if self.optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_type == "adam":
            self.optimizer = optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_type == "adamw":
            self.optimizer = optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer_type == "rmsprop":
            self.optimizer = optim.RMSprop(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            raise NotImplementedError(
                f"{self.optimizer_type} optimizer not implemented"
            )

    def train_model(self, train_loader, val_loader=None):
        self.train()
        total_loss = 0
        train_acc = 0
        for _, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            logits = self.forward(data)
            loss = self.criterion(logits, target)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_acc += (preds == target).float().mean().item()
            loss.backward()
            self.optimizer.step()

        total_loss /= len(train_loader)
        train_acc /= len(train_loader)

        val_loss = None
        if val_loader is not None:
            val_loss, val_acc = self.validate(val_loader)

        return total_loss, val_loss, train_acc, val_acc

    def validate(self, val_loader):
        self.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for _, (data, target) in enumerate(val_loader):
                data, target = data.to(self.device), target.to(self.device)
                logits = self.forward(data)
                loss = self.criterion(logits, target)
                preds = logits.argmax(dim=1)
                val_acc += (preds == target).float().mean().item()
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        return val_loss, val_acc

    def fit(
        self, train_loader, nr_epochs, val_loader=None, verbose=True, print_interval=100
    ):

        train_losses, train_accuracy, val_losses, val_accuracy = ([] for i in range(4))

        for epoch in range(nr_epochs):
            train_loss, val_loss, train_acc, val_acc = self.train_model(
                train_loader, val_loader
            )  # Training and validation in one call
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracy.append(train_acc)
            val_accuracy.append(val_acc)

            if verbose and (epoch + 1) % print_interval == 0:
                message = f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}"
                if val_loss is not None:
                    print(message + f", Val Loss: {val_loss:.4f}")
                else:
                    print(message)
        return train_losses, val_losses, train_accuracy, val_accuracy

    def test(self, test_loader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                logits = self.forward(data)
                predictions = logits.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        test_acc = correct / total
        return test_acc
