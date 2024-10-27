""" KZ todo:
    1. Loss function (volume is approximately (y_pred-y_true)*y_true?)
    2. Maybe introduce an alternative stop criterion where it stops early if loss increases again
    3. Enable extending the k range to larger than 1 and a bit smaller. 
    4. Add the plotting visualization where it plots the triangle volume
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from nn_modules import SeparableApproximation

class SepBKNN:
    def __init__(self, num_terms=2, symm_kind=3, loss_func = 'mse', sub_arch='MLP', device=None):

        self.device = device
        self.model = SeparableApproximation(num_terms=num_terms, symm_kind=1, sub_arch=sub_arch).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, factor=0.3)
        self.loss_func = loss_func
        if loss_func == 'mse':
            self.criterion = nn.MSELoss()

    def train(self, train_loader, val_loader, epochs=300):
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                if self.loss_func == 'mse':
                    loss = self.criterion(outputs, batch_y)
                else:
                    raise TypeError('Loss function not defined')
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            epoch_loss /= len(train_loader)
            train_losses.append(epoch_loss)
            
            if epoch % 10 == 0:
                val_loss = self.evaluate(val_loader)
                val_losses.append(val_loss)
                
                self.scheduler.step(val_loss)
            
            if epoch % 1 == 0:
                print(f"Epoch {epoch}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        return train_losses, val_losses

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                total_loss += self.criterion(outputs, batch_y).item() * batch_y.size(0)
        return total_loss / len(data_loader.dataset)

    def test_mse(self, test_loader):
        mse = self.evaluate(test_loader)
        print(f"Test Mean Squared Error: {mse:.6f}")
        return mse

    def test_results(self, data_loader, train_losses, val_losses):
        all_y = []
        all_y_pred = []
        self.model.eval()
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                y_pred = self.model(batch_X)
                all_y.extend(batch_y.cpu().numpy())
                all_y_pred.extend(y_pred.cpu().numpy())

        all_y = np.array(all_y)
        all_y_pred = np.array(all_y_pred)

        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.scatter(all_y, all_y_pred, alpha=0.1)
        plt.plot([all_y.min(), all_y.max()], [all_y.min(), all_y.max()], 'r--')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('True vs Predicted')

        plt.subplot(132)
        plt.scatter(all_y, all_y_pred - all_y, alpha=0.1)
        plt.xlabel('True Values')
        plt.ylabel('Residuals')
        plt.title('Residuals  (Difference)')

        plt.subplot(133)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig('restuls.pdf')
        plt.show()