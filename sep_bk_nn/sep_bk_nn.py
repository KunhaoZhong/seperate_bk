""" KZ todo:
    1. Loss function (volume is approximately (y_pred-y_true)*y_true?)
    2. Maybe introduce an alternative stop criterion where it stops early if loss increases again
    
    4. Add the plotting visualization where it plots the triangle volume
    5. Add wandb support
    6. maybe a better plotting save? now with too much points and the resulting pdf is massive
    
    7. For the inner product calculation, now it relies on two interpolation. This is mainly for
        future integration with other numerical code. Should test the accuracy of the interpolation
        with a normal scipy's integration method with exact functions. 
        
    8. Maybe I should create a Latin-hyper cube sort of k-vals??

    Done:
    3. Enable extending the k range to larger than 1 and a bit smaller. 
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import interpolate
from scipy import integrate
import os

from .nn_modules import SeparableApproximation
from .bk_functions import create_bk_dataset
from .bk_utils import Delta_fNL_scale_w_interp
from .bk_utils import plot_3d_data


class SepBKNN:
    def __init__(self, num_terms, symm_kind, loss_func = 'mse', sub_arch='MLP', device=None):

        self.device = device
        self.model = SeparableApproximation(num_terms=num_terms, symm_kind=symm_kind, sub_arch=sub_arch).to(self.device)
        # print(self.model)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, factor=0.3)
        self.loss_func = loss_func
        self.num_terms = num_terms
        self.symm_kind = symm_kind
        if loss_func == 'mse':
            self.criterion = nn.MSELoss()
            
    def save_checkpoint(self, val_loss, epoch, checkpoint_dir='./models/checkpoint'):
        """Save model checkpoint"""
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            # 'func_name': func_name,
            # 'func_args': func_args,
            'num_terms': self.num_terms,
            'symm_kind': self.symm_kind
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f'best_model.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f'Model checkpoint saved: val_loss: {val_loss:.6f}')
    
    def load_checkpoint(self, checkpoint_dir='./models/checkpoint'):
        """Load model checkpoint"""
        checkpoint_path = os.path.join(checkpoint_dir, f'best_model.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            return checkpoint['epoch'], checkpoint['val_loss']
        return None, None

    def train(self, train_loader, val_loader, epochs, checkpoint_dir):
        train_losses = []
        val_losses = []
        
        best_val_loss = float('inf')
        
        # Initialize early stopping
        early_stopping = EarlyStopping(patience=5, min_delta=1e-6, verbose=True)
        
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            for batch_X, batch_y in tqdm(train_loader):
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
            
            val_loss = self.evaluate(val_loader)
            val_losses.append(val_loss)

            self.scheduler.step(val_loss)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
                
            # early stop after some epoch
            if epoch >=0:
                # Save checkpoint if best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(val_loss, epoch, checkpoint_dir)

                # Early stopping check
                if early_stopping(val_loss):
                    print(f"EARLY STOPPING TRIGGERED AT EPOCH {epoch+1}")
                    break
        
        # Load best model
        _, best_loss = self.load_checkpoint(checkpoint_dir)
        print(f"Training completed. Best validation loss: {best_loss:.6f}")
        
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

    def plot_results_training(self, path_plot_save, data_loader, train_losses, val_losses, TEST_ID):
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
        plt.savefig(path_plot_save+'/restuls_test_{}.pdf'.format(TEST_ID))
        plt.show()


    def get_Delta_fNL(self, data_loader, kmin):
        """
        This method for now only serve scale-invariant, ns=1 case

        """
        self.model.eval()
        total_loss = 0
        bk_predicted = []
        bk_true = []
        k_vals  = []
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                bk_predicted.append(outputs.cpu().detach().numpy().tolist())
                bk_true.append(batch_y.cpu().detach().numpy().tolist())
                k_vals.append(batch_X.cpu().detach().numpy().tolist())

        bk_predicted = np.concatenate(bk_predicted, axis=0)  
        bk_true = np.concatenate(bk_true, axis=0)  
        k_vals = np.concatenate(k_vals, axis=0)  

        Delta_fNL = Delta_fNL_scale_w_interp(k_vals, bk_predicted, bk_true, kmin, scale_invariant=True)

        return Delta_fNL
        
        
    def save_model(self, filepath, func_name, func_args, num_terms, symm_kind):
        """
        Save the model and training configuration
        
        Args:
            filepath (str): Path to save the model (e.g., 'models/my_model.pth')
        """
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'func_name': func_name,
            'func_args': func_args,
            'num_terms': num_terms,
            'symm_kind': symm_kind
        }
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath, device=None):
        """
        Load a saved model
        
        Args:
            filepath (str): Path to the saved model
            device (torch.device): Device to load the model on
        
        Returns:
            SepBKNN: Loaded model instance
        """
        checkpoint = torch.load(filepath, map_location=device)
        
        # Create a new instance with the saved parameters
        instance = cls(
            # func_name=checkpoint['func_name'].replace('bk_function', 'bk'),  # Convert function name to short form
            num_terms=checkpoint['num_terms'],
            symm_kind = checkpoint['symm_kind'],
            device=device,
            # func_args=checkpoint['func_args']
        )
        
        # Load the saved states
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        instance.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Model loaded from {filepath}")
        return instance
    
    
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.val_loss_min = val_loss
            return False
            
        # Check if validation loss improved
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.val_loss_min = val_loss
            self.counter = 0
            if self.verbose:
                print(f'Validation loss decreased to {val_loss:.6f}')
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        
        # Check if we need to stop
        if self.counter >= self.patience:
            self.early_stop = True
            return True
            
        return False