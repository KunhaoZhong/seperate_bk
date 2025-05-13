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
from .bk_utils import Delta_fNL_scale_w_interp
from .bk_utils import plot_3d_data


class SepBKNN:
    def __init__(self, num_terms, symm_kind, loss_func = 'mse', sub_arch='MLP', log_transform=False, device=None):

        self.device = device
        self.model = SeparableApproximation(num_terms=num_terms, symm_kind=symm_kind, sub_arch=sub_arch, log_transform=log_transform).to(self.device)
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

    def inner_product_loss(self, X, predicted_y, true_y):
        """Compute the inner-product loss"""
        k_weight = X[:,0]*X[:,1]*X[:,2]/(X[:,0]+X[:,1]+X[:,2])
        k_weight *= 0.5*(1-torch.tanh(10*torch.log10(X[:,0]/0.13)))
        k_weight *= 0.5*(1-torch.tanh(10*torch.log10(X[:,0]/0.13)))
        k_weight *= 0.5*(1-torch.tanh(10*torch.log10(X[:,0]/0.13)))
        return torch.sum(k_weight*(predicted_y - true_y)**2)/len(predicted_y)
        
    def train(self, train_loader, val_loader, epochs, checkpoint_dir, patience=5):
        train_losses = []
        val_losses = []
        
        best_val_loss = float('inf')
        
        # Initialize early stopping
        early_stopping = EarlyStopping(patience=patience, min_delta=1e-6, verbose=True)
        
        if self.loss_func == 'inner':
            print("Defining normalization")
            loss_norm = 0
            for batch_X, batch_y in tqdm(train_loader):
                loss_norm += self.inner_product_loss(batch_X, batch_y, 0*batch_y)*len(batch_y)
            self.loss_norm = loss_norm/len(train_loader.dataset)
            
        for epoch in range(epochs):
            print("Starting epoch %d"%epoch)
            self.model.train()
            epoch_loss = 0
            for batch_X, batch_y in tqdm(train_loader):
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                if self.loss_func == 'mse':
                    loss = self.criterion(outputs, batch_y)
                elif self.loss_func == 'inner':
                    loss = self.inner_product_loss(batch_X, outputs, batch_y)* batch_y.size(0)*len(train_loader)/len(train_loader.dataset)/self.loss_norm
                else:
                    raise TypeError('Loss function not defined')
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                if np.isnan(epoch_loss):
                    raise Exception("Loss is nan!")
            
            epoch_loss /= len(train_loader)
            train_losses.append(epoch_loss)
            
            val_loss = self.evaluate(val_loader)
            val_losses.append(val_loss)

            self.scheduler.step(val_loss)
            
            if epoch % 1 == 0:
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
                if self.loss_func == 'mse':
                    loss = self.criterion(outputs, batch_y)
                elif self.loss_func == 'inner':
                    loss = self.inner_product_loss(batch_X, outputs, batch_y)/self.loss_norm
                else:
                    raise TypeError('Loss function not defined')
                total_loss += loss.item() * batch_y.size(0)
        return total_loss / len(data_loader.dataset)

    def test_loss(self, test_loader):
        loss = self.evaluate(test_loader)
        print(f"Test Error: {loss:.6f}")
        return loss

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

    def get_cosine(self, data_loader, kmin=0.001, kmax=1.0):
        """
        This computes the cosine between the approximated template and the truth.
        """
        print("Computing cosine...")
        self.model.eval()
        
        # Define an integration grid
        xmin = kmin/kmax
        _x = np.linspace(xmin,1,100)
        _y = np.linspace(xmin,1,100)
        _k = np.geomspace(kmin,kmax,25)
        xx, yy, kk = np.meshgrid(_x, _y, _k)
        filt = (np.abs(xx-yy)<=1)*(1<=xx+yy)*(xx<=yy)
        filt *= (kk*xx >= kmin)*(kk*yy >= kmin)
        x, y, k = np.min([xx[filt], yy[filt]], 0), np.max([xx[filt], yy[filt]], 0), kk[filt]
        
        # Load true and theory predictions on the test set
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
        k_vals = np.concatenate(k_vals)
        bk_predicted = np.concatenate(bk_predicted)
        bk_true = np.concatenate(bk_true)
        
        # Compute theory on the interpolation grid (only in domain of interest)
        interp_pred, interp_true = np.zeros_like(xx), np.zeros_like(xx)
        interp_pred[filt] = interpolate.LinearNDInterpolator(k_vals[:,:3],bk_predicted.ravel(), 0.)(k*x, k*y, k)
        interp_true[filt] = interpolate.LinearNDInterpolator(k_vals[:,:3],bk_true.ravel(), 0.)(k*x, k*y, k)
        
        def inner(iint1, iint2):
            return integrate.simpson(_k**2*integrate.simpson(integrate.simpson(iint1*iint2, x=_x, axis=0), x=_y, axis=0), x=_k)
        inner_pp = inner(interp_pred, interp_pred)
        inner_pt = inner(interp_pred, interp_true)
        inner_tt = inner(interp_true, interp_true)
        return inner_pt/np.sqrt(inner_tt*inner_pp)

    def get_Delta_fNL(self, data_loader, kmin, method='interpolation'):
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

        if method=='interpolation':
            Delta_fNL = Delta_fNL_scale_w_interp(k_vals, bk_predicted, bk_true, kmin, scale_invariant=True)
        elif method=='direct_sum':
            print('Please ensure the k samples are uniform in order to use direct sum')
            inner_product = np.sum((bk_predicted-bk_true)*(bk_predicted-bk_true)) / np.sum(bk_true*bk_true)
            Delta_fNL = np.sqrt(inner_product)
            print('The bias estimation of fNL is approximately Delta_fNL =  ', Delta_fNL)
        else:
            raise NotImplementedError('To get Delta-fNL, choose either interpolation or direct_sum method')
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