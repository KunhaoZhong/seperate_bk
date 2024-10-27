""" KZ todo:
    1. organize the variables such as grid_points, funcs, to a config/yaml?
"""

from sep_bk_nn import SepBKNN
from bk_functions import create_bk_dataset
from torch.utils.data import TensorDataset, DataLoader

def main():
    # Initialize the SepBKNN
    sep_bk_nn = SepBKNN(num_terms=2, symm_kind=3, sub_arch='MLP', device='cpu') # symm_kind_2 has 3 perm

    # Create datasets
    bk_func = 'bk_sl_collider'
    X_train, y_train = create_bk_dataset(grid_points=1500, func_name=bk_func, scale_invariant=True)
    X_val, y_val = create_bk_dataset(grid_points=1000, func_name=bk_func, scale_invariant=True)
    X_test, y_test = create_bk_dataset(grid_points=1000, func_name=bk_func, scale_invariant=True)

    print('size of training set is {}'.format(X_train.shape[0]))

    # Create data loaders
    batch_size = 256
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Train the model
    train_losses, val_losses = sep_bk_nn.train(train_loader, val_loader, epochs=15)

    # Test the model
    test_mse = sep_bk_nn.test_mse(val_loader)

    # Plot results
    sep_bk_nn.test_results(test_loader, train_losses, val_losses)

if __name__ == "__main__":
    main()