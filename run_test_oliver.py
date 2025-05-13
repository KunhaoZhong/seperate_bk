""" KZ todo:
    1. organize the variables such as grid_points, funcs, to a config/yaml?
"""
import numpy as np
from sep_bk_nn import SepBKNN
from bk_functions import create_bk_dataset
from torch.utils.data import TensorDataset, DataLoader

def main():
    
    ### INITIALIZE CLASS
    
    # Output identifier
    TEST_ID = 0
    
    # Number of terms in the separable representation
    NUM_TERMS = 1
    
    # Choice of symmetrization
    # 1: f(k1,k2,k3) = a(k1)b(k2)c(k3) + 5 perm
    # 2: f(k1,k2,k3) = a(k1)b(k2)b(k3) + 2 perm
    # 3: f(k1,k2,k3) = a(k1)b(k2)c(k3) + 2 cyc perm
    SYMM_KIND = 3
    
    # Initialize the SepBKNN
    sep_bk_nn = SepBKNN(num_terms=NUM_TERMS, symm_kind=SYMM_KIND, sub_arch='MLP', device='cuda')
    
    ### CREATE DATASET
    # Choice of template plus any arguments to pass
    BK_FUNC = 'bk_loc'
    FUN_ARG = None

    # Choose grid-points
    GRID_POINTS_TRAINING = 3000 # size of grid
    KMIN = 0.01 # minimum k
    
    # How to sample K2
    # 0: fixed sampling size
    # 1: more dense near equilateral limit: k1 = k2 = k3 = 1
    # 2: use meshgrid
    K2_SAMPLE_VERSION = 0
    SCALE_INVARIANT = True
    
    # Create a training and test dataset
    # Note: we use a slightly different k-min for the validation / test so the grids don't overlap
    X_train, y_train = create_bk_dataset(grid_points=GRID_POINTS_TRAINING, func_name=BK_FUNC, func_arg=FUN_ARG, kmin=KMIN, kmax=1.0, n_points_k2=None, scale_invariant=SCALE_INVARIANT, k2_sample_version=K2_SAMPLE_VERSION) # expand training set boundaries a little bit. This is hard-coded for now
    X_val,  y_val  = create_bk_dataset(grid_points=1000, func_name=BK_FUNC, func_arg=FUN_ARG, kmin=KMIN+0.003, kmax=1.0, n_points_k2=None, scale_invariant=SCALE_INVARIANT, k2_sample_version=K2_SAMPLE_VERSION)
    X_test, y_test = create_bk_dataset(grid_points=100, func_name=BK_FUNC, func_arg=FUN_ARG, kmin=KMIN+0.006, kmax=1.0, n_points_k2=None, scale_invariant=SCALE_INVARIANT, k2_sample_version=K2_SAMPLE_VERSION)

    print('Function: ', BK_FUNC, 'with arguments: ', FUN_ARG)
    print('Training set size: {}'.format(X_train.shape[0]))
    print('Test set size: {}'.format(X_test.shape[0]))

    ### RUN THE INFERENCE
    
    # Create data loaders
    batch_size = 512
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Train the model
    train_losses, val_losses = sep_bk_nn.train(train_loader, val_loader, epochs=50)

    # Test the model
    test_mse = sep_bk_nn.test_mse(test_loader)

    # Save the model
    sep_bk_nn.save_model('models/sep_bk_nn_test_{}.pth'.format(TEST_ID), BK_FUNC, FUN_ARG, NUM_TERMS, SYMM_KIND)

    # Plot results
    sep_bk_nn.plot_results_training(test_loader, train_losses, val_losses, TEST_ID)

    # Calculate Delta_fNL
    Delta_fNL = sep_bk_nn.get_Delta_fNL(test_loader, kmin = KMIN)
    print('estimated Delta_f_NL is {} '.format(Delta_fNL))


if __name__ == "__main__":
    main()