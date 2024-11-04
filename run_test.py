""" KZ todo:
    1. organize the variables such as grid_points, funcs, to a config/yaml?
"""
import numpy as np
from sep_bk_nn import SepBKNN
from bk_functions import create_bk_dataset
from torch.utils.data import TensorDataset, DataLoader

def main():
    
    TEST_ID = 12
    BK_FUNC = 'bk_sl_collider'
    FUN_ARG = {'alpha': 0.1}
    
    # BK_FUNC = 'bk_eq'
    # FUN_ARG = None
    NUM_TERMS = 2
    SYMM_KIND = 2
    KMIN = 0.01
    K2_SAMPLE_VERSION = 0 # fix step size if verion=0, otherwise more dense in the k1,k2->1 region
    GRID_POINTS_TRAINING = 3000
    
    # Initialize the SepBKNN
    sep_bk_nn = SepBKNN(num_terms=NUM_TERMS, symm_kind=SYMM_KIND, sub_arch='MLP', device='cuda') # symm_kind_2 has 3 perm

    # Create datasets
    
    X_train, y_train = create_bk_dataset(grid_points=GRID_POINTS_TRAINING, func_name=BK_FUNC, func_arg=FUN_ARG, kmin=KMIN, kmax=1.0, n_points_k2=None, scale_invariant=True, k2_sample_version=K2_SAMPLE_VERSION) # expand training set boundaries a little bit. This is hard-coded for now
    X_val,  y_val  = create_bk_dataset(grid_points=1000, func_name=BK_FUNC, func_arg=FUN_ARG, kmin=KMIN+0.01, kmax=1.0, n_points_k2=None, scale_invariant=True, k2_sample_version=K2_SAMPLE_VERSION)
    X_test, y_test = create_bk_dataset(grid_points=100, func_name=BK_FUNC, func_arg=FUN_ARG, kmin=KMIN+0.02, kmax=1.0, n_points_k2=None, scale_invariant=True, k2_sample_version=K2_SAMPLE_VERSION)

    print('size of training set is {}'.format(X_train.shape[0]))
    print('Training for function: ', BK_FUNC, 'with arguments: ', FUN_ARG)
    
    print('size of test set is {}'.format(X_test.shape[0]))

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