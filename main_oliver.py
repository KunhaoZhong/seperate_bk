import numpy as np
import os
import argparse
import json
import datetime
import torch
from torch.utils.data import TensorDataset, DataLoader

from sep_bk_nn import SepBKNN
from sep_bk_nn.bk_functions import create_bk_dataset, create_bk_dataset_new
from sep_bk_nn.utils import ConfigHandler, print_block_separator, parse_test_ids
import warnings

### Create a test dataset
# This is only for testing -- in practice, we can delete this block and just load the dataset of interest in the main code.
def S_function(k1, k2, k3, alpha=0.1, add_eq=False):
    """Shape function (for testing)"""
    out = (k1**2/k2/k3) / (1. + (alpha*k1**2/k2/k3)**2) / 3\
            +(k2**2/k1/k3) / (1. + (alpha*k2**2/k1/k3)**2) / 3\
            +(k3**2/k1/k2) / (1. + (alpha*k3**2/k1/k2)**2) / 3
    if add_eq:
        out += (k1/k2 + k2/k1 + k1/k3 + k3/k1 + k2/k3 + k3/k2)\
         - (k1**2 / (k2 * k3) + k2**2 / (k1 * k3) + k3**2 / (k1 * k2)) - 2
    return out

# We'll generate a test dataset log-spaced in k
kmin = 0.001
kmax = 1.0
n_points_k1 = 100
kk = np.linspace(kmin,kmax,n_points_k1)
data = []
for k1 in kk:
    for k2 in kk:
        for k3 in kk:
            if np.abs(k1-k2)<=k3 and k3<=k1+k2 and k1<=k2<=k3:
                data.append([k1,k2,k3,S_function(k1,k2,k3)])
np.savetxt("test.txt",np.asarray(data))

def run_experiment(config, test_id):
    
    # Define input class
    sep_bk_nn = SepBKNN(**config['model_params'],device=config['device'])
    
    # Define inputs
    NUM_TERMS = config['model_params']['num_terms']
    SYMM_KIND = config['model_params']['symm_kind']
    
    if config['dataset_params']['external_data']:
        
        ## Read in external bispectrum
        bk_input = np.loadtxt(config['dataset_params']['datafile'])
        print("Datafile: %s"%config['dataset_params']['datafile'])
        print("N_triangles: %d"%len(bk_input))
        BK_FUNC = 'external'
        FUN_ARG = None
        
        # Load k1, k2, k3, (dimensionless) shape
        assert len(bk_input[0])==4, "External bispectrum must have format {k1,k2,k3,S(k1,k2,k3)}"
        k1, k2, k3, S = bk_input.T
        kmin = np.min([k1,k2,k3])
        kmax = np.max([k1,k2,k3])
        print("kmin = %.2e, kmax = %.2e"%(kmin,kmax))
        
        # Check that all modes obey the triangle inequality
        assert np.all(np.abs(k1-k2)<=k3), "Not all triangles are valid"
        assert np.all(k1+k2>=k3), "Not all triangles are valid"
        
        # Convert to torch
        X_all = torch.tensor(np.stack([k1,k2,k3],axis=1), dtype=torch.float32)
        y_all = torch.tensor(S, dtype=torch.float32).view(-1, 1)
    
    else:
        ## Compute bispectrum from internal template
        X_all, y_all = create_bk_dataset_new(**config['dataset_params'])
        
        print('Training for function: ', config['dataset_params']['func_name'])
        print('with arguments: ', config['dataset_params']['func_arg'])
        BK_FUNC = config['dataset_params']['func_name']
        FUN_ARG = config['dataset_params']['func_arg']
    
        # Define k range
        kmin = config['dataset_params']['kmin']
        kmax = config['dataset_params']['kmax']
    
    # Split into train, validation and test set
    all_indices = np.arange(len(X_all))
    np.random.shuffle(all_indices)
    train_indices = all_indices[:len(all_indices)//10*8]
    valid_indices = all_indices[len(all_indices)//10*8:len(all_indices)//10*9]
    test_indices = all_indices[len(all_indices)//10*9:]
    
    X_train = X_all[train_indices]
    y_train = y_all[train_indices]
    X_val = X_all[valid_indices]
    y_val = y_all[valid_indices]
    X_test = X_all[test_indices]
    y_test = y_all[test_indices]
    
    # Print info
    print('Training set size: {}'.format(X_train.shape[0]))
    print('Validation set size: {}'.format(X_val.shape[0]))
    print('Test set size: {}'.format(X_test.shape[0]))

    # Create data loaders
    batch_size = config['batch_size']
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Define path for output model
    model_path = config_handler.get_model_path(test_id)
    results_path = config_handler.get_results_path(test_id)
    intermediate_save_path = config['model_dir'] + '/checkpoints_' + test_id
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if not os.path.exists(config['model_dir']):
        os.makedirs(config['model_dir'])
    
    # Train the model over a number of epochs
    train_losses, val_losses = sep_bk_nn.train(train_loader, val_loader, epochs=config['epochs'], checkpoint_dir=intermediate_save_path, patience=config['patience']) # config['epochs']

    # Test the model by computing the loss on the test set
    test_loss = sep_bk_nn.test_loss(test_loader)
    
    # Calculate the cosine between the true and separable template
    cosine = sep_bk_nn.get_cosine(test_loader, kmin = kmin, kmax = kmax)
    print('estimated cosine is {} '.format(cosine))
    
    # Save results
    results = {
        'test_id': test_id,
        'description': config.get('description', ''),
        'timestamp': datetime.datetime.now().isoformat(),
        'test_loss': float(test_loss),
        'cosine': float(cosine),
        'config': config
    }
    
    # Save the model and results
    sep_bk_nn.save_model(model_path, BK_FUNC, FUN_ARG, NUM_TERMS, SYMM_KIND)
    with open(f'{results_path}.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run BK-NN experiments')
    parser.add_argument('--config', type=str, default='configs/experiment_oliver.yaml',
                      help='Path to config file')
    parser.add_argument('--test_id', type=str, default=None,
                      help='Specific test ID to run (e.g., "test_1" or "1")')
    args = parser.parse_args()
    
    # Load configuration
    global config_handler
    config_handler = ConfigHandler(args.config)
    
    if args.test_id:
        # Run specific experiments
        all_results = {}
        test_ids = parse_test_ids(args.test_id)
        
        print_block_separator(f"Experiments to run: {test_ids}", style='hash')
        
        for test_id in test_ids:
            test_id = str(test_id)
            # Convert number to test_X format if needed
            if test_id.isdigit():
                test_id = f"test_{test_id}"
            
            print_block_separator(f"Starting Experiment: {test_id}")
            print(f"\nRunning experiment {test_id}")
            config = config_handler.get_experiment_config(test_id)
            results = run_experiment(config, test_id)
            all_results[test_id] = results
            
        # Print summary for selected experiments
        print("\nSummary of selected experiments:")
        for test_id, results in all_results.items():
            print(f"{test_id}: LOSS = {results['test_loss']}, COSINE = {results['cosine']}")
        print_block_separator("Processing Complete", style='hash')
            
    else:
        # Run all experiments
        all_results = {}
        for test_id in config_handler.get_experiment_ids():
            print_block_separator(f"Starting Experiment: {test_id}")
            print(f"\nRunning experiment {test_id}")
            config = config_handler.get_experiment_config(test_id)
            results = run_experiment(config, test_id)
            all_results[test_id] = results
        
        # Print summary
        print("\nSummary of all experiments:")
        for test_id, results in all_results.items():
            print(f"{test_id}: LOSS = {results['test_loss']}, COSINE = {results['cosine']}")

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()