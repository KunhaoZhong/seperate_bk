import numpy as np
import os
import argparse
import json
import datetime
from torch.utils.data import TensorDataset, DataLoader

from sep_bk_nn import SepBKNN
from sep_bk_nn.bk_functions import create_bk_dataset
from sep_bk_nn.utils import ConfigHandler, print_block_separator, parse_test_ids
import warnings

def run_experiment(config, test_id):
    
    sep_bk_nn = SepBKNN(
        **config['model_params'],
        device=config['device']
    )
    
    BK_FUNC = config['dataset_params_train']['func_name']
    FUN_ARG = config['dataset_params_train']['func_arg']
    NUM_TERMS = config['model_params']['num_terms']
    SYMM_KIND = config['model_params']['symm_kind']
    
    # Create datasets
    X_train, y_train = create_bk_dataset(**config['dataset_params_train']) 
    X_val,  y_val  = create_bk_dataset(**config['dataset_params_vali'])
    X_test, y_test = create_bk_dataset(**config['dataset_params_test'])

    print('size of training set is {}'.format(X_train.shape[0]))
    print('Training for function: ', config['dataset_params_train']['func_name'])
    print('with arguments: ', config['dataset_params_train']['func_arg'])
    
    print('size of test set is {}'.format(X_test.shape[0]))

    # Create data loaders
    batch_size = 512
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    model_path = config_handler.get_model_path(test_id)
    results_path = config_handler.get_results_path(test_id)
    intermediate_save_path = config['model_dir'] + '/checkpoints_' + test_id
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # Train the model
    train_losses, val_losses = sep_bk_nn.train(train_loader, val_loader, epochs=config['epochs'], checkpoint_dir=intermediate_save_path) # config['epochs']

    # Test the model
    test_loss = sep_bk_nn.test_loss(test_loader)

    # Calculate cosine with results
    cosine = sep_bk_nn.get_cosine(test_loader, xmin = config['dataset_params_test']['kmin'], scale_invariant=config['dataset_params_test']['scale_invariant'])
    print('estimated cosine is {} '.format(cosine))
    import sys
    sys.exit()
    
    # Save results
    results = {
        'test_id': test_id,
        'description': config.get('description', ''),
        'timestamp': datetime.datetime.now().isoformat(),
        'test_loss': float(test_loss),
        'config': config
    }
    
    # Save the model and results

    
    sep_bk_nn.save_model(model_path, BK_FUNC, FUN_ARG, NUM_TERMS, SYMM_KIND)
    with open(f'{results_path}.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Plot results
    sep_bk_nn.plot_results_training(results_path, test_loader, train_losses, val_losses, test_id)

    # Calculate Delta_fNL
    # KZ NOTE: this is optional and can take more time. Maybe just choose a few to test/visualize in a notebook
    
    # Delta_fNL = sep_bk_nn.get_Delta_fNL(test_loader, kmin = config['dataset_params_test']['kmin'])
    # print('estimated Delta_f_NL is {} '.format(Delta_fNL))
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run BK-NN experiments')
    parser.add_argument('--config', type=str, default='configs/experiment_bk_sl_collider.yaml',
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
            print(f"{test_id}: MSE = {results['test_loss']}")
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
            print(f"{test_id}: MSE = {results['test_loss']}")

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()