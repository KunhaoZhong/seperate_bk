""" KZ todo:
    1. organize the variables such as grid_points, funcs, to a config/yaml?
"""
import numpy as np
import argparse
import json
import datetime
from torch.utils.data import TensorDataset, DataLoader

from sep_bk_nn import SepBKNN
from sep_bk_nn.bk_functions import create_bk_dataset
from sep_bk_nn.utils import ConfigHandler


def run_experiment(config, test_id):
    
    sep_bk_nn = SepBKNN(
        **config['model_params'],
        device=config['device']
    )
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

    # Train the model
    train_losses, val_losses = sep_bk_nn.train(train_loader, val_loader, epochs=config['epochs'])

    # Test the model
    test_mse = sep_bk_nn.test_mse(test_loader)

    # Save results
    results = {
        'test_id': test_id,
        'description': config.get('description', ''),
        'timestamp': datetime.datetime.now().isoformat(),
        'test_mse': float(test_mse),
        'config': config
    }
    
    # Save the model and results
    model_path = config_handler.get_model_path(test_id)
    results_path = config_handler.get_results_path(test_id)
    
    sep_bk_nn.save_model(model_path)
    with open(f'{results_path}.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Plot results
    sep_bk_nn.plot_results_training(test_loader, train_losses, val_losses, test_id)

    # Calculate Delta_fNL
    Delta_fNL = sep_bk_nn.get_Delta_fNL(test_loader, kmin = config['dataset_params_test']['kmin'])
    print('estimated Delta_f_NL is {} '.format(Delta_fNL))
    
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
        # Run specific experiment
        test_id = args.test_id
        if test_id.isdigit():
            test_id = f"test_{test_id}"
        config = config_handler.get_experiment_config(test_id)
        results = run_experiment(config, test_id)
        print(f"\nResults for {test_id}:")
        print(f"MSE: {results['test_mse']}")
    else:
        # Run all experiments
        all_results = {}
        for test_id in config_handler.get_experiment_ids():
            print(f"\nRunning experiment {test_id}")
            config = config_handler.get_experiment_config(test_id)
            results = run_experiment(config, test_id)
            all_results[test_id] = results
        
        # Print summary
        print("\nSummary of all experiments:")
        for test_id, results in all_results.items():
            print(f"{test_id}: MSE = {results['test_mse']}")


if __name__ == "__main__":
    main()