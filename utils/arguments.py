"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: - Collects arguments from command line, and loads configuration from the yaml files.
             - Prints a summary of all options and arguments.
"""

import os
from argparse import ArgumentParser
import sys
from texttable import Texttable
import torch
import yaml


class ArgParser(ArgumentParser):
    """Inherits from ArgumentParser, and used to print helpful message if an error occurs"""
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)
        
        
def get_arguments():
    """Gets command line arguments"""
    
    # Initialize parser
    parser = ArgParser()

    # Dataset can be provided via command line
    parser.add_argument("-d", "--dataset", type=str, default="sabdab", 
                        help='Name of the dataset to use. It should have a config file with the same name.')
    
    parser.add_argument("-e", "--epochs", type=int, default=None, 
                        help='Defines epoch when loading the model if the model is saved at specific epoch(s)')
    
    parser.add_argument("-bs", "--batch_size", type=int, default=None, 
                        help='Defines batch size. If None, use batch size degfined in config file.')
    
    # Whether to use GPU.
    parser.add_argument("-g", "--gpu", dest='gpu', action='store_true', 
                        help='Used to assign GPU as the device, assuming that GPU is available')
    
    parser.add_argument("-ng", "--no_gpu", dest='gpu', action='store_false', 
                        help='Used to assign CPU as the device')
    
    parser.set_defaults(gpu=True)
        
    # GPU device number as in "cuda:0". Default is 0.
    parser.add_argument("-dn", "--device_number", type=str, default='0', 
                        help='Defines which GPU to use. It is 0 by default')
    
    # Experiment number if MLFlow is on
    parser.add_argument("-ex", "--experiment", type=int, default=1, 
                        help='Used as a suffix to the name of MLFlow experiments if MLFlow is being used')
    
    # Return parser arguments
    return parser.parse_args()

def get_runtime_and_model_config(args):
    """Returns runtime and model/dataset specific config file"""
    try:
        with open(f"./config/{args.dataset}.yaml", "r") as file:
            config = yaml.safe_load(file)
    except Exception as e:
        sys.exit(f"Error reading runtime config file: {e}")

    # Copy dataset names to config to use later
    # config["dataset"] = args.dataset

    return config

def get_config(args):
    """Loads options using yaml files under /config folder and adds command line arguments to it"""
    # Load runtime config from config folder: ./config/ and flatten the runtime config
    config = get_runtime_and_model_config(args)
    # Define which device to use: GPU or CPU
    # config["device"] =torch.device('cuda:' + args.device_number if torch.cuda.is_available() and args.gpu else 'cpu')

    # Dataset
    config["dataset"] = args.dataset
    
    # Define number of workers
    config["num_workers"] = os.cpu_count()
    # config["num_workers"] = 0

    # Define number of epochs
    # config["epochs"] = args.epochs
    
    # Dataset
    # if args.batch_size is not None:
    #     config["batch_size"] = args.batch_size
    
    # Return
    return config


def print_config(args):
    """Prints out options and arguments"""
    # Yaml config is a dictionary while parser arguments is an object. Use vars() only on parser arguments.
    if type(args) is not dict:
        args = vars(args)
    # Sort keys
    keys = sorted(args.keys())
    # Initialize table
    table = Texttable()
    # Add rows to the table under two columns ("Parameter", "Value").
    table.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    # Print the table.
    print(table.draw())


def print_config_summary(config, args=None):
    """Prints out summary of options and arguments used"""
    # Summarize config on the screen as a sanity check
    print(100 * "=")
    print(f"Here is the configuration being used:\n")
    print_config(config)
    print(100 * "=")
    if args is not None:
        print(f"Arguments being used:\n")
        print_config(args)
        print(100 * "=")
