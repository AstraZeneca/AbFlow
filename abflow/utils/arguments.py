"""
Author: Talip Ucar
email: ucabtuc@gmail.com

Description: - Collects arguments from command line, and loads configuration from the yaml files.
             - Prints a summary of all options and arguments.
"""

import sys
import yaml
import os
from argparse import ArgumentParser
from texttable import Texttable


def load_config(config_path: str):
    """Load model config dictionary from a yaml file"""

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config


def get_runtime_and_model_config(args):
    """Returns runtime and model/dataset specific config file"""
    try:
        config = load_config(f"./config/{args.dataset}.yaml")
    except Exception as e:
        sys.exit(f"Error reading runtime config file: {e}")

    return config


class ArgParser(ArgumentParser):
    """Inherits from ArgumentParser, and used to print helpful message if an error occurs"""

    def error(self, message):
        sys.stderr.write("error: %s\n" % message)
        self.print_help()
        sys.exit(2)


def get_arguments():
    """Gets command line arguments"""

    # Initialize parser
    parser = ArgParser()

    # Dataset can be provided via command line
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="sabdab",
        help="Name of the dataset to use. It should have a config file with the same name.",
    )

    # Return parser arguments
    return parser.parse_args()


def get_config(args):
    """Loads options using yaml files under /config folder and adds command line arguments to it"""
    # Load runtime config from config folder: ./config/ and flatten the runtime config
    config = get_runtime_and_model_config(args)

    # Dataset
    config["dataset"] = args.dataset

    # Define number of workers
    config["num_workers"] = os.cpu_count()

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
    table.add_rows(
        [["Parameter", "Value"]]
        + [[k.replace("_", " ").capitalize(), args[k]] for k in keys]
    )
    # Print the table.
    print(table.draw())


def print_config_summary(config, args=None):
    """Prints out summary of options and arguments used"""
    print(100 * "=")
    print(f"Here is the configuration being used:\n")
    print_config(config)
    print(100 * "=")
    if args is not None:
        print(f"Arguments being used:\n")
        print_config(args)
        print(100 * "=")
