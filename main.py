import argparse
import os
import pickle

from model.economy import Economy
from model.frequentist import FrequentistAgent
from analysis.graph import represent_results


def produce_data():

    cognitive_parameters = {
        "memory_span": 250,
        "temp": 0.01,
        "u": 1
    }

    parameters = {
        "repartition_of_roles": [100, 100, 100, 100],
        "agent_model": FrequentistAgent,
        "storing_costs": [0.01, 0.04, 0.09, 0.12],
        "cognitive_parameters": cognitive_parameters,
        "t_max": 100
    }

    e = Economy(
        **parameters
    )

    return parameters, e.run()


def main(args):

    parameters_file = "data/parameters.p"
    backup_file = "data/data.p"

    if os.path.exists(parameters_file) and os.path.exists(backup_file) and not args.force:

        with open(parameters_file, 'rb') as f:
            parameters = pickle.load(f)

        with open(backup_file, 'rb') as f:
            backup = pickle.load(f)

    else:
        parameters, backup = produce_data()

        os.makedirs(os.path.dirname(parameters_file), exist_ok=True)

        with open(parameters_file, 'wb') as f:
            pickle.dump(parameters, f)

        with open(backup_file, 'wb') as f:
            pickle.dump(backup, f)

    represent_results(backup=backup, parameters=parameters, folder='fig')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run money simulations.')

    parser.add_argument('-f', '--force', action="store_true", default=False,
                        help="Force creation of new data.")

    parsed_args = parser.parse_args()

    main(parsed_args)
