import click
import optuna

from optuna.trial import TrialState
from hp_optimizer import objective

import settings


def print_bold(text: str) -> None:
    """
    Print the provided text in bold
    """
    print('\033[1m' + text + '\033[0m')


@click.command()
@click.option('-e', '--epochs', default=5, show_default=True, type=int, help='Number of epochs.')
@click.option('-b', '--batch-size', default=32, show_default=True, type=int, help='Batch size used to load the data.')
@click.option('--train-exemples', default=1_000, show_default=True, type=int, help='Number of data trained on before pruning.')
@click.option('--test-exemples', default=500, show_default=True, type=int, help='Number of data tested on before pruning.')
@click.option('--nb-trials', default=100, show_default=True, type=int, help='Number trials tested to find the optimal model\'s parameters.')
@click.option('--timeout', default=600, show_default=True, type=int, help='Timeout before pruning for each trials.')
def find_best_parameters(epochs, batch_size, train_exemples, test_exemples, nb_trials, timeout):
    settings.init(epochs, batch_size, train_exemples, test_exemples, nb_trials, timeout)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=nb_trials, timeout=timeout) # here is the error

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print_bold("\nStudy statistics")
    print("Number of finished trials:\t", len(study.trials))
    print("Number of pruned trials:\t", len(pruned_trials))
    print("Number of complete trials:\t", len(complete_trials))

    trial = study.best_trial
    print_bold("\nBest trial")
    print("Value:\t", trial.value)

    print("Params:")
    for key, value in trial.params.items():
        print(f"\t{key}: {value}")


if __name__ == "__main__":
    find_best_parameters()
