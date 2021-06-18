import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data import get_mnist
from network import Network

import settings


def define_model(trial) -> "nn.Layer":
    """
    Select the number of hiddens layer and the dropout values for the model.
    """

    layers = []
    in_features = 28 * 28
    n_layers = trial.suggest_int("n_hlayers", 1, 5)


    for i in range(n_layers):
        out_features = trial.suggest_int(f"n_units_l{i}", 10, 784)

        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(trial.suggest_float(f"dropout_l{i}", 0.1, 0.5)))

        in_features = out_features


    layers.append(nn.Linear(in_features, settings.CLASSES))
    layers.append(nn.LogSoftmax(dim=1))

    return layers


def objective(trial) -> float:
    """
    Make one trial to find the best model\'s parameters'
    """

    model = Network(define_model(trial)).to(settings.DEVICE)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=learning_rate)
    train_loader, valid_loader = get_mnist(settings.BATCH_SIZE)

    for epoch in range(settings.EPOCHS):
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx * settings.BATCH_SIZE >= settings.N_TRAIN_EXAMPLES:
                break

            optimizer.zero_grad()
            data, target = data.view(data.size(0), -1).to(settings.DEVICE), target.to(settings.DEVICE)

            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        correct = 0
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                if batch_idx * settings.BATCH_SIZE >= settings.N_VALID_EXAMPLES:
                    break

                data, target = data.view(data.size(0), -1).to(settings.DEVICE), target.to(settings.DEVICE)

                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)

                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(valid_loader.dataset), settings.N_VALID_EXAMPLES)

        trial.report(accuracy, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy
