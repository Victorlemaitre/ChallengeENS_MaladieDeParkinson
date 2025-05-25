
import torch
import torch.optim as optim
import time
import numpy as np

from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset,TensorDataset,Dataset
import torch.nn.functional as F

def fold_training(X, dataset, config):
    cross_entropy = (config.criterion == F.cross_entropy)
    folds = KFold(n_splits=config.n_folds, shuffle=True, random_state=43)

    loss_fold = []
    for fold, (train_index, val_index) in enumerate(folds.split(X)):
        ensemble = [config.model(config, n_features=X.size(1) // config.context_length) for _ in
                    range(config.n_deep_ensemble)]
        optimizers = [optim.AdamW(model.parameters(), lr=config.lr) for model in ensemble]
        schedulers = [optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5) for optimizer in optimizers]

        print(f"fold {fold}")
        train_subset = Subset(dataset, train_index)
        val_subset = Subset(dataset, val_index)
        train_dataloader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=True)
        loss_fold.append(
            training_loop(ensemble, train_dataloader, val_dataloader, optimizers, schedulers, config, cross_entropy))

    print("==============================================================================")
    for f in range(len(loss_fold)):
        print(f"Fold {f}, val loss  : {loss_fold[f]}")
    print(f"moyenne : {sum(loss_fold) / len(loss_fold)}")
    print("==============================================================================")

def complete_training( X, dataset, config):
    #Trains on the whole dataset, used for submitting models
    cross_entropy = (config.criterion == F.cross_entropy)

    ensemble = [config.model(config, n_features = X.size(1)//config.context_length) for _ in range(config.n_deep_ensemble)]
    optimizers = [optim.AdamW(model.parameters(), lr = config.lr, weight_decay= config.weight_decay) for model in ensemble]
    schedulers = [optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma) for optimizer in optimizers]

    train_dataloader =  DataLoader(dataset, batch_size = config.batch_size, shuffle = True)
    training_loop(ensemble, train_dataloader, None, optimizers, schedulers, config,cross_entropy)

    return ensemble


def training_loop(ensemble, train_dataloader, val_dataloader, optimizers, schedulers, config, cross_entropy):
    for epoch in range(config.epochs):

        print("-------------------------------------------------------")
        for param_group in optimizers[0].param_groups:  # lr is the same among all optimizers
            print(f"epoch : {epoch} : Learning Rate = {param_group['lr']}")
        loss_train_individual = []
        duration_epochs = []
        for i, model in enumerate(ensemble):
            loss, duration = do_a_training_step(model, train_dataloader, optimizers[i], schedulers[i], config,
                                                cross_entropy)
            loss_train_individual.append(loss)
            duration_epochs.append(duration)
        duration_epochs = np.array(duration_epochs)
        print(f"Average time per epoch {duration_epochs.mean():.6f}")
        print(f"loss per model : {loss_train_individual}")
        if val_dataloader is not None:
            loss_val = eval_models(ensemble, val_dataloader, config, cross_entropy)
            print(f"loss ensemble val : {loss_val}")
            print("-------------------------------------------------------")

    if val_dataloader is not None:
        # returns the last validation lost
        return loss_val


def do_a_training_step(model, dataloader, optimizer, scheduler, config, cross_entropy):
    start_time = time.time()
    loss_train = []
    model.train()
    if cross_entropy:
        for inputs, targets in dataloader:
            optimizer.zero_grad()

            pred = model(inputs)
            non_zero_idx = torch.where(targets != config.null_value)
            targets = torch.round(targets).to(torch.int64)
            loss = config.criterion(pred[non_zero_idx], targets[non_zero_idx])
            loss.backward()
            optimizer.step()
            loss_train.append(loss.item())

        scheduler.step()
        loss_train = sum(loss_train) / len(loss_train)

        return loss_train

    else:
        nb_non_null = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()

            pred = model(inputs).squeeze(-1)

            non_zero_idx = torch.where(targets != config.null_value)

            loss = config.criterion(pred[non_zero_idx], targets[non_zero_idx])
            loss.backward()
            optimizer.step()

            nb_non_null += len(non_zero_idx[0])

            loss_train.append(loss.item() * len(non_zero_idx[0]))

        scheduler.step()
        MSE_loss_train = (sum(loss_train) / nb_non_null)
        end_time = time.time()
        duration = end_time - start_time

        return MSE_loss_train, duration


def eval_models(ensemble, dataloader, config, cross_entropy):
    loss_test = []
    for model in ensemble:
        model.eval()
    nb_non_null = 0
    with torch.inference_mode():
        for inputs, targets in dataloader:
            if cross_entropy:
                pred_models = [torch.argmax(model(inputs), dim=-1) for model in ensemble]
                pred_models = torch.stack(pred_models).float()
            else:
                pred_models = [model(inputs).squeeze(-1) for model in ensemble]
                pred_models = torch.stack(pred_models)
            pred_ensemble = torch.mean(pred_models, dim=0)

            non_zero_idx = torch.where(targets != config.null_value)
            loss = F.mse_loss(pred_ensemble[non_zero_idx], targets[non_zero_idx])
            nb_non_null += len(non_zero_idx[0])
            loss_test.append(loss.item() * len(non_zero_idx[0]))
    MSE_loss_test = (sum(loss_test) / nb_non_null)
    return MSE_loss_test








