import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU, CrossEntropyLoss
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from Utils import *
from torch.utils.data import *
from sklearn.datasets import load_digits
from torch import tensor, Tensor
from torch.utils.tensorboard import SummaryWriter

torch.set_default_tensor_type(torch.FloatTensor)

class Landau(Dataset):

    def __init__(self):
        mnist = load_digits()
        self.data = mnist['data']
        self.target = mnist['target']

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return tensor(self.data[idx]).float(), tensor(self.target[idx]).long()


if __name__ == '__main__':

    writer = SummaryWriter()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # train_loader, validation_loader = get_data_loaders(train_batch_size=BATCH_SIZE,
    #                                                    validation_batch_size=BATCH_SIZE)
    train_loader = DataLoader(Landau(), batch_size=128)
    validation_loader = DataLoader(Landau(), batch_size=128)

    model = Sequential(
        Linear(input_size, hidden_size),
        ReLU(),
        Linear(hidden_size, out_features=num_classes),
        # nn.Softmax(dim=1)
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    criterion = CrossEntropyLoss()

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

    val_metrics = {
        "accuracy": Accuracy(),
        "nll": Loss(criterion)
    }


    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(validation_loader)
        metrics = evaluator.state.metrics
        print("Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics["accuracy"], metrics["nll"]))


    evaluator = create_supervised_evaluator(model, metrics=val_metrics)

    trainer.run(train_loader, max_epochs=num_epochs)