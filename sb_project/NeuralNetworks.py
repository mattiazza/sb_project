import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import hiddenlayer as hl


# Define the FNN model
class base_NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_layers_dim: list = []):
        super(NeuralNetwork, self).__init__()
        if num_classes == 2:
            num_classes = num_classes - 1
        self.layers = nn.ModuleList()

        if len(hidden_layers_dim) == 0:
            self.layers = self.layers.append(nn.Linear(input_dim, num_classes))
        else:
            for layer_idx in range(len(hidden_layers_dim)):
                if layer_idx == 0:  # first layer, from input to hidden
                    self.layers = self.layers.append(
                        nn.Linear(input_dim, hidden_layers_dim[layer_idx])
                    )
                else:  # hidden layers, depending on the input
                    self.layers = self.layers.append(
                        nn.Linear(
                            hidden_layers_dim[layer_idx - 1],
                            hidden_layers_dim[layer_idx],
                        )
                    )
            self.layers = self.layers.append(
                nn.Linear(hidden_layers_dim[-1], num_classes)
            )  # final output layer
        # self.apply(self._init_weights)

    def forward(self, x):
        if len(self.layers) == 1:
            return torch.sigmoid(self.layers[0](x))
        else:
            for layer in self.layers[:-1]:
                x = F.relu(layer(x))
            return torch.sigmoid(self.layers[-1](x))


# Train the model
def train(
    model: NeuralNetwork,
    optimizer: optim.Adam,
    criterion: nn.CrossEntropyLoss,
    dataloader_train: DataLoader,
    dataloader_val: DataLoader,
    EPOCHS,
):
    loss_train, loss_val = [], []
    acc_train, acc_val = [], []
    history_train = hl.History()  # This is a simple tool for logging
    canvas_train = hl.Canvas()  # This is a simple tool for plotting
    for epoch in range(EPOCHS):
        model.train()
        total_acc_train, total_count_train, n_train_batches, total_loss_train = (
            0,
            0,
            0,
            0,
        )
        for inputs, labels in dataloader_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, torch.max(labels, 1)[1])
            total_loss_train += loss.item()
            loss.backward()
            optimizer.step()

            pred_label = torch.argmax(outputs, dim=1)
            accuracy = (pred_label == torch.max(labels, 1)[1]).sum().item()
            total_acc_train += accuracy
            total_count_train += labels.size(0)
            n_train_batches += 1

        avg_loss_train = total_loss_train / n_train_batches
        loss_train.append(avg_loss_train)
        accuracy_train = total_acc_train / total_count_train
        acc_train.append(accuracy_train)

        total_acc_val, total_count_val, n_val_batches, total_loss_val = 0, 0, 0, 0
        with torch.no_grad():
            model.eval()
            for inputs, labels in dataloader_val:
                outputs = model(inputs)
                loss = criterion(outputs, torch.max(labels, 1)[1])
                total_loss_val += loss.item()
                pred_label = torch.argmax(outputs, dim=1)
                total_acc_val += (pred_label == torch.max(labels, 1)[1]).sum().item()
                total_count_val += labels.size(0)
                n_val_batches += 1

        avg_loss_val = total_loss_val / n_val_batches
        loss_val.append(avg_loss_val)
        accuracy_val = total_acc_val / total_count_val
        acc_val.append(accuracy_val)

        if epoch % 1 == 0:
            history_train.log(
                epoch,
                train_loss=avg_loss_train,
                train_accuracy=accuracy_train,
                val_loss=avg_loss_val,
                val_accuracy=accuracy_val,
            )  # ,

            with canvas_train:
                canvas_train.draw_plot(
                    [history_train["train_loss"], history_train["val_loss"]]
                )
                canvas_train.draw_plot(
                    [history_train["train_accuracy"], history_train["val_accuracy"]]
                )

    return loss_train, acc_train, loss_val, acc_val


def test(model, dataloader_test, criterion):
    model.eval()
    total_loss_test, total_acc_test = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader_test:
            logits = model(inputs)

            labels = torch.max(labels, 1)[1]

            loss = criterion(logits, labels)
            total_loss_test += loss.item()

            _, preds = torch.max(logits, 1)
            total_acc_test += torch.sum(preds == labels).item()

    avg_loss_test = total_loss_test / len(dataloader_test)
    accuracy_test = total_acc_test / len(dataloader_test.dataset)

    print(f"Test Loss: {avg_loss_test:.4f}, Test Accuracy: {accuracy_test:.4f}")

    return avg_loss_test, accuracy_test
