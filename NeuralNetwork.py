import torch.nn as nn
import torch
import numpy as np
import torch.utils.data as data
from torch.utils.data import WeightedRandomSampler
import torch.optim as optim

from BinaryClassificationModel import BinaryClassificationModel


class SimpleClassifier(nn.Module):
    def __init__(self, num_inputs, num_hidden1, num_hidden2, num_outputs):
        # Initialize the modules we need to build the network
        super().__init__()
        self.lin1 = nn.Linear(num_inputs, num_hidden1)
        self.d1 = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(num_hidden1)
        self.act1 = nn.ReLU()

        self.lin2 = nn.Linear(num_hidden1, num_hidden2)
        self.d2 = nn.Dropout(0.3)
        self.bn2 = nn.BatchNorm1d(num_hidden2)
        self.act2 = nn.ReLU()

        self.lin3 = nn.Linear(num_hidden2, num_outputs)

    def forward(self, x):
        # Perform the calculation of the model to determine the prediction
        x = self.lin1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.d1(x)

        x = self.lin2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.d2(x)

        x = self.lin3(x)
        return x


class NeuralNetwork(BinaryClassificationModel):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'NeuralNetwork'
        self.NUMBER_INPUTS = 16
        self.NUMBER_EPOCHS = 100
        self.BATCH_SIZE = 64
        self.LEARNING_RATE = 0.005
        self.model = SimpleClassifier(num_inputs=self.NUMBER_INPUTS,
                                      num_hidden1=1000,
                                      num_hidden2=900,
                                      num_outputs=2)

    def predict(self, X):
        self.model.eval()

        inputs = torch.from_numpy(np.array(X))
        preds = self.model(inputs.float())
        preds = torch.sigmoid(preds)
        preds = torch.softmax(preds, axis=-1)
        preds = preds.detach().numpy()
        preds = preds.squeeze()

        return [pred[1] for pred in preds]

    def fit(self, X, y) -> None:
        train_dataset = data.TensorDataset(torch.from_numpy(np.array(X)),
                                           self.one_hot_class(y))
        classes = {0: 0.2, 1: 0.6}
        y_classified = [int(np.argmax(y)) for _, y in train_dataset]
        example_weights = [classes[e] for e in y_classified]

        sampler = WeightedRandomSampler(example_weights, len(train_dataset))
        train_data_loader = data.DataLoader(train_dataset,
                                            batch_size=self.BATCH_SIZE,
                                            sampler=sampler)

        MSELoss = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.LEARNING_RATE)

        self.model.train()

        for epoch in range(self.NUMBER_EPOCHS):
            sum_loss = 0
            for inputs, labels in train_data_loader:
                optimizer.zero_grad()
                preds = self.model(inputs.float())
                preds = torch.sigmoid(preds)
                loss = MSELoss(preds, labels.float())

                loss.backward()
                optimizer.step()

                sum_loss += loss

            print(epoch, sum_loss / X.shape[0])

    def one_hot_class(self, dataset):
        dataset = np.array(dataset)
        return torch.tensor([[int(i == x) for i in range(2)] for x in dataset])
