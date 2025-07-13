import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import optuna
import csv

# Dataset Class
class MyDataset(Dataset):
    def __init__(self, file_path):
        self.samples = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                hours = float(row[0])
                marks = float(row[1])
                grade = float(row[2])
                self.samples.append((hours, marks, grade))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        hours, marks, grade = self.samples[idx]
        x = torch.tensor([hours], dtype=torch.float32)
        y = torch.tensor([marks, grade], dtype=torch.float32)
        return x, y

#Load Data
csv_path ="pytorch_project/csv_folder/csvpost_files.csv"
dataset = MyDataset(csv_path)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden1 = trial.suggest_int("hidden1", 32, 128)
    hidden2 = trial.suggest_int("hidden2", 32, 128)
    act_name = trial.suggest_categorical("activation", ["relu", "tanh", "sigmoid", "gelu", "leaky_relu", "silu"])

    activation_map = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "gelu": nn.GELU(),
        "leaky_relu": nn.LeakyReLU(),
        "silu": nn.SiLU(),
    }
    activation = activation_map[act_name]

    model = nn.Sequential(
        nn.Linear(1, hidden1),
        activation,
        nn.Linear(hidden1, hidden2),
        activation,
        nn.Linear(hidden2, 2)
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(10):  # Increase to 50+ for real training
        for x, y in loader:
            preds = model(x)
            loss = loss_fn(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if trial.number == 0 or loss.item() < objective.best_loss:
        torch.save(model.state_dict(),"pytorch_project/csv_folder/pytorch_traindata.pth")
        torch.save({
            "hidden1": hidden1,
            "hidden2": hidden2,
            "activation": act_name
        }, "pytorch_project/csv_folder/model_config.pth")
        objective.best_loss = loss.item()
    return loss.item()

objective.best_loss = float('inf')
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)
print("Best Hyperparams:", study.best_trial.params)


