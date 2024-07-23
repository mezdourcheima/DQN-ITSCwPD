# MetaNet.py
import torch
import torch.nn as nn
import torch.optim as optim

class MetaNet(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(MetaNet, self).__init__()
        self.fc1 = nn.Linear(input_shape[0], 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, output_shape[0])
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict(self, state):
        state = torch.FloatTensor(state)
        self.eval()
        with torch.no_grad():
            return self.forward(state).numpy()

    def fit(self, state, target, epochs=1, verbose=0):
        state = torch.FloatTensor(state)
        target = torch.FloatTensor(target)
        self.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output = self.forward(state)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if verbose:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

    def save(self, name):
        torch.save(self.state_dict(), name)

    def load(self, name):
        self.load_state_dict(torch.load(name))
