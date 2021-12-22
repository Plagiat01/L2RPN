from torch_model_wrapper import TorchWrapper
import torch
import torch.nn as nn

from utils import RANDOM_SEED

class NN(nn.Module):
  def __init__(self, input_features, output_features):
    super().__init__()

    self.linears = nn.Sequential(
                    nn.Linear(input_features, 2048),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(2048, output_features)
                  )
  
  def forward(self, x):
    return self.linears(x)

class DenseNN:
  def __init__(self, input_features, nb_action):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss = nn.MSELoss()
    network = NN(input_features, nb_action)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    network.to(device)

    self.wrapper = TorchWrapper(network, device, optimizer, loss)
    print(f"{self.wrapper.get_parameters(trainable=True)} trainable parameters.")

  def predict(self, X):
    return self.wrapper.predict(X)

  def fit(self, X, y, batch_size, shuffle):
    self.wrapper.fit(X, y, verbose=0, batch_size=batch_size, shuffle=shuffle)
  
  def copy_weights(self, model):
    model.wrapper.nn.load_state_dict(self.wrapper.nn.state_dict())
  
  def save(self, filename):
    self.wrapper.save(filename)
  
  def load(self, filename):
    self.wrapper.load(filename)