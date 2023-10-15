import json
import torch
import random
import torch.nn as nn
import torch.optim as optim

from schemas_request import State

with open('config.json') as f:
    config = json.load(f)
    config['lr'] = 0.01 if 'lr' not in config else float(config['lr'])
    config['gamma'] = 0.8 if 'gamma' not in config else float(config['gamma'])
    config['epsilon'] = 0.05 if 'epsilon' not in config else float(config['epsilon'])


class MLP(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: list[int]):
        super().__init__()
        assert len(hidden) > 0, "Model needs at least 1 hidden layer"
        layers = []
        layers.append(nn.Linear(state_dim, hidden[0]))
        layers.append(nn.ReLU())
        for i in range(1, len(hidden)):
            layers.append(nn.Linear(hidden[i-1], hidden[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden[-1], action_dim))
        layers.append(nn.ReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def create_model():
    # architecture from https://arxiv.org/pdf/1803.03916.pdf
    model = MLP(8, 2, [32] * 5)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    return model, optimizer


def save_model(model: nn.Module, optimizer: optim.Optimizer, id: str):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, f'data/checkpoint_{id}.pth')
    return id


def load_model(id: str):
    model, optimizer = create_model()
    state = torch.load(f'data/checkpoint_{id}.pth')
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    return model, optimizer


def get_state(data: State):
    return torch.tensor([
        data.price,
        data.volume,
        data.rsi,
        data.macd,
        data.EMA_12,
        data.EMA_26,
        data.value_percent_in_account,
        data.value_percent_in_assets
    ], dtype=torch.float32)

def get_best_action(id, state: torch.Tensor):
    model, _ = load_model(id)
    predicted_rewards = model(state)
    if random.random() < config['epsilon']:
        return int(random.random() * len(predicted_rewards))
    return int(torch.argmax(predicted_rewards))


def learning_step(id,
          state: torch.Tensor,
          action: int,
          next_state: torch.Tensor,
          reward: float):
    model, optimizer = load_model(id)
    predicted_rewards = model(state)
    predicted_rewards_next = model(next_state)
    target = reward + torch.max(predicted_rewards_next) * config['gamma']
    predicted_rewards_with_actual_reward = predicted_rewards.clone()
    predicted_rewards_with_actual_reward[action] = target
    loss_fn = nn.MSELoss()
    loss = loss_fn(predicted_rewards_with_actual_reward, predicted_rewards)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    save_model(model, optimizer, id)
    return float(loss)
   
    