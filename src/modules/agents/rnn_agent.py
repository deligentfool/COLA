from audioop import bias
import torch.nn as nn
import torch.nn.functional as F
import torch


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim * 2, args.n_actions)

        self.latent_state_encoder = nn.Sequential(
            nn.Linear(self.args.perceive_dim + 1, args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
        )

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()


    def calc_hidden(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        return h

    def calc_value(self, latent_state, hidden_state):
        latent_state_embedding = self.latent_state_encoder(latent_state)
        x = torch.cat([hidden_state, latent_state_embedding], dim=-1)
        q = self.fc2(x)
        return q
