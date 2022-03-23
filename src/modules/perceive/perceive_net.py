import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, mlp_hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class Perceive(nn.Module):
    def __init__(self, state_dim, obs_dim, args):
        super(Perceive, self).__init__()
        self.args = args
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.perceive_hidden_dim = self.args.perceive_hidden_dim
        self.perceive_dim = self.args.perceive_dim

        # * student
        self.view_obs_net = MLP(self.obs_dim, self.perceive_hidden_dim * 2, self.perceive_hidden_dim)
        self.project_net = MLP(self.perceive_hidden_dim, self.perceive_hidden_dim, self.perceive_dim)

        # * teacher
        self.teacher_view_obs_net = MLP(self.obs_dim, self.perceive_hidden_dim * 2, self.perceive_hidden_dim)
        self.teacher_project_net = MLP(self.perceive_hidden_dim, self.perceive_hidden_dim, self.perceive_dim)
        

    def calc_student(self, inputs):
        representation = self.view_obs_net(inputs)    
        project = self.project_net(representation)
        return project

    def calc_teacher(self, inputs):
        representation = self.teacher_view_obs_net(inputs)
        project = self.teacher_project_net(representation)
        return project


    def update(self):
        for param_o, param_t in zip(self.view_obs_net.parameters(), self.teacher_view_obs_net.parameters()):
            param_t.data = param_t.data * self.args.tau + param_o.data * (1. - self.args.tau)
            
        for param_o, param_t in zip(self.project_net.parameters(), self.teacher_project_net.parameters()):
            param_t.data = param_t.data * self.args.tau + param_o.data * (1. - self.args.tau)


    def update_parameters(self):
        return list(self.view_obs_net.parameters()) \
         + list(self.project_net.parameters())
