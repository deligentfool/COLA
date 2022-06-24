from modules.agents import REGISTRY as agent_REGISTRY
from modules.perceive.perceive_net import Perceive
from modules.embedding.embedding_net import Embedding_net
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        if self.args.input == 'hidden':
            self._build_perceive(self.args.rnn_hidden_dim)
        elif self.args.input == 'obs':
            self._build_perceive(input_shape)
        self._build_embedding_net()
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None
        self.obs_center = th.zeros(1, self.args.perceive_dim).cuda()


    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        self.hidden_states = self.agent.calc_hidden(agent_inputs, self.hidden_states)
        with th.no_grad():
            if self.args.input == 'hidden':
                latent_state = self.perceive.calc_student(self.hidden_states)
            elif self.args.input == 'obs':
                latent_state = self.perceive.calc_student(agent_inputs)

            # latent_state = latent_state - latent_state.max(-1, keepdim=True)[0].detach()
            latent_state_id = F.softmax(latent_state, dim=-1).detach().max(-1)[1].unsqueeze(-1)
            latent_state_id[ep_batch['alive_allies'][:, t].reshape(*latent_state_id.size()) == 0] = self.args.perceive_dim
            latent_state_embedding = self.embedding_net(latent_state_id.squeeze(-1))
        agent_outs = self.agent.calc_value(latent_state_embedding, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                # Epsilon floor
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # With probability epsilon, we will pick an available action uniformly
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    # Zero out the unavailable actions
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return list(self.agent.parameters()) + list(self.embedding_net.parameters())

    def perceive_update_parameters(self):
        return self.perceive.update_parameters()
    
    def perceive_all_parameters(self):
        return self.perceive.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.embedding_net.load_state_dict(other_mac.embedding_net.state_dict())

    def load_perceive_state(self, other_mac):
        self.perceive.load_state_dict(other_mac.perceive.state_dict())
        self.obs_center = other_mac.obs_center.detach().clone()

    def cuda(self):
        self.agent.cuda()
        self.perceive.cuda()
        self.embedding_net.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.perceive.state_dict(), "{}/perceive.th".format(path))
        th.save(self.embedding_net.state_dict(), "{}/embedding.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.perceive.load_state_dict(th.load("{}/perceive.th".format(path), map_location=lambda storage, loc: storage))
        self.embedding_net.load_state_dict(th.load("{}/embedding.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_embedding_net(self):
        self.embedding_net = Embedding_net(self.args)

    def _build_perceive(self, obs_shape):
        state_dim = int(np.prod(self.args.state_shape))
        self.perceive = Perceive(state_dim, obs_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t <= 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
