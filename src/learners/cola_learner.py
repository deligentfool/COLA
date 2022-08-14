from asyncio.proactor_events import constants
import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class COLALearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.consensus_builder_params = list(mac.consensus_builder_update_parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.consensus_builder_optimiser = RMSprop(params=self.consensus_builder_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1



    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        inputs = self._build_inputs(batch)

        # Calculate estimated Q-Values
        mac_out = []
        hidden_states = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            hidden_states.append(self.mac.hidden_states.view(self.args.batch_size, self.args.n_agents, -1))
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        hidden_states = th.stack(hidden_states, dim=1)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        target_hidden_states = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_hidden_states.append(self.target_mac.hidden_states.view(self.args.batch_size, self.args.n_agents, -1))
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
        target_hidden_states = th.stack(target_hidden_states, dim=1)
        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        if self.args.input == 'hidden':
            origin_obs = hidden_states[:, :-1].reshape(batch.batch_size * (batch.max_seq_length - 1) * self.args.n_agents, -1).detach()
        elif self.args.input == 'obs':
            origin_obs = inputs[:, :-1].reshape(batch.batch_size * (batch.max_seq_length - 1) * self.args.n_agents, -1)

        alive_mask = batch['alive_allies'][:, :-1]
        alive_mask[:, 1:] = alive_mask[:, 1:] * (1 - terminated[:, :-1])
        valid_state_mask = (alive_mask.sum(-1, keepdim=True) > 0).flatten(start_dim=0, end_dim=1).bool()
        valid_obs_mask = valid_state_mask.unsqueeze(-1).repeat([1, self.args.n_agents, 1]).flatten(0, 1).bool()
        alive_obs_mask = alive_mask.flatten(0, 2).bool().unsqueeze(-1)

        valid_obs = th.masked_select(origin_obs, valid_obs_mask).view(-1, origin_obs.size()[-1])
        alive_obs = th.masked_select(origin_obs, alive_obs_mask).view(-1, origin_obs.size()[-1])

        obs_projection = self.mac.consensus_builder.calc_student(valid_obs)
        teacher_obs_projection = self.mac.consensus_builder.calc_teacher(valid_obs)
        real_teacher_obs_projection = self.mac.consensus_builder.calc_teacher(alive_obs)

        online_obs_prediction = obs_projection.view(-1, self.args.n_agents, self.args.consensus_builder_dim)
        teacher_obs_projection = teacher_obs_projection.view(-1, self.args.n_agents, self.args.consensus_builder_dim)
        real_teacher_obs_projection = real_teacher_obs_projection.view(-1, self.args.consensus_builder_dim).detach()

        # online_obs_prediction = online_obs_prediction - online_obs_prediction.max(dim=-1, keepdim=True)[0].detach()
        centering_teacher_obs_projection = teacher_obs_projection - self.mac.obs_center.detach()
        # centering_teacher_obs_projection = centering_teacher_obs_projection - centering_teacher_obs_projection.max(dim=-1, keepdim=True)[0].detach()

        online_obs_prediction_sharp = online_obs_prediction / self.args.online_temp
        target_obs_projection_z = F.softmax(centering_teacher_obs_projection / self.args.target_temp, dim=-1)

        contrastive_loss = - th.bmm(target_obs_projection_z.detach(), F.log_softmax(online_obs_prediction_sharp, dim=-1).transpose(1, 2))

        contrastive_mask = th.masked_select(alive_mask.flatten().unsqueeze(-1), valid_obs_mask).view(-1, self.args.n_agents)
        contrastive_mask = contrastive_mask.unsqueeze(-1)
        contrastive_mask = th.bmm(contrastive_mask, contrastive_mask.transpose(1, 2))
        contrastive_mask = contrastive_mask * (1 - th.diag_embed(th.ones(self.args.n_agents))).unsqueeze(0).to(contrastive_mask.device)

        contrastive_loss = (contrastive_loss * contrastive_mask).sum() / contrastive_mask.sum()

        # Optimise
        self.consensus_builder_optimiser.zero_grad()
        contrastive_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.consensus_builder_params, self.args.grad_norm_clip)
        self.consensus_builder_optimiser.step()

        self.mac.obs_center = (self.args.center_tau * self.mac.obs_center + (1 - self.args.center_tau) * real_teacher_obs_projection.mean(0, keepdim=True)).detach()
        self.mac.consensus_builder.update()


        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("co_loss", contrastive_loss.item(), t_env)



        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]


        with th.no_grad():
            if self.args.input == 'hidden':
                mixing_state_projection = self.mac.consensus_builder.calc_student(hidden_states)
            elif self.args.input == 'obs':
                mixing_state_projection = self.mac.consensus_builder.calc_student(inputs)
            # mixing_state_projection = mixing_state_projection - mixing_state_projection.max(-1, keepdim=True)[0].detach()
            mixing_state_projection_z = F.softmax(mixing_state_projection / self.args.online_temp, dim=-1)
            mixing_state_projection_z = mixing_state_projection_z * batch['alive_allies'].unsqueeze(-1)
            latent_state_id = mixing_state_projection_z.sum(-2).detach().max(-1)[1]
            latent_state_onehot = th.zeros(*latent_state_id.size(), self.args.consensus_builder_dim).cuda().scatter_(-1, latent_state_id.unsqueeze(-1), 1)
            latent_state_embedding = self.mac.embedding_net(latent_state_id)

            latent_state_id_count = ((latent_state_onehot[:, :-1]).sum([0, 1]) > 0).sum().float()

            if self.args.input == 'hidden':
                target_mixing_state_projection = self.target_mac.consensus_builder.calc_student(target_hidden_states)
            elif self.args.input == 'obs':
                target_mixing_state_projection = self.target_mac.consensus_builder.calc_student(inputs)
            # target_mixing_state_projection = target_mixing_state_projection - target_mixing_state_projection.max(-1, keepdim=True)[0].detach()
            target_mixing_state_projection_z = F.softmax(target_mixing_state_projection / self.args.online_temp, dim=-1)
            target_mixing_state_projection_z = target_mixing_state_projection_z * batch['alive_allies'].unsqueeze(-1)
            target_latent_state_id = target_mixing_state_projection_z.sum(-2).detach().max(-1)[1]
            target_latent_state_embedding = self.target_mac.embedding_net(target_latent_state_id)

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch['state'][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch['state'][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        rl_mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * rl_mask

        # Normal L2 loss, take mean over actual data
        rl_loss = (masked_td_error ** 2).sum() / rl_mask.sum()


        # Optimise
        self.optimiser.zero_grad()
        rl_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self.target_mac.load_consensus_builder_state(self.mac)
            self._update_targets()
            self.last_target_update_episode = episode_num


        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("rl_loss", rl_loss.item(), t_env)
            self.logger.log_stat("latent_state_id_count", latent_state_id_count.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env



    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))


    def _build_inputs(self, batch):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, :batch.max_seq_length])  # b1av
        if self.args.obs_last_action:
            inputs.append(th.cat([
                th.zeros_like(batch["actions_onehot"][:, 0]).unsqueeze(1), batch["actions_onehot"][:, :batch.max_seq_length-1]
            ], dim=1))
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.args.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, batch.max_seq_length, -1, -1))

        inputs = th.cat(inputs, dim=-1)
        return inputs
