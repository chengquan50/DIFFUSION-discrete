import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from memory import *
from models.critic import *
from models.diffusion_policy import *
def get_mixed_actions_logits(critic_target: Critic,
                             batch_size: int,
                             diffusion_memory: DiffusionMemory,
                             diffusion_memory_expert: HighRewardTrajectoryMemory,
                             encoder_fn,
                             device='cpu',
                             mix_ratio: float=0.25):
    action_dim = critic_target.q1.out_features
    num_expert_samples = int(batch_size*mix_ratio)
    num_q_samples = batch_size - num_expert_samples
    available_exp = diffusion_memory_expert.getlength()
    if available_exp < num_expert_samples:
        num_expert_samples = 0
        num_q_samples = batch_size
    states_q_list, _, _ = diffusion_memory.sample(num_q_samples)
    states_q_t = encoder_fn(states_q_list).detach().to(device)
    with torch.no_grad():
        q1_q, q2_q = critic_target(states_q_t)
        q_ = torch.min(q1_q, q2_q)
        best_actions_idx = q_.argmax(dim=1)
        B, n_actions = q_.shape
        best_actions_logits_q = -10.0 * torch.ones_like(q_)
        best_actions_logits_q[range(B), best_actions_idx] = 10.0
    states_q_final = states_q_t
    best_logits_q_final = best_actions_logits_q
    if num_expert_samples > 0:
        states_exp_list, actions_exp, _ = diffusion_memory_expert.sample(num_expert_samples)
        states_exp_t = encoder_fn(states_exp_list).detach().to(device)
        actions_exp_t = torch.FloatTensor(actions_exp).to(device)
        B2, _ = actions_exp_t.shape
        best_actions_logits_exp = -10.0 * torch.ones((B2, n_actions), device=device)
        idx_exp = actions_exp_t.argmax(dim=-1).long()
        best_actions_logits_exp[range(B2), idx_exp] = 10.0
        states_mix = torch.cat([states_q_final, states_exp_t], dim=0)
        best_logits_mix = torch.cat([best_logits_q_final, best_actions_logits_exp], dim=0)
    else:
        states_mix = states_q_final
        best_logits_mix = best_logits_q_final
    return states_mix, best_logits_mix


class DPO:
    def __init__(self,
                 state_dim,
                 action_dim,
                 device='cpu',
                 gamma=0.99,
                 tau=0.01,  
                 actor_lr=1e-4, 
                 critic_lr=3e-5,
                 n_timesteps=50,
                 temperature=1.0,
                 noise_ratio=0.05,
                 beta_schedule='linear',
                 alpha=0.1,
                 policy_type='diffusion',
                 use_ddim=True,
                 ddim_steps=10,
                 ddim_eta=0.0,
                 K_epochs=5,
                 eps_clip=0.2,
                 initial_temperature=1.0,
                 final_temperature=0.5,
                 temperature_decay_steps=20000  
                ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.log_alpha = torch.nn.Parameter(torch.log(torch.tensor(alpha, device=device)), requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=actor_lr)
        self.target_entropy = -np.log(1.0 / action_dim)
        self.policy_type = policy_type
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.temperature_decay_steps = temperature_decay_steps
        if policy_type == 'diffusion':
            self.actor = DiffusionPolicy(
                state_dim=state_dim,
                action_dim=action_dim,
                noise_ratio=noise_ratio,
                beta_schedule=beta_schedule,
                n_timesteps=n_timesteps,
                temperature=temperature,
                device=device,
                use_ddim=use_ddim,
                ddim_steps=ddim_steps,
                ddim_eta=ddim_eta
            ).to(device)
        elif policy_type == 'mlp':
            self.actor = MLPPolicy(
                state_dim=state_dim,
                action_dim=action_dim,
                temperature=temperature,
                device=device
            ).to(device)
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=1000, gamma=0.99)
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=1000, gamma=0.99)
        self.action_dim = action_dim
        self.total_it = 0
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip

    def select_action(self, state, mask=None, eval=False):
        if isinstance(state, dict):
            state_t = state
            mask_t = torch.BoolTensor(mask.copy()).unsqueeze(0).to(self.device) if mask is not None else None
        elif len(state.shape)==1:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            mask_t = torch.BoolTensor(mask).unsqueeze(0).to(self.device) if mask is not None else None
        else:
            state_t = torch.FloatTensor(state).to(self.device)
            mask_t = torch.BoolTensor(mask).to(self.device) if mask is not None else None
        action, log_prob = self.actor.sample_action(state_t, mask=mask_t, eval=eval)
        return action[0], log_prob[0] if len(action)==1 else (action, log_prob)

    def train(self, memory: ReplayMemory, diffusion_memory: DiffusionMemory, highreward_memory: HighRewardTrajectoryMemory,
            batch_size=64, mix_ratio=0.25, critic_update_steps=10, actor_update_interval=4):
        if len(memory) < batch_size:
            return
        self.total_it += 1

        new_temp = self.initial_temperature + (self.final_temperature - self.initial_temperature) * (self.total_it / self.temperature_decay_steps)
        new_temp = max(new_temp, self.final_temperature)
        self.actor.temperature = new_temp

        states_list, actions_np, rewards_np, next_states_list, dones_np, masks_np, old_log_probs_np = memory.sample(batch_size)
        states = self.actor.encode_states_batch(states_list).detach()
        next_states_all = self.actor.encode_states_batch(next_states_list).detach()
        dones_tensor = torch.tensor(dones_np, device=self.device, dtype=next_states_all.dtype).unsqueeze(1)
        next_states = next_states_all * (1 - dones_tensor)
        actions = torch.LongTensor(actions_np).to(self.device)
        rewards = torch.FloatTensor(rewards_np).to(self.device)
        dones = torch.FloatTensor(dones_np).to(self.device)
        masks_t = torch.BoolTensor(masks_np).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs_np).to(self.device)

        dataset_size = states.shape[0]
        idx_arr = np.arange(dataset_size)

        for step in range(critic_update_steps):
            np.random.shuffle(idx_arr)
            start = 0
            while start < dataset_size:
                end = min(start + batch_size, dataset_size)
                batch_idx = idx_arr[start:end]
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_rewards = rewards[batch_idx]
                batch_next_states = next_states[batch_idx]
                batch_dones = dones[batch_idx]

                with torch.no_grad():
                    q1_next, q2_next = self.critic_target(batch_next_states)
                    q_next = torch.min(q1_next, q2_next)
                    next_probs, _ = self.actor.train_forward(batch_next_states, mask=None, eval=False)
                    log_next_probs = torch.log(next_probs + 1e-8)
                    alpha_val = self.log_alpha.exp().detach()
                    soft_value_next = (next_probs * (q_next - alpha_val * log_next_probs)).sum(dim=1)
                    target_q = batch_rewards + (1 - batch_dones) * self.gamma * soft_value_next
                    target_q = target_q.detach()

                q1, q2 = self.critic(batch_states)
                current_q1 = q1.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                current_q2 = q2.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
                start = end

            for param, tgt_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                tgt_param.data.copy_(self.tau * param.data + (1 - self.tau) * tgt_param.data)

            if self.total_it < 1000:
                continue

            # Actor update block
            if step % actor_update_interval == 0:
                np.random.shuffle(idx_arr)
                start = 0
                while start < dataset_size:
                    end = min(start + batch_size, dataset_size)
                    batch_idx = idx_arr[start:end]
                    mini_states = [states_list[i] for i in batch_idx]
                    mini_masks = torch.BoolTensor(masks_np[batch_idx]).to(self.device)
                    mini_old_log_probs = old_log_probs[batch_idx]
                    batch_state_enc = self.actor.encode_states_batch(mini_states).detach()

                    final_probs, raw_logits = self.actor.train_forward(batch_state_enc, mask=mini_masks, eval=False)
                    log_probs = torch.log(final_probs + 1e-8)
                    batch_actions = torch.LongTensor(actions_np[batch_idx]).to(self.device)
                    new_log_probs = log_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                    ratio = torch.exp(new_log_probs - mini_old_log_probs)

                    q1_pi, q2_pi = self.critic(batch_state_enc)
                    q_pi = torch.min(q1_pi, q2_pi)
                    q_pi_a = q_pi.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                    soft_v = (final_probs * (q_pi - self.log_alpha.exp() * log_probs)).sum(dim=1)
                    advantage = q_pi_a - soft_v
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantage
                    actor_loss = -torch.min(surr1, surr2).mean()
                    entropy_term = -(final_probs * log_probs).sum(dim=1).mean()
                    lambda_entropy = 0.01
                    actor_loss_total = actor_loss  + lambda_entropy * entropy_term


                    
                    aux_loss = 0.0
                    if len(diffusion_memory) > 0:
                        states_best, best_actions_logits = get_mixed_actions_logits(
                            critic_target=self.critic_target,
                            batch_size=batch_size,
                            diffusion_memory=diffusion_memory,
                            diffusion_memory_expert=highreward_memory,
                            encoder_fn=self.actor.encode_states_batch,
                            device=self.device,
                            mix_ratio=mix_ratio
                        )
                        aux_loss = self.actor.loss(best_actions_logits, states_best.to(self.device))

                    lambda_aux = 0.05
                    actor_loss_total += lambda_aux * aux_loss
                    self.actor_optimizer.zero_grad()
                    actor_loss_total.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    self.actor_optimizer.step()
                    start = end

                with torch.no_grad():
                    batch_state_enc_full = self.actor.encode_states_batch(states_list).detach()
                    mask_full = torch.BoolTensor(masks_np).to(self.device)
                    final_probs_alpha, _ = self.actor.train_forward(batch_state_enc_full, mask=mask_full, eval=False)
                    log_pi_alpha = torch.log(final_probs_alpha + 1e-8)
                    entropy = -(final_probs_alpha * log_pi_alpha).sum(dim=-1).mean()
                    alpha_target_diff = entropy - self.target_entropy

                alpha_loss = -(self.log_alpha * alpha_target_diff).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

        self.actor_scheduler.step()
        self.critic_scheduler.step()
