
import torch
import jumanji
import jumanji.wrappers
from memory import *
from training import *
from utils import *
import config
import os
import csv
def main():
    set_seed(config.SEED)
    csv_file = f"episode_rewards_graphcoloring.csv"
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Episode", "Reward"])
    env_base = jumanji.make('GraphColoring-v0')
    env = jumanji.wrappers.JumanjiToGymWrapper(env_base)
    env.action_space.seed(config.SEED)
    env.observation_space.seed(config.SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = DIPO(
        state_dim=config.STATE_DIM,
        action_dim=env.action_space.n,
        device=device,
        gamma=0.99,
        tau=config.TAU,
        actor_lr=config.ACTOR_LR,
        critic_lr=config.CRITIC_LR,
        n_timesteps=config.N_TIMESTEPS,
        temperature=config.TEMPERATURE,
        noise_ratio=config.NOISE_RATIO,
        beta_schedule=config.BETA_SCHEDULE,
        policy_type=config.POLICY_TYPE,
        alpha=config.ALPHA,
        use_ddim=config.USE_DDIM,
        ddim_steps=config.DDIM_STEPS,
        ddim_eta=config.DDIM_ETA,
        distill_interval=config.DISTILL_INTERVAL,
        K_epochs=config.K_EPOCHS,
        eps_clip=config.EPS_CLIP,
        initial_temperature=config.INITIAL_TEMPERATURE,
        final_temperature=config.FINAL_TEMPERATURE,
        temperature_decay_steps=config.TEMPERATURE_DECAY_STEPS
    )

    memory = ReplayMemory(capacity=int(config.MEMORY_CAPACITY*2))
    diffusion_memory = DiffusionMemory(capacity=int(config.MEMORY_CAPACITY/5))
    highreward_memory = HighRewardTrajectoryMemory(capacity=config.EXPERT_MEMORY_CAPACITY, action_dim=env.action_space.n)

    for ep in range(config.NUM_EPISODES):
        s, _ = env.reset(seed=config.SEED)
        ep_reward = 0.0
        done = False
        terminal = False
        states_list, actions_list = [], []
        while not done and not terminal:
            action_mask = s["action_mask"]
            a, log_p = agent.select_action(s, mask=action_mask, eval=False)
            s_next, r, done, terminal, info = env.step(a)
            done = not done
            memory.push(s, a, r, s_next, float(done), action_mask, log_p)
            a_onehot = torch.nn.functional.one_hot(torch.tensor(a), num_classes=env.action_space.n).float().numpy()
            diffusion_memory.append(s, a_onehot)
            states_list.append(s)
            actions_list.append(a)
            s = s_next
            ep_reward += r
            mix_ratio = max(1 - ep / 400, 0)
            agent.train(memory, diffusion_memory, highreward_memory, batch_size=config.BATCH_SIZE, mix_ratio=mix_ratio,
                        critic_update_steps=10, actor_update_interval=8)

        highreward_memory.add_trajectory(states_list, actions_list, ep_reward)
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([ep, ep_reward])
    print("Done.")


if __name__ == "__main__":
    main()
