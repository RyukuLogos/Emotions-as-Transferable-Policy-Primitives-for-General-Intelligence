import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
from tqdm import tqdm

from game2048 import Game2048
from model import (
    Actor, Hippocampus, ValueNetwork, PolicyNetwork, 
    ObservationEncoder, NUM_EMOTIONS
)


class MCTS_Fixed:
    """MCTS that uses hippocampus for emotion simulation"""
    
    def __init__(self, hippocampus, value_net, num_simulations=16, ucb_constant=2.0):
        self.hippocampus = hippocampus
        self.value_net = value_net
        self.num_simulations = num_simulations
        self.ucb_constant = ucb_constant
    
    @torch.no_grad()
    def search(self, z, policy_logits):
        """
        z: (batch, num_slots, z_dim)
        policy_logits: (batch, num_emotions)
        Returns: best_emotion (batch,), visit_counts (batch, num_emotions)
        
        理論通り: 各感情eで z→z' を海馬でシミュレーションし、value_netで評価
        """
        batch_size = z.shape[0]
        device = z.device
        
        # Initialize visit counts and values
        visit_counts = torch.zeros(batch_size, NUM_EMOTIONS, device=device)
        total_values = torch.zeros(batch_size, NUM_EMOTIONS, device=device)
        
        # Prior probabilities from policy network
        prior_probs = F.softmax(policy_logits, dim=-1)
        
        for sim in range(self.num_simulations):
            # Select emotions using UCB
            if visit_counts.sum() == 0:
                # First iteration: sample from prior
                selected = torch.multinomial(prior_probs, 1).squeeze(-1)
            else:
                # UCB formula: Q(s,a) + c * P(s,a) * sqrt(N) / (1 + n(s,a))
                mean_values = torch.where(
                    visit_counts > 0,
                    total_values / (visit_counts + 1e-8),
                    torch.zeros_like(total_values)
                )
                
                # Exploration bonus
                total_visits = visit_counts.sum(dim=-1, keepdim=True)
                exploration = prior_probs * torch.sqrt(total_visits + 1) / (1 + visit_counts)
                
                # UCB score
                ucb_scores = mean_values + self.ucb_constant * exploration
                
                # Select best UCB action
                selected = ucb_scores.argmax(dim=-1)
            
            # === 理論通り: 海馬でシミュレーション ===
            emotion_onehot = F.one_hot(selected, NUM_EMOTIONS).float()
            z_sim = self.hippocampus(z, emotion_onehot)  # z → z' prediction
            values = self.value_net(z_sim)  # Evaluate z'
            
            # Update statistics
            for b in range(batch_size):
                e = selected[b].item()
                visit_counts[b, e] += 1
                total_values[b, e] += values[b].item()
        
        # Select emotion based on visit counts
        temperature = 0.5
        visit_probs = F.softmax(visit_counts / temperature, dim=-1)
        best_emotion = torch.multinomial(visit_probs, 1).squeeze(-1)
        
        return best_emotion, visit_counts


class ReplayBuffer:
    """Store trajectories for training"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, trajectory):
        self.buffer.append(trajectory)
    
    def sample(self, batch_size, seq_len):
        trajectories = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        batch = []
        for traj in trajectories:
            if len(traj) < seq_len:
                padding = [traj[0]] * (seq_len - len(traj))
                traj = padding + traj
            elif len(traj) > seq_len:
                start = random.randint(0, len(traj) - seq_len)
                traj = traj[start:start + seq_len]
            batch.append(traj)
        
        return batch
    
    def __len__(self):
        return len(self.buffer)


class Trainer:
    """Training with proper model-based advantage"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # Hyperparameters
        self.z_dim = 128
        self.num_slots = 4
        self.seq_len = 64
        self.batch_size = 32
        self.lr = 1e-4
        self.gamma = 0.99
        
        # Models
        self.encoder = ObservationEncoder(
            obs_dim=16, z_dim=self.z_dim, num_slots=self.num_slots
        ).to(device)
        
        self.actor = Actor(
            obs_dim=16, z_dim=self.z_dim, num_slots=self.num_slots,
            d_model=256, nhead=4, num_layers=4, seq_len=self.seq_len
        ).to(device)
        
        # 統一的な海馬 (理論通り)
        self.hippocampus = Hippocampus(
            z_dim=self.z_dim, num_slots=self.num_slots, hidden_dim=256
        ).to(device)
        
        self.value_net = ValueNetwork(
            z_dim=self.z_dim, num_slots=self.num_slots
        ).to(device)
        
        self.policy_net = PolicyNetwork(
            z_dim=self.z_dim, num_slots=self.num_slots
        ).to(device)
        
        # MCTS with hippocampus (理論通り)
        self.mcts = MCTS_Fixed(
            self.hippocampus, self.value_net, 
            num_simulations=16, ucb_constant=2.0
        )
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.hippo_optimizer = optim.Adam(self.hippocampus.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=2000)
        
        # Stats
        self.episode_rewards = []
        self.max_tiles = []
    
    def collect_trajectory(self, epsilon=0.1):
        """Collect one episode"""
        env = Game2048()
        obs = env.reset()
        
        trajectory = []
        
        # Initialize
        z = torch.zeros(1, self.num_slots, self.z_dim).to(self.device)
        memory = torch.zeros(1, self.num_slots, self.z_dim).to(self.device)
        emotion = 0
        action = 0
        
        done = False
        episode_reward = 0
        
        while not done:
            obs_tensor = torch.FloatTensor(obs.flatten()).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Encode
                z_encoded = self.encoder(obs_tensor)
                
                # Get policy for emotion selection
                policy_logits = self.policy_net(z_encoded)
                
                # === MCTS for emotion selection (理論通り: 海馬を使う) ===
                if random.random() < epsilon:
                    emotion_next = random.randint(0, NUM_EMOTIONS - 1)
                    visit_counts = torch.ones(1, NUM_EMOTIONS) / NUM_EMOTIONS
                else:
                    emotion_next, visit_counts = self.mcts.search(z_encoded, policy_logits)
                    emotion_next = emotion_next.item()
                
                # Get action from actor
                obs_seq = obs_tensor.unsqueeze(1)
                z_seq = z.unsqueeze(1)
                memory_seq = memory.unsqueeze(1)
                emotion_seq = torch.LongTensor([[emotion]]).to(self.device)
                action_seq = torch.LongTensor([[action]]).to(self.device)
                
                z_new, query, action_logits = self.actor(
                    obs_seq, z_seq, memory_seq, emotion_seq, action_seq
                )
                z_new = z_new.squeeze(1)
                
                # === 海馬でメモリ取得 (理論通り: queryを送る) ===
                memory_new = self.hippocampus(z_new, query)
                
                # Select action
                action_probs = torch.softmax(action_logits.squeeze(0), dim=-1)
                action_next = action_probs.argmax().item()
            
            # Store transition
            trajectory.append({
                'obs': obs.copy(),
                'z': z.cpu().numpy(),
                'memory': memory.cpu().numpy(),
                'emotion': emotion,
                'action': action,
                'z_new': z_new.cpu().numpy(),
                'query': query.cpu().numpy(),
                'emotion_next': emotion_next,
                'visit_counts': visit_counts.cpu().numpy()
            })
            
            # Step
            obs, reward, done = env.step(action_next)
            episode_reward += reward
            
            # Update
            z = z_new
            memory = memory_new
            emotion = emotion_next
            action = action_next
        
        # Add returns
        returns = []
        G = 0
        for t in reversed(trajectory):
            returns.insert(0, G)
            G = G * self.gamma
        
        for t, r in zip(trajectory, returns):
            t['return'] = r
        
        self.episode_rewards.append(episode_reward)
        self.max_tiles.append(env.max_tile)
        
        return trajectory
    
    def train_step(self):
        """Single training step"""
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size, self.seq_len)
        
        # Prepare data
        obs_list = []
        z_list = []
        memory_list = []
        emotion_list = []
        action_list = []
        z_new_list = []
        returns_list = []
        visit_counts_list = []
        query_list = []
        emotion_next_list = []
        
        for traj in batch:
            obs_seq = [t['obs'].flatten() for t in traj]
            z_seq = [t['z'] for t in traj]
            memory_seq = [t['memory'] for t in traj]
            emotion_seq = [t['emotion'] for t in traj]
            action_seq = [t['action'] for t in traj]
            z_new_seq = [t['z_new'] for t in traj]
            returns_seq = [t['return'] for t in traj]
            visit_counts_seq = [t['visit_counts'] for t in traj]
            query_seq = [t['query'] for t in traj]
            emotion_next_seq = [t['emotion_next'] for t in traj]
            
            obs_list.append(obs_seq)
            z_list.append(z_seq)
            memory_list.append(memory_seq)
            emotion_list.append(emotion_seq)
            action_list.append(action_seq)
            z_new_list.append(z_new_seq)
            returns_list.append(returns_seq)
            visit_counts_list.append(visit_counts_seq)
            query_list.append(query_seq)
            emotion_next_list.append(emotion_next_seq)
        
        obs_batch = torch.FloatTensor(np.stack(obs_list)).to(self.device)
        z_batch = torch.FloatTensor(np.stack(z_list)).to(self.device)
        memory_batch = torch.FloatTensor(np.stack(memory_list)).to(self.device)
        emotion_batch = torch.LongTensor(np.stack(emotion_list)).to(self.device)
        action_batch = torch.LongTensor(np.stack(action_list)).to(self.device)
        z_new_batch = torch.FloatTensor(np.stack(z_new_list)).to(self.device)
        returns_batch = torch.FloatTensor(np.stack(returns_list)).to(self.device)
        visit_counts_batch = torch.FloatTensor(np.stack(visit_counts_list)).to(self.device)
        query_batch = torch.FloatTensor(np.stack(query_list)).to(self.device)
        emotion_next_batch = torch.LongTensor(np.stack(emotion_next_list)).to(self.device)
        
        # Forward pass
        z_pred, query_pred, action_logits = self.actor(
            obs_batch, z_batch, memory_batch, emotion_batch, action_batch
        )
        
        # === Model-based Advantage (理論通り) ===
        # 1. 海馬でメモリ取得
        z_mem = self.hippocampus(z_pred, query_pred)
        
        # 2. V(z_mem) を計算
        value_with_memory = self.value_net(z_mem)
        
        # 3. baseline として V(z_prev_mem) を使う
        with torch.no_grad():
            # 前ステップのメモリ付きzの価値
            baseline = self.value_net(memory_batch[:, -1])
        
        # 4. Advantage = V(z_mem) - baseline
        advantage = value_with_memory - baseline
        
        # === 理論通り: Actorの全出力(z, query, action)を advantage で学習 ===
        # advantage を最大化 = -advantage を最小化
        loss_actor = -advantage.mean()
        
        # Policy output
        policy_logits = self.policy_net(z_pred)
        
        # === 海馬のダイナミクス学習 (理論通り: MCTSが選んだ e で z→z' を学習) ===
        emotion_onehot = F.one_hot(emotion_next_batch[:, -1], NUM_EMOTIONS).float()
        z_next_pred = self.hippocampus(z_pred, emotion_onehot)
        loss_dynamics = F.mse_loss(z_next_pred, z_new_batch[:, -1])
        
        # Value network loss
        loss_value = F.mse_loss(value_with_memory, returns_batch[:, -1])
        
        # Policy loss (MCTS supervision)
        visit_probs = visit_counts_batch[:, -1] / (
            visit_counts_batch[:, -1].sum(dim=-1, keepdim=True) + 1e-8
        )
        loss_policy = F.kl_div(
            F.log_softmax(policy_logits, dim=-1),
            visit_probs,
            reduction='batchmean'
        )
        
        # Total loss
        total_loss = (
            1.0 * loss_actor +      # Model-based advantage
            0.5 * loss_value + 
            0.3 * loss_policy + 
            0.5 * loss_dynamics     # 海馬の学習
        )
        
        # Optimize
        self.actor_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()
        self.hippo_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.hippocampus.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        
        self.actor_optimizer.step()
        self.value_optimizer.step()
        self.policy_optimizer.step()
        self.hippo_optimizer.step()
        self.encoder_optimizer.step()
        
        losses = {
            'actor': loss_actor.item(),
            'value': loss_value.item(),
            'policy': loss_policy.item(),
            'dynamics': loss_dynamics.item(),
            'total': total_loss.item(),
            'advantage': advantage.mean().item()
        }
        
        return losses
    
    def train(self, num_episodes=1000, update_freq=4):
        """Main training loop"""
        print(f"Training on {self.device}")
        print("理論通り実装:")
        print("  - MCTSが海馬を使って感情シミュレーション")
        print("  - Model-based Advantage で Actor の全出力を学習")
        print("  - 海馬は MCTS が選んだ感情で z→z' を学習")
        
        for episode in tqdm(range(num_episodes)):
            epsilon = max(0.05, 0.5 - episode / (num_episodes * 0.7))
            trajectory = self.collect_trajectory(epsilon=epsilon)
            self.replay_buffer.push(trajectory)
            
            if episode % update_freq == 0 and len(self.replay_buffer) >= self.batch_size:
                losses = self.train_step()
                
                if episode % 50 == 0 and losses:
                    avg_reward = np.mean(self.episode_rewards[-50:])
                    max_tile = np.max(self.max_tiles[-50:])
                    avg_tile = np.mean(self.max_tiles[-50:])
                    
                    print(f"\nEp {episode} | Reward: {avg_reward:.3f} | "
                          f"MaxTile: {max_tile} (avg: {avg_tile:.0f})")
                    print(f"  Actor: {losses['actor']:.3f}, Value: {losses['value']:.3f}, "
                          f"Advantage: {losses['advantage']:.3f}")
                    print(f"  Dynamics: {losses['dynamics']:.3f}, Policy: {losses['policy']:.3f}")
        
        return self.episode_rewards, self.max_tiles
    
    def save(self, path='model_2048.pt'):
        torch.save({
            'actor': self.actor.state_dict(),
            'hippocampus': self.hippocampus.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_net': self.policy_net.state_dict(),
            'encoder': self.encoder.state_dict(),
        }, path)
        print(f"Models saved to {path}")
    
    def load(self, path='model_2048.pt'):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.hippocampus.load_state_dict(checkpoint['hippocampus'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.encoder.load_state_dict(checkpoint['encoder'])
        print(f"Models loaded from {path}")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    trainer = Trainer(device=device)
    rewards, max_tiles = trainer.train(num_episodes=500, update_freq=2)
    
    trainer.save()
    
    # Plot results
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    
    axes[0, 1].plot(max_tiles)
    axes[0, 1].set_title('Max Tile Achieved')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Max Tile')
    
    # Rolling average
    window = 50
    rolling_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    rolling_tiles = np.convolve(max_tiles, np.ones(window)/window, mode='valid')
    
    axes[1, 0].plot(rolling_rewards)
    axes[1, 0].set_title(f'Rewards (Rolling Avg, window={window})')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Reward')
    
    axes[1, 1].plot(rolling_tiles)
    axes[1, 1].set_title(f'Max Tile (Rolling Avg, window={window})')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Max Tile')
    
    plt.tight_layout()
    plt.savefig('training_fixed.png', dpi=150)
    print("Results saved to training_fixed.png")
