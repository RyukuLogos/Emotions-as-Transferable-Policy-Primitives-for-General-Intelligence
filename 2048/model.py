import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List

# Emotions (discrete policy primitives)
EMOTIONS = ["Aggressive", "Cautious", "Corner", "Patient", "Exploratory"]
NUM_EMOTIONS = len(EMOTIONS)

class Hippocampus(nn.Module):
    """Predicts next z given current z and condition (emotion or query)"""
    
    def __init__(self, z_dim=128, num_slots=4, hidden_dim=256):
        super().__init__()
        self.z_dim = z_dim
        self.num_slots = num_slots
        self.total_dim = z_dim * num_slots
        
        # Memory bank (learnable)
        self.memory = nn.Parameter(torch.randn(64, hidden_dim))
        
        # Attention
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # MLP for combining z and condition
        self.condition_proj = nn.Linear(NUM_EMOTIONS, hidden_dim)
        self.z_proj = nn.Linear(self.total_dim, hidden_dim)
        
        # Output projection
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.total_dim)
        )
    
    def forward(self, z, condition):
        """
        z: (batch, num_slots, z_dim)
        condition: (batch, condition_dim) - one-hot emotion or query vector
        """
        batch_size = z.shape[0]
        
        # Flatten z
        z_flat = z.reshape(batch_size, -1)  # (batch, total_dim)
        
        # Project inputs
        z_proj = self.z_proj(z_flat)  # (batch, hidden_dim)
        
        # Handle condition (emotion one-hot or query vector)
        if condition.shape[-1] == NUM_EMOTIONS:
            cond_proj = self.condition_proj(condition)
        else:
            cond_proj = condition  # Already projected query
        
        # Combine as query
        query = (z_proj + cond_proj).unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Retrieve from memory
        memory_expanded = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        attn_out, _ = self.attn(query, memory_expanded, memory_expanded)
        attn_out = attn_out.squeeze(1)  # (batch, hidden_dim)
        
        # Combine and project to z'
        combined = torch.cat([z_proj, attn_out], dim=-1)
        z_next_flat = self.output_mlp(combined)
        z_next = z_next_flat.reshape(batch_size, self.num_slots, self.z_dim)
        
        return z_next


class Actor(nn.Module):
    """Transformer that outputs z, query, and action in parallel"""
    
    def __init__(self, obs_dim=16, z_dim=128, num_slots=4, 
                 d_model=256, nhead=4, num_layers=4, seq_len=64):
        super().__init__()
        self.z_dim = z_dim
        self.num_slots = num_slots
        self.total_z_dim = z_dim * num_slots
        self.seq_len = seq_len
        
        # Input embeddings
        self.obs_embed = nn.Linear(obs_dim, d_model)
        self.z_embed = nn.Linear(self.total_z_dim, d_model)
        self.emotion_embed = nn.Embedding(NUM_EMOTIONS, d_model)
        self.action_embed = nn.Embedding(4, d_model)  # 4 actions in 2048
        self.memory_embed = nn.Linear(self.total_z_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads (parallel outputs)
        self.z_head = nn.Linear(d_model, self.total_z_dim)
        self.query_head = nn.Linear(d_model, d_model)  # Query for hippocampus
        self.action_head = nn.Linear(d_model, 4)  # Action logits
    
    def forward(self, obs, z_prev, memory_prev, emotion_prev, action_prev):
        """
        All inputs: (batch, seq_len, ...)
        Returns: z, query, action_logits (all for last timestep)
        """
        batch_size = obs.shape[0]
        seq_len = obs.shape[1]
        
        # Embed each component
        obs_emb = self.obs_embed(obs)  # (batch, seq, d_model)
        z_emb = self.z_embed(z_prev.reshape(batch_size, seq_len, -1))
        emotion_emb = self.emotion_embed(emotion_prev)
        action_emb = self.action_embed(action_prev)
        memory_emb = self.memory_embed(memory_prev.reshape(batch_size, seq_len, -1))
        
        # Combine (simple sum for now)
        tokens = obs_emb + z_emb + emotion_emb + action_emb + memory_emb
        
        # Add positional encoding
        tokens = tokens + self.pos_encoding[:, :seq_len, :]
        
        # Transformer
        out = self.transformer(tokens)  # (batch, seq, d_model)
        
        # Use last token for predictions
        final = out[:, -1, :]  # (batch, d_model)
        
        # Parallel outputs
        z = self.z_head(final).reshape(batch_size, self.num_slots, self.z_dim)
        query = self.query_head(final)
        action_logits = self.action_head(final)
        
        return z, query, action_logits


class ValueNetwork(nn.Module):
    """Estimates value of a state z"""
    
    def __init__(self, z_dim=128, num_slots=4):
        super().__init__()
        total_dim = z_dim * num_slots
        
        self.net = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, z):
        """z: (batch, num_slots, z_dim)"""
        z_flat = z.reshape(z.shape[0], -1)
        return self.net(z_flat).squeeze(-1)


class PolicyNetwork(nn.Module):
    """Outputs policy over emotions given z"""
    
    def __init__(self, z_dim=128, num_slots=4):
        super().__init__()
        total_dim = z_dim * num_slots
        
        self.net = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_EMOTIONS)
        )
    
    def forward(self, z):
        """z: (batch, num_slots, z_dim)"""
        z_flat = z.reshape(z.shape[0], -1)
        return self.net(z_flat)


class MCTS:
    """Monte Carlo Tree Search for emotion selection"""
    
    def __init__(self, hippocampus, value_net, num_simulations=8, ucb_constant=2.0):
        self.hippocampus = hippocampus
        self.value_net = value_net
        self.num_simulations = num_simulations
        self.ucb_constant = ucb_constant  # Increased for more exploration
    
    @torch.no_grad()
    def search(self, z, policy_logits):
        """
        z: (batch, num_slots, z_dim)
        policy_logits: (batch, num_emotions)
        Returns: best_emotion (batch,), visit_counts (batch, num_emotions)
        """
        batch_size = z.shape[0]
        device = z.device
        
        # Initialize visit counts and values
        visit_counts = torch.zeros(batch_size, NUM_EMOTIONS, device=device)
        total_values = torch.zeros(batch_size, NUM_EMOTIONS, device=device)
        
        # Prior probabilities
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
                
                # Exploration bonus (higher for less-visited actions)
                total_visits = visit_counts.sum(dim=-1, keepdim=True)
                exploration = prior_probs * torch.sqrt(total_visits + 1) / (1 + visit_counts)
                
                # UCB score
                ucb_scores = mean_values + self.ucb_constant * exploration
                
                # Select best UCB action
                selected = ucb_scores.argmax(dim=-1)
            
            # Simulate each selected emotion
            emotion_onehot = F.one_hot(selected, NUM_EMOTIONS).float()
            z_sim = self.hippocampus(z, emotion_onehot)
            values = self.value_net(z_sim)
            
            # Update statistics
            for b in range(batch_size):
                e = selected[b].item()
                visit_counts[b, e] += 1
                total_values[b, e] += values[b].item()
        
        # Select emotion based on visit counts (use temperature for exploration)
        # Softmax with temperature to convert visit counts to probabilities
        temperature = 0.5  # Lower = more deterministic
        visit_probs = F.softmax(visit_counts / temperature, dim=-1)
        
        # During training: sample from visit distribution
        # During inference: argmax
        best_emotion = torch.multinomial(visit_probs, 1).squeeze(-1)
        
        return best_emotion, visit_counts


class ObservationEncoder(nn.Module):
    """Encodes raw observation to z"""
    
    def __init__(self, obs_dim=16, z_dim=128, num_slots=4):
        super().__init__()
        self.z_dim = z_dim
        self.num_slots = num_slots
        total_dim = z_dim * num_slots
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, total_dim)
        )
    
    def forward(self, obs):
        """obs: (batch, obs_dim)"""
        z_flat = self.net(obs)
        return z_flat.reshape(obs.shape[0], self.num_slots, self.z_dim)
