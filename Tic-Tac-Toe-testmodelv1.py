import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import random
from typing import Dict, List, Tuple
import math

# ========================================
# ゲーム環境: Tic-Tac-Toe
# ========================================
class TicTacToe:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.board = np.zeros((3, 3), dtype=np.float32)
        self.current_player = 1  # 1 or -1
        self.done = False
        self.winner = None
        return self.get_state()
    
    def get_state(self):
        """盤面を返す (3, 3)"""
        return self.board.copy()
    
    def get_legal_actions(self):
        """合法手のリスト"""
        return [i for i in range(9) if self.board[i // 3, i % 3] == 0]
    
    def step(self, action):
        """行動を実行"""
        if self.done:
            raise ValueError("Game already finished")
        
        row, col = action // 3, action % 3
        
        if self.board[row, col] != 0:
            # 違法手 -> ゲーム終了、負け
            self.done = True
            self.winner = -self.current_player
            return self.get_state(), -1, True
        
        # 石を置く
        self.board[row, col] = self.current_player
        
        # 勝敗判定
        if self.check_win(self.current_player):
            self.done = True
            self.winner = self.current_player
            return self.get_state(), 1, True
        
        # 引き分け判定
        if len(self.get_legal_actions()) == 0:
            self.done = True
            self.winner = 0
            return self.get_state(), 0, True
        
        # 手番交代
        self.current_player *= -1
        return self.get_state(), 0, False
    
    def check_win(self, player):
        """勝利判定"""
        board = self.board
        
        # 横
        for i in range(3):
            if np.all(board[i, :] == player):
                return True
        
        # 縦
        for i in range(3):
            if np.all(board[:, i] == player):
                return True
        
        # 斜め
        if board[0, 0] == player and board[1, 1] == player and board[2, 2] == player:
            return True
        if board[0, 2] == player and board[1, 1] == player and board[2, 0] == player:
            return True
        
        return False
    
    def render(self):
        """盤面表示"""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        for row in self.board:
            print(' '.join(symbols[int(x)] for x in row))
        print()


# ========================================
# ニューラルネットワーク
# ========================================
class Encoder(nn.Module):
    """(z_prev, a_prev, s) -> z"""
    def __init__(self, z_dim=32):
        super().__init__()
        self.z_dim = z_dim
        
        # 盤面を処理
        self.board_encoder = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # 行動埋め込み
        self.action_embed = nn.Embedding(10, 16)  # 0-8 + 初期値9
        
        # GRUで統合
        self.gru = nn.GRUCell(64 + 16, z_dim)
    
    def forward(self, z_prev, a_prev, s):
        """
        z_prev: (batch, z_dim)
        a_prev: (batch,) 行動ID (0-8, 初期は9)
        s: (batch, 3, 3) 盤面
        """
        batch_size = s.shape[0]
        
        # 盤面をフラット化
        s_flat = s.reshape(batch_size, -1)
        
        # 盤面エンコード
        board_features = self.board_encoder(s_flat)
        
        # 行動埋め込み
        action_features = self.action_embed(a_prev)
        
        # 結合
        combined = torch.cat([board_features, action_features], dim=-1)
        
        # GRUで更新
        z_next = self.gru(combined, z_prev)
        
        return z_next


class Hippocampus(nn.Module):
    """海馬: (z, e) -> z_next 予測"""
    def __init__(self, z_dim=32, num_emotions=4, num_tokens=256):
        super().__init__()
        self.z_dim = z_dim
        self.num_emotions = num_emotions
        self.num_tokens = num_tokens
        
        # 固定数の学習可能メモリトークン
        self.memory = nn.Parameter(torch.randn(num_tokens, z_dim))
        
        # 感情埋め込み
        self.emotion_embed = nn.Embedding(num_emotions, 16)
        
        # クエリ投影
        self.query_proj = nn.Linear(z_dim + 16, z_dim)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(z_dim, num_heads=4, batch_first=True)
        
        # 出力MLP
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, z_dim)
        )
    
    def predict(self, z, e):
        """
        z: (batch, z_dim)
        e: (batch,) 感情ID
        -> z_next: (batch, z_dim)
        """
        batch_size = z.shape[0]
        
        # 感情埋め込み
        e_vec = self.emotion_embed(e)
        
        # クエリ作成
        query = torch.cat([z, e_vec], dim=-1)
        query = self.query_proj(query).unsqueeze(1)  # (batch, 1, z_dim)
        
        # メモリを全バッチで共有
        memory = self.memory.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, num_tokens, z_dim)
        
        # Cross-attention
        attended, _ = self.cross_attn(query, memory, memory)
        attended = attended.squeeze(1)  # (batch, z_dim)
        
        # MLP
        z_next = self.mlp(attended)
        
        return z_next


class PolicyNetwork(nn.Module):
    """z -> 感情分布"""
    def __init__(self, z_dim=32, num_emotions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_emotions)
        )
    
    def forward(self, z):
        logits = self.net(z)
        return F.softmax(logits, dim=-1)


class Motor(nn.Module):
    """(z, e) -> 行動分布"""
    def __init__(self, z_dim=32, num_emotions=4, num_actions=9):
        super().__init__()
        self.emotion_embed = nn.Embedding(num_emotions, 16)
        
        self.net = nn.Sequential(
            nn.Linear(z_dim + 16, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
    
    def forward(self, z, e):
        """
        z: (batch, z_dim)
        e: (batch,) 感情ID
        -> action_logits: (batch, 9)
        """
        e_vec = self.emotion_embed(e)
        x = torch.cat([z, e_vec], dim=-1)
        logits = self.net(x)
        return logits


class ValueNetwork(nn.Module):
    """z -> 勝率予測"""
    def __init__(self, z_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # -1 ~ 1
        )
    
    def forward(self, z):
        return self.net(z).squeeze(-1)


# ========================================
# MCTS
# ========================================
class MCTS:
    def __init__(self, num_emotions=4, c_puct=1.0, device='cuda'):
        self.num_emotions = num_emotions
        self.c_puct = c_puct
        self.device = device
        self.Q = defaultdict(float)  # Q[z_hash, e]
        self.N = defaultdict(int)    # N[z_hash, e]
    
    def search(self, z, hippocampus, value_net, num_simulations=50):
        """
        z: (z_dim,) tensor
        -> emotion_visits: {e: count}
        """
        z_hash = self._hash_z(z)
        
        for _ in range(num_simulations):
            e = self._select_emotion_ucb(z_hash)
            
            # 海馬で次状態予測
            z_next = hippocampus.predict(z.unsqueeze(0), torch.tensor([e]).to(self.device)).squeeze(0)
            
            # 価値評価
            v = value_net(z_next.unsqueeze(0)).item()
            
            # 統計更新
            key = (z_hash, e)
            self.Q[key] = (self.Q[key] * self.N[key] + v) / (self.N[key] + 1)
            self.N[key] += 1
        
        # 訪問回数を返す
        visits = {e: self.N[(z_hash, e)] for e in range(self.num_emotions)}
        return visits
    
    def _select_emotion_ucb(self, z_hash):
        """UCB1で感情選択"""
        total_visits = sum(self.N[(z_hash, e)] for e in range(self.num_emotions))
        
        if total_visits == 0:
            return random.randint(0, self.num_emotions - 1)
        
        best_score = -float('inf')
        best_emotion = 0
        
        for e in range(self.num_emotions):
            key = (z_hash, e)
            n = self.N[key]
            
            if n == 0:
                return e  # 未探索を優先
            
            q = self.Q[key]
            u = self.c_puct * math.sqrt(math.log(total_visits) / n)
            score = q + u
            
            if score > best_score:
                best_score = score
                best_emotion = e
        
        return best_emotion
    
    def _hash_z(self, z):
        """zをハッシュ化（簡易版）"""
        return tuple(z.detach().cpu().numpy().round(2))


# ========================================
# 学習エージェント
# ========================================
class EmotionalAgent:
    def __init__(self, device='cuda', z_dim=32, num_emotions=4, lr=1e-3):
        self.device = device
        self.z_dim = z_dim
        self.num_emotions = num_emotions
        
        # モジュール
        self.encoder = Encoder(z_dim).to(device)
        self.hippocampus = Hippocampus(z_dim, num_emotions, num_tokens=256).to(device)
        self.policy_net = PolicyNetwork(z_dim, num_emotions).to(device)
        self.motor = Motor(z_dim, num_emotions, num_actions=9).to(device)
        self.value_net = ValueNetwork(z_dim).to(device)
        
        # MCTS
        self.mcts = MCTS(num_emotions, device=device)
        
        # Optimizer (Valueは別)
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.hippocampus.parameters()) +
            list(self.policy_net.parameters()) +
            list(self.motor.parameters()),
            lr=lr
        )
        
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)
    
    def self_play_episode(self):
        """1試合の自己対戦"""
        game = TicTacToe()
        s = game.reset()
        
        # 初期化
        z = torch.zeros(1, self.z_dim).to(self.device)
        a_prev = torch.tensor([9]).to(self.device)  # 初期値
        
        episode_buffer = []
        
        while not game.done:
            # エンコード
            s_tensor = torch.from_numpy(s).float().unsqueeze(0).to(self.device)
            z = self.encoder(z, a_prev, s_tensor)
            
            # 合法手マスク
            legal_actions = game.get_legal_actions()
            
            # MCTS
            emotion_visits = self.mcts.search(
                z.squeeze(0),
                self.hippocampus,
                self.value_net,
                num_simulations=50
            )
            
            # 感情サンプリング
            e = self._sample_emotion(emotion_visits)
            
            # 行動生成
            action_logits = self.motor(z, torch.tensor([e]).to(self.device))
            
            # 合法手のみに制限
            action_probs = F.softmax(action_logits, dim=-1).squeeze(0)
            action_probs_masked = torch.zeros(9).to(self.device)
            action_probs_masked[legal_actions] = action_probs[legal_actions]
            action_probs_masked /= action_probs_masked.sum()
            
            a = torch.multinomial(action_probs_masked, 1).item()
            
            # 実行
            s_next, reward, done = game.step(a)
            
            # 次状態エンコード（detachして計算グラフを保存）
            s_next_tensor = torch.from_numpy(s_next).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                z_next_actual = self.encoder(z.detach(), torch.tensor([a]).to(self.device), s_next_tensor)
            
            # ========== 即座に学習（Dynamics, Motor, Policy） ==========
            
            # 1. Dynamics
            z_next_pred = self.hippocampus.predict(z, torch.tensor([e]).to(self.device))
            L_dynamics = F.mse_loss(z_next_pred, z_next_actual)
            
            # 2. Motor
            a_target = torch.tensor([a]).to(self.device)
            L_motor = F.cross_entropy(action_logits, a_target)
            
            # 3. Policy
            emotion_probs = self.policy_net(z)
            target_probs = self._normalize_visits(emotion_visits)
            target_probs_tensor = torch.tensor(target_probs).float().unsqueeze(0).to(self.device)
            L_policy = F.kl_div(
                torch.log(emotion_probs + 1e-8),
                target_probs_tensor,
                reduction='batchmean'
            )
            
            # 更新
            total_loss = L_dynamics + L_motor + L_policy
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # ========== Valueは保存 ==========
            episode_buffer.append({
                'z': z.detach().clone(),
                'player': game.current_player
            })
            
            # 次へ
            s = s_next
            z = z_next_actual  # 次のステップのzとして使う
            a_prev = torch.tensor([a]).to(self.device)
        
        # ========== 試合終了後、Value学習 ==========
        if game.winner is not None:
            outcome = game.winner
        else:
            outcome = 0
        
        for t in reversed(episode_buffer):
            v_pred = self.value_net(t['z'])
            v_target = torch.tensor([outcome]).float().to(self.device)
            
            L_value = F.mse_loss(v_pred, v_target)
            
            self.value_optimizer.zero_grad()
            L_value.backward()
            self.value_optimizer.step()
            
            # 手番交代
            outcome *= -1
        
        return game.winner
    
    def _sample_emotion(self, emotion_visits):
        """訪問回数に基づいて感情サンプリング"""
        emotions = list(emotion_visits.keys())
        visits = list(emotion_visits.values())
        
        if sum(visits) == 0:
            return random.choice(emotions)
        
        probs = np.array(visits, dtype=np.float32)
        probs /= probs.sum()
        
        return np.random.choice(emotions, p=probs)
    
    def _normalize_visits(self, emotion_visits):
        """訪問回数を確率分布に"""
        total = sum(emotion_visits.values())
        if total == 0:
            return [1.0 / self.num_emotions] * self.num_emotions
        
        probs = [emotion_visits.get(e, 0) / total for e in range(self.num_emotions)]
        return probs
    
    def evaluate(self, num_games=100):
        """ランダムプレイヤーと対戦"""
        wins = 0
        draws = 0
        losses = 0
        
        for _ in range(num_games):
            game = TicTacToe()
            s = game.reset()
            
            z = torch.zeros(1, self.z_dim).to(self.device)
            a_prev = torch.tensor([9]).to(self.device)
            
            is_agent_turn = True
            
            while not game.done:
                if is_agent_turn:
                    # エージェント
                    s_tensor = torch.from_numpy(s).float().unsqueeze(0).to(self.device)
                    z = self.encoder(z, a_prev, s_tensor)
                    
                    emotion_visits = self.mcts.search(
                        z.squeeze(0),
                        self.hippocampus,
                        self.value_net,
                        num_simulations=30
                    )
                    e = max(emotion_visits, key=emotion_visits.get)
                    
                    action_logits = self.motor(z, torch.tensor([e]).to(self.device))
                    legal_actions = game.get_legal_actions()
                    
                    action_probs = F.softmax(action_logits, dim=-1).squeeze(0)
                    action_probs_masked = torch.zeros(9).to(self.device)
                    action_probs_masked[legal_actions] = action_probs[legal_actions]
                    
                    a = action_probs_masked.argmax().item()
                    a_prev = torch.tensor([a]).to(self.device)
                else:
                    # ランダム
                    legal_actions = game.get_legal_actions()
                    a = random.choice(legal_actions)
                
                s, reward, done = game.step(a)
                is_agent_turn = not is_agent_turn
            
            if game.winner == 1:
                wins += 1
            elif game.winner == 0:
                draws += 1
            else:
                losses += 1
        
        return wins, draws, losses


# ========================================
# メイン
# ========================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    agent = EmotionalAgent(device=device, z_dim=32, num_emotions=4, lr=1e-3)
    
    print("Starting training...")
    
    for iteration in range(100):
        # 自己対戦
        results = {'X': 0, 'O': 0, 'Draw': 0}
        
        for _ in range(10):
            winner = agent.self_play_episode()
            
            if winner == 1:
                results['X'] += 1
            elif winner == -1:
                results['O'] += 1
            else:
                results['Draw'] += 1
        
        # 評価
        if iteration % 10 == 0:
            wins, draws, losses = agent.evaluate(num_games=50)
            print(f"Iteration {iteration}: Self-play {results} | vs Random: W={wins} D={draws} L={losses}")
    
    print("Training complete!")


if __name__ == '__main__':
    main()
