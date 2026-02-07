import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Dict, List, Optional
import math

# ========================================
# ゲーム環境: 2048
# ========================================
class Game2048:
    """
    4×4グリッドで2048ゲーム
    報酬: 生存ターン数（長く生き残るほど良い）
    """
    def __init__(self):
        self.size = 4
        self.reset()
    
    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.score = 0
        self.done = False
        self.num_moves = 0
        
        # 初期タイル2つ配置
        self._add_random_tile()
        self._add_random_tile()
        
        return self.get_state()
    
    def get_state(self):
        """盤面をlog2正規化して返す (4, 4)"""
        # 0 -> 0, 2 -> 1, 4 -> 2, 8 -> 3, ..., 2048 -> 11
        state = np.zeros_like(self.board, dtype=np.float32)
        mask = self.board > 0
        state[mask] = np.log2(self.board[mask])
        return state
    
    def get_legal_actions(self):
        """合法手のリスト: 0=上, 1=右, 2=下, 3=左"""
        legal = []
        for action in range(4):
            if self._can_move(action):
                legal.append(action)
        return legal
    
    def step(self, action):
        """
        行動を実行
        action: 0=上, 1=右, 2=下, 3=左
        報酬: +1 (1ターン生存)
        """
        if self.done:
            return self.get_state(), 0, True
        
        # 盤面コピー
        old_board = self.board.copy()
        
        # スライド＆マージ
        moved = self._move(action)
        
        if not moved:
            # 動けなかった = 違法手
            self.done = True
            return self.get_state(), -10, True  # ペナルティ
        
        # 新しいタイル追加
        self._add_random_tile()
        self.num_moves += 1
        
        # 報酬: 生存ターン = +1
        reward = 1.0
        
        # ゲーム終了判定
        if not self.get_legal_actions():
            self.done = True
        
        return self.get_state(), reward, self.done
    
    def _can_move(self, action):
        """その方向に動けるかチェック"""
        temp_board = self.board.copy()
        rotated = self._rotate_board(temp_board, action)
        moved, _ = self._slide_and_merge(rotated)
        return moved
    
    def _move(self, action):
        """盤面を動かす（実際に盤面を更新）"""
        rotated = self._rotate_board(self.board, action)
        moved, rotated = self._slide_and_merge(rotated)
        
        if moved:
            self.board = self._rotate_board(rotated, (4 - action) % 4)
        
        return moved
    
    def _rotate_board(self, board, times):
        """盤面を90度反時計回りにtimes回回転（左方向に統一するため）"""
        rotated = board.copy()
        for _ in range(times):
            rotated = np.rot90(rotated)
        return rotated
    
    def _slide_and_merge(self, board):
        """左方向にスライド＆マージ"""
        moved = False
        new_board = np.zeros_like(board)
        
        for i in range(self.size):
            row = board[i]
            # 0を除去
            tiles = row[row != 0]
            
            # マージ処理
            merged = []
            skip = False
            for j in range(len(tiles)):
                if skip:
                    skip = False
                    continue
                
                if j + 1 < len(tiles) and tiles[j] == tiles[j + 1]:
                    # マージ
                    merged.append(tiles[j] * 2)
                    self.score += tiles[j] * 2
                    skip = True
                else:
                    merged.append(tiles[j])
            
            # 新しい行を作成
            new_row = np.zeros(self.size, dtype=np.int32)
            new_row[:len(merged)] = merged
            
            if not np.array_equal(new_row, board[i]):
                moved = True
            
            new_board[i] = new_row
        
        return moved, new_board
    
    def _add_random_tile(self):
        """空いているマスにランダムに2か4を配置"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        
        if not empty_cells:
            return
        
        i, j = random.choice(empty_cells)
        self.board[i, j] = 2 if random.random() < 0.9 else 4
    
    def render(self):
        """盤面表示"""
        print(f"Score: {self.score} | Moves: {self.num_moves}")
        for row in self.board:
            print(' '.join(f'{x:5d}' if x > 0 else '    .' for x in row))
        print()


# ========================================
# MuZero型MCTSノード（ハッシュを使わない）
# ========================================
class MCTSNode:
    """
    探索木のノード
    z（潜在状態）をそのまま保持し、ハッシュに頼らない
    """
    def __init__(self, z: torch.Tensor, parent: Optional['MCTSNode'] = None, emotion: Optional[int] = None):
        self.z = z  # (z_dim,) 潜在状態
        self.parent = parent
        self.emotion = emotion  # 親から到達した感情ID
        
        self.children: Dict[int, 'MCTSNode'] = {}  # {emotion_id: child_node}
        
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = 1.0  # 事前確率（将来的にポリシーから取得可能）
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, parent_visit_count, c_puct=1.0):
        """UCB1スコア"""
        if self.visit_count == 0:
            return float('inf')
        
        exploitation = self.value()
        exploration = c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        
        return exploitation + exploration
    
    def select_child(self, c_puct=1.0):
        """UCBスコアが最大の子ノードを選択"""
        return max(self.children.values(), key=lambda child: child.ucb_score(self.visit_count, c_puct))
    
    def expand(self, emotion_probs, hippocampus, device):
        """
        全感情について子ノードを展開
        emotion_probs: (num_emotions,) ポリシーネットワークからの確率分布
        """
        num_emotions = len(emotion_probs)
        
        for e in range(num_emotions):
            if e in self.children:
                continue
            
            # Hippocampusで次の潜在状態を予測
            with torch.no_grad():
                e_tensor = torch.tensor([e], dtype=torch.long).to(device)
                z_next = hippocampus.predict(self.z.unsqueeze(0), e_tensor).squeeze(0)
            
            # 子ノード作成
            child = MCTSNode(z=z_next, parent=self, emotion=e)
            child.prior = emotion_probs[e]
            self.children[e] = child
    
    def backup(self, value):
        """バックアップ: ルートまで価値を伝播"""
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # 相手視点に切り替え（2048は1人ゲームだが一般性のため）
            node = node.parent


class MCTS:
    """MuZero型MCTS（ノードベース探索木）"""
    def __init__(self, num_emotions=4, c_puct=1.0):
        self.num_emotions = num_emotions
        self.c_puct = c_puct
    
    def search(self, z_root, policy_net, hippocampus, value_net, num_simulations, device):
        """
        z_root: (z_dim,) ルートの潜在状態
        返り値: {emotion_id: visit_count} の辞書
        """
        # ルートノード作成
        root = MCTSNode(z=z_root, parent=None, emotion=None)
        
        # ルートを展開
        with torch.no_grad():
            emotion_probs = policy_net(z_root.unsqueeze(0)).squeeze(0).cpu().numpy()
        root.expand(emotion_probs, hippocampus, device)
        
        # シミュレーション
        for _ in range(num_simulations):
            node = root
            
            # 1. Selection: 葉ノードまで降下
            while not node.is_leaf():
                node = node.select_child(self.c_puct)
            
            # 2. Expansion: 葉ノードを展開
            with torch.no_grad():
                emotion_probs = policy_net(node.z.unsqueeze(0)).squeeze(0).cpu().numpy()
            node.expand(emotion_probs, hippocampus, device)
            
            # 3. Evaluation: 価値推定
            with torch.no_grad():
                value = value_net(node.z.unsqueeze(0)).item()
            
            # 4. Backup: 価値を伝播
            node.backup(value)
        
        # ルートの子ノードの訪問回数を集計
        emotion_visits = {e: child.visit_count for e, child in root.children.items()}
        
        return emotion_visits


# ========================================
# ニューラルネットワーク
# ========================================
class Encoder(nn.Module):
    """(z_prev, a_prev, s) -> z"""
    def __init__(self, z_dim=32):
        super().__init__()
        self.z_dim = z_dim
        
        # 盤面を処理（4×4=16マス、log2正規化済み）
        self.board_encoder = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # 行動埋め込み（0-3の方向 + 初期値4）
        self.action_embed = nn.Embedding(5, 16)
        
        # GRUで統合
        self.gru = nn.GRUCell(64 + 16, z_dim)
    
    def forward(self, z_prev, a_prev, s):
        """
        z_prev: (batch, z_dim)
        a_prev: (batch,) 行動ID (0-3, 初期は4)
        s: (batch, 4, 4) 盤面（log2正規化済み）
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
        memory = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Cross-attention
        attended, _ = self.cross_attn(query, memory, memory)
        attended = attended.squeeze(1)
        
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


class ValueNetwork(nn.Module):
    """z -> 状態価値"""
    def __init__(self, z_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, z):
        return self.net(z).squeeze(-1)


class AdvantageActorCriticMotor(nn.Module):
    """Advantage Actor-Critic: (z, e) -> 行動分布"""
    def __init__(self, z_dim=32, num_emotions=4, num_actions=4):
        super().__init__()
        self.num_actions = num_actions
        
        # 感情埋め込み
        self.emotion_embed = nn.Embedding(num_emotions, 16)
        
        # Actor（行動分布）
        self.actor = nn.Sequential(
            nn.Linear(z_dim + 16, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
    
    def forward(self, z, e):
        """
        z: (batch, z_dim)
        e: (batch,) 感情ID
        -> action_logits: (batch, num_actions)
        """
        e_vec = self.emotion_embed(e)
        combined = torch.cat([z, e_vec], dim=-1)
        action_logits = self.actor(combined)
        return action_logits


# ========================================
# エージェント
# ========================================
class EmotionalAgent:
    def __init__(self, device='cpu', z_dim=32, num_emotions=4, lr=1e-3):
        self.device = device
        self.z_dim = z_dim
        self.num_emotions = num_emotions
        
        # ネットワーク
        self.encoder = Encoder(z_dim).to(device)
        self.hippocampus = Hippocampus(z_dim, num_emotions).to(device)
        self.policy_net = PolicyNetwork(z_dim, num_emotions).to(device)
        self.value_net = ValueNetwork(z_dim).to(device)
        self.motor = AdvantageActorCriticMotor(z_dim, num_emotions, num_actions=4).to(device)
        
        # MCTS
        self.mcts = MCTS(num_emotions=num_emotions, c_puct=1.0)
        
        # オプティマイザ
        params = (
            list(self.encoder.parameters()) +
            list(self.hippocampus.parameters()) +
            list(self.policy_net.parameters()) +
            list(self.value_net.parameters()) +
            list(self.motor.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=lr)
    
    def play_episode(self):
        """1エピソードプレイ"""
        game = Game2048()
        s = game.reset()
        
        z = torch.zeros(1, self.z_dim).to(self.device)
        a_prev = torch.tensor([4]).to(self.device)  # 初期値（4方向の外）
        
        episode_buffer = []
        
        while not game.done:
            # エンコード
            s_tensor = torch.from_numpy(s).float().unsqueeze(0).to(self.device)
            z = self.encoder(z, a_prev, s_tensor)
            
            # 合法手マスク
            legal_actions = game.get_legal_actions()
            
            if not legal_actions:
                break
            
            # MCTS
            emotion_visits = self.mcts.search(
                z.squeeze(0),
                self.policy_net,
                self.hippocampus,
                self.value_net,
                num_simulations=50,
                device=self.device
            )
            
            # 感情サンプリング
            e = self._sample_emotion(emotion_visits)
            
            # 行動生成
            action_logits = self.motor(z, torch.tensor([e]).to(self.device))
            
            # 合法手のみに制限
            action_probs = F.softmax(action_logits, dim=-1).squeeze(0)
            action_probs_masked = torch.zeros(4).to(self.device)
            action_probs_masked[legal_actions] = action_probs[legal_actions]
            action_probs_masked /= action_probs_masked.sum()
            
            a = torch.multinomial(action_probs_masked, 1).item()
            
            # 実行
            s_next, reward, done = game.step(a)
            
            # 次状態エンコード
            s_next_tensor = torch.from_numpy(s_next).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                z_next_actual = self.encoder(z.detach(), torch.tensor([a]).to(self.device), s_next_tensor)
            
            # バッファに保存
            episode_buffer.append({
                'z': z.clone(),
                'e': e,
                'a': a,
                'z_next_actual': z_next_actual.detach().clone(),
                'reward': reward,
                'done': done,
                'emotion_visits': emotion_visits
            })
            
            # 次へ
            s = s_next
            z = z_next_actual
            a_prev = torch.tensor([a]).to(self.device)
        
        # エピソード終了後に学習
        self._train_episode(episode_buffer)
        
        return game.score, game.num_moves
    
    def _train_episode(self, episode_buffer):
        """エピソード全体を使ってEnd-to-End学習"""
        if len(episode_buffer) == 0:
            return
        
        # 逆順でリターンを計算（割引率0.99）
        returns = []
        R = 0
        gamma = 0.99
        
        for t in reversed(episode_buffer):
            R = t['reward'] + gamma * R
            returns.insert(0, R)
        
        total_loss = 0
        
        for i, step in enumerate(episode_buffer):
            z = step['z']
            e = step['e']
            a = step['a']
            z_next_actual = step['z_next_actual']
            emotion_visits = step['emotion_visits']
            G_t = returns[i]
            
            # 1. Dynamics Loss
            z_next_pred = self.hippocampus.predict(z, torch.tensor([e]).to(self.device))
            L_dynamics = F.mse_loss(z_next_pred, z_next_actual)
            
            # 2. Policy Loss
            emotion_probs = self.policy_net(z)
            target_probs = self._normalize_visits(emotion_visits)
            target_probs_tensor = torch.tensor(target_probs).float().unsqueeze(0).to(self.device)
            L_policy_emotion = F.kl_div(
                torch.log(emotion_probs + 1e-8),
                target_probs_tensor,
                reduction='batchmean'
            )
            
            # 3. Advantage Actor-Critic Loss
            state_value_pred = self.value_net(z)
            advantage = G_t - state_value_pred.item()
            
            action_logits = self.motor(z, torch.tensor([e]).to(self.device))
            a_tensor = torch.tensor([a]).to(self.device)
            action_log_prob = F.log_softmax(action_logits, dim=-1)[0, a]
            L_actor = -action_log_prob * advantage
            
            # 4. Value Network Loss
            G_t_tensor = torch.tensor([G_t]).float().to(self.device)
            L_value = F.mse_loss(state_value_pred, G_t_tensor)
            
            # 総合Loss
            loss = L_dynamics + L_policy_emotion + L_actor + L_value
            total_loss += loss
        
        # バックプロパゲーション
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) +
            list(self.hippocampus.parameters()) +
            list(self.policy_net.parameters()) +
            list(self.value_net.parameters()) +
            list(self.motor.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()
    
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


# ========================================
# メイン
# ========================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("=" * 60)
    print("2048 with Emotional AGI (MuZero-style MCTS)")
    print("=" * 60)
    print("Game: 4×4 grid, survive as long as possible")
    print("Reward: +1 per turn survived")
    print("Architecture:")
    print("  - z_dim: 32")
    print("  - num_emotions: 4")
    print("  - MCTS: Node-based (no hashing)")
    print("  - MCTS simulations: 50")
    print("=" * 60)
    print()
    
    agent = EmotionalAgent(device=device, z_dim=32, num_emotions=4, lr=1e-3)
    
    print("Starting training...")
    
    best_score = 0
    best_moves = 0
    
    for iteration in range(200):
        scores = []
        moves_list = []
        
        # 10エピソード実行
        for _ in range(10):
            score, num_moves = agent.play_episode()
            scores.append(score)
            moves_list.append(num_moves)
        
        avg_score = np.mean(scores)
        avg_moves = np.mean(moves_list)
        max_score = max(scores)
        max_moves = max(moves_list)
        
        if max_score > best_score:
            best_score = max_score
        if max_moves > best_moves:
            best_moves = max_moves
        
        if iteration % 10 == 0:
            print(f"Iter {iteration:3d}: Avg Score={avg_score:7.1f} Avg Moves={avg_moves:5.1f} | "
                  f"Max Score={max_score:5.0f} Max Moves={max_moves:4.0f} | "
                  f"Best Score={best_score:5.0f} Best Moves={best_moves:4.0f}")
    
    print()
    print("Training complete!")
    print(f"Best Score: {best_score}")
    print(f"Best Moves: {best_moves}")


if __name__ == '__main__':
    main()
