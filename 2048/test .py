"""
Simple test script to watch a trained agent play 2048
"""
import torch
import numpy as np
from game2048 import Game2048
from model import Actor, Hippocampus, ValueNetwork, PolicyNetwork, MCTS, ObservationEncoder, NUM_EMOTIONS
import time

def play_game(trainer, render=True, delay=0.3):
    """Play one game and optionally render"""
    env = Game2048()
    obs = env.reset()
    
    # Initialize
    z = torch.zeros(1, trainer.num_slots, trainer.z_dim).to(trainer.device)
    memory = torch.zeros(1, trainer.num_slots, trainer.z_dim).to(trainer.device)
    emotion = 0
    action = 0
    
    done = False
    step = 0
    
    if render:
        print("\n" + "="*40)
        print("Starting new game!")
        print("="*40)
    
    while not done:
        if render:
            print(f"\nStep {step}")
            print_board(env.board)
            print(f"Score: {env.score} | Max Tile: {env.max_tile}")
        
        obs_tensor = torch.FloatTensor(obs.flatten()).unsqueeze(0).to(trainer.device)
        
        with torch.no_grad():
            # Encode
            z_encoded = trainer.encoder(obs_tensor)
            
            # Get emotion via MCTS
            policy_logits = trainer.policy_net(z_encoded)
            emotion_next, visit_counts = trainer.mcts.search(z_encoded, policy_logits)
            emotion_next = emotion_next.item()
            
            # Get action
            obs_seq = obs_tensor.unsqueeze(1)
            z_seq = z.unsqueeze(1)
            memory_seq = memory.unsqueeze(1)
            emotion_seq = torch.LongTensor([[emotion]]).to(trainer.device)
            action_seq = torch.LongTensor([[action]]).to(trainer.device)
            
            z_new, query, action_logits = trainer.actor(
                obs_seq, z_seq, memory_seq, emotion_seq, action_seq
            )
            z_new = z_new.squeeze(1)
            
            # Get memory
            emotion_onehot = torch.zeros(1, NUM_EMOTIONS).to(trainer.device)
            emotion_onehot[0, emotion_next] = 1.0
            memory_new = trainer.hippocampus(z_new, query.unsqueeze(0))
            
            # Select action
            action_probs = torch.softmax(action_logits.squeeze(0), dim=-1)
            action_next = action_probs.argmax().item()
        
        if render:
            emotions = ["Aggressive", "Cautious", "Corner", "Patient", "Exploratory"]
            actions = ["UP", "RIGHT", "DOWN", "LEFT"]
            print(f"Emotion: {emotions[emotion_next]} | Action: {actions[action_next]}")
            time.sleep(delay)
        
        # Step
        obs, reward, done = env.step(action_next)
        
        # Update
        z = z_new
        memory = memory_new
        emotion = emotion_next
        action = action_next
        step += 1
    
    if render:
        print("\n" + "="*40)
        print("GAME OVER!")
        print(f"Final Score: {env.score}")
        print(f"Max Tile: {env.max_tile}")
        print(f"Steps: {step}")
        print("="*40)
    
    return env.score, env.max_tile, step

def print_board(board):
    """Pretty print the board"""
    print("\n" + "-" * 25)
    for row in board:
        print("|", end="")
        for cell in row:
            if cell == 0:
                print("    .", end=" ")
            else:
                print(f"{cell:5d}", end=" ")
        print("|")
    print("-" * 25)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    from train import Trainer
    
    # Load trained model
    trainer = Trainer(device=device)
    try:
        trainer.load('model_2048.pt')
        print("Loaded trained model")
    except:
        print("No trained model found, using random initialization")
    
    # Play a few games
    scores = []
    max_tiles = []
    
    for i in range(5):
        score, max_tile, steps = play_game(trainer, render=(i==0), delay=0.1)
        scores.append(score)
        max_tiles.append(max_tile)
        print(f"Game {i+1}: Score={score}, MaxTile={max_tile}, Steps={steps}")
    
    print(f"\nAverage Score: {np.mean(scores):.1f}")
    print(f"Average Max Tile: {np.mean(max_tiles):.1f}")
    print(f"Best Max Tile: {max(max_tiles)}")
