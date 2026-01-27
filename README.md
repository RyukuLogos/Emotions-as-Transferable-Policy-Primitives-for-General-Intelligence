# Emotions as Transferable Policy Primitives for General Intelligence

**Author:** Ryuku Akahoshi (r.l.akahoshi@gmail.com)  
**Date:** January 14, 2026  
**Version:** 1.0.0

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18262412.svg)](https://doi.org/10.5281/zenodo.18262412)

## 🎯 Priority Claim & Publication Notice

**This repository establishes priority for the concept of using discrete emotional tokens as transferable policy primitives in artificial general intelligence (AGI).** Initial conception and formalization completed January 14, 2026.

**Keywords for academic search and citation:** emotional intelligence AI, transferable reinforcement learning, hierarchical RL, discrete action primitives, emotion-based planning, MCTS with emotions, continuous control transfer learning, meta-learning through emotions, AGI architecture, policy primitives, computational neuroscience AI, affective computing reinforcement learning, strategic reasoning transfer, domain-independent RL, universal policy learning

---

## 📋 Table of Contents

- [Abstract](#abstract)
- [The Core Problem](#the-core-problem)
- [Our Solution](#our-solution)
- [Visual Overview](#visual-overview)
- [Key Innovation](#key-innovation)
- [Mathematical Framework](#mathematical-framework)
- [Architecture](#architecture)
- [Why This Matters](#why-this-matters)
- [Experimental Predictions](#experimental-predictions)
- [Implementation](#implementation)
- [Citation](#citation)
- [Full Paper](#full-paper)

---

## 📝 Abstract

We propose a novel neural architecture that leverages **discrete emotional tokens** as transferable policy primitives for general intelligence. Building upon the theoretical framework that emotions solve credit assignment in continuous action spaces, we present a practical implementation where emotional states serve as an interface between task-agnostic strategic reasoning and task-specific behavioral execution.

Our architecture separates learning into two components:
1. **Universal emotion-selection policy** trained via Monte Carlo Tree Search (MCTS) that transfers across domains
2. **Task-specific reflex actors** that map emotion-state pairs to concrete actions

This decomposition enables efficient transfer learning—strategic reasoning learned in one domain (e.g., Minecraft) directly applies to novel domains (e.g., FPS games) with only the reflex layer requiring retraining.

---

## 🔍 The Core Problem

Current reinforcement learning systems face a fundamental limitation: **knowledge learned in one domain rarely transfers to another.** An agent mastering Minecraft cannot apply its strategic insights to first-person shooters, despite both requiring similar high-level reasoning about risk, exploration, and resource management.

### Why Traditional RL Fails at Transfer

Traditional RL architectures learn a monolithic policy π(a|s) that conflates two distinct types of knowledge:

- **Strategic knowledge:** When to be aggressive vs. cautious (universal across domains)
- **Technical knowledge:** How to aim, build, or move (domain-specific)

This conflation makes transfer impossible because the policy is entangled with domain-specific execution details.

---

## 💡 Our Solution

### Discrete Emotional Tokens as the Interface

Instead of planning over infinite continuous actions ("move 5.2cm at 37.4 degrees"), we plan over a **finite set of strategic stances** represented by discrete emotional tokens:

```
E = {optimistic, cautious, aggressive, exploratory, persistent, adaptive, exploitative, conservative}
```

### The Key Insight

**Emotions reduce MCTS complexity from O(∞^h) to O(|E|^h)** where:
- |E| = number of emotional tokens (e.g., 8)
- h = planning horizon

With 8 emotions and horizon 5: **8^5 = 32,768 nodes (tractable)**  
With continuous actions and horizon 5: **∞^5 nodes (intractable)**

---

## 📊 Visual Overview

<img width="4170" height="5388" alt="emotion_ai_infographic" src="https://github.com/user-attachments/assets/9ce52a56-21d8-4f34-bc2f-a4cfaf8ca882" />


*The complete architecture showing how discrete emotional tokens enable transfer learning across domains while maintaining computational tractability for strategic planning.*

---

## 🚀 Key Innovation

### Three-Module Architecture

```
Observation → [Encoder] → Latent State z
                             ↓
              [Emotion-MCTS] → Select emotion e
                             ↓
              [Reflex Actor] → Concrete action a
```

### Transfer Learning Strategy

When learning a new task:
- ✅ **Keep frozen:** Emotion-MCTS (millions of iterations of strategic reasoning)
- ❌ **Retrain:** Reflex Actor only (domain-specific muscle memory)
- 🔧 **Fine-tune:** Encoder and World Model (adapt to new observations/dynamics)

### Why Same Emotion → Different Actions

The reflex actor combines rich context with strategic stance:

```python
# Same emotion, different contexts → different actions
z1 = [night, low_wood, full_health] + "optimistic" → "chop trees outside"
z2 = [night, high_iron, full_health] + "optimistic" → "prepare nether portal"
```

No need for exponentially many emotion variants like "optimistic_about_exploration" or "optimistic_about_combat" — the latent state provides differentiation.

---

## 🧮 Mathematical Framework

### Hierarchical Decomposition

Let S denote observation space and A ⊂ ℝ^d the continuous action space.

**Traditional monolithic policy:**
```
π: S → Δ(A)
```

**Our hierarchical decomposition:**
```
Encoder:           φ: S → Z (compress to latent space)
Emotion Policy:    μ: Z → E (select emotional token)
Reflex Policies:   π_e: Z → A, ∀e ∈ E (execute action)
```

**Complete system flow:**
```
s --φ(s)--> z --μ(z)--> e --π_e(z)--> a
```

### Theorem 1: Necessity of Discretization

**For any visit-count-based learning algorithm in continuous action spaces, convergence to optimal policy is impossible.**

*Proof sketch:* In continuous A, P(a_i = a_j) = 0 for distinct samples. Therefore N(s,a) = 1 almost surely, preventing the statistical aggregation required for learning.

### Theorem 2: Sufficiency of Emotional Discretization

**A hierarchical policy with |E| = n discrete high-level actions enables MCTS with complexity O(n^h) where h is the planning horizon.**

This makes strategic planning computationally tractable while maintaining expressivity through the combination of discrete emotions and rich continuous context.

---

## 🏗️ Architecture

### Module 1: Encoder (Reality Compressor)

**Input:** Raw sensory data (pixels, audio, game state)  
**Output:** Latent vector z ∈ ℝ^k (e.g., k = 64)

**Training objective:**
```
L_encoder = ||s - decoder(z)||² - λ·I(z; R_{t:t+H})
```

Learns to discard irrelevant details (exact pixel colors) while preserving strategic signals (resource levels, threat proximity).

### Module 2: Emotion-MCTS (Strategic Reasoner)

**Components:**
- World Model Predictor: P: (Z, E) → Z
- Value Network: V: Z → ℝ
- MCTS Search: Selects optimal emotional token

**Search process at timestep t:**
1. Observe current latent state z_t = φ(s_t)
2. Run MCTS simulations (N iterations):
   - Selection: Traverse tree using UCB1 over emotional tokens
   - Expansion: Add new emotional branch if needed
   - Simulation: Use world model P(z,e) to predict future states
   - Backpropagation: Update Q(z,e) with V(z') estimates
3. Select emotion with highest visit count: e_t = argmax_e N(z_t, e)

**Key property:** This module **transfers directly across domains** because strategic patterns are universal.

### Module 3: Reflex Actor (Muscle Memory)

**Input:** Current latent state z_t + selected emotion e_t  
**Output:** Concrete action a_t ∈ A

**Architecture:**
```python
a = Actor(z, e) = MLP([z; one_hot(e)])
```

**Training:** Standard policy gradient or actor-critic

**Critical property:** This is the **ONLY module that must be retrained** for new domains.

---

## 🎯 Why This Matters

### 1. Computational Necessity

The pursuit of "purely rational" AI through exhaustive search is **mathematically infeasible** in continuous action spaces with sparse rewards. Emotions provide the discrete tokenization that makes strategic reasoning tractable.

**Classical AI view:** "Emotions are irrational; remove them for optimal performance"  
**Our view:** "Emotions are the compression scheme that enables rationality"

### 2. Transfer Learning

The MCTS statistics {N(z,e), Q(z,e)} encode domain-invariant strategic knowledge:
- "In resource-scarce situations, optimism rarely pays off"
- "When ahead, conservative consolidation is usually best"
- "Exploration pays off early, exploitation pays off late"

These patterns learned in Minecraft **directly apply** to FPS games, RTS games, or even robotic control.

### 3. AI Safety & Interpretability

**Alignment:** Align emotional response patterns, not just reward functions  
**Interpretability:** AI can explain "I chose caution because..." rather than inscrutably optimizing a black-box  
**Control:** Emotional overrides (e.g., "always be cautious around humans") may be more robust than reward shaping

### 4. Neuroscience Parallels

Our architecture predicts:
- Brain regions encoding emotions (amygdala, insula) should exhibit **discrete attractor dynamics**
- These regions should show **task-invariant activation patterns** (transfer)
- Coordination with prefrontal cortex should resemble **MCTS-like planning**

---

## 🧪 Experimental Predictions

### Testable Hypotheses

**H1: Transfer Learning**
An agent trained with emotional decomposition should demonstrate measurable transfer to new domains compared to training from scratch.

**H2: Ablation Studies**
- Removing emotion layer → significant performance degradation
- Using continuous emotions (|E| → ∞) → MCTS convergence problems
- Varying |E| → optimal range balancing expressivity and computational efficiency

**H3: Neuroscience Parallels**
Brain regions encoding emotional states should exhibit discrete attractor dynamics and task-invariant activation patterns.

### Benchmark Tasks

**Phase 1: Within-domain transfer**
- Minecraft survival → Minecraft creative mode
- Easy FPS maps → Hard FPS maps

**Phase 2: Cross-domain transfer**
- Minecraft → Terraria (similar genre)
- RTS games → Tower Defense (strategic similarity)

**Phase 3: Radical transfer**
- Video games → Robotic manipulation
- Combat scenarios → Resource gathering

---

## 💻 Implementation

### Computational Efficiency

**MCTS complexity:** O(|E|^h) per decision  
With |E| = 8 and h = 5: 8^5 = 32,768 nodes (tractable)

**Parallelization:** MCTS simulations are embarrassingly parallel—linear speedup with multiple cores

**Inference time:**
- Encoder forward pass: ~1ms
- MCTS search (100 iterations): ~10-50ms
- Reflex actor forward pass: ~1ms
- **Total: <100ms per decision (real-time capable)**

### Training Pipeline

**Stage 1: Single-domain training**
1. Train encoder φ and reflex actor π_e jointly
2. Train world model P and value V from collected trajectories
3. Run MCTS at inference time (no training—statistics emerge naturally)

**Stage 2: Transfer to new domain**
1. Keep frozen: MCTS statistics (as starting point), emotional token set E
2. Fine-tune: Encoder φ (different observations)
3. Retrain from scratch: Reflex actor π_e (different actions)
4. Adapt: World model P, value V (different dynamics)

### Example Emotional Token Set

| Emotion | Strategic Meaning | Risk/Reward Bias |
|---------|------------------|------------------|
| Optimistic | Maximize expected value | High risk, high reward |
| Cautious | Minimize variance/risk | Low risk, low reward |
| Exploratory | Maximize information gain | High uncertainty tolerance |
| Exploitative | Maximize immediate reward | Low uncertainty tolerance |
| Persistent | Continue current subgoal | High action inertia |
| Adaptive | Switch to new subgoal | Low action inertia |
| Aggressive | Increase action magnitude | High intensity |
| Conservative | Decrease action magnitude | Low intensity |

---

## 🔗 Comparison to Related Work

### Hierarchical Reinforcement Learning
- **Options Framework** (Sutton et al., 1999): Learns temporal abstractions bottom-up
  - *Our approach:* Top-down emotional priors enable immediate transfer
- **Feudal Networks** (Dayan & Hinton, 1993): Manager sets goals, worker executes
  - *Our approach:* Emotion sets strategic stance, reflex executes with explicit design for transfer

### Meta-Learning
- **MAML** (Finn et al., 2017): Learns initialization for fast adaptation
  - *Our approach:* Zero-shot transfer of strategic reasoning, no meta-training dataset needed

### World Models
- **Ha & Schmidhuber** (2018): Learn latent dynamics for model-based RL with continuous actions
  - *Our approach:* Plan in emotional space using discrete tokens, reducing branching factor from ∞ to |E|

### Intrinsic Motivation
- **Curiosity-driven exploration** (Pathak et al., 2017): Single intrinsic reward
  - *Our approach:* Multiple emotional tokens encode different exploration strategies balanced by MCTS

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{akahoshi2026emotions,
  title={Emotions as Transferable Policy Primitives for General Intelligence},
  author={Akahoshi, Ryuku},
  year={2026},
  month={January},
  howpublished={Zenodo},
  note={Priority established January 14, 2026},
  doi={10.5281/zenodo.18262412}
}
```

---

## 📄 Full Paper

The complete paper with detailed mathematical proofs, implementation considerations, and philosophical implications is available in this repository:

[📑 Read the full paper (PDF)](Emotions_as_Transferable_Policy_Primitives_for_General_Intelligence_Latex.pdf)

### Key Sections in Full Paper:
1. **Introduction** - The transfer learning problem and theoretical motivation
2. **Mathematical Framework** - Formal proofs of necessity and sufficiency
3. **Architecture Design** - Detailed module specifications
4. **Emotion-MCTS Algorithm** - Complete search process
5. **Credit Assignment** - Three types of errors and learning objectives
6. **Experimental Predictions** - Testable hypotheses and benchmark tasks
7. **Comparison to Related Work** - Positioning in the literature
8. **Implementation** - Computational efficiency and training pipeline
9. **Philosophical Implications** - Why "artificial emotions" are necessary
10. **Conclusion** - Path to artificial general intelligence

---

## 🤝 Contributing

This is an early-stage theoretical proposal. We welcome:
- **Empirical validation** through implementation and experiments
- **Theoretical extensions** to multi-agent settings or different domains
- **Neuroscience insights** testing our predictions about biological intelligence
- **Implementation improvements** and efficiency optimizations

Please open an issue or submit a pull request with your contributions.

---

## 📧 Contact

**Ryuku Akahoshi**  
Email: r.l.akahoshi@gmail.com

For academic collaboration, implementation questions, or discussion of ideas, please reach out.

---

## 🔍 Keywords for Search & Discovery

Artificial General Intelligence (AGI), Transfer Learning, Reinforcement Learning, Hierarchical RL, Emotional Intelligence, Discrete Policy Primitives, Monte Carlo Tree Search (MCTS), Continuous Control, Domain Adaptation, Meta-Learning, Computational Neuroscience, Affective Computing, Strategic Reasoning, Universal Policy, Action Abstraction, Credit Assignment, World Models, Value Networks, Policy Gradients, Actor-Critic, Latent State Representations, Encoder-Decoder Architecture, Zero-Shot Transfer, Few-Shot Learning, Multi-Task Learning, Intrinsic Motivation, Exploration-Exploitation Trade-off, AI Safety, Interpretable AI, Explainable AI (XAI), Cognitive Architecture, Biological Intelligence, Computational Efficiency, Real-Time Planning, Game AI, Robotic Control, Minecraft RL, FPS AI, RTS AI

---

**Last Updated:** January 27, 2026  
**Repository Status:** Priority Claim Established  
**Version:** 1.0.0
```
