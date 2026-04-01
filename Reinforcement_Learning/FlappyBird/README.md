# 🐦 Flappy Bird AI using Deep Q-Learning (DQN)

This project implements a Deep Q-Network (DQN) agent to play the Flappy Bird game using reinforcement learning. The agent learns to maximize rewards by interacting with the environment.

---

## 🚀 Project Overview

- Built using PyTorch and Gymnasium  
- Uses Deep Q-Learning (DQN) algorithm  
- Implements Experience Replay and Target Network  
- Trains an agent to play Flappy Bird autonomously  

---

## 📂 Project Structure

```
.
├── Agent.py                # Main training & testing script
├── dqn.py                  # Neural Network model (DQN)
├── experience_replay.py    # Replay Memory implementation
├── parameters.yaml         # Hyperparameters configuration
├── .gitignore              # Files to ignore in Git
├── __pycache__/            # Python cache files (auto-generated)
├── runs/                   # Saved models & logs
```

---

## 🧠 Key Concepts Used

- Reinforcement Learning  
- Deep Q-Network (DQN)  
- Epsilon-Greedy Strategy  
- Experience Replay Buffer  
- Target Network Synchronization  

---

## ⚙️ Hyperparameters

Defined in `parameters.yaml`:

- alpha (learning rate)  
- gamma (discount factor)  
- epsilon_init (starting exploration rate)  
- epsilon_min (minimum exploration rate)  
- epsilon_decay (decay rate)  
- replay_memory_size  
- mini_batch_size  
- network_sync_rate  
- reward_threshold  

---

## 🛠️ Installation

```bash
pip install torch gymnasium flappy-bird-gymnasium pyyaml
```

---

## ▶️ How to Run

### Train the Agent

```bash
python Agent.py flappybirdv0 --train
```

### Test the Agent

```bash
python Agent.py flappybirdv0
```

---

## 📈 How It Works

1. Initialize the Flappy Bird environment  
2. Use epsilon-greedy strategy to select actions  
3. Store experiences in replay memory  
4. Sample mini-batches and train the neural network  
5. Update target network periodically  
6. Save the best performing model  

---

## 📊 Output

- Model is saved in `runs/` directory  
- Training logs are stored in `.log` file  
- Best model is saved as `.pt` file  

---

## 🚫 Ignored Files (.gitignore)

Typical entries in `.gitignore`:

```
__pycache__/
*.pyc
runs/
```

---

## 💡 Future Improvements

- Implement Double DQN  
- Add visualization graphs for training performance  
- Hyperparameter tuning  
- Extend to other Gym environments  

---

## 👨‍💻 Author

Developed as a Reinforcement Learning project using PyTorch.