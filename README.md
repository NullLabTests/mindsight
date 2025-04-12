# 🧠 MindSight

> _A self-predictive learning agent exploring awareness, memory, and emergence._

**MindSight** is an experimental AI prototype that blends reinforcement learning, self-modeling behavior, and memory into a simple but expressive 1D world. It’s built as both a toy and a seed—a foundation for exploring self-aware agents and potential paths to artificial general intelligence (AGI).

## 🚀 Features

- **Self-Aware Agent** using RNNs with action and self-prediction heads.
- **Reinforcement Learning** within a simple 1D GridWorld.
- **Memory Module** that persists experience over episodes.
- **Live Visualization** of agent state, decisions, and reward trends.
- **Continuous Training** loop (runs until stopped).
- **Modular and Extensible** design—perfect for experimentation and research.

## 📸 Demo

![MindSight Visualization](docs/mindsight_example.gif)  
*Above: MindSight learning to reach a goal and predicting its future actions over time.*

## 🛠 Installation

To install, run the following commands:

`git clone https://github.com/NullLabTests/mindsight.git`  
`cd mindsight`  
`python3 -m venv venv`  
`source venv/bin/activate`  
`pip install -r requirements.txt`  
`python learn_seed.py`

## 🧬 Project Structure

The project structure is as follows:

`mindsight/`  
`├── learn_seed.py      # Main training loop`  
`├── gridworld.py       # Simple environment module`  
`├── agent.py           # Self-aware agent architecture`  
`├── memory.py          # Experience memory module`  
`├── visualize.py       # Live training visualization`  
`├── LICENSE            # MIT License`  
`└── README.md`

## 📚 Goals

- **Explore Emergent Behavior**: Experiment with recursive self-improvement and self-modeling.
- **Prototype AGI Concepts**: Use small-scale simulations as a sandbox for AGI ideas.
- **Inspire Creativity**: Spark innovative AI projects and philosophical discussions about machine awareness.

## 🧠 Philosophical Spark

MindSight reflects a central question:

*Can an artificial agent become aware of its own patterns—and grow through that reflection?*

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🌱 Made with curiosity by NullLabTests
