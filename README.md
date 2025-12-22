# 🌍 Evolving AI Biosphere

### A self-evolving digital ecosystem where AI organisms fight for survival.

[![Status](https://img.shields.io/badge/Status-Active-success)]()
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)]()
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)]()
[![Ollama](https://img.shields.io/badge/LLM-Llama3.2-orange)]()

---

## 📖 Overview

**Evolving AI Biosphere** is a complex artificial life simulation that combines **Deep Reinforcement Learning**, **Genetic Algorithms**, and **Large Language Models** to create a living, breathing ecosystem.

Organisms in this world are not programmed with rules—they *learn* how to survive.

- **Predators** share a collective "Hive Mind" (LSTM) that learns from the experiences of the pack.
- **Herbivores** evolve individually via genetic mutation of their neural weights.
- **The World** is watched by an "AI Council" (Gaia, Entropy, Arbiter) that can intervene to balance the system.

👉 **[Read the Technical Documentation](https://alwaysvivek.github.io/evolving-ai-biosphere/docs/index.html)** for a deep dive into the architecture and theory.

---

## 🛠 Tech Stack

| Component | Technology | Description |
|-----------|------------|-------------|
| **Core Engine** | Python, Pygame | Real-time simulation loop and rendering. |
| **Neural Nets** | PyTorch | LSTM networks for creature brains. |
| **LLM Orchestration** | LangChain, LangGraph | Multi-agent council system (Gaia/Entropy). |
| **Inference** | Ollama | Local LLM serving (Llama 3.2). |
| **Observability** | MLflow | Experiment tracking and metric logging. |

---

## 🧩 Architecture

```mermaid
graph TD
    User[User] -->|Run| Main[main.py Orchestrator]
    Main -->|Launches| Ollama[Ollama Service]
    Main -->|Launches| MLflow[MLflow UI]
    Main -->|Starts| Sim[Simulation Core]
    
    subgraph Ecosystem
        Sim -->|Updates| Grid[Game Grid]
        Sim -->|Manages| Council[AI Council (LangGraph)]
    end
    
    subgraph Agents
        Predators[Predator Hive (Shared LSTM)]
        Herbivores[Herbivore Individuals (Genome)]
    end
    
    Council -->|Intervenes| Grid
    Sim -->|Logs| MLflow
```

---

## ✨ Key Features

- **🧠 Hive Mind Intelligence:** Predators learn collectively using a shared experience buffer and policy gradient training.
- **🧬 Genetic Evolution:** Herbivores evolve over generations through random mutations in their neural weights.
- **🤖 AI God Mode:** A LangGraph-based "Council" (Gaia vs Entropy) debates and intervenes in the simulation autonomously.
- **📊 MLflow Tracking:** Live dashboard of population statistics, extinction events, and system health.
- **🧪 Pheromone System:** Gradient-based scent maps for indirect agent communication.

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9+
- Ollama (installed and running with `llama3.2`)

### 1. Clone the Repository
```bash
git clone https://github.com/alwaysvivek/evolving-ai-biosphere.git
cd evolving-ai-biosphere
```

### 2. Auto-Setup
We provide a unified orchestrator that handles venv creation, dependencies, and service launching.
```bash
python3 main.py
```
*This command will check for Ollama, start MLflow, create the virtualenv if needed, and run the simulation.*

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `SPACE` | Pause/Resume |
| `R` | Generate Console Report |
| `T` | Toggle Predator Training |
| `K` | Kill All Predators (Test resilience) |
| `E` | Trigger Scarcity Event |

---

## 📊 Monitoring

Once the simulation starts, open MLflow to see live metrics:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

**Author**: Vivek Dagar
