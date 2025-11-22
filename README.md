# 🌍 **Evolving AI Biosphere**
### A living digital world where AI organisms evolve and adapt through memory, learning, and survival.

---

## 📚 Table of Contents
1. [🧠 Overview](#-overview)  
2. [🧬 Species Overview](#-species-overview)  
   - [🌱 Plants (Green)](#-plants-green)  
   - [🐇 Herbivores (Blue)](#-herbivores-blue)  
   - [🦊 Predators (Red)](#-predators-red)  
3. [🌦️ Environment Dynamics](#️-environment-dynamics)  
4. [⚙️ Energy Flow](#️-energy-flow)  
5. [🧪 Scent Diffusion System](#-scent-diffusion-system)  
6. [🧭 Emergent Phenomena](#-emergent-phenomena)  
7. [🧬 Reproduction & Inheritance](#-reproduction--inheritance)  
8. [💀 Death & Decay](#-death--decay)  
9. [🕹️ Interactive Controls](#️-interactive-controls)  
10. [🎨 Visual Feedback](#-visual-feedback)  
11. [📊 Long-Term Dynamics](#-long-term-dynamics)  
12. [⚙️ Installation & Setup](#️-installation--setup)  
13. [▶️ Running the Simulation](#️-running-the-simulation)  
14. [🌋 Test Ecosystem Resilience](#-test-ecosystem-resilience)  
15. [🧠 Study Hive Learning](#-study-hive-learning)  
16. [🧩 Custom Scenarios](#-custom-scenarios)  
17. [📈 Reporting](#-reporting)  
18. [🌍 Summary](#-summary)

--- 

## 🧠 Overview

**Evolving AI Biosphere** (formerly *AI Ecosphere*) is a **self-evolving artificial life simulation** where three species — **plants**, **herbivores**, and **predators** — interact in a dynamic, learning-based ecosystem.

Unlike static simulations, organisms here use **neural networks** and **reinforcement learning** to evolve **emergent behaviors** — adapting, learning, and surviving across generations.

> A digital petri dish where **machine learning meets natural selection**.

🧩 **Technical Documentation:**  
Basically, how it all comes together — visit the full docs at  
👉 [https://alwaysvivek.github.io/evolving-ai-biosphere/](https://alwaysvivek.github.io/evolving-ai-biosphere/)

---

## 🧬 Species Overview

### 🌱 **Plants (Green)**
Sessile organisms that generate energy via photosynthesis and reproduce asexually when conditions allow.

**Life Cycle:**
- Gain energy from light; reproduce at **130+ energy** (splitting energy with offspring)
- Die from **old age (350+ cycles)**, **overcrowding**, **low light**, or **herbivore consumption**
- Emit **scent** signals to attract herbivores

**Key Stats:**
- Max Energy: 150  
- Photosynthesis: +3.5 energy/cycle  
- Metabolism: -0.4 energy/cycle  
- Death in low light (< 0.18 intensity)

---

### 🐇 **Herbivores (Blue)**
Mobile agents that eat plants and flee predators using their own **LSTM neural networks**.

**Behavior Systems:**
- **Foraging:** Seek plant-rich areas via scent gradients  
- **Predator Avoidance:** Escape zones with predator presence  
- **Energy Management:** Balance eating, resting, and reproducing  
- **Reproduction:** At **90+ energy** (with mutated neural weights)

**LSTM Inputs (8):**
1. Nearby plant count  
2. Herbivore count  
3. Predator count  
4. Nutrient count  
5. Energy level  
6. Age factor  
7–8. Random noise (exploration)

**Outputs:** Reproduce, move/hunt, rest, or idle.

---

### 🦊 **Predators (Red)**
Apex hunters governed by a **collective LSTM Hive Mind** — all predators share one evolving neural brain.

**Life Cycle:**
- Feed on herbivores (+120 energy per kill)
- Reproduce at **100+ energy**
- Die from starvation or old age (600+ cycles)

**Hive Mind Learning:**
- All predators share experiences (observations, rewards, actions)
- Every 20 generations, the **Hive trains via REINFORCE**
- Collective intelligence leads to evolved group strategies (e.g., flanking, trapping prey)

---

## 🌦️ Environment Dynamics

### 🔆 **Spatial Light Field**
- Varies across the map (bright → plant-rich, dark → barren)
- Affects plant growth and energy efficiency

### 🌡️ **Temperature**
- Fluctuates between 0.0–1.0  
- Alters metabolism and photosynthesis efficiency

### 🌘 **Global Light Cycle**
- Varies between 0.3–1.0 to simulate day/night cycles

### 🪨 **Nutrient Spawning**
- Random nutrient deposits boost local ecosystem growth

---

## ⚙️ Energy Flow

Energy drives survival — all species gain and spend it differently:

| Flow | Source → Target | Description |
|------|-----------------|--------------|
| ☀️ | Light → Plants | Photosynthesis |
| 🌿 | Plants → Herbivores | Foraging |
| 🩸 | Herbivores → Predators | Hunting |
| 🔥 | All | Metabolism & movement drain |

**Balance:**  
Overgrowth in one level causes cascading effects — predator crashes, herbivore booms, plant depletion, etc.

---

## 🧪 Scent Diffusion System

Chemical scent fields create **indirect perception**:

| Scent Type | Emitted By | Function |
|-------------|-------------|-----------|
| 🌿 **Plant Scent** | Plants | Attracts herbivores |
| 🐾 **Herbivore Scent** | Herbivores | Attracts predators |

Each scent diffuses outward over 3 steps, creating **gradient maps** for navigation.

---

## 🧭 Emergent Phenomena

The AI-driven evolution leads to realistic ecological dynamics:

- **Boom–Bust Cycles:** Natural oscillations in population sizes  
- **Spatial Clustering:** Territory and colony formation  
- **Behavioral Evolution:** Learned evasion and hunting strategies  
- **Extinction Events:** Permanent loss of species reshapes balance  
- **Monoculture Dominance:** Single-species takeovers causing fragility  

---

## 🧬 Reproduction & Inheritance

| Species | Inheritance Type | Mutation |
|----------|------------------|-----------|
| 🌱 Plants | Asexual cloning | None |
| 🐇 Herbivores | Neural weight mutation | ±0.01 (2% chance) |
| 🦊 Predators | Hive-mind training | Collective evolution |

---

## 💀 Death & Decay

| Cause | Description |
|--------|--------------|
| Starvation | Energy depletion |
| Old Age | Beyond max lifespan |
| Predation | Eaten by higher species |
| Overcrowding | Excess neighbors (plants) |
| Light Starvation | Insufficient local light |
| Random Events | Manual extinction events |

---

## 🕹️ Interactive Controls

| Key | Action |
|-----|--------|
| **SPACE** | Play/Pause simulation |
| **C** | Clear all organisms |
| **R** | Generate statistical report |
| **T** | Toggle Hive training |
| **F** | Spawn flower pattern |
| **S** | Spawn spiral formation |
| **O** | Predator swarm |
| **N** | Nutrient field |
| **K** | Kill all predators |
| **L** | Kill all herbivores |
| **P** | Kill all plants |
| **E** | Scarcity event |
| **Q** | Quit simulation |

---

## 🎨 Visual Feedback

- **Colors:** Green=Plant, Blue=Herbivore, Red=Predator  
- **Brightness:** Indicates energy level  
- **Scent Halos:** Faint green/red glows show chemical concentrations  
- **Background:** Noise-based soil texture that shifts with temperature and light  
- **HUD Overlay:** Displays population stats, generation count, diversity, temperature, and hive experience

---

## 📊 Long-Term Dynamics

| Phase | Description |
|--------|-------------|
| **0–50 generations** | Unstable bursts and population crashes |
| **50–200 generations** | Evolved equilibrium and specialization |
| **200+ generations** | Stable biomes, learned behaviors, and possible extinctions |

> Each major extinction or scarcity event permanently alters ecosystem balance.

---

## ⚙️ Installation & Setup

### �
 **Prerequisites**
- Python 3.7+  
- Basic GPU/CPU for 800×800 rendering (30 FPS)  

---

### 📦 **Installation**

1. **Clone or Download Repository**
   ```bash
   git clone https://github.com/<your-username>/evolving-ai-biosphere.git
   cd evolving-ai-biosphere
2. **Install dependencies**
   ```bash
    pip3 install -r requirements.txt
   ```
3. Run simulation
   ```bash
     python3 main.py
   ```

### ▶️ Running the Simulation

Then press:
1. **SPACE** → Start simulation  
2. **R** → View stats report periodically  
3. **Watch** the populations evolve for 50–100 generations  

---

### 🌋 Test Ecosystem Resilience

- Let the world stabilize  
- Press **K** → Wipe predators  
- Observe herbivore explosion and eventual plant collapse  

---

### 🧠 Study Hive Learning

- Press **T** → Disable predator training (baseline)  
- Press **T** again → Re-enable and wait 20+ generations  
- Compare predator coordination and efficiency  

---

### 🧩 Custom Scenarios

- **C** → Clear world  
- **F** → Add plants  
- **O** → Add predators  
- **SPACE** → Begin evolution  

---

## 📈 Reporting

Press **R** anytime to generate:

- Generation count  
- Total and per-species population  
- Temperature and light levels  
- Diversity score *(Shannon entropy)*  
- Total births and deaths  
- Hive training status  

---

## 🌍 Summary

**Evolving AI Biosphere** creates a sandbox of **digital evolution**, where:

- Neural agents **adapt and learn**  
- The **Hive Mind** evolves collectively  
- Nature’s balance unfolds through **machine learning**  

> A glimpse into how life might look when evolution is powered by AI.
