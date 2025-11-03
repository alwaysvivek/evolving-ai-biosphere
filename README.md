# evolving-ai-biosphere
A living digital world where AI organisms evolve and adapt through memory, learning, and survival.
# AI Ecosphere - Predator Hive LSTM

## Complete Documentation

---

## Section 1: The Living Digital Ecosystem

### Overview

AI Ecosphere is a self-evolving artificial life simulation where three species—plants, herbivores, and predators—interact in a dynamic grid-based world. Unlike traditional simulations with fixed behaviors, this ecosystem uses neural networks and reinforcement learning to create emergent behaviors, allowing species to adapt, learn, and evolve over hundreds of generations.

The simulation creates a living digital petri dish where natural selection, energy economics, and machine learning converge. Every organism is an autonomous agent making real-time decisions about survival, reproduction, and movement based on its perception of the environment.

---

### The Three Species

#### **Plants (Green)**
Plants are the foundation of the ecosystem, converting light into energy through photosynthesis. They are sessile organisms that cannot move but can reproduce when they accumulate sufficient energy.

**Life Cycle:**
- Plants grow by absorbing light, with energy gain influenced by both global light levels and local spatial light variations
- They reproduce asexually when their energy exceeds 130 units, creating offspring in adjacent empty tiles
- Energy is split during reproduction (parent retains 50%)
- Plants face death from: energy depletion, old age (350+ cycles), overcrowding (more than 5 plant neighbors), or insufficient local light
- Herbivore predation is their primary mortality factor

**Survival Mechanics:**
- Maximum energy capacity: 150 units
- Metabolism rate: 0.4 units per cycle
- Photosynthesis provides 3.5 energy per cycle (modified by light and temperature)
- Each plant occupies a spatial grid cell and emits "scent" that attracts herbivores
- Overcrowding causes energy penalties and increased death probability
- Dark patches (local light < 0.18) cause accelerated energy loss

#### **Herbivores (Blue)**
Herbivores are mobile creatures that feed on plants while fleeing from predators. They possess basic neural networks (LSTM) that help them make survival decisions.

**Life Cycle:**
- Herbivores must constantly balance energy intake (eating plants) with energy expenditure (movement and metabolism)
- They reproduce at 90+ energy, creating offspring that inherit mutated versions of their parent's neural network
- Maximum lifespan: 400 cycles
- Death occurs from: starvation, old age, or predator attacks

**Behavior Systems:**
- **Predator Avoidance:** When predators are detected nearby, herbivores immediately switch to escape mode, calculating the safest adjacent tile based on predator density
- **Foraging:** In safe conditions, herbivores use their neural networks to decide between reproducing, moving toward plant-rich areas, or resting
- **Scent Navigation:** They follow plant scent gradients diffused across the grid, with stronger concentrations indicating food sources
- **Energy Economics:** Movement costs 1.0 energy, escaping costs 2.5 energy, hunting plants rewards up to 60 energy

**Intelligence:**
Each herbivore has a small LSTM neural network that processes 8 inputs:
- Nearby plant count (normalized)
- Nearby herbivore count
- Nearby predator count
- Nearby nutrient count
- Personal energy level
- Age factor
- Two random noise inputs for exploration

The network outputs 4 action preferences: reproduce, move/hunt, rest, or default behavior.

#### **Predators (Red)**
Predators are apex hunters that feed exclusively on herbivores. Unlike other species, all predators share a single collective intelligence—the **Predator Hive Mind**.

**Life Cycle:**
- Predators must hunt herbivores to survive, with each successful kill providing up to 120 energy
- Reproduction threshold: 100+ energy
- Maximum lifespan: 600 cycles (longest of all species)
- Death from: starvation or old age

**Hunting Behavior:**
- Predators use herbivore scent trails to track prey across the map
- They can detect herbivore presence through diffused chemical signals
- Movement is biased toward areas with high herbivore concentration
- Active hunting costs energy but provides large rewards upon success

**The Hive Mind:**
The most unique feature of predators is their shared LSTM policy network. Every predator in the simulation uses the same neural network, creating a collective intelligence that learns from all predator experiences simultaneously.

- All predators contribute experiences (observation, action, reward) to a shared memory pool
- Every 20 generations, the hive network trains on accumulated experiences using REINFORCE algorithm
- Successful hunting strategies discovered by one predator benefit the entire species
- The hive can learn to coordinate indirect strategies like cutting off escape routes or targeting weakened prey

---

### Environmental Dynamics

#### **Spatial Light Field**
Unlike uniform lighting, the world has a spatially varying light map with brighter regions near the top and darker patches scattered throughout. This creates:
- Light gradients that favor certain zones for plant growth
- Strategic value in territorial positioning
- Natural biome formation (bright plant-rich areas vs. dark barren zones)

#### **Temperature Fluctuations**
Temperature drifts randomly between 0.0 and 1.0, affecting:
- Metabolic rates (higher temperature = faster energy consumption)
- Photosynthesis efficiency (optimal at mid-temperatures)

#### **Global Light Levels**
Light intensity varies between 0.3 and 1.0, directly impacting plant energy production and creating day/night-like cycles.

#### **Nutrient Spawning**
Occasionally, nutrient deposits appear randomly across the map, providing opportunities for rapid population growth.

---

### Energy Economics

Energy is the fundamental currency of survival. Every action has an energy cost, and every organism must maintain positive energy balance or face death.

**Energy Flows:**
- **Plants:** Gain energy from photosynthesis (light → energy)
- **Herbivores:** Gain energy from eating plants (biomass transfer)
- **Predators:** Gain energy from eating herbivores (predation)
- **All Species:** Lose energy from metabolism, movement, and reproduction

**The Balance:**
The ecosystem self-regulates through energy cascades. If plants flourish, herbivore populations explode. This feeds predator growth, which then suppresses herbivores, allowing plant recovery. Disrupting any link causes cascading effects.

---

### Scent Diffusion System

The simulation implements a chemical signaling system where organisms leave scent trails:

**Plant Scent:**
- Emitted by all plants, proportional to their energy
- Diffuses outward over 3 steps, creating concentration gradients
- Attracts herbivores toward food sources
- Stronger scents indicate healthier, more energy-rich plants

**Herbivore Scent:**
- Emitted by herbivores based on their energy levels
- Diffuses similarly to plant scent
- Attracts predators to hunting grounds
- Creates "heat maps" of herbivore activity

This system enables indirect perception—organisms don't need direct line-of-sight to detect food or threats.

---

### Emergent Phenomena

Because organisms use learning algorithms rather than fixed rules, the ecosystem develops unexpected patterns:

**Boom-Bust Cycles:**
Populations oscillate in predator-prey relationships, sometimes crashing to near-extinction before recovering.

**Spatial Clustering:**
Species form colonies and territories based on resource distribution and predator pressure.

**Behavioral Evolution:**
Over hundreds of generations, the hive mind develops hunting strategies, herbivores learn better escape routes, and plants optimize reproductive timing.

**Extinction Events:**
Species can go completely extinct, fundamentally altering ecosystem dynamics. A world without predators sees herbivore explosion followed by plant collapse.

**Biodiversity Collapse:**
Occasionally, one species becomes so dominant that diversity drops to near-zero, creating fragile monocultures.

---

### Reproduction and Inheritance

**Plants:** Simple cloning with no genetic variation—all offspring are identical to parents.

**Herbivores:** Neural network weights are copied to offspring with small random mutations (2% chance per weight, ±0.01 perturbation). This creates gradual evolution of behaviors.

**Predators:** Offspring don't inherit individual brains—they immediately connect to the shared hive mind. Evolution happens at the species level through collective learning rather than individual inheritance.

---

### Death and Decay

Organisms die from multiple causes, each tracked in ecosystem statistics:

- **Starvation:** Energy drops to zero
- **Old Age:** Exceeds maximum lifespan (species-dependent)
- **Predation:** Eaten by a higher trophic level
- **Overcrowding:** Plants in dense clusters die from competition
- **Light Starvation:** Plants in areas with insufficient local light wither
- **Random Events:** Triggered manually via keyboard controls

Dead organisms disappear instantly—there is no decomposition or nutrient recycling beyond occasional nutrient spawning events.

---

### Interactive Controls & Features

The simulation provides extensive real-time manipulation:

**Spawning Patterns:**
- **F:** Flower pattern (radial plant arrangement with herbivore ring)
- **S:** Spiral galaxy (mixed species in spiral formation)
- **O:** Predator swarm (6x6 predator cluster)
- **N:** Nutrient field (random nutrient deposits)

**Extinction Tools:**
- **K:** Kill all predators (observe herbivore explosion)
- **L:** Kill all herbivores (study predator starvation)
- **P:** Kill all plants (total ecosystem collapse)
- **E:** Scarcity event (drastically reduce plant energy and growth rate)

**Training Control:**
- **T:** Toggle predator hive training ON/OFF (pause learning to observe static behavior)

**Simulation Control:**
- **SPACE:** Play/Pause evolution
- **C:** Clear all organisms and reset
- **R:** Generate detailed statistical report
- **Q:** Quit simulation

---

### Visual Feedback

The rendering system provides rich visual information:

**Organism Representation:**
- Colored circles represent each organism (green=plant, blue=herbivore, red=predator)
- Circle brightness indicates energy level (brighter = more energy)
- White dots appear on organisms older than 180 cycles
- Smooth interpolated movement creates fluid animation

**Scent Visualization:**
- Faint green halos show plant scent concentrations
- Faint red halos show herbivore scent concentrations
- Stronger colors indicate higher concentrations

**Background Environment:**
- Procedurally generated soil-like texture using layered noise
- Color tint shifts based on temperature (warm = reddish, cool = bluish)
- Darkness increases when global light is low

**Statistics Overlay:**
Real-time display shows:
- Generation count
- Total population and per-species counts
- Temperature and light levels
- Species diversity score (0-1, based on Shannon entropy)
- Total births and deaths
- Hive experience count and training status

---

### Long-Term Dynamics

Running the simulation for hundreds of generations reveals:

**Generation 0-50:** Explosive growth from initial spawn, rapid population swings, unstable predator-prey ratios.

**Generation 50-200:** Stabilization as herbivore neural networks evolve predator avoidance, plants learn optimal reproduction timing, predators develop hunting efficiency through hive learning.

**Generation 200+:** Complex equilibrium states, possible species extinctions, emergence of spatial structure (plant forests, herbivore migrations, predator territories).

**Critical Events:** Manual extinction events or scarcity triggers can cause permanent regime shifts—ecosystems rarely return to previous states after major disruptions.

---

## Section 2: Installation & Getting Started

### Prerequisites

**System Requirements:**
- Python 3.7 or higher
- 4GB RAM minimum (8GB recommended for large simulations)
- Graphics capable of rendering 800x800 window at 30 FPS
- Windows, macOS, or Linux

### Installation Steps

**1. Install Python Dependencies**

Open terminal/command prompt and run:

```bash
pip install pygame torch numpy
```

**Package Details:**
- `pygame`: Graphics rendering and user input handling
- `torch`: PyTorch for neural networks (CPU version is sufficient)
- `numpy`: Numerical computations and array operations

**2. Download the Simulation**

Save the main Python file as `ai_ecosphere_hive.py`

**3. Run the Simulation**

```bash
python ai_ecosphere_hive.py
```

Or if you have multiple Python versions:

```bash
python3 ai_ecosphere_hive.py
```

### First Launch

When you start the simulation:

1. **Window Opens:** An 800x800 pixel window appears showing the initial ecosystem
2. **Paused State:** Simulation starts paused—organisms are visible but frozen
3. **Control Hints:** Terminal displays all keyboard controls
4. **Initial Population:** Includes a flower pattern of plants with herbivores, a predator swarm, and scattered nutrients

**Recommended First Steps:**

1. Press **SPACE** to unpause and watch natural evolution
2. Wait 20-30 generations to observe population dynamics
3. Press **R** to generate a statistical report
4. Try **F** or **S** to spawn new organism patterns
5. Experiment with **E** (scarcity) to observe crisis response
6. Use **K**, **L**, or **P** to trigger extinction events

---



### Basic Usage Patterns

**Observing Natural Evolution:**
```
1. Start simulation (python ai_ecosphere_hive.py)
2. Press SPACE to unpause
3. Wait and watch for 100+ generations
4. Press R periodically for reports
```

**Testing Ecosystem Resilience:**
```
1. Let ecosystem stabilize (50+ generations)
2. Press K to kill all predators
3. Observe herbivore explosion and plant collapse
4. Watch for natural predator recovery (if any)
```

**Studying Hive Learning:**
```
1. Press T to disable training
2. Observe predator behavior (baseline)
3. Press T to re-enable training
4. Wait 20+ generations for learning to occur
5. Compare predator hunting efficiency
```

**Creating Custom Scenarios:**
```
1. Press C to clear everything
2. Press F multiple times to create plant clusters
3. Press O to add predator swarms
4. Let herbivores spawn naturally from plant proximity
5. Press SPACE to begin evolution
```

---

