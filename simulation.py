# ai_ecosphere_hive.py
# Requires: pygame, torch, numpy
# pip install pygame torch numpy
import warnings
# Suppress specific SSL warnings from urllib3
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
# Suppress LangChain pydantic v1 deprecation if we can't fully upgrade yet, but cleaning the import helps
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pygame
import time
import random
import numpy as np
import math
import copy
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import queue
from typing import TypedDict, Annotated, List, Literal
from langgraph.graph import StateGraph, START, END
from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
# UPDATED: Use pydantic directly to avoid deprecation warning
from pydantic import BaseModel, Field
import uuid
import datetime
import re
import mlflow


# ------- Utilities -------
def clamp(x, a, b): return max(a, min(b, x))


def lerp(a, b, t): return a + (b - a) * t


# Grid directions (dx,dy) - index 0..4 for actions we use
DIRS_4 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
DIRS_MOVE_ONLY = [(1, 0), (-1, 0), (0, 1), (0, -1)]


# -----------------------------
# Shared Predator Hive (LSTM policy)
# -----------------------------
class PredatorHive(nn.Module):
    """
    Shared LSTM policy for predators.
    Input: small observation vector (same as Cell.perceive_vector).
    Output: logits for 4 high-level actions (0=reproduce,1=move,2=rest,3=hunt)
    We'll use a simple REINFORCE-like training (log_prob * reward).
    """

    def __init__(self, input_size=8, hidden=64, output_size=4, device=None):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.input_size = input_size
        self.hidden = hidden
        self.output_size = output_size

        # single-step LSTM cell; we'll run it fresh per observation during training (no inplace)
        self.lstm = nn.LSTMCell(input_size, hidden).to(self.device)
        self.fc = nn.Linear(hidden, output_size).to(self.device)

        # init
        for name, p in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)

        self.to(self.device)

    def forward_logits(self, x_tensor):
        """
        x_tensor: (B, input_size)
        returns logits tensor (B, output_size)
        We implement loop to use LSTMCell with zeros hidden per sample to keep things stable.
        """
        B = x_tensor.shape[0]
        h0 = torch.zeros(B, self.hidden, device=self.device)
        c0 = torch.zeros(B, self.hidden, device=self.device)
        # run single-step LSTMCell for each batch sample
        h, c = self.lstm(x_tensor, (h0, c0))
        logits = self.fc(h)
        return logits

    def act_np(self, obs_np):
        """
        obs_np: single observation numpy array shape (input_size,)
        returns: probs (numpy), used for sampling action at runtime (no grad)
        """
        xt = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.forward_logits(xt)  # (1, output_size)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            # Sample action based on probabilities (instead of greedy argmax)
            action_idx = np.random.choice(len(probs), p=probs)
        return probs, action_idx  # Return both for debug/tracking

    def train_from_experiences(self, experiences, lr=1e-3, epochs=6, batch_size=64):
        """
        experiences: list of (obs_np, action_idx, reward_float)
        We'll do a simple REINFORCE: maximize expected reward by minimizing -logpi(a|s)*R
        Normalize rewards before training.
        """
        if len(experiences) == 0:
            return

        device = self.device
        obs_list = [e[0] for e in experiences]
        acts = np.array([e[1] for e in experiences], dtype=np.int64)
        rews = np.array([e[2] for e in experiences], dtype=np.float32)

        # normalize rewards (advantage)
        if rews.std() > 1e-6:
            rews = (rews - rews.mean()) / (rews.std() + 1e-6)
        else:
            rews = rews - rews.mean()

        X = torch.tensor(np.stack(obs_list, axis=0).astype(np.float32), device=device)
        A = torch.tensor(acts, dtype=torch.long, device=device)
        R = torch.tensor(rews, dtype=torch.float32, device=device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        N = X.shape[0]
        for ep in range(epochs):
            perm = torch.randperm(N, device=device)
            for i in range(0, N, batch_size):
                idx = perm[i:i + batch_size]
                xb = X[idx]
                ab = A[idx]
                rb = R[idx]

                optimizer.zero_grad()
                logits = self.forward_logits(xb)  # (B, out)
                logp = F.log_softmax(logits, dim=-1)
                chosen_logp = logp[range(len(ab)), ab]  # (B,)
                # loss = -(chosen_logp * rb).mean()  => minimize negative expected reward
                loss = -(chosen_logp * rb).mean()
                loss.backward()
                optimizer.step()

        # done


# -----------------------------
# Per-cell LSTM for plants/herbivores (if you want later); not used for predators now
# (we keep CellLSTM for compatibility)
# -----------------------------
class CellLSTM(nn.Module):
    def __init__(self, input_size=8, hidden_size=32, output_size=4, device=None):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.lstm = nn.LSTMCell(input_size, hidden_size).to(self.device)
        self.fc = nn.Linear(hidden_size, output_size).to(self.device)
        self.hidden_size = hidden_size
        # init weights
        for name, p in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
        self.h = torch.zeros(1, hidden_size, device=self.device)
        self.c = torch.zeros(1, hidden_size, device=self.device)

    def forward_np(self, x_np):
        with torch.no_grad():
            xt = torch.tensor(x_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.h, self.c = self.lstm(xt, (self.h, self.c))
            out = self.fc(self.h)
            probs = F.softmax(out, dim=-1).cpu().numpy()[0]
            return probs

    def reset_state(self):
        self.h.zero_();
        self.c.zero_()

    def clone(self):
        new = CellLSTM(device=self.device)
        new.load_state_dict(copy.deepcopy(self.state_dict()))
        new.reset_state()
        return new


# -----------------------------
# Simple Q-learning agent (kept for compatibility)
# -----------------------------
def _state_key(local_counts, energy):
    pc = clamp(local_counts.get(0, 0), 0, 3)
    hc = clamp(local_counts.get(1, 0), 0, 3)
    pr = clamp(local_counts.get(2, 0), 0, 3)
    eb = int(clamp(energy / 100.0 * 4, 0, 3))
    return pc, hc, pr, eb


class QAgent:
    def __init__(self, actions, alpha=0.2, gamma=0.9, eps=0.15):
        self.q = {}
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def get_q(self, state_key, a):
        return self.q.get((state_key, a), 0.0)

    def choose(self, local_counts, energy):
        key = _state_key(local_counts, energy)
        if random.random() < self.eps:
            return random.randrange(len(self.actions)), key
        vals = [self.get_q(key, i) for i in range(len(self.actions))]
        maxv = max(vals)
        bests = [i for i, v in enumerate(vals) if v == maxv]
        return random.choice(bests), key

    def learn(self, state_key, action_index, reward, next_state_key):
        old = self.get_q(state_key, action_index)
        next_max = max([self.get_q(next_state_key, i) for i in range(len(self.actions))], default=0.0)
        new = (1 - self.alpha) * old + self.alpha * (reward + self.gamma * next_max)
        self.q[(state_key, action_index)] = new

    def clone(self):
        na = QAgent(self.actions, self.alpha, self.gamma, self.eps)
        na.q = copy.deepcopy(self.q)
        return na



# -----------------------------
# LangGraph Commentary Agent
# -----------------------------
class SimState(TypedDict):
    generation: int
    stats: str
    commentary: str

class CommentaryAgent:
    def __init__(self):
        self.llm = None
        self.graph = None
        
        # Async init
        def _init():
            try:
                self.llm = OllamaLLM(model="llama3.2")
                builder = StateGraph(SimState)
                builder.add_node("commentator", self.generate_commentary)
                builder.add_edge(START, "commentator")
                builder.add_edge("commentator", END)
                self.graph = builder.compile()
                print("[CommentaryAgent] Connected to Ollama (llama3.2)")
            except Exception as e:
                print(f"[CommentaryAgent] Init failed: {e}")
        
        t = threading.Thread(target=_init)
        t.daemon = True
        t.start()

    def generate_commentary(self, state: SimState):
        if not self.llm:
            return {"commentary": "Ollama initializing..."}
        
        prompt = (f"You are a nature documentary narrator observing an artificial life simulation. "
                  f"Current Generation: {state['generation']}. "
                  f"Ecosystem Stats: {state['stats']}. "
                  f"Provide a 1-sentence dramatic, scientific, or philosophical commentary on the current state of the ecosystem. "
                  f"Focus on the balance between predators, herbivores, and plants.")
        try:
            response = self.llm.invoke(prompt)
            # clean up newlines
            response = response.replace("\n", " ")
        except Exception as e:
            response = f"Commentary error: {str(e)}"
        
        return {"commentary": response}


    def invoke(self, generation, stats_str):
        return self.graph.invoke({"generation": generation, "stats": stats_str, "commentary": ""})


# -----------------------------
# History & Forensics
# -----------------------------
class EventLogger:
    def __init__(self, max_history=1000):
        self.history = deque(maxlen=max_history)
        self.significant_events = [] # Persist significant events (extinctions) forever

    def log(self, gen, event_type, details):
        entry = {
            "gen": gen,
            "type": event_type,
            "details": details,
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
        }
        self.history.append(entry)
        
        # Check for significance
        if "LAST_SURVIVOR" in details.get("tags", []):
            self.significant_events.append(entry)
            print(f"*** SIGNIFICANT EVENT: {event_type} - {details} ***")

GLOBAL_EVENT_LOGGER = EventLogger()







# -----------------------------
# God Mode (Tool Calling)
# -----------------------------
# We define tools that will be bound to the LLM.

GOD_MODE_QUEUE = queue.Queue()





@tool
def consult_history(query: str):
    """Query the simulation history. Use this to answer questions about 'who died', 'when', 'extinction', etc.
    query: Natural language query or keyword (e.g., 'predator extinction', 'last death', 'cause of death').
    """
    # Simple keyword search implementation
    hits = []
    q_lower = query.lower()
    
    # helper to format entry
    def fmt(e):
        return f"[Gen {e['gen']}] {e['type']}: {e['details']}"

    # Search significant first
    for e in GLOBAL_EVENT_LOGGER.significant_events:
        if any(w in str(e).lower() for w in q_lower.split()):
            hits.append(fmt(e))
            
    # Search recent history
    for e in reversed(GLOBAL_EVENT_LOGGER.history):
        if len(hits) > 5: break
        if any(w in str(e).lower() for w in q_lower.split()):
            hits.append(fmt(e))
            
    # Generic fallback if no specific keywords found but query asks for "recent"
    if not hits and "recent" in q_lower:
        for e in list(GLOBAL_EVENT_LOGGER.history)[-5:]:
            hits.append(fmt(e))
            
    if not hits:
        return "No specific history records found matching that query. (Found 0)"
    
    return "History Records:\n" + "\n".join(hits)

@tool
def spawn_creatures(species: str, count: int):
    """Spawn a specific number of creatures of a given species.
    species: One of 'plant', 'herbivore', 'predator'.
    count: Number to spawn (1-50).
    """
    GOD_MODE_QUEUE.put(("spawn", species, count))
    return f"Queued spawn: {count} {species}"

@tool
def change_environment(temp_delta: float, light_delta: float):
    """Change global temperature or light.
    temp_delta: Change in temperature (-0.5 to 0.5). Negative is colder.
    light_delta: Change in light (-0.5 to 0.5). Negative is darker.
    """
    GOD_MODE_QUEUE.put(("env", temp_delta, light_delta))
    return f"Queued env change: T{temp_delta:+.2f}, L{light_delta:+.2f}"

@tool
def cull_species(species: str, percentage: float):
    """Kill a percentage of a species.
    species: One of 'plant', 'herbivore', 'predator'.
    percentage: 0.0 to 1.0 (e.g., 0.5 removes half).
    """
    GOD_MODE_QUEUE.put(("cull", species, percentage))
    return f"Queued cull: {percentage*100}% of {species}"

# -----------------------------
# Multi-Agent Council
# -----------------------------
class CouncilState(TypedDict):
    stats: str
    gaia_proposal: str
    entropy_critique: str
    final_decision: str
    messages: List[str]

class CouncilSystem:
    def __init__(self):
        self.llm = OllamaLLM(model="llama3.2")
        self.arbiter_llm = ChatOllama(model="llama3.2", temperature=0) # Arbiter needs tools
        
        # Tools for Arbiter
        self.tools = [spawn_creatures, change_environment, cull_species]
        self.arbiter_llm_with_tools = self.arbiter_llm.bind_tools(self.tools)

        # Build Graph
        builder = StateGraph(CouncilState)
        builder.add_node("gaia", self.node_gaia)
        builder.add_node("entropy", self.node_entropy)
        builder.add_node("arbiter", self.node_arbiter)
        
        builder.add_edge(START, "gaia")
        builder.add_edge("gaia", "entropy")
        builder.add_edge("entropy", "arbiter")
        builder.add_edge("arbiter", END)
        
        self.graph = builder.compile()
        print("[Council] Graph compiled.")

    def node_gaia(self, state: CouncilState):
        prompt = f"Role: Gaia (Life/Growth). Stats: {state['stats']}. Propose an intervention to help life flourish. Be brief."
        res = self.llm.invoke(prompt)
        return {"gaia_proposal": res, "messages": [f"Gaia: {res}"]}

    def node_entropy(self, state: CouncilState):
        prompt = f"Role: Entropy (Death/Balance). Proposal: {state['gaia_proposal']}. Critique this. Is it too safe? Should we add chaos? Be brief."
        res = self.llm.invoke(prompt)
        return {"entropy_critique": res, "messages": [f"Entropy: {res}"]}

    def node_arbiter(self, state: CouncilState):
        # Arbiter takes previous args and calls a tool if needed
        prompt = [
            SystemMessage(f"Role: Arbiter. Weigh Gaia ({state['gaia_proposal']}) and Entropy ({state['entropy_critique']}). Decide on the single best action."),
            HumanMessage("Execute the decision using a tool, or reply 'No Action' if none needed.")
        ]
        res = self.arbiter_llm_with_tools.invoke(prompt)
        
        # execute if tool call
        decision_text = res.content
        if res.tool_calls:
            results = []
            for tool_call in res.tool_calls:
                func_name = tool_call["name"]
                args = tool_call["args"]
                if func_name == "spawn_creatures": s_res = spawn_creatures.invoke(args)
                elif func_name == "change_environment": s_res = change_environment.invoke(args)
                elif func_name == "cull_species": s_res = cull_species.invoke(args)
                else: s_res = "Unknown"
                results.append(s_res)
            decision_text = "ACTION: " + "; ".join(results)
            
        return {"final_decision": decision_text, "messages": [f"Arbiter: {decision_text}"]}

    def invoke(self, stats_str):
        return self.graph.invoke({"stats": stats_str, "messages": []})

GLOBAL_COUNCIL = None

def get_council():
    global GLOBAL_COUNCIL
    if GLOBAL_COUNCIL is None:
        print("[DEBUG] Lazy-initializing Council System...")
        GLOBAL_COUNCIL = CouncilSystem()
    return GLOBAL_COUNCIL

@tool
def summon_council(query: str):
    """Summon the Multi-Agent Council (Gaia, Entropy, Arbiter) to debate and decide on an ecosystem intervention.
    Use this when the user asks for 'council', 'debate', 'autonomy', or 'help me decide'.
    query: Optional topic for the council to focus on (can be empty).
    """
    council = get_council()
    
    # Grab current stats (using global hack or assuming GodModeAgent passes it? 
    # GodModeAgent doesn't inherently have stats. We'll use a placeholder or read from Sim if possible.
    # ideally we pass stats in query or have a way to access SIM.
    # For now, we'll let main loop update a global STATS variable? Or just generic debate.
    # Let's assume the AGENT prompt included stats, or we just ask general advice.
    
    res = council.invoke("Current Ecosystem Status: Unknown (User requested intervention)")
    # Format the debate
    log = "\n".join(res["messages"])
    return f"--- Council Debate ---\n{log}\n----------------------"

class GodModeAgent:
    def __init__(self):
        self.llm = None
        self.llm_with_tools = None
        
        def _init():
            try:
                # function calling usually requires the chat model
                llm = ChatOllama(model="llama3.2", temperature=0)
                self.tools = [spawn_creatures, change_environment, cull_species, consult_history, summon_council]
                self.llm_with_tools = llm.bind_tools(self.tools)
                self.llm = llm # set last to imply readiness
                print("[GodMode] Connected to Ollama (llama3.2) with tools.")
            except Exception as e:
                print(f"[GodMode] Failed to connect: {e}")
        
        t = threading.Thread(target=_init)
        t.daemon = True
        t.start()

    def process_command(self, user_text: str):
        if not self.llm: return "God Mode offline."
        
        # Updated system prompt to encourage chat if no tool is needed, but prioritize tools for queries
        messages = [SystemMessage("You are an omnipotent ecosystem controller. Use 'consult_history' for events, 'summon_council' for multi-agent decisions, and tools to act. If the user just wants to chat, reply normally."), 
                    HumanMessage(user_text)]
        try:
            ai_msg = self.llm_with_tools.invoke(messages)
            # execute tools if any
            if ai_msg.tool_calls:
                messages.append(ai_msg) # Add the AIMessage with tool calls to history
                
                results = []
                for tool_call in ai_msg.tool_calls:
                    # We manually map to our functions 
                    # (In a full LangGraph we'd use ToolNode, but here we just run them)
                    func_name = tool_call["name"]
                    args = tool_call["args"]
                    call_id = tool_call.get("id", "call_default") # Manual tool calls might not have distinct IDs in this simple loop, but LangChain objects usually do
                    
                    if func_name == "spawn_creatures":
                        res = spawn_creatures.invoke(args)
                    elif func_name == "change_environment":
                        res = change_environment.invoke(args)
                    elif func_name == "cull_species":
                        res = cull_species.invoke(args)
                    elif func_name == "consult_history":
                        res = consult_history.invoke(args)
                    elif func_name == "summon_council":
                        res = summon_council.invoke(args)
                    else:
                        res = "Unknown tool: " + func_name
                    
                    results.append(str(res))
                    # Append ToolMessage
                    messages.append(ToolMessage(content=str(res), tool_call_id=call_id, name=func_name))
                
                # Turn 2: Get final response from LLM based on tool outputs
                final_response = self.llm_with_tools.invoke(messages)
                return final_response.content
            else:
                return ai_msg.content
        except Exception as e:
            return f"Error: {e}"


# -----------------------------
# Core simulation classes
# -----------------------------
class CellType:
    PLANT = 0
    HERBIVORE = 1
    PREDATOR = 2
    NUTRIENT = 3


class Cell:
    def __init__(self, pos, cell_type, brain=None, qagent=None):
        self.pos = pos
        self.type = cell_type
        self.energy = 100.0
        self.age = 0
        self.generation = 0
        self.draw_pos = np.array([pos[0], pos[1]], dtype=float)
        self.id = uuid.uuid4().hex[:6] # Unique short ID
        
        # Biology: Metabolism (Gene)
        # Default 1.0. Higher = faster movement, higher energy burn.
        self.metabolism = 1.0 
        
        if brain is None:
            # only plants/herbivores keep small NN; predators will use hive mostly
            if cell_type in (CellType.PLANT, CellType.HERBIVORE):
                self.brain = CellLSTM()
            else:
                self.brain = None
        else:
            self.brain = brain

        if qagent is None and cell_type in (CellType.HERBIVORE, CellType.PREDATOR):
            self.qagent = QAgent(actions=DIRS_4)
        else:
            self.qagent = qagent

        if cell_type == CellType.PLANT:
            self.max_energy = 150.0
            self.metabolism = 0.4  # Increased metabolism for plants
            # MODIFIED: Increased reproduction threshold for energy-based reproduction
            self.reproduction_threshold = 130.0
        elif cell_type == CellType.HERBIVORE:
            self.max_energy = 120.0
            self.metabolism = 0.9
            self.reproduction_threshold = 90.0
        elif cell_type == CellType.PREDATOR:
            self.max_energy = 140.0
            self.metabolism = 1.2  # Reduced metabolism for better survival
            # MODIFIED: Reduced reproduction threshold for faster predator response
            self.reproduction_threshold = 85.0
        else:
            self.max_energy = 40.0
            self.metabolism = 0.0
            self.reproduction_threshold = float('inf')

        if hasattr(self, 'brain') and self.brain is not None:
            try:
                self.brain.reset_state()
            except Exception:
                pass

    def perceive_vector(self, neighbors_info):
        plant_count = neighbors_info.get(CellType.PLANT, 0) / 8.0
        herb_count = neighbors_info.get(CellType.HERBIVORE, 0) / 8.0
        pred_count = neighbors_info.get(CellType.PREDATOR, 0) / 8.0
        nutrient_count = neighbors_info.get(CellType.NUTRIENT, 0) / 8.0
        energy_level = clamp(self.energy / (self.max_energy or 1.0), 0.0, 1.0)
        age_factor = clamp(self.age / 200.0, 0.0, 1.0)
        return np.array([plant_count, herb_count, pred_count, nutrient_count,
                         energy_level, age_factor, random.random(), random.random()], dtype=np.float32)

    def high_level_action(self, neighbors_info, global_bias=None):
        # For herbivores/plants with local LSTM
        if self.brain is not None:
            vec = self.perceive_vector(neighbors_info)
            probs = self.brain.forward_np(vec)
            if global_bias is not None:
                probs = probs * global_bias
                probs = probs / np.sum(probs)
            return int(np.argmax(probs)), probs
        else:
            # fallback random
            return random.randrange(4), np.ones(4) / 4.0


# -----------------------------
# Ecosystem simulation with hive
# -----------------------------
def _diffuse_once(grid, decay):
    w, h = grid.shape
    new = grid * decay
    for x in range(w):
        for y in range(h):
            s = grid[x, y]
            share = s * (1.0 - decay) * 0.25
            if share <= 0: continue
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    new[nx, ny] += share
    return new


class EcoLifeSimulation:
    # Set a maximum global plant count for stability
    MAX_PLANTS_ALLOWED = 700

    def __init__(self, width, height, tile_size, fps, device=None):
        print("[DEBUG] Initializing Pygame...")
        pygame.init()
        print("[DEBUG] Pygame initialized.")

        try:
             # Connect to MLflow in a non-blocking way or tolerate failure
             # We spin up a thread to do the logging connection so it doesn't freeze the UI
             def init_mlflow():
                try:
                    time.sleep(2) # Wait for service to possibly come up
                    mlflow.set_tracking_uri("http://127.0.0.1:5001")
                    mlflow.set_experiment("EcoLife_Simulation")
                    self.mlflow_run = mlflow.start_run()
                    mlflow.log_param("width", width)
                    mlflow.log_param("height", height)
                    print("[DEBUG] MLflow connected (Async).")
                except Exception as e:
                    print(f"[DEBUG] MLflow async connect failed: {e}")
             
             t = threading.Thread(target=init_mlflow)
             t.daemon = True
             t.start()
        except:
             pass
        
        self.sidebar_width = 400  # Increased from 300
        self.sim_width = width
        self.sim_height = height
        self.width = width + self.sidebar_width
        self.height = height
        
        self.tile_size = tile_size;
        self.fps = fps
        self.grid_width = self.sim_width // tile_size;
        self.grid_height = self.sim_height // tile_size

        print("[DEBUG] Creating Display...")
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("AI Ecosphere - Predator Hive LSTM")
        print("[DEBUG] Display created.")

        self.background_color = (15, 20, 30)
        self.cell_colors = {
            CellType.PLANT: (50, 200, 80),
            CellType.HERBIVORE: (100, 150, 255),
            CellType.PREDATOR: (255, 80, 80),
            CellType.NUTRIENT: (255, 220, 100)
        }

        self.running = True
        self.playing = False
        self.count = 0
        self.update_freq = 8
        self.generation = 0

        self.cells = {}
        self.stats = {'total_births': 0, 'total_deaths': 0, 'max_population': 0, 'dominant_species': None}

        self.temperature = 0.5
        self.light = 0.8

        # Predator hive (shared LSTM)
        print("[DEBUG] Initializing Hive Device...")
        self.device = device or (torch.device("mps") if torch.backends.mps.is_available() else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")))
        print(f"[DEBUG] Using device: {self.device}")
        
        self.hive = PredatorHive(input_size=8, hidden=64, output_size=4, device=self.device)
        self.hive_experiences = []  # list of (obs_np, action_idx, reward_float)
        self.TRAIN_INTERVAL = 20  # train hive every N generations
        self.MIN_EXPERIENCES_TO_TRAIN = 16

        # spawn defaults
        self.spawn_flower_pattern(self.grid_width // 2, self.grid_height // 2)
        self.spawn_predator_swarm(6, 6)
        self.spawn_nutrients_field(3, 3, 6)

        # Passive plant growth chance removed later in update_ecosystem
        self.plant_passive_grow_chance = 0.01
        self.evolution_mode = True
        self.mutation_base = 0.06

        # scent maps
        self.plant_scent = np.zeros((self.grid_width, self.grid_height), dtype=np.float32)
        self.herbivore_scent = np.zeros((self.grid_width, self.grid_height), dtype=np.float32)
        self.scent_decay = 0.85
        self.scent_diffuse_steps = 3

        # NEW: spatial light map (per-grid-cell) and textured background
        self.light_map = self.generate_light_map()
        self.background_surface = self.generate_noise_background()

        self.font = pygame.font.Font(None, 24)
        self.extinction_events = []
        self.trend_history = deque(maxlen=50)
        # God Mode / Director
        print("[DEBUG] Initializing God Mode Agent...")
        self.god_mode_agent = GodModeAgent()
        
        # Input Box
        self.input_rect = pygame.Rect(self.sim_width + 20, 560, self.sidebar_width - 40, 32)
        self.input_text = ""
        self.input_active = False
        self.input_feedback = ""
        self.god_mode_scroll_y = 0
        self.god_mode_content_height = 0
        
        # Key bindings shown at start
        self.print_bindings()
        print("[DEBUG] Initialization Complete.")

    def print_bindings(self):
        print("\n=== Controls (keyboard) ===")
        print("SPACE - Play/Pause")
        print("F     - Spawn flower pattern")
        print("S     - Spawn spiral galaxy")
        print("O     - Spawn predator swarm")
        print("N     - Spawn nutrient field")
        print("C     - Clear")
        print("R     - Print report")
        print("T     - Toggle predator training ON/OFF (training runs automatically when ON)")
        print("E     - Trigger scarcity event")
        print("K     - Kill predators")
        print("L     - Kill herbivores")
        print("P     - Kill plants")
        print("Q     - Quit")
        print("==========================\n")

        # training toggle state
        self.training_enabled = True

    # ---------- Light map & background generation ----------
    def generate_light_map(self):
        """
        Create a spatial light map over the grid.
        Brighter near the top, with gentle noise to create patches.
        Returns an array shape (grid_width, grid_height) with values 0..1.
        """
        w, h = self.grid_width, self.grid_height
        scale_base = 0.12
        arr = np.zeros((w, h), dtype=np.float32)

        # base vertical gradient (top brighter)
        y_grad = np.linspace(1.0, 0.5, h)
        for x in range(w):
            arr[x, :] = y_grad

        # add layered sinusoidal/noise patterns for organic variation
        for x in range(w):
            for y in range(h):
                nx = x * scale_base
                ny = y * scale_base
                v = 0.12 * math.sin(1.3 * nx + 0.5 * ny)
                v += 0.08 * math.cos(2.1 * nx - 0.4 * ny)
                v += 0.06 * math.sin(0.8 * nx * ny + 0.2)
                # small random smoothing
                v += (random.random() - 0.5) * 0.03
                arr[x, y] = clamp(arr[x, y] + v, 0.0, 1.0)
        # final normalization
        if arr.max() > 0:
            arr = arr / (arr.max() + 1e-9)
        return arr

    def generate_noise_background(self):
        """
        Generate a soil-like textured background surface using value-noise style
        (a lightweight Perlin-like approach). The surface size equals the
        pixel screen size (grid_width*tile_size, grid_height*tile_size).
        """
        surf_w = self.grid_width * self.tile_size
        surf_h = self.grid_height * self.tile_size
        texture = np.zeros((self.grid_width, self.grid_height), dtype=np.float32)

        # Produce low-frequency field using sin/cos + layered scales
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                nx = x / max(1, self.grid_width)
                ny = y / max(1, self.grid_height)
                v = 0.0
                v += 0.6 * math.sin(6.28 * (nx * 1.0 + 0.13 * ny))
                v += 0.35 * math.cos(6.28 * (nx * 2.2 - 0.07 * ny))
                v += 0.2 * math.sin(6.28 * (nx * 3.7 + ny * 1.4))
                v += (random.random() - 0.5) * 0.08
                texture[x, y] = v

        # normalize texture to 0..1
        mn = texture.min();
        mx = texture.max()
        if mx - mn > 1e-6:
            texture = (texture - mn) / (mx - mn)
        else:
            texture = np.clip(texture, 0.0, 1.0)

        # create pygame surface and paint per-grid-cell color then scale up by tile_size
        surf = pygame.Surface((self.grid_width, self.grid_height))
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                v = texture[x, y]
                # soil-like color palette (tuned)
                r = int(40 + v * 40)  # 40..80
                g = int(30 + v * 25)  # 30..55
                b = int(20 + v * 18)  # 20..38
                surf.set_at((x, y), (r, g, b))
        surf = pygame.transform.smoothscale(surf, (surf_w, surf_h))
        return surf

    # spawn helpers
    def spawn_flower_pattern(self, cx, cy):
        if 0 <= cx < self.grid_width and 0 <= cy < self.grid_height:
            self.cells[(cx, cy)] = Cell((cx, cy), CellType.PLANT)
        for angle in range(0, 360, 72):
            rad = math.radians(angle)
            for r in range(1, 4):
                x = int(cx + math.cos(rad) * r);
                y = int(cy + math.sin(rad) * r)
                if 0 <= x < self.grid_width and 0 <= y < self.grid_height and (x, y) not in self.cells:
                    self.cells[(x, y)] = Cell((x, y), CellType.PLANT)
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            x = int(cx + math.cos(rad) * 5);
            y = int(cy + math.sin(rad) * 5)
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height and (x, y) not in self.cells:
                self.cells[(x, y)] = Cell((x, y), CellType.HERBIVORE)

    def spawn_predator_swarm(self, cx, cy, n=6):
        for i in range(n):
            for sign in (-1, 1):
                x = cx + sign * i;
                y = cy + i
                if 0 <= x < self.grid_width and 0 <= y < self.grid_height and (x, y) not in self.cells:
                    self.cells[(x, y)] = Cell((x, y), CellType.PREDATOR)

    def spawn_nutrients_field(self, cx, cy, size):
        for _ in range(size * 4):
            x = cx + random.randint(-size, size);
            y = cy + random.randint(-size, size)
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height and (x, y) not in self.cells:
                self.cells[(x, y)] = Cell((x, y), CellType.NUTRIENT)

    def spawn_spiral_galaxy(self, cx, cy):
        for angle in range(0, 720, 12):
            rad = math.radians(angle)
            r = angle / 60.0
            x = int(cx + math.cos(rad) * r);
            y = int(cy + math.sin(rad) * r)
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height and (x, y) not in self.cells:
                t = random.choice([CellType.PLANT, CellType.HERBIVORE, CellType.PREDATOR])
                self.cells[(x, y)] = Cell((x, y), t)

    def get_neighbors(self, pos):
        x, y = pos
        out = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if dx == 0 and dy == 0: continue
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height: out.append((nx, ny))
        return out

    def get_neighbor_info(self, pos):
        info = defaultdict(int)
        for n in self.get_neighbors(pos):
            c = self.cells.get(n)
            if c: info[c.type] += 1
        return info

    def compute_global_obs(self):
        counts = defaultdict(int)
        for c in self.cells.values(): counts[c.type] += 1
        obs = np.array([counts[CellType.PLANT], counts[CellType.HERBIVORE],
                        counts[CellType.PREDATOR], counts[CellType.NUTRIENT]], dtype=np.float32)
        obs = obs / max(1.0, (self.grid_width * self.grid_height) / 16.0)
        return obs

    # scent maps: plant_scent attracts herbivores; herbivore_scent attracts predators
    def compute_scent_maps(self):
        w, h = self.grid_width, self.grid_height
        plant_src = np.zeros((w, h), dtype=np.float32)
        herb_src = np.zeros((w, h), dtype=np.float32)
        for (x, y), c in self.cells.items():
            if c.type == CellType.PLANT:
                plant_src[x, y] += 1.0 + (c.energy / c.max_energy)
            if c.type == CellType.HERBIVORE:
                herb_src[x, y] += 1.0 + (c.energy / c.max_energy) * 0.9
        ps = plant_src.copy();
        hs = herb_src.copy()
        for _ in range(self.scent_diffuse_steps):
            ps = _diffuse_once(ps, self.scent_decay)
            hs = _diffuse_once(hs, self.scent_decay)
        if ps.max() > 0: ps = ps / (ps.max() + 1e-9)
        if hs.max() > 0: hs = hs / (hs.max() + 1e-9)
        self.plant_scent = ps;
        self.herbivore_scent = hs

        # optional: small dynamic modulation for light map (e.g., day-night) could be added here
        # For now, light_map is static spatial field multiplied by self.light when used.

    def scent_at(self, pos, typ):
        x, y = pos
        if typ == CellType.HERBIVORE:
            return float(self.plant_scent[x, y])
        if typ == CellType.PREDATOR:
            return float(self.herbivore_scent[x, y])
        return 0.0

    # predator training
    def maybe_train_hive(self):
        if not self.training_enabled:
            return
        if self.generation % self.TRAIN_INTERVAL != 0:
            return
        if len(self.hive_experiences) < self.MIN_EXPERIENCES_TO_TRAIN:
            # not enough data
            return
        # train
        print(f"[Hive] Training on {len(self.hive_experiences)} experiences at gen {self.generation}")
        try:
            self.hive.train_from_experiences(self.hive_experiences, lr=1e-3, epochs=8, batch_size=64)
        except Exception as e:
            print("[Hive] Training error:", e)
        # clear experiences after training
        self.hive_experiences.clear()

    # helper: find best empty neighbor for herbivore to flee predators
    def choose_escape_tile(self, pos):
        # evaluate each neighbor empty tile; choose tile with minimal predator count in its neighborhood
        neighbors = self.get_neighbors(pos) + [pos]
        best = pos
        best_score = 1e9
        for n in neighbors:
            if n in self.cells and n != pos:
                continue
            # predator count within 2-cell radius from n
            px, py = n
            score = 0
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = px + dx, py + dy
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        c = self.cells.get((nx, ny))
                        if c and c.type == CellType.PREDATOR:
                            # closer predators weigh more
                            dist = max(abs(dx), abs(dy))
                            score += (3.0 / (dist + 1))
            if score < best_score:
                best_score = score
                best = n
        return best

    # choose predator movement biased toward herbivores (herbivore_scent)
    def choose_predator_move_by_scent(self, pos):
        # choose neighbor (or stay) with max herbivore_scent value (prefers moving toward herbivores)
        candidates = [pos] + self.get_neighbors(pos)
        best = pos
        best_val = -1.0
        for cpos in candidates:
            if cpos in self.cells and cpos != pos:
                # occupied
                continue
            val = self.herbivore_scent[cpos[0], cpos[1]]
            if val > best_val:
                best_val = val;
                best = cpos
        return best

    # NEW FUNCTION: Handles reintroduction of extinct species
    def maybe_reintroduce_species(self):
        counts = defaultdict(int)
        for c in self.cells.values(): counts[c.type] += 1

        # Herbivore reintroduction logic
        if counts[CellType.HERBIVORE] == 0:
            # MODIFIED: Increased chance from 0.005 to 0.015
            if random.random() < 0.015:
                cx = random.randint(5, self.grid_width - 6)
                cy = random.randint(5, self.grid_height - 6)

                # Spawn a tiny starting population (1-3 individuals)
                for _ in range(random.randint(1, 3)):
                    x = cx + random.randint(-1, 1)
                    y = cy + random.randint(-1, 1)
                    if 0 <= x < self.grid_width and 0 <= y < self.grid_height and (x, y) not in self.cells:
                        self.cells[(x, y)] = Cell((x, y), CellType.HERBIVORE)
                        # Ensure the new herbivore has a brain if required by the Cell class,
                        # though Cell() defaults should handle it.

                print(f"**🌱 Herbivores Reintroduced at Gen {self.generation}**")
                return  # Only reintroduce one species per step

        # Predator reintroduction logic (less frequent/higher hurdle)
        # Only reintroduce predators if there are enough herbivores to sustain them (> 10)
        if counts[CellType.PREDATOR] == 0 and counts[CellType.HERBIVORE] > 10:
            # MODIFIED: Increased chance from 0.002 to 0.01
            if random.random() < 0.01:
                cx = random.randint(5, self.grid_width - 6)
                cy = random.randint(5, self.grid_height - 6)
                # Spawn 1 predator
                self.cells[(cx, cy)] = Cell((cx, cy), CellType.PREDATOR)
                print(f"**🦁 Predators Reintroduced at Gen {self.generation}**")

    def update_ecosystem(self):
        # Define MAX_PLANTS_ALLOWED here (or as a class variable as done above)
        MAX_PLANTS_ALLOWED = self.MAX_PLANTS_ALLOWED  # 700

        # Calculate initial population counts for cap checks
        species_count = defaultdict(int)
        for c in self.cells.values(): species_count[c.type] += 1
        current_plant_count = species_count[CellType.PLANT]

        # update scents first
        self.compute_scent_maps()

        new_cells = {}
        to_remove = set()
        to_add = []

        obs = self.compute_global_obs()
        self.trend_history.append(obs)
        
        if self.generation % 5 == 0:
            try:
                mlflow.log_metric("plant_count", species_count[CellType.PLANT], step=self.generation)
                mlflow.log_metric("herbivore_count", species_count[CellType.HERBIVORE], step=self.generation)
                mlflow.log_metric("predator_count", species_count[CellType.PREDATOR], step=self.generation)
                mlflow.log_metric("global_temperature", self.temperature, step=self.generation)
            except: pass

        # passive plant growth (uses modified self.plant_passive_grow_chance)
        # DELETED: Removed passive plant growth loop to enforce energy-threshold reproduction

        # process cells in random order
        items = list(self.cells.items())
        random.shuffle(items)
        for pos, cell in items:
            cell.age += 1
            env_metab = 1.0 + (self.temperature - 0.5) * 0.8
            cell.energy -= cell.metabolism * env_metab

            # MODIFIED: Define max age based on species type
            if cell.type == CellType.PREDATOR:
                max_age = 600  # Increased predator lifespan for better learning
            elif cell.type == CellType.HERBIVORE:
                max_age = 400  # Slightly reduced herbivore lifespan
            else:
                max_age = 350  # Plant lifespan

            # Determine removal cause
            death_cause = "UNKNOWN"
            if cell.energy <= 0: death_cause = "STARVATION"
            elif cell.age > max_age: death_cause = "OLD_AGE"
            
            if cell.energy <= 0 or cell.age > max_age:
                to_remove.add(pos)
                self.stats['total_deaths'] += 1
                
                # Check for significant extinction event (last 5 of species)
                s_count = species_count[cell.type]
                tags = []
                if s_count <= 5:
                    tags.append("LAST_SURVIVOR")
                    
                GLOBAL_EVENT_LOGGER.log(self.generation, "DEATH", 
                                        {"id": cell.id, "type": cell.type, "cause": death_cause, "tags": tags})
                                        
                if cell.type == CellType.PLANT:
                    current_plant_count -= 1  # Decrement if a plant dies
                    species_count[CellType.PLANT] -= 1 # Update local count for next iter check? (Actually this loop is over frozen keys, but useful for logs)
                continue

            neighbor_info = self.get_neighbor_info(pos)

            # PREDATOR: use hive for high-level decisions
            if cell.type == CellType.PREDATOR:
                obs_vec = cell.perceive_vector(neighbor_info)
                # use hive to get sampled action index
                probs, action_idx = self.hive.act_np(obs_vec)

                # action logic similar to before but predator uses hive for decision
                if action_idx == 0 and cell.energy > cell.reproduction_threshold:
                    empty = [n for n in self.get_neighbors(pos) if
                             n not in new_cells and n not in to_remove]  # Use new_cells/to_remove for immediate checks
                    if empty:
                        chosen = random.choice(empty)
                        # child inherits nothing from hive (hive is shared). just reproduce clone of cell brain if any.
                        child = Cell(chosen, CellType.PREDATOR)
                        # Mutation: Metabolism
                        mutation = random.uniform(-0.1, 0.1)
                        child.metabolism = max(0.5, min(2.0, cell.metabolism + mutation))
                        
                        child.generation = cell.generation + 1
                        to_add.append((chosen, child))
                        cell.energy *= 0.5
                        self.stats['total_births'] += 1

                elif action_idx in (1, 3) and cell.energy > 10:
                    # predator movement biased by herbivore scent / direct chase
                    # prefer move towards herbivore scent; we still select among DIRS_4 moves
                    best_target = self.choose_predator_move_by_scent(pos)
                    if best_target != pos and best_target not in self.cells:
                        new_pos = best_target
                    else:
                        # default: try small random move or stay
                        a = random.randrange(len(DIRS_4))
                        dx, dy = DIRS_4[a]
                        new_pos = (pos[0] + dx, pos[1] + dy)
                        if not (0 <= new_pos[0] < self.grid_width and 0 <= new_pos[1] < self.grid_height):
                            new_pos = pos

                    if new_pos in self.cells:
                        target = self.cells[new_pos]
                        if target.type == CellType.HERBIVORE and action_idx == 3:
                            # successful hunt
                            # MODIFIED: Increased energy gain multiplier and max energy
                            energy_gain = min(target.energy * 1.5, 120)
                            cell.energy += energy_gain;
                            cell.energy = min(cell.energy, cell.max_energy)
                            to_remove.add(new_pos)
                            self.stats['total_deaths'] += 1
                            
                            # Log predation
                            victim = self.cells[new_pos]
                            s_count = species_count[victim.type]
                            tags = []
                            if s_count <= 5: tags.append("LAST_SURVIVOR") # Approximation (count is from start of frame)
                            
                            GLOBAL_EVENT_LOGGER.log(self.generation, "DEATH", 
                                                    {"id": victim.id, "type": victim.type, "cause": f"EATEN_BY_{cell.id}", "tags": tags})
                                                    
                            # add experience: reward proportional to energy_gain (un-normalized)
                            self.hive_experiences.append((obs_vec, 3, float(energy_gain)))
                        else:
                            # nothing eaten
                            pass
                    else:
                        # move into empty
                        if new_pos != pos:
                            cell.pos = new_pos
                            cell.draw_pos = np.array([new_pos[0], new_pos[1]], dtype=float)
                        # small energy cost
                    cell.energy -= 1.2

                elif action_idx == 2:
                    # rest -> regain some energy
                    cell.energy += 3.0;
                    cell.energy = min(cell.max_energy, cell.energy)

            # HERBIVORE: run away heuristics + their own LSTM for high-level decisions
            elif cell.type == CellType.HERBIVORE:
                # first check immediate predators nearby; if present, try to escape
                neigh_preds = neighbor_info.get(CellType.PREDATOR, 0)
                if neigh_preds > 0:
                    escape_to = self.choose_escape_tile(pos)
                    if escape_to != pos and escape_to not in new_cells and escape_to not in to_remove:
                        cell.pos = escape_to
                        cell.draw_pos = np.array([escape_to[0], escape_to[1]], dtype=float)
                        # MODIFIED: Increased escape energy cost
                        cell.energy -= 2.5
                    else:
                        # if can't move away, try to hunt (eat plant) if adjacent
                        eaten = False
                        for n in self.get_neighbors(pos):
                            if n in self.cells and self.cells[n].type == CellType.PLANT:
                                target = self.cells[n]
                                energy_gain = min(target.energy * 0.9, 60)
                                cell.energy += energy_gain;
                                cell.energy = min(cell.energy, cell.max_energy)
                                to_remove.add(n)
                                current_plant_count -= 1  # Decrement if a plant is eaten
                                self.stats['total_deaths'] += 1
                                eaten = True
                                break
                        if not eaten:
                            cell.energy -= 0.8
                else:
                    # no predator immediate threat -> normal LSTM decision (if available)
                    if cell.brain is not None:
                        action_idx, probs = cell.high_level_action(neighbor_info)
                    else:
                        action_idx = random.randrange(4)

                    if action_idx == 0 and cell.energy > cell.reproduction_threshold:
                        empty = [n for n in self.get_neighbors(pos) if
                                 n not in new_cells and n not in to_remove]
                        if empty:
                            chosen = random.choice(empty)
                            child_brain = cell.brain.clone() if cell.brain else None
                            if child_brain:
                                # small mutation
                                for p in child_brain.parameters():
                                    if random.random() < 0.02:
                                        p.data += torch.randn_like(p.data) * 0.01
                            child = Cell(chosen, CellType.HERBIVORE, brain=child_brain)
                            # Mutation: Metabolism
                            mutation = random.uniform(-0.1, 0.1)
                            child.metabolism = max(0.5, min(2.0, cell.metabolism + mutation))
                            
                            child.generation = cell.generation + 1
                            to_add.append((chosen, child))
                            cell.energy *= 0.5
                            self.stats['total_births'] += 1

                    elif action_idx in (1, 3) and cell.energy > 10:
                        # move - prefer tiles with better plant scent and lower predator presence
                        candidates = [pos] + [n for n in self.get_neighbors(pos) if
                                              n not in new_cells and n not in to_remove]
                        best = pos;
                        best_score = -1e9
                        for cpos in candidates:
                            # score = plant_scent - predator_penalty
                            plant_val = self.plant_scent[cpos[0], cpos[1]]
                            pred_penalty = 0
                            for dx in range(-1, 2):
                                for dy in range(-1, 2):
                                    nx, ny = cpos[0] + dx, cpos[1] + dy
                                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                                        c = self.cells.get((nx, ny))
                                        if c and c.type == CellType.PREDATOR:
                                            pred_penalty += 1.5
                            score = plant_val - pred_penalty * 0.7 + random.random() * 0.02
                            if score > best_score:
                                best_score = score;
                                best = cpos
                        if best != pos and best not in self.cells:
                            cell.pos = best
                            cell.draw_pos = np.array([best[0], best[1]], dtype=float)
                        cell.energy -= 1.0

                    elif action_idx == 2:
                        # rest/photosynth
                        cell.energy += 6.0 * self.light * (1.0 - abs(self.temperature - 0.5))
                        cell.energy = min(cell.energy, cell.max_energy)

            # PLANT: simple growth/rest/reproduce (now with overcrowding & local light death)
            elif cell.type == CellType.PLANT:
                # MODIFIED: Reproduction is now exclusively based on high energy threshold AND GLOBAL CAP
                if current_plant_count < MAX_PLANTS_ALLOWED and cell.energy > cell.reproduction_threshold:
                    empty = [n for n in self.get_neighbors(pos) if n not in new_cells and n not in to_remove]
                    if empty:
                        newp = random.choice(empty)
                        newp = random.choice(empty)
                        child_plant = Cell(newp, CellType.PLANT)
                        # Mutation: Metabolism
                        mutation = random.uniform(-0.1, 0.1)
                        child_plant.metabolism = max(0.5, min(2.0, cell.metabolism + mutation))
                        
                        to_add.append((newp, child_plant))
                        cell.energy *= 0.5  # Energy cost of reproduction
                        self.stats['total_births'] += 1
                        current_plant_count += 1  # Increment counter

                # photosynth uses both global self.light and local self.light_map
                lx, ly = pos
                local_light = float(self.light_map[lx, ly]) * self.light  # combine spatial field and global intensity
                # MODIFIED: Reduced energy gain from 5.0 to 3.5
                cell.energy += 3.5 * local_light * (1.0 - abs(self.temperature - 0.5))
                cell.energy = min(cell.energy, cell.max_energy)

                # ---------- NEW: Overcrowding ----------
                nearby = 0
                for n in self.get_neighbors(pos):
                    c = self.cells.get(n)
                    if c and c.type == CellType.PLANT:
                        nearby += 1
                # include self in count region (optionally)
                if nearby > 5:
                    # overcrowding penalty
                    cell.energy -= 0.6  # stronger decay when overcrowded
                    # chance to die from overcrowding
                    if random.random() < 0.06:
                        to_remove.add(pos)
                        current_plant_count -= 1  # Decrement counter
                        self.stats['total_deaths'] += 1
                        continue

                # ---------- NEW: Low local light death ----------
                if local_light < 0.18:
                    # plants in very dark patches lose energy faster and may die
                    cell.energy -= 0.4
                    if cell.energy <= 0 or random.random() < 0.05:
                        to_remove.add(pos)
                        current_plant_count -= 1  # Decrement counter
                        self.stats['total_deaths'] += 1
                        continue

            # keep cell if not removed
            if pos not in to_remove:
                new_cells[pos] = cell

        # remove & add
        for r in to_remove:
            new_cells.pop(r, None)
            self.cells.pop(r, None)
        for pos, c in to_add:
            if pos not in new_cells:
                new_cells[pos] = c
        self.cells = new_cells

        # stats
        pop = len(self.cells)
        self.stats['max_population'] = max(self.stats['max_population'], pop)

        # occasional nutrients spawn
        if random.random() < 0.015:
            self.spawn_nutrients_field(random.randint(1, self.grid_width - 3), random.randint(1, self.grid_height - 3),
                                       3)

        # NEW: Attempt reintroduction of extinct species
        self.maybe_reintroduce_species()

        self.temperature = clamp(self.temperature + random.uniform(-0.02, 0.02), 0.0, 1.0)
        self.light = clamp(self.light + random.uniform(-0.03, 0.03), 0.3, 1.0)

        # maybe train hive
        self.maybe_train_hive()
        
        # REMOVED: Auto-trigger commentary logic. Now handled by manual button in handle_events.


    def species_diversity_score(self):
        counts = defaultdict(int)
        for c in self.cells.values(): counts[c.type] += 1
        total = sum(counts.values()) or 1
        freqs = [counts[t] / total for t in (CellType.PLANT, CellType.HERBIVORE, CellType.PREDATOR)]
        import math
        H = -sum([f * math.log(f + 1e-9) for f in freqs])
        maxH = math.log(3)
        return H / (maxH + 1e-9)

    # DRAW / UI
    def draw_grid(self):
        # draw textured background
        try:
            self.screen.blit(self.background_surface, (0, 0))
            # overlay small tint based on temperature/light to keep original mood
            tint = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            # compute tint color from temperature/light (warm when hot, cool when cold)
            r = int(10 + (self.temperature - 0.5) * 80)
            g = int(10 + (self.light - 0.5) * 40)
            b = int(10 + (0.5 - self.temperature) * 40)
            alpha = int(40 * (1.0 - (self.light - 0.3)))  # darker when light low
            tint.fill((clamp(r, 0, 255), clamp(g, 0, 255), clamp(b, 0, 255), clamp(alpha, 5, 120)))
            self.screen.blit(tint, (0, 0))
        except Exception:
            # fallback to solid fill if background not available
            r = int(20 + self.temperature * 70)
            g = int(20 + self.light * 60)
            b = int(30 + (1.0 - self.temperature) * 30)
            self.screen.fill((r, g, b))

        for pos, cell in list(self.cells.items()):
            tx, ty = cell.pos
            cell.draw_pos = cell.draw_pos * 0.75 + np.array([tx, ty]) * 0.25
            cx = int(cell.draw_pos[0] * self.tile_size + self.tile_size / 2)
            cy = int(cell.draw_pos[1] * self.tile_size + self.tile_size / 2)
            base = self.cell_colors[cell.type]
            energy_factor = clamp(cell.energy / (cell.max_energy or 1.0), 0.1, 1.0)
            color = tuple(int(clamp(c * (0.5 + 0.5 * energy_factor), 0, 255)) for c in base)
            glow = tuple(int(c * 0.25) for c in base)
            glow_r = int(self.tile_size * 0.6)
            pygame.draw.circle(self.screen, glow, (cx, cy), glow_r)
            radius = int(self.tile_size * 0.45)
            pygame.draw.circle(self.screen, color, (cx, cy), radius)
            if cell.age > 180:
                pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), 2)

        species_count = defaultdict(int)
        for c in self.cells.values(): species_count[c.type] += 1

        # faint scent overlay (debug)
        N = 40
        if self.plant_scent.size > 0:
            flat_idxs = np.argpartition(-self.plant_scent.flatten(), min(N, self.plant_scent.size) - 1)[:N]
            for idx in flat_idxs:
                x = idx // self.plant_scent.shape[1];
                y = idx % self.plant_scent.shape[1]
                val = self.plant_scent[x, y]
                if val <= 0.02: continue
                px = int((x + 0.5) * self.tile_size);
                py = int((y + 0.5) * self.tile_size)
                s = int(30 * val)
                surf = pygame.Surface((s * 2, s * 2), pygame.SRCALPHA)
                surf.fill((40, 200, 40, int(30 * val)))
                self.screen.blit(surf, (px - s, py - s))
        if self.herbivore_scent.size > 0:
            flat_idxs = np.argpartition(-self.herbivore_scent.flatten(), min(N, self.herbivore_scent.size) - 1)[:N]
            for idx in flat_idxs:
                x = idx // self.herbivore_scent.shape[1];
                y = idx % self.herbivore_scent.shape[1]
                val = self.herbivore_scent[x, y]
                if val <= 0.02: continue
                px = int((x + 0.5) * self.tile_size);
                py = int((y + 0.5) * self.tile_size)
                s = int(28 * val)
                surf = pygame.Surface((s * 2, s * 2), pygame.SRCALPHA)
                surf.fill((220, 60, 60, int(28 * val)))
                self.screen.blit(surf, (px - s, py - s))

        pygame.display.flip()

    def draw_dashboard(self):
        # Draw background for sidebar
        sidebar_rect = pygame.Rect(self.sim_width, 0, self.sidebar_width, self.height)
        pygame.draw.rect(self.screen, (30, 35, 45), sidebar_rect)
        pygame.draw.line(self.screen, (100, 100, 120), (self.sim_width, 0), (self.sim_width, self.height), 2)
        
        # Stats
        species_count = defaultdict(int)
        for c in self.cells.values(): species_count[c.type] += 1
        
        start_x = self.sim_width + 15
        y = 20
        
        def draw_text(txt, size=24, color=(220, 220, 220)):
            nonlocal y
            # Use default system font or fallback
            f = pygame.font.SysFont("Arial", size)
            s = f.render(txt, True, color)
            self.screen.blit(s, (start_x, y))
            y += size + 8
            
        draw_text("AI ECOSPHERE", 28, (255, 255, 255))
        y += 10
        # Stats
        infos = [
            f"Generation: {self.generation}",
            f"Population: {len(self.cells)}",
            f"Plants: {species_count[CellType.PLANT]}",
            f"Herbivores: {species_count[CellType.HERBIVORE]}",
            f"Predators: {species_count[CellType.PREDATOR]}",
            "",
            f"Temp: {self.temperature:.2f}",
            f"Light: {self.light:.2f}",
            f"Diversity: {len(set([c.type for c in self.cells.values()]))/3.0:.2f}"
        ]
        
        y = 60
        start_x = self.sim_width + 20
        
        for i, line in enumerate(infos):
            c = (200, 200, 200)
            if "Plants" in line: c = self.cell_colors[CellType.PLANT]
            if "Herbivores" in line: c = self.cell_colors[CellType.HERBIVORE]
            if "Predators" in line: c = self.cell_colors[CellType.PREDATOR]
            
            s = self.font.render(line, True, c)
            self.screen.blit(s, (start_x, y))
            y += 24

        # God Mode Input
        # Position it below stats
        input_y = y + 50
        self.input_rect.x = start_x
        self.input_rect.y = input_y
        self.input_rect.width = self.sidebar_width - 40
        self.input_rect.height = 32
        
        lbl = self.font.render("God Mode Input:", True, (255, 200, 50))
        self.screen.blit(lbl, (start_x, input_y - 25))
        
        pygame.draw.rect(self.screen, (20, 20, 30), self.input_rect)
        border = (100, 100, 200) if self.input_active else (80, 80, 80)
        pygame.draw.rect(self.screen, border, self.input_rect, 2)
        
        # Input Text
        disp = self.input_text
        while self.font.size(disp)[0] > self.input_rect.width - 20: 
            disp = disp[1:]
        
        in_s = self.font.render(disp, True, (255, 255, 255))
        self.screen.blit(in_s, (self.input_rect.x + 5, self.input_rect.y + 5))
        
        if not self.input_text and not self.input_active:
            ph = self.font.render("Type command...", True, (100, 100, 100))
            self.screen.blit(ph, (self.input_rect.x + 5, self.input_rect.y + 5))

        # Feedback Area
        if self.input_feedback:
             # Helper locally defined since we stripped the old one
             def draw_wrapped_text(text, font, color, max_width, x, start_y):
                paragraphs = text.replace('\r', '').split('\n')
                current_y = start_y
                for paragraph in paragraphs:
                    words = paragraph.split(' ')
                    lines = []
                    curr_line = []
                    for word in words:
                        test_words = curr_line + [word]
                        test_line = ' '.join(test_words)
                        if font.size(test_line)[0] < max_width:
                            curr_line.append(word)
                        else:
                            lines.append(' '.join(curr_line))
                            curr_line = [word]
                    if curr_line: lines.append(' '.join(curr_line))
                    for l in lines:
                        s = font.render(l, True, color)
                        self.screen.blit(s, (x, current_y))
                        current_y += 18
                return current_y

             fb_font = pygame.font.Font(None, 18)
             view_y = self.input_rect.y + 40
             view_h = self.height - view_y - 10
             view_rect = pygame.Rect(self.input_rect.x, view_y, self.sidebar_width - 10, view_h)
             
             old_clip = self.screen.get_clip()
             self.screen.set_clip(view_rect)
             
             final_y = draw_wrapped_text(self.input_feedback, fb_font, (150, 255, 150), 
                                         self.sidebar_width - 15, view_rect.x, view_rect.y - self.god_mode_scroll_y)
             
             self.god_mode_content_height = final_y - (view_rect.y - self.god_mode_scroll_y)
             self.screen.set_clip(old_clip)
             
             # Scrollbar
             if self.god_mode_content_height > view_h:
                  sb_h = max(20, int(view_h * (view_h / self.god_mode_content_height)))
                  sb_y = view_y + int((self.god_mode_scroll_y / self.god_mode_content_height) * view_h)
                  pygame.draw.rect(self.screen, (100, 100, 100), (self.width - 6, sb_y, 4, sb_h))

    def process_god_mode_queue(self):
        while not GOD_MODE_QUEUE.empty():
            cmd, arg1, arg2 = GOD_MODE_QUEUE.get()
            print(f"[GodMode] Processing: {cmd} {arg1} {arg2}")
            
            if cmd == "spawn":
                # arg1=species str, arg2=count
                s_map = {"plant": CellType.PLANT, "herbivore": CellType.HERBIVORE, "predator": CellType.PREDATOR}
                ctype = s_map.get(arg1.lower(), CellType.HERBIVORE)
                for _ in range(int(arg2)):
                    pos = (random.randint(0, self.grid_width-1), random.randint(0, self.grid_height-1))
                    if pos not in self.cells:
                        self.cells[pos] = Cell(pos, ctype)
            elif cmd == "env":
                # arg1=temp_delta, arg2=light_delta
                self.temperature = clamp(self.temperature + float(arg1), 0.0, 1.0)
                self.light = clamp(self.light + float(arg2), 0.2, 1.0)
            elif cmd == "cull":
                # arg1=species, arg2=pct
                s_map = {"plant": CellType.PLANT, "herbivore": CellType.HERBIVORE, "predator": CellType.PREDATOR}
                ctype = s_map.get(arg1.lower())
                if ctype is not None:
                    to_kill = []
                    for pos, c in self.cells.items():
                        if c.type == ctype and random.random() < float(arg2):
                            to_kill.append(pos)
                    for k in to_kill:
                        c = self.cells[k]
                        GLOBAL_EVENT_LOGGER.log(self.generation, "DEATH", {"id": c.id, "type": c.type, "cause": "GOD_CULL"})
                        del self.cells[k]
                    self.stats['total_deaths'] += len(to_kill)

    def run_god_mode_thread(self, text):
        res = self.god_mode_agent.process_command(text)
        self.input_feedback = res

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEWHEEL:
                # Check if mouse is in sidebar feedback area
                mx, my = pygame.mouse.get_pos()
                if mx > self.sim_width:
                    # scroll sensitivity
                    self.god_mode_scroll_y -= event.y * 20
                    # clamp
                    view_h = self.height - (self.input_rect.y + 40) - 10
                    max_scroll = max(0, self.god_mode_content_height - view_h)
                    self.god_mode_scroll_y = max(0, min(self.god_mode_scroll_y, max_scroll))
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.input_rect.collidepoint(event.pos):
                    self.input_active = True
                else:
                    self.input_active = False
                

            elif event.type == pygame.KEYDOWN:
                if self.input_active:
                    if event.key == pygame.K_RETURN:
                       if self.input_text:
                           self.input_feedback = "Processing..."
                           threading.Thread(target=self.run_god_mode_thread, args=(self.input_text,)).start()
                           self.input_text = ""
                    elif event.key == pygame.K_BACKSPACE:
                        self.input_text = self.input_text[:-1]
                    else:
                        self.input_text += event.unicode
                else:
                    # Normal hotkeys
                    if event.key == pygame.K_SPACE:
                        self.playing = not self.playing
                    elif event.key == pygame.K_c:
                        self.cells.clear();
                        if "GLOBAL_EVENT_LOGGER" in globals():
                             GLOBAL_EVENT_LOGGER.log(self.generation, "EXTINCTION", {"cause": "USER_CLEAR"})
                        self.playing = False;
                        self.generation = 0
                    elif event.key == pygame.K_f:
                        self.spawn_flower_pattern(random.randint(6, self.grid_width - 8),
                                                  random.randint(6, self.grid_height - 8))
                    elif event.key == pygame.K_o:
                        self.spawn_predator_swarm(random.randint(5, self.grid_width - 6),
                                                  random.randint(5, self.grid_height - 6))
                    elif event.key == pygame.K_n:
                        self.spawn_nutrients_field(random.randint(5, self.grid_width - 6),
                                                   random.randint(5, self.grid_height - 6), 4)
                    elif event.key == pygame.K_s:
                        self.spawn_spiral_galaxy(random.randint(8, self.grid_width - 10),
                                                 random.randint(8, self.grid_height - 10))
                    elif event.key == pygame.K_r:
                        self.generate_report()
                    elif event.key == pygame.K_t:
                        self.training_enabled = not getattr(self, 'training_enabled', True)
                        print("[Hive] Training toggled to", self.training_enabled)
                    elif event.key == pygame.K_e:
                        print("➡️ Scarcity event triggered (plants weakened).")
                        # simple scarcity: reduce plant energy and growth chance
                        for c in list(self.cells.values()):
                            if c.type == CellType.PLANT:
                                c.energy *= 0.25
                        self.plant_passive_grow_chance *= 0.2
                    elif event.key == pygame.K_k:
                        print("➡️ Killing all predators.")
                        self.exterminate_species(CellType.PREDATOR)
                    elif event.key == pygame.K_l:
                        print("➡️ Killing all herbivores.")
                        self.exterminate_species(CellType.HERBIVORE)
                    elif event.key == pygame.K_p:
                        print("➡️ Killing all plants.")
                        self.exterminate_species(CellType.PLANT)
                    elif event.key == pygame.K_q:
                        print("➡️ Quit requested.")
                        self.running = False

    def exterminate_species(self, species_type):
        species_names = {CellType.PLANT: "Plants", CellType.HERBIVORE: "Herbivores", CellType.PREDATOR: "Predators",
                         CellType.NUTRIENT: "Nutrients"}
        before = len(self.cells)
        self.cells = {pos: cell for pos, cell in self.cells.items() if cell.type != species_type}
        killed = before - len(self.cells)
        if killed > 0:
            print(f"EXTINCTION EVENT: Killed {killed} {species_names[species_type]}")
            self.stats['total_deaths'] += killed
            self.extinction_events.append((self.generation, f"Killed {killed} {species_names[species_type]}"))

    def generate_report(self):
        counts = defaultdict(int)
        for c in self.cells.values(): counts[c.type] += 1
        names = {0: "Plants", 1: "Herbivores", 2: "Predators", 3: "Nutrients"}
        dom = max([(counts[t], t) for t in (0, 1, 2)], default=(0, 0))[1]
        print("\n" + "=" * 60)
        print(f"Generation {self.generation} REPORT")
        print(f"Population: {len(self.cells)} | Dominant: {names.get(dom, 'None')}")
        print(f"Plants: {counts[0]} | Herbivores: {counts[1]} | Predators: {counts[2]}")
        print(f"Temperature: {self.temperature:.2f} | Light: {self.light:.2f}")
        print(f"Diversity score: {self.species_diversity_score():.3f}")
        print(f"Hive experiences stored: {len(self.hive_experiences)}")
        print("=" * 60 + "\n")

    def run(self):
        while self.running:
            self.clock.tick(self.fps)
            self.handle_events()
            if self.playing:
                self.count += 1
            if self.count >= self.update_freq:
                self.count = 0
                self.update_ecosystem()
                self.process_god_mode_queue() # Execute commands if any
                self.generation += 1
                if self.generation % 20 == 0:
                    self.generate_report()
            
            
            pygame.display.set_caption(f"AI Ecosphere - {'EVOLVING' if self.playing else 'PAUSED'} | Gen {self.generation}")
            
            # Draw
            self.draw_grid()      # Draws game board
            self.draw_dashboard() # Draws sidebar overlay
            pygame.display.flip() # Flip buffer

        self.generate_report()
        if hasattr(self, 'mlflow_run') and self.mlflow_run:
            mlflow.end_run()
        pygame.quit()


# Entrypoint
def main():
    print("""
AI ECOSPHERE — Predator Hive LSTM
Controls printed in terminal on startup.
""")
    sim = EcoLifeSimulation(800, 800, 16, 30)
    sim.run()


if __name__ == "__main__":
    main()