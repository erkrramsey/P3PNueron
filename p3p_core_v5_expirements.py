#!/usr/bin/env python3
# mesh_cortex_v6_all_experiments.py
#
# 3-node spiking mesh with:
# - logger / echo / mesh / cortex7 (LIF-like)
# - mesocortex_main (regions: AB, BC, Pattern, Osc, Memory, Cart, Flock, Seq, Anomaly)
# - metrics aggregator
# - snapshot persist / load
# - scheduler warmup_demo
# - control_port (JSON over TCP) for all experiments
#
# Experiments (all wired):
#   - experiment_pattern_classify
#   - experiment_rhythm_gen
#   - experiment_hopfield_assoc
#   - experiment_cart_balance
#   - experiment_boid_swarm
#   - experiment_seq_gen
#   - experiment_seizure_detect

import os
import json
import time
import math
import random
import socket
import threading
from collections import defaultdict, deque
from typing import Dict, Any, List, Tuple, Optional

# -----------------------
# GLOBAL CONFIG / SEEDING
# -----------------------

GLOBAL_SEED = 1337
random.seed(GLOBAL_SEED)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

METRICS_PATH = os.path.join(BASE_DIR, "metrics_dump_core.json")

CONFIG = {
    "cluster_secret": "super_secret_cluster_key_42",
    "nodes": {
        "nodeA": {"mesh_port": 10001},
        "nodeB": {"mesh_port": 10002},
        "nodeC": {"mesh_port": 10003},
    },
    "control_port": 10080,
    "cortex": {
        "n_neurons": 4,
        "v_rest": 0.0,
        "v_reset": 0.0,
        "thr_base": [1.0, 1.2, 1.3, 1.5],
        "decay": 0.9,
        "stdp_lr": 0.02,
    },
}

# Region wiring (conceptual mapping)
REGION_LAYOUT = {
    "Region_AB": {
        "columns": {
            "col_A1": ("nodeA", 0),
            "col_B1": ("nodeB", 0),
        }
    },
    "Region_BC": {
        "columns": {
            "col_B2": ("nodeB", 0),
            "col_C1": ("nodeC", 0),
        }
    },
    # Additional regions primarily use nodeA's cortex but are kept explicit for extension
    "Region_Pattern": {
        "columns": {
            "pat_in_0": ("nodeA", 0),
            "pat_in_1": ("nodeA", 1),
        }
    },
    "Region_Osc": {
        "columns": {
            "osc_exc": ("nodeA", 0),
            "osc_inh": ("nodeA", 1),
        }
    },
    "Region_Memory": {
        "columns": {
            "mem_0": ("nodeA", 0),
            "mem_1": ("nodeA", 1),
            "mem_2": ("nodeA", 2),
            "mem_3": ("nodeA", 3),
        }
    },
    "Region_Cart": {
        "columns": {
            "cart_s0": ("nodeA", 0),
            "cart_s1": ("nodeA", 1),
            "cart_s2": ("nodeA", 2),
            "cart_s3": ("nodeA", 3),
        }
    },
    "Region_Flock": {
        "columns": {
            "flock_pos": ("nodeA", 0),
            "flock_vel": ("nodeB", 0),
            "flock_align": ("nodeC", 0),
        }
    },
    "Region_Seq": {
        "columns": {
            "seq_in": ("nodeA", 0),
            "seq_hidden": ("nodeA", 1),
            "seq_out": ("nodeA", 2),
        }
    },
    "Region_Anomaly": {
        "columns": {
            "an_norm": ("nodeA", 0),
            "an_detect": ("nodeA", 1),
            "an_alert": ("nodeA", 2),
        }
    },
}

# --------------
# LOGGING HELPER
# --------------

def log(node_name: str, module: str, payload: Dict[str, Any]) -> None:
    prefix = f"[{node_name}::{module}]"
    print(f"{prefix} {payload}", flush=True)


# --------------
# CORTEX / STDP
# --------------

class CortexNeuron:
    def __init__(self, thr: float, v_rest: float, v_reset: float, decay: float):
        self.v = v_rest
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.thr = thr
        self.last_spike_t = -10**9
        self.spikes: List[int] = []
        self.decay = decay


class Cortex7:
    def __init__(self, node_name: str, n_neurons: int, cfg: Dict[str, Any]):
        self.node_name = node_name
        self.n_neurons = n_neurons
        self.neurons: List[CortexNeuron] = []
        thr_base = cfg["thr_base"]
        for i in range(n_neurons):
            self.neurons.append(
                CortexNeuron(
                    thr=thr_base[min(i, len(thr_base) - 1)],
                    v_rest=cfg["v_rest"],
                    v_reset=cfg["v_reset"],
                    decay=cfg["decay"],
                )
            )
        self.t = 0
        # Simple fully connected excitatory synapses w_ij (i -> j)
        self.weights = [[0.0 for _ in range(n_neurons)] for _ in range(n_neurons)]
        for i in range(n_neurons):
            for j in range(n_neurons):
                if i != j:
                    self.weights[i][j] = 0.1 * random.random()
        self.stdp_lr = cfg["stdp_lr"]

    def inject_current(self, idx: int, amount: float) -> None:
        if 0 <= idx < self.n_neurons:
            self.neurons[idx].v += amount

    def advance(self, ticks: int, metrics: "MetricsModule", region: Optional[str] = None):
        for _ in range(ticks):
            self._step(metrics, region)

    def _step(self, metrics: "MetricsModule", region: Optional[str]):
        self.t += 1
        # decay
        for n in self.neurons:
            n.v = n.v_rest + (n.v - n.v_rest) * n.decay

        # synaptic input
        for i, pre in enumerate(self.neurons):
            if self.t - pre.last_spike_t == 1:
                for j, post in enumerate(self.neurons):
                    if i != j:
                        self.neurons[j].v += self.weights[i][j]

        # spikes
        spike_mask = [False] * self.n_neurons
        for i, n in enumerate(self.neurons):
            if n.v >= n.thr:
                n.spikes.append(self.t)
                n.last_spike_t = self.t
                n.v = n.v_reset
                spike_mask[i] = True

        # STDP (very simple)
        for pre_idx, pre in enumerate(self.neurons):
            if spike_mask[pre_idx]:
                for post_idx, post in enumerate(self.neurons):
                    dt = post.last_spike_t - pre.last_spike_t
                    if dt > 0 and dt <= 5:
                        # LTP
                        self.weights[pre_idx][post_idx] += self.stdp_lr * math.exp(-dt / 5.0)
                    elif dt < 0 and dt >= -5:
                        # LTD
                        self.weights[pre_idx][post_idx] -= self.stdp_lr * math.exp(dt / 5.0)

        # Metrics
        for i, fired in enumerate(spike_mask):
            if fired:
                metrics.record_spike(self.node_name, "cortex7", i, self.t, region)

    def dump_state(self) -> Dict[str, Any]:
        return {
            "t": self.t,
            "neurons": {
                i: {
                    "v": n.v,
                    "thr": n.thr,
                    "last_spike_t": n.last_spike_t,
                    "spikes": list(n.spikes),
                }
                for i, n in enumerate(self.neurons)
            },
            "weights": self.weights,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.t = state["t"]
        for i, n_state in state["neurons"].items():
            i = int(i)
            n = self.neurons[i]
            n.v = n_state["v"]
            n.thr = n_state["thr"]
            n.last_spike_t = n_state["last_spike_t"]
            n.spikes = list(n_state["spikes"])
        if "weights" in state:
            self.weights = state["weights"]


# --------------
# METRICS MODULE
# --------------

class MetricsModule:
    def __init__(self):
        self.reset()

    def reset(self):
        self.spike_events: List[Tuple[str, str, int, int, Optional[str]]] = []
        self.last_t: Dict[str, int] = {}
        self.extra: Dict[str, Any] = {}

    def record_spike(
        self, node: str, module: str, neuron_idx: int, t: int, region: Optional[str]
    ):
        self.spike_events.append((node, module, neuron_idx, t, region))
        self.last_t[node] = t

    def attach_extra(self, key: str, value: Any):
        self.extra[key] = value

    def summarize(self) -> Dict[str, Any]:
        total_spikes = len(self.spike_events)
        spikes_per_node: Dict[str, int] = defaultdict(int)
        spikes_per_region: Dict[str, int] = defaultdict(int)
        top_counts: Dict[Tuple[str, str, int], int] = defaultdict(int)

        for node, module, neuron_idx, t, region in self.spike_events:
            spikes_per_node[node] += 1
            if region:
                spikes_per_region[region] += 1
            top_counts[(node, module, neuron_idx)] += 1

        top_sorted = sorted(
            top_counts.items(), key=lambda kv: kv[1], reverse=True
        )

        last_t = dict(self.last_t)

        summary = {
            "total_spikes": total_spikes,
            "spikes_per_node": dict(spikes_per_node),
            "spikes_per_region": dict(spikes_per_region),
            "top_neurons": [
                {"node": n, "module": m, "neuron": i, "count": c}
                for (n, m, i), c in top_sorted
            ],
            "last_t_per_node": last_t,
            "extra": self.extra,
        }
        return summary

    def write_summary(self, path: str = METRICS_PATH):
        summary = self.summarize()
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

        # Pretty console dump
        print("\n==== METRICS SUMMARY ====")
        print(f"Total spikes: {summary['total_spikes']}\n")
        print("Spikes per node:")
        for n, c in summary["spikes_per_node"].items():
            print(f"  {n}: {c}")
        print("\nSpikes per region:")
        for r, c in summary["spikes_per_region"].items():
            print(f"  {r}: {c}")
        print("\nTop (node, module, neuron) spike counts:")
        for item in summary["top_neurons"][:10]:
            print(
                f"  {item['node']}.{item['module']}[{item['neuron']}] = {item['count']} spikes"
            )
        print("\nLast t per node:")
        for n, t in summary["last_t_per_node"].items():
            print(f"  {n}: t={t}")
        print("==========================\n")
        print(f"[cluster] metrics written to {path}")


# --------------
# SNAPSHOT / PERSIST
# --------------

def snapshot_path(node: str, ts: Optional[int] = None) -> str:
    if ts is None:
        ts = int(time.time())
    return os.path.join(SNAPSHOT_DIR, f"snapshot_{node}_{ts}.json")


def write_snapshot(node: str, cortex_state: Dict[str, Any]) -> str:
    path = snapshot_path(node)
    payload = {"node": node, "cortex7": cortex_state}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    log(node, "persist", {"persist": "snapshot_written", "path": path})
    return path


def find_latest_snapshot(node: str) -> Optional[str]:
    latest = None
    latest_ts = -1
    for fname in os.listdir(SNAPSHOT_DIR):
        if not fname.startswith(f"snapshot_{node}_"):
            continue
        try:
            ts = int(fname.split("_")[-1].split(".")[0])
        except Exception:
            continue
        if ts > latest_ts:
            latest_ts = ts
            latest = os.path.join(SNAPSHOT_DIR, fname)
    return latest


def load_snapshot(node: str, cortex: Cortex7, path: str = "latest") -> Optional[str]:
    if path == "latest":
        p = find_latest_snapshot(node)
        if p is None:
            return None
    else:
        p = path
    with open(p, "r") as f:
        payload = json.load(f)
    cortex_state = payload["cortex7"]
    cortex.load_state(cortex_state)
    log(node, "persist", {"persist": "load_success", "path": p})
    return p


# --------------
# MESH (SIMPLIFIED)
# --------------

class MeshRegistry:
    def __init__(self):
        self.nodes: Dict[str, "Node"] = {}

    def register(self, node_name: str, node: "Node"):
        self.nodes[node_name] = node

    def send_spike(
        self,
        src_node: str,
        dst_node: str,
        dst_neuron_idx: int,
        amount: float,
        region: Optional[str],
        metrics: MetricsModule,
    ):
        if dst_node not in self.nodes:
            return
        node = self.nodes[dst_node]
        node.cortex.inject_current(dst_neuron_idx, amount)
        # we rely on node-specific mesocortex / control to advance cortex;
        # remote spike identity flows via metrics when spike occurs
        # (we tag region when advancing)


# --------------
# MESOCORTEX CONTROLLER
# --------------

class MesocortexMain:
    def __init__(
        self,
        node_name: str,
        mesh: MeshRegistry,
        metrics: MetricsModule,
        nodes: Dict[str, "Node"],
    ):
        self.node_name = node_name
        self.mesh = mesh
        self.metrics = metrics
        self.nodes = nodes
        self.regions = list(REGION_LAYOUT.keys())
        log(
            node_name,
            "mesocortex_main",
            {"mesocortex": "online", "regions": self.regions},
        )

    def stimulate_region(
        self,
        region: str,
        pattern: Dict[str, float],
        ticks: int = 1,
    ):
        layout = REGION_LAYOUT.get(region)
        if not layout:
            return

        cols = layout["columns"]
        # inject currents
        for col_name, strength in pattern.items():
            if col_name not in cols:
                continue
            node_name, neuron_idx = cols[col_name]
            node = self.nodes[node_name]
            node.cortex.inject_current(neuron_idx, strength)

        log(
            self.node_name,
            "mesocortex_main",
            {
                "mesocortex": "stimulate_region",
                "region": region,
                "columns": list(pattern.keys()),
                "pattern_len": len(pattern),
            },
        )

        # advance all nodes' cortex for a few ticks under this region
        for _ in range(ticks):
            for node_name, node in self.nodes.items():
                node.cortex.advance(1, self.metrics, region=region)

    def advance_all(self, ticks: int, region: Optional[str] = None):
        for _ in range(ticks):
            for node_name, node in self.nodes.items():
                node.cortex.advance(1, self.metrics, region=region)
        log(
            self.node_name,
            "mesocortex_main",
            {"mesocortex": "advance_all", "ticks": ticks},
        )

    def dump_all(self):
        # only snapshot nodeA for now (same pattern you had)
        nodeA = self.nodes["nodeA"]
        stateA = nodeA.cortex.dump_state()
        path = write_snapshot("nodeA", stateA)
        log(self.node_name, "mesocortex_main", {"mesocortex": "dump_all"})
        return path


# --------------
# NODE CONTAINER
# --------------

class Node:
    def __init__(
        self,
        name: str,
        mesh_registry: MeshRegistry,
        metrics: MetricsModule,
        cortex_cfg: Dict[str, Any],
    ):
        self.name = name
        self.metrics = metrics
        self.mesh_registry = mesh_registry

        # logger
        log(name, "logger", {"logger": "online"})

        # echo
        log(name, "echo", {"echo": "online"})

        # cortex7
        self.cortex = Cortex7(name, cortex_cfg["n_neurons"], cortex_cfg)
        log(
            name,
            "cortex7",
            {"cortex7": "online", "neurons": cortex_cfg["n_neurons"]},
        )

        # mesh (logical)
        self.mesh_port = CONFIG["nodes"][name]["mesh_port"]
        log(
            name,
            "mesh",
            {"mesh": "online", "addr": f"127.0.0.1:{self.mesh_port}"},
        )

    # simple echo call for sanity pings
    def echo_ping(self, from_node: str, from_module: str):
        payload = {
            "echo_ping": {
                "hello": f"from {from_node}",
                "_remote_from_node": from_node,
                "_remote_from_module": from_module,
            }
        }
        log(self.name, "echo", payload)


# --------------
# CONTROL MODULE
# --------------

class ControlModule:
    def __init__(
        self,
        node_name: str,
        nodes: Dict[str, Node],
        mesocortex: MesocortexMain,
        metrics: MetricsModule,
    ):
        self.node_name = node_name
        self.nodes = nodes
        self.mesocortex = mesocortex
        self.metrics = metrics
        log(node_name, "control", {"control": "online"})

    # ---- High-level demo used by scheduler ----

    def run_demo(self):
        log(
            self.node_name,
            "control",
            {
                "control": "run_demo_begin",
                "regions": ["Region_AB", "Region_BC"],
                "stim_cycles": 5,
                "advance_ticks": 5,
            },
        )
        for _ in range(5):
            # AB
            self.mesocortex.stimulate_region(
                "Region_AB", {"col_A1": 0.8, "col_B1": 0.6}, ticks=1
            )
            # BC
            self.mesocortex.stimulate_region(
                "Region_BC", {"col_B2": 0.9, "col_C1": 0.7}, ticks=1
            )

        self.mesocortex.advance_all(5, region=None)
        path = self.mesocortex.dump_all()
        log(self.node_name, "control", {"control": "run_demo_complete"})
        log(
            self.node_name,
            "control",
            {"control": "snapshot_requested", "args": {"reason": "after_run_demo"}},
        )
        return path

    def status(self):
        modules = [
            "logger",
            "echo",
            "cortex7",
            "mesh",
            "metrics",
            "control",
            "persist",
            "scheduler",
            "control_port",
            "mesocortex_main",
        ]
        log(
            self.node_name,
            "control",
            {"control": "status", "node": self.node_name, "modules": modules},
        )

    def load_snapshot_latest(self):
        log(
            self.node_name,
            "control",
            {"control": "load_snapshot_requested", "args": {"path": "latest"}},
        )
        p = load_snapshot(self.node_name, self.nodes[self.node_name].cortex, "latest")
        return p

    # ---- Experiment Drivers (ALL) ----

    # 1. Pattern Replay Classifier
    def experiment_pattern_classify(self, epochs: int = 50):
        random.seed(GLOBAL_SEED + 1)
        self.metrics.reset()
        correct = 0
        total = 0

        # Two binary patterns (0 and 1) over 2 columns
        patterns = {
            0: {"pat_in_0": 0.9, "pat_in_1": 0.1},
            1: {"pat_in_0": 0.1, "pat_in_1": 0.9},
        }

        for epoch in range(epochs):
            for label in [0, 1]:
                pattern = patterns[label]
                noise0 = (random.random() - 0.5) * 0.1
                noise1 = (random.random() - 0.5) * 0.1
                p = {
                    "pat_in_0": max(0.0, pattern["pat_in_0"] + noise0),
                    "pat_in_1": max(0.0, pattern["pat_in_1"] + noise1),
                }
                self.mesocortex.stimulate_region(
                    "Region_Pattern", p, ticks=5
                )
                # decode: take nodeA.cortex spikes difference between neurons 0 and 1
                nodeA = self.nodes["nodeA"].cortex
                sp0 = len(nodeA.neurons[0].spikes)
                sp1 = len(nodeA.neurons[1].spikes)
                pred = 0 if sp0 >= sp1 else 1
                if pred == label:
                    correct += 1
                total += 1

        acc = correct / max(1, total)
        self.metrics.attach_extra("pattern_classify_acc", acc)
        return acc

    # 2. Oscillatory Rhythm Generator
    def experiment_rhythm_gen(self, epochs: int = 100, ticks_per_epoch: int = 20):
        random.seed(GLOBAL_SEED + 2)
        self.metrics.reset()

        nodeA = self.nodes["nodeA"].cortex

        # Kick the oscillator once at start
        nodeA.inject_current(0, 1.2)
        for epoch in range(epochs):
            # every 10 epochs, small extra kick
            if epoch % 10 == 0:
                nodeA.inject_current(0, 0.5)
            self.mesocortex.advance_all(ticks_per_epoch, region="Region_Osc")

        # compute simple "frequency" as spikes / total_time for neuron 0
        t_total = nodeA.t
        spike_count = len(nodeA.neurons[0].spikes)
        freq = spike_count / max(1, t_total)
        self.metrics.attach_extra("osc_freq_nodeA_n0", freq)
        return freq

    # 3. Associative Memory (Hopfield-like)
    def experiment_hopfield_assoc(self, epochs: int = 30):
        random.seed(GLOBAL_SEED + 3)
        self.metrics.reset()
        nodeA = self.nodes["nodeA"].cortex

        # patterns over 4 neurons (Region_Memory)
        patterns = [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
        ]

        def stim_pattern(vec, noise: float = 0.0):
            cols = REGION_LAYOUT["Region_Memory"]["columns"]
            pattern = {}
            for i, bit in enumerate(vec):
                name = f"mem_{i}"
                base = 1.0 if bit == 1 else 0.2
                base += (random.random() - 0.5) * noise
                base = max(0.0, base)
                pattern[name] = base
            self.mesocortex.stimulate_region("Region_Memory", pattern, ticks=5)

        # training: just repeatedly stimulate clean patterns
        for ep in range(epochs):
            for p in patterns:
                stim_pattern(p, noise=0.0)

        # testing: noisy versions
        correct = 0
        total = 0
        for p in patterns:
            noisy = [(bit if random.random() > 0.2 else 1 - bit) for bit in p]
            stim_pattern(noisy, noise=0.05)
            # decode by neuron spike counts
            counts = [len(nodeA.neurons[i].spikes) for i in range(4)]
            # reconstruct by threshold
            recon = [1 if c > (sum(counts) / (4 * 0.8) if sum(counts) > 0 else 0.5) else 0 for c in counts]
            if recon == p:
                correct += 1
            total += 1

        acc = correct / max(1, total)
        self.metrics.attach_extra("hopfield_recall_acc", acc)
        return acc

    # 4. Spiking CartPole-like agent (very simplified)
    def experiment_cart_balance(self, episodes: int = 50, steps_per_episode: int = 40):
        random.seed(GLOBAL_SEED + 4)
        self.metrics.reset()
        nodeA = self.nodes["nodeA"].cortex

        def reset_env():
            return [0.0, 0.0, random.uniform(-0.05, 0.05), 0.0]

        def step_env(state, action):
            # toy dynamics: keep pole angle small
            x, v, theta, w = state
            force = -1.0 if action == 0 else 1.0
            theta_dot = w + 0.05 * force
            theta_new = theta + theta_dot
            x_new = x + 0.1 * force
            v_new = v + 0.01 * force
            done = abs(theta_new) > 0.5 or abs(x_new) > 2.4
            reward = 1.0 if not done else 0.0
            return [x_new, v_new, theta_new, theta_dot], reward, done

        def encode_state(state):
            # map to injections on 4 mem neurons
            pattern = {}
            cols = REGION_LAYOUT["Region_Cart"]["columns"]
            for i, s in enumerate(state):
                name = f"cart_s{i}"
                val = (s + 2.4) / 4.8 if i == 0 else (s + 1.0) / 2.0
                val = max(0.0, min(1.0, abs(val)))
                pattern[name] = 0.2 + 0.8 * val
            return pattern

        total_reward = 0.0

        for ep in range(episodes):
            state = reset_env()
            for t in range(steps_per_episode):
                pattern = encode_state(state)
                self.mesocortex.stimulate_region("Region_Cart", pattern, ticks=3)
                # choose action by comparing neuron 0 vs 1 spikes
                sp0 = len(nodeA.neurons[0].spikes)
                sp1 = len(nodeA.neurons[1].spikes)
                action = 0 if sp0 >= sp1 else 1
                state, r, done = step_env(state, action)
                total_reward += r
                if done:
                    break

        avg_reward = total_reward / max(1, episodes)
        self.metrics.attach_extra("cart_avg_reward", avg_reward)
        return avg_reward

    # 5. Flocking / Swarm
    def experiment_boid_swarm(self, epochs: int = 50, n_boids: int = 8):
        random.seed(GLOBAL_SEED + 5)
        self.metrics.reset()

        # simple 1D positions for boids
        positions = [random.random() for _ in range(n_boids)]

        def step_boids(pos):
            new_pos = []
            mean = sum(pos) / len(pos)
            for p in pos:
                # move slightly toward mean
                p_new = p + 0.1 * (mean - p) + (random.random() - 0.5) * 0.02
                new_pos.append(p_new)
            return new_pos

        for ep in range(epochs):
            # encode global mean position into Region_Flock
            mean = sum(positions) / len(positions)
            pattern = {
                "flock_pos": 0.2 + 0.8 * max(0.0, min(1.0, mean)),
                "flock_vel": 0.5,
                "flock_align": 0.5,
            }
            self.mesocortex.stimulate_region("Region_Flock", pattern, ticks=3)
            positions = step_boids(positions)

        mean = sum(positions) / len(positions)
        var = sum((p - mean) ** 2 for p in positions) / len(positions)
        self.metrics.attach_extra("flock_pos_variance", var)
        return var

    # 6. Sequence generator (binary pattern)
    def experiment_seq_gen(self, epochs: int = 30, seq_len: int = 20):
        random.seed(GLOBAL_SEED + 6)
        self.metrics.reset()
        nodeA = self.nodes["nodeA"].cortex

        # base sequence 010101...
        base_seq = [i % 2 for i in range(seq_len)]

        def stim_bit(bit):
            pattern = {
                "seq_in": 0.9 if bit == 1 else 0.1,
                "seq_hidden": 0.5,
                "seq_out": 0.3,
            }
            self.mesocortex.stimulate_region("Region_Seq", pattern, ticks=3)

        # training
        for ep in range(epochs):
            for b in base_seq:
                stim_bit(b)

        # generation: feed first bit, let spikes drive others (simplified)
        pred_seq = []
        for i, b in enumerate(base_seq):
            stim_bit(b)
            sp0 = len(nodeA.neurons[0].spikes)
            sp2 = len(nodeA.neurons[2].spikes)
            pred = 1 if sp2 >= sp0 else 0
            pred_seq.append(pred)

        mismatches = sum(int(a != b) for a, b in zip(base_seq, pred_seq))
        acc = 1.0 - mismatches / max(1, len(base_seq))
        self.metrics.attach_extra("seq_acc", acc)
        return acc

    # 7. Anomaly / seizure detection
    def experiment_seizure_detect(self, epochs: int = 40, burst_every: int = 7):
        random.seed(GLOBAL_SEED + 7)
        self.metrics.reset()
        nodeA = self.nodes["nodeA"].cortex

        true_pos = 0
        false_pos = 0
        total_bursts = 0
        total_norm = 0

        for ep in range(epochs):
            is_burst = (ep % burst_every == 0)
            pattern = {}
            if is_burst:
                # strong bursts
                pattern = {
                    "an_norm": 0.2,
                    "an_detect": 1.0,
                    "an_alert": 1.0,
                }
                total_bursts += 1
            else:
                pattern = {
                    "an_norm": 0.5,
                    "an_detect": 0.1,
                    "an_alert": 0.1,
                }
                total_norm += 1

            self.mesocortex.stimulate_region("Region_Anomaly", pattern, ticks=5)

            # decode: many spikes on alert neuron => anomaly
            sp_alert = len(nodeA.neurons[2].spikes)
            threshold = 3
            alert = sp_alert >= threshold
            if alert and is_burst:
                true_pos += 1
            elif alert and not is_burst:
                false_pos += 1

        tpr = true_pos / max(1, total_bursts)
        fpr = false_pos / max(1, total_norm)
        self.metrics.attach_extra(
            "anomaly_tpr_fpr",
            {"tpr": tpr, "fpr": fpr, "true_pos": true_pos, "false_pos": false_pos},
        )
        return tpr, fpr


# --------------
# SCHEDULER (SIMPLE)
# --------------

class Scheduler:
    def __init__(self, node_name: str, control: ControlModule):
        self.node_name = node_name
        self.control = control
        log(self.node_name, "scheduler", {"scheduler": "online"})

    def run_warmup(self):
        job_id = "warmup_demo"
        log(
            self.node_name,
            "scheduler",
            {
                "scheduler": "job_starting",
                "job_id": job_id,
                "type": "control_run_demo",
                "repeat": 1,
            },
        )
        self.control.run_demo()
        log(
            self.node_name,
            "scheduler",
            {
                "scheduler": "job_finished",
                "job_id": job_id,
                "type": "control_run_demo",
            },
        )
        log(self.node_name, "scheduler", {"scheduler": "all_jobs_complete"})


# --------------
# CONTROL PORT (TCP JSON)
# --------------

class ControlPortServer(threading.Thread):
    def __init__(self, node_name: str, port: int, control: ControlModule, metrics: MetricsModule):
        super().__init__(daemon=True)
        self.node_name = node_name
        self.port = port
        self.control = control
        self.metrics = metrics
        self._stop = threading.Event()
        log(
            node_name,
            "control_port",
            {"control_port": "online", "addr": f"127.0.0.1:{port}"},
        )

    def run(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", self.port))
        s.listen(5)
        while not self._stop.is_set():
            try:
                s.settimeout(0.5)
                conn, addr = s.accept()
            except socket.timeout:
                continue
            with conn:
                data = b""
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                if not data:
                    continue
                try:
                    cmd = json.loads(data.decode("utf-8").strip())
                    self.handle_cmd(cmd)
                except Exception as e:
                    log(self.node_name, "control_port", {"error": str(e)})
        s.close()

    def stop(self):
        self._stop.set()

    def handle_cmd(self, cmd: Dict[str, Any]):
        name = cmd.get("cmd")
        args = cmd.get("args", {})
        if not name:
            return
        if name == "status":
            self.control.status()
            return
        if name == "load_snapshot_latest":
            self.control.load_snapshot_latest()
            return

        # experiment routing
        if name == "experiment_pattern_classify":
            acc = self.control.experiment_pattern_classify(
                epochs=int(args.get("epochs", 50))
            )
            log(
                self.node_name,
                "control_port",
                {"experiment": name, "acc": acc},
            )
        elif name == "experiment_rhythm_gen":
            freq = self.control.experiment_rhythm_gen(
                epochs=int(args.get("epochs", 100)),
                ticks_per_epoch=int(args.get("ticks_per_epoch", 20)),
            )
            log(
                self.node_name,
                "control_port",
                {"experiment": name, "freq": freq},
            )
        elif name == "experiment_hopfield_assoc":
            acc = self.control.experiment_hopfield_assoc(
                epochs=int(args.get("epochs", 30))
            )
            log(
                self.node_name,
                "control_port",
                {"experiment": name, "acc": acc},
            )
        elif name == "experiment_cart_balance":
            avg_reward = self.control.experiment_cart_balance(
                episodes=int(args.get("episodes", 50)),
                steps_per_episode=int(args.get("steps_per_episode", 40)),
            )
            log(
                self.node_name,
                "control_port",
                {"experiment": name, "avg_reward": avg_reward},
            )
        elif name == "experiment_boid_swarm":
            var = self.control.experiment_boid_swarm(
                epochs=int(args.get("epochs", 50)),
                n_boids=int(args.get("n_boids", 8)),
            )
            log(
                self.node_name,
                "control_port",
                {"experiment": name, "variance": var},
            )
        elif name == "experiment_seq_gen":
            acc = self.control.experiment_seq_gen(
                epochs=int(args.get("epochs", 30)),
                seq_len=int(args.get("seq_len", 20)),
            )
            log(
                self.node_name,
                "control_port",
                {"experiment": name, "acc": acc},
            )
        elif name == "experiment_seizure_detect":
            tpr, fpr = self.control.experiment_seizure_detect(
                epochs=int(args.get("epochs", 40)),
                burst_every=int(args.get("burst_every", 7)),
            )
            log(
                self.node_name,
                "control_port",
                {"experiment": name, "tpr": tpr, "fpr": fpr},
            )
        else:
            log(
                self.node_name,
                "control_port",
                {"unknown_cmd": name},
            )
        # after each experiment, dump metrics summary
        self.metrics.write_summary()


# --------------
# CLUSTER BOOT / MAIN
# --------------

def run_cluster():
    # shared metrics + mesh registry
    metrics = MetricsModule()
    mesh_registry = MeshRegistry()

    # nodes
    nodes: Dict[str, Node] = {}
    for n in ["nodeA", "nodeB", "nodeC"]:
        nodes[n] = Node(n, mesh_registry, metrics, CONFIG["cortex"])
        mesh_registry.register(n, nodes[n])

    # mesocortex lives on nodeA (control plane)
    mesocortex = MesocortexMain("nodeA", mesh_registry, metrics, nodes)

    # control + scheduler on nodeA
    control = ControlModule("nodeA", nodes, mesocortex, metrics)
    scheduler = Scheduler("nodeA", control)

    # control port (TCP) on nodeA
    cp = ControlPortServer("nodeA", CONFIG["control_port"], control, metrics)
    cp.start()

    # warmup sanity
    # Sanity ping nodeA -> nodeB.echo
    nodes["nodeB"].echo_ping("nodeA", "testerA")
    # warmup demo from scheduler
    scheduler.run_warmup()
    # basic status
    control.status()

    # load latest snapshot back into nodeA cortex (if exists)
    control.load_snapshot_latest()

    # write basic metrics so there's always a core summary
    metrics.write_summary()

    # keep process alive so control_port can be used
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        cp.stop()


if __name__ == "__main__":
    run_cluster()
