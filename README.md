P3P Neuron — Distributed Mesocortex Simulation (Core v5)

Overview

P3P Neuron Core v5 is a fully-functional, distributed neural-mesh simulation engine built on top of a multi-node asynchronous runtime.
It combines:

Secure mesh networking (HMAC-signed envelopes)

Distributed spiking neural cores (Cortex-7)

Mesocortex orchestrator (region-based co-activation across nodes)

STDP plasticity with dopamine/reward modulation

Cluster-wide metrics aggregation

Snapshot + replay system

External control plane (JSON/TCP)

Experiment suite (7 cognitive modules)


This system behaves like a tiny distributed brain, spanning three cooperating nodes connected by an authenticated mesh fabric.
It can learn, spike, oscillate, classify patterns, generate sequences, detect anomalies, and respond to external reinforcement signals.


---

Architecture

The project creates three simulated nodes, each with modular subsystems:

nodeA ── cortex7, mesocortex_main, metrics, control-plane,
           scheduler, persistence, control_port, mesh
 nodeB ── cortex7, mesh, logger, echo
 nodeC ── cortex7, mesh, logger, echo

All nodes communicate via HMAC-authenticated TCP envelopes.

Each node hosts multiple modules:

Core Modules

Cortex7Module — spiking neural network with STDP + dopamine

MesocortexModule — orchestrates multi-node regions

MetricsModule — collects spikes, region activity, snapshots

PersistenceModule — JSON snapshots and replay

JobSchedulerModule — automatic demo runner

ControlPlaneModule — dispatches system/experiment commands

ControlPortModule — external TCP JSON-line API

NetworkMeshModule — HMAC-protected inter-node network

LoggerModule — structured logging

EchoModule — connectivity tests

NodeRuntime — scheduling, dispatch, messaging



---

Cortex-7: Spiking Neural Core

Each node contains Cortex7Core, a 4-neuron leaky-integrate-and-fire model with:

membrane voltage

thresholds

leak + baseline + random noise

spike history

STDP synaptic plasticity

recurrent + mild feedback connections

dopamine-modulated learning


Reward Modulation
Weights change according to:

dw = STDP(pre, post) * (1 + 2 * reward_level)

Send reward globally via control port:

{"cmd": "reward_global", "args": {"value": 1.0}}


---

Mesocortex Regions

High-level cognitive "regions" are defined across the network:

Region_Pattern   (pattern recognition)
Region_Osc       (oscillatory generator)
Region_Memory    (associative recall)
Region_Cart      (actor-critic agent)
Region_Flock     (multi-agent boids)
Region_Seq       (sequence generator)
Region_Anomaly   (burst/anomaly detector)
Region_AB / Region_BC (demo regions)

Each region is composed of columns mapped to Cortex7 cores on different nodes, creating a distributed mesocortical topology.

Example:

"Region_Pattern": [
  {"id": "pattern_sens_A", "node": "nodeA"},
  {"id": "pattern_hidden_B", "node": "nodeB"},
  {"id": "pattern_out_C", "node": "nodeC"}
]


---

Experiments Included

This core ships with 7 complete cognitive experiment modes, each runnable from the control port.

1. Pattern Replay Classifier

Stimulates sensory neurons with different patterns (00,01,10,11-ish).
Learns via STDP + optional reward.

{"cmd": "experiment_pattern_classify", "args": {"epochs": 20}}


---

2. Oscillatory Rhythm Generator

Excitatory → inhibitory → excitatory loop across nodes creates self-sustaining oscillations.

{"cmd": "experiment_rhythm_gen"}


---

3. Hopfield-Like Associative Memory

Trains on clean patterns, probes with noisy versions.

{"cmd": "experiment_hopfield_assoc"}


---

4. CartPole-Like Agent (Actor–Critic)

Sensor → actor → critic across nodes.
Supports real reward shaping via dopamine broadcast.

{"cmd": "experiment_cart_balance"}


---

5. Boids Multi-Agent Swarm

Simple flocking/coordination dynamics via alternating patterns.

{"cmd": "experiment_boid_swarm"}


---

6. Generative Sequence Model

Outputs a symbolic sequence such as 010101…

{"cmd": "experiment_seq_gen"}


---

7. Seizure / Anomaly Detector

Alternates between normal activity and burst "seizure" events.

{"cmd": "experiment_seizure_detect"}


---

Debugging, Control & Reinforcement

Low Threshold Debug Mode

Force cortex into high-activity mode:

{"cmd": "debug_low_thresholds", "args": {"threshold": 0.4}}

Restore normal behavior:

{"cmd": "debug_low_thresholds", "args": {"threshold": 1.0}}


---

Snapshot / Replay

Create snapshot:

{"cmd": "snapshot", "args": {"reason": "manual_checkpoint"}}

Load latest snapshot:

{"cmd": "load_snapshot", "args": {"path": "latest"}}

Snapshots live in:

snapshots/snapshot_nodeA_<timestamp>.json


---

Metrics + Telemetry

Metrics module aggregates cluster-wide:

per-neuron spike counts

spikes per region

global tick counters

compact cortex state dumps

collected states for offline inspection


Written to:

metrics_dump_core.json

Example summary:

==== METRICS SUMMARY ====
Total spikes: 381
Spikes per node:
  nodeA: 132
  nodeB: 117
  nodeC: 132
Spikes per region:
  Region_AB: 51
  Region_Pattern: 80
...


---

Mesh Networking (HMAC Auth)

All inter-node messages include:

{
  "from_node": "nodeA",
  "dest_module": "cortex7",
  "kind": "advance",
  "payload": {...},
  "mac": "<hmac>"
}

MAC is computed via:

HMAC-SHA256(secret, canonicalized_json)

Nodes reject invalid signatures, ensuring:

integrity

authenticity

no rogue nodes joining the mesh



---

External Control Port

NodeA exposes:

127.0.0.1:10080

Protocol: JSON Lines
One JSON object per line.

Examples:

Status

{"cmd": "status"}

Run the default demo

{"cmd": "run_demo"}

Stimulate any region

{"cmd": "stimulate_region", "args": {...}}

Reward

{"cmd": "reward_global", "args": {"value": 1.0}}

Run experiments

{"cmd": "experiment_boid_swarm"}


---

How to Run Everything

1. Clone the repo

git clone https://github.com/erkrramsey/P3PNueron
cd P3PNueron

2. Run the simulation

python3 p3p_core_expirements_more.py

3. Use the control port

Open another terminal:

nc 127.0.0.1 10080

Send commands:

{"cmd": "status"}
{"cmd": "experiment_seq_gen", "args": {"epochs": 15}}
{"cmd": "reward_global", "args": {"value": 1.0}}


---

What This System Can Do

✓ Distributed neural simulations across multiple nodes

✓ Learn patterns using dopamine + STDP

✓ Generate oscillations, sequences, and recalls

✓ Detect anomalies and classify bursts

✓ Run actor-critic reinforcement experiments

✓ Snapshot/restore cognitive state

✓ Provide full telemetry for ML research

✓ Serve as a P3P node prototype

✓ Serve as an RL playground

✓ Provide a secure HMAC-based mesh control fabric


---

What You Can Build Next (Recommended)

1. Real-time dashboard (HTML/JS)

Visualize:

spikes per node

region activity

synaptic weights

reward influences

timeline graphs


2. Real-world anomaly detection engine

Pipe real telemetry → turn bursts into anomaly events → reward correct detections.

3. Reinforcement agent with real reward loop

CartPole agent but extended to:

text games

robotics simulation

web interactions


4. P3P computing fabric

Replace Cortex7 modules with:

storage

semantic memory

distributed vector search

routing layer


Using the same mesh + modules API.

5. P3P Cluster Playground UI

Browser-based UI to run experiments like a neuroscience control room.

---

License

MIT
