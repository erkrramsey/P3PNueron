#!/usr/bin/env python3
"""
P3P Core v5 — Mesh + Cortex + Mesocortex + Metrics + Control + Persistence + Scheduler + Replay + Identity (HMAC)
+ Cluster-wide Metrics + External Control Port + Experiment Suite

New in this version:
- Cluster-wide metrics: remote nodes (B, C) forward metrics_spike / metrics_state
  messages to nodeA.metrics over the mesh.
- External control port on nodeA (TCP, JSON line protocol) forwards commands
  into the local ControlPlaneModule (127.0.0.1:10080).
- Experiment commands wired into control-plane:
    * experiment_pattern_classify      (Region_Pattern)
    * experiment_rhythm_gen           (Region_Osc)
    * experiment_hopfield_assoc       (Region_Memory)
    * experiment_cart_balance         (Region_Cart)
    * experiment_boid_swarm           (Region_Flock)
    * experiment_seq_gen              (Region_Seq)
    * experiment_seizure_detect       (Region_Anomaly)
    * run_all_experiments             (runs all of the above in sequence)

Each experiment:
- Uses CONFIG['regions'] topology (columns spread across nodes).
- Uses mesocortex_main to stimulate appropriate region.
- Advances cortex and optionally snapshots via persist.
- Emits control/log messages so you can track behavior and cross-check metrics_dump_core.json.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple
import asyncio
import time
import uuid
import json
import math
import os
import hmac
import hashlib

# =========================
# CONFIGURATION
# =========================

CONFIG: Dict[str, Any] = {
    "listen_host": "127.0.0.1",
    "nodes": [
        {
            "id": "nodeA",
            "port": 10001,
            "modules": [
                "logger",
                "echo",
                "cortex7",
                "mesocortex",
                "mesh",
                "metrics",
                "control",
                "persist",
                "scheduler",
                "control_port",
            ],
        },
        {
            "id": "nodeB",
            "port": 10002,
            "modules": ["logger", "echo", "cortex7", "mesh"],
        },
        {
            "id": "nodeC",
            "port": 10003,
            "modules": ["logger", "echo", "cortex7", "mesh"],
        },
    ],
    # Regions: base + experiment topologies (columns are logical identities; node+module actual routing)
    "regions": {
        # Base AB/BC linkage used by warmup_demo
        "Region_AB": [
            {"id": "col_A1", "node": "nodeA", "module": "cortex7"},
            {"id": "col_B1", "node": "nodeB", "module": "cortex7"},
        ],
        "Region_BC": [
            {"id": "col_B2", "node": "nodeB", "module": "cortex7"},
            {"id": "col_C1", "node": "nodeC", "module": "cortex7"},
        ],

        # 1. Pattern Replay Classifier (MNIST-lite / XOR-ish)
        # Sensory on nodeA, hidden on nodeB, output on nodeC.
        "Region_Pattern": [
            {"id": "pattern_sens_A", "node": "nodeA", "module": "cortex7"},
            {"id": "pattern_hidden_B", "node": "nodeB", "module": "cortex7"},
            {"id": "pattern_out_C", "node": "nodeC", "module": "cortex7"},
        ],

        # 2. Oscillatory Rhythm Generator
        # Exc A -> inh B -> exc C loop across nodes.
        "Region_Osc": [
            {"id": "osc_exc_A", "node": "nodeA", "module": "cortex7"},
            {"id": "osc_inh_B", "node": "nodeB", "module": "cortex7"},
            {"id": "osc_exc_C", "node": "nodeC", "module": "cortex7"},
        ],

        # 3. Associative Memory (Hopfield-like)
        "Region_Memory": [
            {"id": "mem_in_A",  "node": "nodeA", "module": "cortex7"},
            {"id": "mem_assoc_B", "node": "nodeB", "module": "cortex7"},
            {"id": "mem_out_C", "node": "nodeC", "module": "cortex7"},
        ],

        # 4. Procedural Game Agent (CartPole-ish)
        "Region_Cart": [
            {"id": "cart_sens_A", "node": "nodeA", "module": "cortex7"},
            {"id": "cart_actor_B", "node": "nodeB", "module": "cortex7"},
            {"id": "cart_critic_C", "node": "nodeC", "module": "cortex7"},
        ],

        # 5. Multi-Agent Coordination (Flocking / Boids)
        "Region_Flock": [
            {"id": "flock_A1", "node": "nodeA", "module": "cortex7"},
            {"id": "flock_B1", "node": "nodeB", "module": "cortex7"},
            {"id": "flock_C1", "node": "nodeC", "module": "cortex7"},
        ],

        # 6. Generative Sequence Model
        "Region_Seq": [
            {"id": "seq_embed_A", "node": "nodeA", "module": "cortex7"},
            {"id": "seq_hidden_B", "node": "nodeB", "module": "cortex7"},
            {"id": "seq_out_C", "node": "nodeC", "module": "cortex7"},
        ],

        # 7. Anomaly Detection / Epileptic-like Detect
        "Region_Anomaly": [
            {"id": "anom_norm_A", "node": "nodeA", "module": "cortex7"},
            {"id": "anom_detect_B", "node": "nodeB", "module": "cortex7"},
            {"id": "anom_alert_C", "node": "nodeC", "module": "cortex7"},
        ],
    },
    "mesocortex_controllers": [
        {
            "node": "nodeA",
            "module_name": "mesocortex_main",
            # Controller manages all regions (base + experiments)
            "regions": [
                "Region_AB",
                "Region_BC",
                "Region_Pattern",
                "Region_Osc",
                "Region_Memory",
                "Region_Cart",
                "Region_Flock",
                "Region_Seq",
                "Region_Anomaly",
            ],
        }
    ],
    "demo": {
        "stim_cycles": 5,
        "advance_ticks": 5,
    },
    "metrics_dump_path": "metrics_dump_core.json",
    "snapshot_dir": "snapshots",
    # Simple job configuration for the scheduler (only warmup demo by default)
    "jobs": [
        {
            "id": "warmup_demo",
            "type": "control_run_demo",
            "node": "nodeA",
            "control_module": "control",
            "delay_s": 0.2,
            "repeat": 1,
            "interval_s": 0.5,
            "args": {},
        },
        # NOTE: experiment jobs are not auto-run here; trigger them via control_port or run_all_experiments.
    ],
    # Shared secret for HMAC signing of mesh envelopes
    "cluster_secret": "p3p_cluster_demo_secret_01",
    # External control port (on nodeA.control_port)
    "control_port": 10080,
}

# =========================
# Message Envelope
# =========================

@dataclass
class Message:
    msg_id: str
    src: str
    dest: str
    kind: str
    ts: float
    payload: Dict[str, Any] = field(default_factory=dict)


# =========================
# Base Module
# =========================

class BaseModule:
    def __init__(self, name: str):
        self.name = name
        self.runtime: "NodeRuntime" = None  # type: ignore

    async def on_start(self):
        pass

    async def on_message(self, msg: Message):
        pass

    async def emit(self, dest: str, kind: str, payload: Dict[str, Any]):
        await self.runtime.emit(self.name, dest, kind, payload)


# =========================
# Node Runtime (with shutdown sentinel)
# =========================

class NodeRuntime:
    """
    Single-node event loop.

    Clean shutdown:
      - stop() sets _running = False and puts a '__shutdown__' message
      - start() exits after processing that message
    """

    def __init__(self, node_id: Optional[str] = None):
        self.node_id = node_id or uuid.uuid4().hex
        self.modules: Dict[str, BaseModule] = {}
        self.queue: "asyncio.Queue[Message]" = asyncio.Queue()
        self._running = False

    def register(self, module: BaseModule):
        if module.name in self.modules:
            raise ValueError(f"Module '{module.name}' already registered on node {self.node_id}")
        module.runtime = self
        self.modules[module.name] = module

    async def emit(self, src: str, dest: str, kind: str, payload: Dict[str, Any]):
        msg = Message(
            msg_id=uuid.uuid4().hex,
            src=src,
            dest=dest,
            kind=kind,
            ts=time.time(),
            payload=payload
        )
        await self.queue.put(msg)

    async def _dispatch(self, msg: Message):
        # Shutdown sentinel is not dispatched to modules
        if msg.kind == "__shutdown__":
            return

        if msg.dest == "broadcast":
            for m in self.modules.values():
                if m.name != msg.src:
                    await m.on_message(msg)
        else:
            m = self.modules.get(msg.dest)
            if m:
                await m.on_message(msg)

    async def start(self):
        self._running = True
        # on_start hooks
        for m in self.modules.values():
            await m.on_start()

        while self._running:
            msg = await self.queue.get()
            await self._dispatch(msg)

    def stop(self):
        """
        Signal loop to end and unblock queue.get() via a shutdown sentinel.
        """
        self._running = False
        try:
            self.queue.put_nowait(
                Message(
                    msg_id=uuid.uuid4().hex,
                    src="runtime",
                    dest="broadcast",
                    kind="__shutdown__",
                    ts=time.time(),
                    payload={}
                )
            )
        except asyncio.QueueFull:
            pass


# =========================
# Utility Modules
# =========================

class LoggerModule(BaseModule):
    async def on_start(self):
        print(f"[{self.runtime.node_id}::{self.name}] logger online")

    async def on_message(self, msg: Message):
        print(f"[{self.runtime.node_id}::{self.name}] got {msg.kind} from {msg.src}: {msg.payload}")


class EchoModule(BaseModule):
    async def on_start(self):
        print(f"[{self.runtime.node_id}::{self.name}] echo online")
        await self.emit("logger", "log", {"status": "echo online"})

    async def on_message(self, msg: Message):
        if msg.kind == "ping":
            await self.emit("logger", "log", {"echo_ping": msg.payload})
            await self.emit(msg.src, "pong", {"echo": msg.payload})


# =========================
# Metrics Module
# =========================

class MetricsModule(BaseModule):
    """
    Aggregates global metrics:

    - spike_counts[(node, module, neuron)] -> int
    - region_spike_counts[region] -> int
    - last_t[node] -> int
    - state_snapshots: arbitrary dumps for later offline inspection
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.spike_counts: Dict[Tuple[str, str, int], int] = {}
        self.region_spike_counts: Dict[str, int] = {}
        self.last_t: Dict[str, int] = {}
        self.state_snapshots: List[Dict[str, Any]] = []

    async def on_start(self):
        print(f"[{self.runtime.node_id}::{self.name}] metrics online")

    async def on_message(self, msg: Message):
        if msg.kind == "metrics_spike":
            node = msg.payload["node"]
            module = msg.payload["module"]
            neuron = int(msg.payload["neuron"])
            t_global = int(msg.payload["t_global"])
            region = msg.payload.get("region")

            key = (node, module, neuron)
            self.spike_counts[key] = self.spike_counts.get(key, 0) + 1
            self.last_t[node] = max(self.last_t.get(node, 0), t_global)

            if region is not None:
                self.region_spike_counts[region] = self.region_spike_counts.get(region, 0) + 1

        elif msg.kind == "metrics_state":
            self.state_snapshots.append(msg.payload)

    def summary(self) -> str:
        lines: List[str] = []
        lines.append("\n==== METRICS SUMMARY ====")

        total_spikes = sum(self.spike_counts.values())
        lines.append(f"Total spikes: {total_spikes}")

        # Per node
        node_totals: Dict[str, int] = {}
        for (node, module, neuron), count in self.spike_counts.items():
            node_totals[node] = node_totals.get(node, 0) + count

        lines.append("\nSpikes per node:")
        for node, c in sorted(node_totals.items()):
            lines.append(f"  {node}: {c}")

        # Per region
        if self.region_spike_counts:
            lines.append("\nSpikes per region:")
            for r, c in sorted(self.region_spike_counts.items()):
                lines.append(f"  {r}: {c}")

        # Per neuron
        lines.append("\nTop (node, module, neuron) spike counts:")
        for (node, module, neuron), c in sorted(
            self.spike_counts.items(), key=lambda kv: kv[1], reverse=True
        ):
            lines.append(f"  {node}.{module}[{neuron}] = {c} spikes")

        # Last t per node
        lines.append("\nLast t per node:")
        for node, tval in sorted(self.last_t.items()):
            lines.append(f"  {node}: t={tval}")

        lines.append("==========================\n")
        return "\n".join(lines)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "spike_counts": {
                f"{node}.{module}.{neuron}": count
                for (node, module, neuron), count in self.spike_counts.items()
            },
            "region_spike_counts": dict(self.region_spike_counts),
            "last_t": dict(self.last_t),
            "state_snapshots": self.state_snapshots,
        }


# =========================
# Mesh Security Helpers (HMAC)
# =========================

def compute_mac(secret: str, envelope: Dict[str, Any]) -> str:
    """
    Compute HMAC-SHA256 over a canonical JSON representation (without 'mac').
    """
    env_copy = dict(envelope)
    env_copy.pop("mac", None)
    blob = json.dumps(env_copy, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(secret.encode("utf-8"), blob, hashlib.sha256).hexdigest()


def verify_mac(secret: str, envelope: Dict[str, Any]) -> bool:
    expected = envelope.get("mac")
    if not isinstance(expected, str):
        return False
    actual = compute_mac(secret, envelope)
    return hmac.compare_digest(expected, actual)


# =========================
# Network Mesh Module (TCP-based, HMAC-protected)
# =========================

class NetworkMeshModule(BaseModule):
    def __init__(self, name: str, listen_host: str, listen_port: int, peers: Dict[str, tuple], cluster_secret: str):
        super().__init__(name)
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.peers = peers
        self.server: Optional[asyncio.AbstractServer] = None
        self.cluster_secret = cluster_secret

    async def on_start(self):
        async def handle_conn(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            addr = writer.get_extra_info("peername")
            try:
                while True:
                    line = await reader.readline()
                    if not line:
                        break
                    try:
                        obj = json.loads(line.decode("utf-8"))
                    except Exception as e:
                        print(f"[{self.runtime.node_id}::mesh] bad JSON from {addr}: {e}")
                        continue

                    if not verify_mac(self.cluster_secret, obj):
                        await self.emit("logger", "log", {
                            "mesh": "recv_mac_invalid",
                            "from_node": obj.get("from_node"),
                            "src_module": obj.get("src_module"),
                        })
                        continue

                    payload = dict(obj.get("payload", {}))
                    payload.setdefault("_remote_from_node", obj.get("from_node"))
                    payload.setdefault("_remote_from_module", obj.get("src_module"))

                    await self.emit(obj["dest_module"], obj["kind"], payload)
            finally:
                writer.close()
                await writer.wait_closed()

        self.server = await asyncio.start_server(
            handle_conn, self.listen_host, self.listen_port
        )
        addr = self.server.sockets[0].getsockname()
        print(f"[{self.runtime.node_id}::mesh] listening on {addr}")
        await self.emit("logger", "log", {"mesh": "online", "addr": f"{addr[0]}:{addr[1]}"})

    async def on_message(self, msg: Message):
        if msg.kind == "send_remote":
            target_node = msg.payload["target_node"]
            dest_module = msg.payload["dest_module"]
            remote_kind = msg.payload["remote_kind"]
            remote_payload = dict(msg.payload.get("payload", {}))

            if target_node not in self.peers:
                await self.emit("logger", "log", {
                    "mesh": "send_remote_failed",
                    "reason": "unknown_target",
                    "target_node": target_node
                })
                return

            host, port = self.peers[target_node]

            try:
                reader, writer = await asyncio.open_connection(host, port)
            except Exception as e:
                await self.emit("logger", "log", {
                    "mesh": "send_remote_failed",
                    "reason": repr(e),
                    "target_node": target_node
                })
                return

            envelope = (
                {
                    "from_node": self.runtime.node_id,
                    "src_module": msg.src,
                    "dest_module": dest_module,
                    "kind": remote_kind,
                    "payload": remote_payload,
                }
            )
            envelope["mac"] = compute_mac(self.cluster_secret, envelope)

            data = (json.dumps(envelope) + "\n").encode("utf-8")
            writer.write(data)
            await writer.drain()
            writer.close()
            await writer.wait_closed()


# =========================
# CORTEX-7 Core
# =========================

@dataclass
class NeuronState:
    v: float = 0.0
    threshold: float = 1.0
    last_spike_t: int = -10**9
    spikes: List[int] = field(default_factory=list)


@dataclass
class Synapse:
    pre: int
    post: int
    w: float
    last_update_t: int = 0


class Cortex7Core:
    def __init__(self, n_neurons: int):
        self.t: int = 0
        self.neurons: Dict[int, NeuronState] = {i: NeuronState() for i in range(n_neurons)}
        self.synapses: Dict[Tuple[int, int], Synapse] = {}
        self.outgoing: Dict[int, List[Synapse]] = {i: [] for i in range(n_neurons)}
        self.incoming: Dict[int, List[Synapse]] = {i: [] for i in range(n_neurons)}
        self.leak = 0.95
        self.stdplast_params = {"a_plus": 0.01, "a_minus": 0.012, "tau": 20.0}

    def set_threshold(self, n: int, thr: float):
        self.neurons[n].threshold = thr

    def link(self, pre: int, post: int, w: float):
        key = (pre, post)
        if key in self.synapses:
            self.synapses[key].w = w
            return
        s = Synapse(pre=pre, post=post, w=w)
        self.synapses[key] = s
        if pre not in self.outgoing:
            self.outgoing[pre] = []
        if post not in self.incoming:
            self.incoming[post] = []
        self.outgoing[pre].append(s)
        self.incoming[post].append(s)

    def add_current(self, n: int, current: float):
        self.neurons[n].v += current

    def _stdp_update(self, syn: Synapse):
        pre_n = self.neurons[syn.pre]
        post_n = self.neurons[syn.post]
        dt = post_n.last_spike_t - pre_n.last_spike_t
        a_plus = self.stdplast_params["a_plus"]
        a_minus = self.stdplast_params["a_minus"]
        tau = self.stdplast_params["tau"]

        if dt > 0:
            dw = a_plus * math.exp(-dt / tau)
        else:
            dw = -a_minus * math.exp(dt / tau)

        syn.w += dw
        syn.w = max(-2.0, min(2.0, syn.w))
        syn.last_update_t = self.t

    def step(self) -> List[int]:
        for n in self.neurons.values():
            n.v *= self.leak

        spiked: List[int] = []
        for idx, n in self.neurons.items():
            if n.v >= n.threshold:
                n.v = 0.0
                n.spikes.append(self.t)
                n.last_spike_t = self.t
                spiked.append(idx)

        for pre_idx in spiked:
            for s in self.outgoing.get(pre_idx, []):
                self.neurons[s.post].v += s.w
                self._stdp_update(s)

        self.t += 1
        return spiked

    def advance(self, ticks: int) -> List[Tuple[int, int]]:
        all_spikes: List[Tuple[int, int]] = []
        for _ in range(ticks):
            sp = self.step()
            for n_idx in sp:
                all_spikes.append((n_idx, self.t - 1))
        return all_spikes

    def dump_state(self) -> Dict[str, Any]:
        return {
            "t": self.t,
            "neurons": {
                i: {
                    "v": n.v,
                    "threshold": n.threshold,
                    "last_spike_t": n.last_spike_t,
                    "spikes": n.spikes[-20:],
                }
                for i, n in self.neurons.items()
            },
            "synapses": {
                f"{s.pre}->{s.post}": {"w": s.w, "last_update_t": s.last_update_t}
                for s in self.synapses.values()
            },
        }

    def load_state(self, state: Dict[str, Any]):
        """
        Restore from dump_state() format.
        Ensures outgoing/incoming are consistent and list-typed.
        """
        self.t = int(state.get("t", 0))

        neurons_state = state.get("neurons", {})
        indices = []
        for k in neurons_state.keys():
            try:
                indices.append(int(k))
            except Exception:
                pass
        max_idx = max(indices + [len(self.neurons) - 1])

        for i in range(len(self.neurons), max_idx + 1):
            self.neurons[i] = NeuronState()

        self.outgoing = {i: [] for i in self.neurons.keys()}
        self.incoming = {i: [] for i in self.neurons.keys()}

        for i_str, n_state in neurons_state.items():
            try:
                i = int(i_str)
            except Exception:
                continue
            if i not in self.neurons:
                self.neurons[i] = NeuronState()
                self.outgoing[i] = []
                self.incoming[i] = []
            n = self.neurons[i]
            n.v = float(n_state.get("v", 0.0))
            n.threshold = float(
                n_state.get("threshold", n_state.get("thr", 1.0))
            )
            n.last_spike_t = int(n_state.get("last_spike_t", -10**9))
            n.spikes = [int(s) for s in n_state.get("spikes", [])]

        self.synapses.clear()
        syn_state = state.get("synapses", {})
        for key, sdict in syn_state.items():
            try:
                pre_str, post_str = key.split("->", 1)
                pre = int(pre_str)
                post = int(post_str)
            except Exception:
                continue
            w = float(sdict.get("w", 0.0))
            last_update = int(sdict.get("last_update_t", 0))
            s = Synapse(pre=pre, post=post, w=w, last_update_t=last_update)
            self.synapses[(pre, post)] = s
            if pre not in self.outgoing:
                self.outgoing[pre] = []
            if post not in self.incoming:
                self.incoming[post] = []
            self.outgoing[pre].append(s)
            self.incoming[post].append(s)


# =========================
# Cortex7 Module (cluster-wide metrics aware)
# =========================

class Cortex7Module(BaseModule):
    def __init__(
        self,
        name: str,
        n_neurons: int = 4,
        has_local_metrics: bool = False,
        remote_metrics_node: Optional[str] = None,
    ):
        super().__init__(name)
        self.core = Cortex7Core(n_neurons=n_neurons)
        self.core.set_threshold(0, 1.0)
        self.core.set_threshold(1, 1.2)
        self.core.set_threshold(2, 1.3)
        self.core.set_threshold(3, 1.5)
        self.core.link(0, 1, 0.6)
        self.core.link(1, 2, 0.5)
        self.core.link(2, 3, 0.4)

        self.has_local_metrics = has_local_metrics
        self.remote_metrics_node = remote_metrics_node

    async def on_start(self):
        await self.emit("logger", "log", {"cortex7": "online", "neurons": len(self.core.neurons)})

    async def _emit_spikes(
        self,
        spikes: List[Tuple[int, int]],
        remote_source: Optional[str],
        region: Optional[str],
    ):
        if not spikes:
            return
        node_id = self.runtime.node_id
        for n_idx, tick in spikes:
            payload = {
                "module": self.name,
                "neuron": n_idx,
                "tick": tick,
                "t_global": self.core.t,
                "source": remote_source,
                "region": region,
            }
            await self.emit("logger", "cortex_spike", payload)

            mp = {
                "node": node_id,
                "module": self.name,
                "neuron": n_idx,
                "tick": tick,
                "t_global": self.core.t,
                "region": region,
            }

            if self.has_local_metrics and "metrics" in self.runtime.modules:
                await self.emit("metrics", "metrics_spike", mp)
            elif self.remote_metrics_node is not None:
                await self.emit(
                    "mesh",
                    "send_remote",
                    {
                        "target_node": self.remote_metrics_node,
                        "dest_module": "metrics",
                        "remote_kind": "metrics_spike",
                        "payload": mp,
                    },
                )

    async def on_message(self, msg: Message):
        region = msg.payload.get("region")

        if msg.kind == "inject":
            neuron = int(msg.payload.get("neuron", 0))
            current = float(msg.payload.get("current", 0.0))
            remote_source = msg.payload.get("_remote_from_node")
            self.core.add_current(neuron, current)
            spikes = self.core.advance(1)
            await self._emit_spikes(spikes, remote_source, region)

        elif msg.kind == "advance":
            ticks = int(msg.payload.get("ticks", 1))
            remote_source = msg.payload.get("_remote_from_node")
            spikes = self.core.advance(ticks)
            await self._emit_spikes(spikes, remote_source, region)

        elif msg.kind == "dump":
            state = self.core.dump_state()
            compact = {
                "t": state["t"],
                "neurons": {
                    i: {
                        "v": n["v"],
                        "thr": n["threshold"],
                        "last_spike_t": n["last_spike_t"],
                        "spikes": n["spikes"],
                    }
                    for i, n in state["neurons"].items()
                },
            }
            await self.emit("logger", "cortex_state", compact)

            state_payload = {
                "node": self.runtime.node_id,
                "module": self.name,
                "state": compact,
            }

            if self.has_local_metrics and "metrics" in self.runtime.modules:
                await self.emit("metrics", "metrics_state", state_payload)
            elif self.remote_metrics_node is not None:
                await self.emit(
                    "mesh",
                    "send_remote",
                    {
                        "target_node": self.remote_metrics_node,
                        "dest_module": "metrics",
                        "remote_kind": "metrics_state",
                        "payload": state_payload,
                    },
                )


# =========================
# Mesocortex Module (tagging region)
# =========================

class MesocortexModule(BaseModule):
    def __init__(self, name: str, topo: Dict[str, List[Dict[str, Any]]]):
        super().__init__(name)
        self.regions = topo

    async def on_start(self):
        await self.emit("logger", "log", {"mesocortex": "online", "regions": list(self.regions.keys())})

    async def _send_to_column(self, col: Dict[str, Any], kind: str, payload: Dict[str, Any]):
        target_node = col["node"]
        module_name = col["module"]

        if target_node == self.runtime.node_id:
            await self.emit(module_name, kind, payload)
        else:
            await self.emit(
                "mesh",
                "send_remote",
                {
                    "target_node": target_node,
                    "dest_module": module_name,
                    "remote_kind": kind,
                    "payload": payload,
                },
            )

    async def on_message(self, msg: Message):
        if msg.kind == "stimulate_region":
            region = msg.payload.get("region")
            pattern = msg.payload.get("pattern", [])
            cols = self.regions.get(region, [])

            await self.emit("logger", "log", {
                "mesocortex": "stimulate_region",
                "region": region,
                "columns": [c["id"] for c in cols],
                "pattern_len": len(pattern),
            })

            for col in cols:
                for p in pattern:
                    payload = {
                        "neuron": int(p.get("neuron", 0)),
                        "current": float(p.get("current", 0.0)),
                        "region": region,
                    }
                    await self._send_to_column(col, "inject", payload)

        elif msg.kind == "advance_all":
            ticks = int(msg.payload.get("ticks", 1))
            for region, cols in self.regions.items():
                for col in cols:
                    await self._send_to_column(col, "advance", {"ticks": ticks})
            await self.emit("logger", "log", {
                "mesocortex": "advance_all",
                "ticks": ticks,
            })

        elif msg.kind == "dump_all":
            for region, cols in self.regions.items():
                for col in cols:
                    await self._send_to_column(col, "dump", {})
            await self.emit("logger", "log", {
                "mesocortex": "dump_all",
            })

        elif msg.kind == "update_regions":
            new_regions = msg.payload.get("regions")
            if isinstance(new_regions, dict):
                self.regions = new_regions
                await self.emit("logger", "log", {
                    "mesocortex": "regions_updated",
                    "regions": list(self.regions.keys()),
                })


# =========================
# Persistence Module
# =========================

class PersistenceModule(BaseModule):
    """
    Handles snapshots of local node state to JSON and loading them back.

    Receives 'persist' messages with payload:
        {'cmd': 'snapshot', 'args': {...}}
        {'cmd': 'load', 'args': {...}}

    Snapshot includes (if present locally):
        - cortex7.core.dump_state()
        - metrics.as_dict()
        - node id, timestamp
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name)
        self.config = config

    async def on_start(self):
        await self.emit("logger", "log", {"persist": "online"})

    async def on_message(self, msg: Message):
        if msg.kind != "persist":
            return

        cmd = msg.payload.get("cmd")
        args = msg.payload.get("args", {}) or {}

        if cmd == "snapshot":
            await self._cmd_snapshot(args)
        elif cmd == "load":
            await self._cmd_load(args)
        else:
            await self.emit("logger", "log", {
                "persist": "unknown_cmd",
                "cmd": cmd,
            })

    async def _cmd_snapshot(self, args: Dict[str, Any]):
        node_id = self.runtime.node_id
        ts = time.time()

        data: Dict[str, Any] = {
            "node": node_id,
            "ts": ts,
            "cortex7": None,
            "metrics": None,
        }

        if "cortex7" in self.runtime.modules:
            cortex_mod: Cortex7Module = self.runtime.modules["cortex7"]  # type: ignore
            data["cortex7"] = cortex_mod.core.dump_state()

        if "metrics" in self.runtime.modules:
            metrics_mod: MetricsModule = self.runtime.modules["metrics"]  # type: ignore
            data["metrics"] = metrics_mod.as_dict()

        snapshot_dir = self.config.get("snapshot_dir", "snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)
        fname = f"snapshot_{node_id}_{int(ts)}.json"
        path = os.path.join(snapshot_dir, fname)

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            await self.emit("logger", "log", {
                "persist": "snapshot_written",
                "path": os.path.abspath(path),
            })
        except Exception as e:
            await self.emit("logger", "log", {
                "persist": "snapshot_failed",
                "error": repr(e),
            })

    async def _cmd_load(self, args: Dict[str, Any]):
        node_id = self.runtime.node_id
        snapshot_dir = self.config.get("snapshot_dir", "snapshots")
        path = args.get("path")

        if not path or str(path) == "latest":
            try:
                files = [
                    f for f in os.listdir(snapshot_dir)
                    if f.startswith(f"snapshot_{node_id}_") and f.endswith(".json")
                ]
            except FileNotFoundError:
                files = []

            if not files:
                await self.emit("logger", "log", {
                    "persist": "load_failed",
                    "reason": "no_snapshots_found",
                    "snapshot_dir": snapshot_dir,
                })
                return

            def parse_ts(fname: str) -> int:
                try:
                    base = fname.rsplit(".", 1)[0]
                    ts_str = base.split("_")[-1]
                    return int(ts_str)
                except Exception:
                    return 0

            files.sort(key=parse_ts, reverse=True)
            path = os.path.join(snapshot_dir, files[0])

        path = os.path.abspath(str(path))

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            await self.emit("logger", "log", {
                "persist": "load_failed",
                "reason": "io_error",
                "error": repr(e),
                "path": path,
            })
            return

        if "cortex7" in self.runtime.modules and data.get("cortex7") is not None:
            cortex_mod: Cortex7Module = self.runtime.modules["cortex7"]  # type: ignore
            cortex_mod.core.load_state(data["cortex7"])

        # Metrics are live telemetry for the current run; we don't restore them.
        await self.emit("logger", "log", {
            "persist": "load_success",
            "path": path,
        })


# =========================
# Control Plane Module
# =========================

class ControlPlaneModule(BaseModule):
    """
    Control-plane for this node:

    Receives 'control' messages with payload {'cmd': ..., 'args': {...}}.

    Supported commands:
      - 'status'
      - 'run_demo'
      - 'update_regions'
      - 'snapshot'
      - 'load_snapshot'
      - 'experiment_pattern_classify'
      - 'experiment_rhythm_gen'
      - 'experiment_hopfield_assoc'
      - 'experiment_cart_balance'
      - 'experiment_boid_swarm'
      - 'experiment_seq_gen'
      - 'experiment_seizure_detect'
      - 'run_all_experiments'
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name)
        self.config = config

    async def on_start(self):
        await self.emit("logger", "log", {"control": "online"})

    async def on_message(self, msg: Message):
        if msg.kind != "control":
            return

        cmd = msg.payload.get("cmd")
        args = msg.payload.get("args", {}) or {}

        if cmd == "status":
            await self._cmd_status(args)
        elif cmd == "run_demo":
            await self._cmd_run_demo(args)
        elif cmd == "update_regions":
            await self._cmd_update_regions(args)
        elif cmd == "snapshot":
            await self._cmd_snapshot(args)
        elif cmd == "load_snapshot":
            await self._cmd_load_snapshot(args)
        elif cmd == "run_all_experiments":
            await self._cmd_run_all_experiments(args)

        # Experiment suite
        elif cmd == "experiment_pattern_classify":
            await self._cmd_experiment_pattern_classify(args)
        elif cmd == "experiment_rhythm_gen":
            await self._cmd_experiment_rhythm_gen(args)
        elif cmd == "experiment_hopfield_assoc":
            await self._cmd_experiment_hopfield_assoc(args)
        elif cmd == "experiment_cart_balance":
            await self._cmd_experiment_cart_balance(args)
        elif cmd == "experiment_boid_swarm":
            await self._cmd_experiment_boid_swarm(args)
        elif cmd == "experiment_seq_gen":
            await self._cmd_experiment_seq_gen(args)
        elif cmd == "experiment_seizure_detect":
            await self._cmd_experiment_seizure_detect(args)

        else:
            await self.emit("logger", "log", {
                "control": "unknown_cmd",
                "cmd": cmd,
            })

    # ----- Basic control -----

    async def _cmd_status(self, args: Dict[str, Any]):
        modules = list(self.runtime.modules.keys())
        await self.emit("logger", "log", {
            "control": "status",
            "node": self.runtime.node_id,
            "modules": modules,
        })

    async def _cmd_run_demo(self, args: Dict[str, Any]):
        demo = self.config.get("demo", {})
        stim_cycles = int(args.get("stim_cycles", demo.get("stim_cycles", 5)))
        advance_ticks = int(args.get("advance_ticks", demo.get("advance_ticks", 5)))

        controllers = self.config.get("mesocortex_controllers", [])
        if not controllers:
            await self.emit("logger", "log", {
                "control": "run_demo_failed",
                "reason": "no_controllers_in_config",
            })
            return

        ctrl = controllers[0]
        mname = ctrl["module_name"]
        regions = ["Region_AB", "Region_BC"]

        await self.emit("logger", "log", {
            "control": "run_demo_begin",
            "regions": regions,
            "stim_cycles": stim_cycles,
            "advance_ticks": advance_ticks,
        })

        for _ in range(stim_cycles):
            for region in regions:
                await self.emit(
                    mname,
                    "stimulate_region",
                    {
                        "region": region,
                        "pattern": [{"neuron": 0, "current": 0.7}],
                    },
                )
            await asyncio.sleep(0.05)

        await self.emit(
            mname,
            "advance_all",
            {"ticks": advance_ticks},
        )
        await asyncio.sleep(0.2)

        await self.emit(
            mname,
            "dump_all",
            {},
        )
        await asyncio.sleep(0.2)

        await self.emit("logger", "log", {
            "control": "run_demo_complete",
        })

        if "persist" in self.runtime.modules:
            await self._cmd_snapshot({"reason": "after_run_demo"})

    async def _cmd_update_regions(self, args: Dict[str, Any]):
        new_regions = args.get("regions")
        if not isinstance(new_regions, dict):
            await self.emit("logger", "log", {
                "control": "update_regions_failed",
                "reason": "regions_not_dict",
            })
            return

        controllers = self.config.get("mesocortex_controllers", [])
        if not controllers:
            await self.emit("logger", "log", {
                "control": "update_regions_failed",
                "reason": "no_controllers_in_config",
            })
            return

        ctrl = controllers[0]
        mname = ctrl["module_name"]

        await self.emit(
            mname,
            "update_regions",
            {"regions": new_regions},
        )
        await self.emit("logger", "log", {
            "control": "update_regions_sent",
            "regions": list(new_regions.keys()),
        })

    async def _cmd_snapshot(self, args: Dict[str, Any]):
        if "persist" not in self.runtime.modules:
            await self.emit("logger", "log", {
                "control": "snapshot_failed",
                "reason": "no_persist_module",
            })
            return

        await self.emit(
            "persist",
            "persist",
            {"cmd": "snapshot", "args": args},
        )
        await self.emit("logger", "log", {
            "control": "snapshot_requested",
            "args": args,
        })

    async def _cmd_load_snapshot(self, args: Dict[str, Any]):
        if "persist" not in self.runtime.modules:
            await self.emit("logger", "log", {
                "control": "load_snapshot_failed",
                "reason": "no_persist_module",
            })
            return

        await self.emit(
            "persist",
            "persist",
            {"cmd": "load", "args": args},
        )
        await self.emit("logger", "log", {
            "control": "load_snapshot_requested",
            "args": args,
        })

    async def _cmd_run_all_experiments(self, args: Dict[str, Any]):
        """
        Run the entire experiment suite in sequence on this node.
        Args can optionally contain per-experiment sub-args:
            {
              "pattern": {...},
              "rhythm": {...},
              "hopfield": {...},
              "cart": {...},
              "flock": {...},
              "seq": {...},
              "anomaly": {...}
            }
        Any missing sub-args fall back to defaults inside each experiment method.
        """
        await self.emit("logger", "log", {
            "control": "run_all_experiments_begin",
        })

        pattern_args = args.get("pattern", {})
        rhythm_args = args.get("rhythm", {})
        hopfield_args = args.get("hopfield", {})
        cart_args = args.get("cart", {})
        flock_args = args.get("flock", {})
        seq_args = args.get("seq", {})
        anomaly_args = args.get("anomaly", {})

        await self._cmd_experiment_pattern_classify(pattern_args)
        await self._cmd_experiment_rhythm_gen(rhythm_args)
        await self._cmd_experiment_hopfield_assoc(hopfield_args)
        await self._cmd_experiment_cart_balance(cart_args)
        await self._cmd_experiment_boid_swarm(flock_args)
        await self._cmd_experiment_seq_gen(seq_args)
        await self._cmd_experiment_seizure_detect(anomaly_args)

        await self.emit("logger", "log", {
            "control": "run_all_experiments_complete",
        })

    # ----- Experiment helpers -----

    async def _get_mesocortex_main_name(self) -> Optional[str]:
        ctrls = self.config.get("mesocortex_controllers", [])
        if not ctrls:
            return None
        return ctrls[0]["module_name"]

    # 1. Pattern Replay Classifier (scaffold)
    async def _cmd_experiment_pattern_classify(self, args: Dict[str, Any]):
        epochs = int(args.get("epochs", 10))
        mname = await self._get_mesocortex_main_name()
        if not mname:
            await self.emit("logger", "log", {
                "control": "experiment_pattern_classify_failed",
                "reason": "no_mesocortex",
            })
            return

        await self.emit("logger", "log", {
            "control": "experiment_pattern_classify_begin",
            "epochs": epochs,
            "region": "Region_Pattern",
        })

        # Four simple patterns on neuron 0/1 to emulate 00/01/10/11-ish rate codes
        patterns = [
            [{"neuron": 0, "current": 0.3}],                 # pattern 0
            [{"neuron": 1, "current": 0.3}],                 # pattern 1
            [{"neuron": 0, "current": 0.5}],                 # pattern 2
            [{"neuron": 1, "current": 0.5}],                 # pattern 3
        ]

        for epoch in range(epochs):
            for p in patterns:
                await self.emit(
                    mname,
                    "stimulate_region",
                    {
                        "region": "Region_Pattern",
                        "pattern": p,
                    },
                )
                # short evolution window
                await asyncio.sleep(0.01)

            # Let the region settle
            await self.emit(
                mname,
                "advance_all",
                {"ticks": 5},
            )

            if epoch % 5 == 0:
                await self.emit(
                    mname,
                    "dump_all",
                    {},
                )
                if "persist" in self.runtime.modules:
                    await self._cmd_snapshot({"reason": f"pattern_epoch_{epoch}"})

        await self.emit("logger", "log", {
            "control": "experiment_pattern_classify_complete",
            "epochs": epochs,
        })

    # 2. Oscillatory Rhythm Generator (scaffold)
    async def _cmd_experiment_rhythm_gen(self, args: Dict[str, Any]):
        epochs = int(args.get("epochs", 20))
        ticks_per_epoch = int(args.get("ticks_per_epoch", 20))
        mname = await self._get_mesocortex_main_name()
        if not mname:
            await self.emit("logger", "log", {
                "control": "experiment_rhythm_gen_failed",
                "reason": "no_mesocortex",
            })
            return

        await self.emit("logger", "log", {
            "control": "experiment_rhythm_gen_begin",
            "epochs": epochs,
            "ticks_per_epoch": ticks_per_epoch,
            "region": "Region_Osc",
        })

        # Initial kick to excite oscillation
        await self.emit(
            mname,
            "stimulate_region",
            {
                "region": "Region_Osc",
                "pattern": [{"neuron": 0, "current": 1.0}],
            },
        )

        for epoch in range(epochs):
            await self.emit(
                mname,
                "advance_all",
                {"ticks": ticks_per_epoch},
            )

            if epoch % 5 == 0:
                await self.emit(
                    mname,
                    "dump_all",
                    {},
                )
                if "persist" in self.runtime.modules:
                    await self._cmd_snapshot({"reason": f"osc_epoch_{epoch}"})

            await asyncio.sleep(0.02)

        await self.emit("logger", "log", {
            "control": "experiment_rhythm_gen_complete",
            "epochs": epochs,
        })

    # 3. Associative Memory (scaffold)
    async def _cmd_experiment_hopfield_assoc(self, args: Dict[str, Any]):
        epochs = int(args.get("epochs", 15))
        mname = await self._get_mesocortex_main_name()
        if not mname:
            await self.emit("logger", "log", {
                "control": "experiment_hopfield_assoc_failed",
                "reason": "no_mesocortex",
            })
            return

        await self.emit("logger", "log", {
            "control": "experiment_hopfield_assoc_begin",
            "epochs": epochs,
            "region": "Region_Memory",
        })

        base_patterns = [
            [{"neuron": 0, "current": 0.6}],
            [{"neuron": 1, "current": 0.6}],
            [{"neuron": 2, "current": 0.6}],
        ]

        for epoch in range(epochs):
            # "Train" phase: clean patterns
            for p in base_patterns:
                await self.emit(
                    mname,
                    "stimulate_region",
                    {
                        "region": "Region_Memory",
                        "pattern": p,
                    },
                )
                await self.emit(
                    mname,
                    "advance_all",
                    {"ticks": 3},
                )

            # "Probe" with noisy version (just scaled currents as a toy)
            for p in base_patterns:
                noisy = []
                for entry in p:
                    noisy.append({
                        "neuron": entry["neuron"],
                        "current": entry["current"] * 0.7,
                    })
                await self.emit(
                    mname,
                    "stimulate_region",
                    {
                        "region": "Region_Memory",
                        "pattern": noisy,
                    },
                )
                await self.emit(
                    mname,
                    "advance_all",
                    {"ticks": 3},
                )

            if epoch % 3 == 0:
                await self.emit(
                    mname,
                    "dump_all",
                    {},
                )
                if "persist" in self.runtime.modules:
                    await self._cmd_snapshot({"reason": f"hopfield_epoch_{epoch}"})

            await asyncio.sleep(0.02)

        await self.emit("logger", "log", {
            "control": "experiment_hopfield_assoc_complete",
            "epochs": epochs,
        })

    # 4. Procedural Game Agent (CartPole-ish, scaffold)
    async def _cmd_experiment_cart_balance(self, args: Dict[str, Any]):
        episodes = int(args.get("episodes", 20))
        steps_per_episode = int(args.get("steps_per_episode", 30))
        mname = await self._get_mesocortex_main_name()
        if not mname:
            await self.emit("logger", "log", {
                "control": "experiment_cart_balance_failed",
                "reason": "no_mesocortex",
            })
            return

        await self.emit("logger", "log", {
            "control": "experiment_cart_balance_begin",
            "episodes": episodes,
            "steps_per_episode": steps_per_episode,
            "region": "Region_Cart",
        })

        for ep in range(episodes):
            # crude "state" ramp via currents on sensor neuron 0
            for step in range(steps_per_episode):
                current_level = 0.2 + 0.01 * step
                await self.emit(
                    mname,
                    "stimulate_region",
                    {
                        "region": "Region_Cart",
                        "pattern": [{"neuron": 0, "current": current_level}],
                    },
                )
                await self.emit(
                    mname,
                    "advance_all",
                    {"ticks": 1},
                )
                await asyncio.sleep(0.003)

            if ep % 5 == 0:
                await self.emit(
                    mname,
                    "dump_all",
                    {},
                )
                if "persist" in self.runtime.modules:
                    await self._cmd_snapshot({"reason": f"cart_ep_{ep}"})

        await self.emit("logger", "log", {
            "control": "experiment_cart_balance_complete",
            "episodes": episodes,
        })

    # 5. Multi-Agent Coordination / Flocking (scaffold)
    async def _cmd_experiment_boid_swarm(self, args: Dict[str, Any]):
        epochs = int(args.get("epochs", 25))
        mname = await self._get_mesocortex_main_name()
        if not mname:
            await self.emit("logger", "log", {
                "control": "experiment_boid_swarm_failed",
                "reason": "no_mesocortex",
            })
            return

        await self.emit("logger", "log", {
            "control": "experiment_boid_swarm_begin",
            "epochs": epochs,
            "region": "Region_Flock",
        })

        for epoch in range(epochs):
            # Simulate "neighbors" by alternating currents on different neurons
            pat = []
            base_current = 0.3 + 0.02 * (epoch % 5)
            pat.append({"neuron": 0, "current": base_current})
            pat.append({"neuron": 1, "current": base_current * 0.9})

            await self.emit(
                mname,
                "stimulate_region",
                {
                    "region": "Region_Flock",
                    "pattern": pat,
                },
            )
            await self.emit(
                mname,
                "advance_all",
                {"ticks": 5},
            )

            if epoch % 5 == 0:
                await self.emit(
                    mname,
                    "dump_all",
                    {},
                )
                if "persist" in self.runtime.modules:
                    await self._cmd_snapshot({"reason": f"flock_epoch_{epoch}"})

            await asyncio.sleep(0.01)

        await self.emit("logger", "log", {
            "control": "experiment_boid_swarm_complete",
            "epochs": epochs,
        })

    # 6. Generative Sequence Model (scaffold)
    async def _cmd_experiment_seq_gen(self, args: Dict[str, Any]):
        epochs = int(args.get("epochs", 20))
        seq_len = int(args.get("seq_len", 10))
        mname = await self._get_mesocortex_main_name()
        if not mname:
            await self.emit("logger", "log", {
                "control": "experiment_seq_gen_failed",
                "reason": "no_mesocortex",
            })
            return

        await self.emit("logger", "log", {
            "control": "experiment_seq_gen_begin",
            "epochs": epochs,
            "seq_len": seq_len,
            "region": "Region_Seq",
        })

        for epoch in range(epochs):
            # Simple alternating "0101..." sequence encoded as current pulses on n0
            for t in range(seq_len):
                bit = t % 2
                current = 0.6 if bit == 1 else 0.3
                await self.emit(
                    mname,
                    "stimulate_region",
                    {
                        "region": "Region_Seq",
                        "pattern": [{"neuron": 0, "current": current}],
                    },
                )
                await self.emit(
                    mname,
                    "advance_all",
                    {"ticks": 1},
                )
                await asyncio.sleep(0.003)

            if epoch % 5 == 0:
                await self.emit(
                    mname,
                    "dump_all",
                    {},
                )
                if "persist" in self.runtime.modules:
                    await self._cmd_snapshot({"reason": f"seq_epoch_{epoch}"})

        await self.emit("logger", "log", {
            "control": "experiment_seq_gen_complete",
            "epochs": epochs,
        })

    # 7. Anomaly Detection (scaffold)
    async def _cmd_experiment_seizure_detect(self, args: Dict[str, Any]):
        epochs = int(args.get("epochs", 30))
        burst_every = int(args.get("burst_every", 7))
        mname = await self._get_mesocortex_main_name()
        if not mname:
            await self.emit("logger", "log", {
                "control": "experiment_seizure_detect_failed",
                "reason": "no_mesocortex",
            })
            return

        await self.emit("logger", "log", {
            "control": "experiment_seizure_detect_begin",
            "epochs": epochs,
            "burst_every": burst_every,
            "region": "Region_Anomaly",
        })

        for epoch in range(epochs):
            # "Normal" low-amplitude sine-ish waveform approximated by alternating small currents
            normal_current = 0.2 + 0.05 * ((epoch % 4) / 3.0)
            pattern = [{"neuron": 0, "current": normal_current}]

            # Occasionally inject a "burst" as anomaly
            is_burst = (epoch % burst_every) == 0
            if is_burst:
                pattern.append({"neuron": 1, "current": 1.0})

            await self.emit(
                mname,
                "stimulate_region",
                {
                    "region": "Region_Anomaly",
                    "pattern": pattern,
                },
            )
            await self.emit(
                mname,
                "advance_all",
                {"ticks": 4},
            )

            if epoch % 5 == 0 or is_burst:
                await self.emit(
                    mname,
                    "dump_all",
                    {},
                )
                if "persist" in self.runtime.modules:
                    reason = "anom_burst" if is_burst else f"anom_epoch_{epoch}"
                    await self._cmd_snapshot({"reason": reason})

            await asyncio.sleep(0.01)

        await self.emit("logger", "log", {
            "control": "experiment_seizure_detect_complete",
            "epochs": epochs,
        })


# =========================
# Job Scheduler Module
# =========================

class JobSchedulerModule(BaseModule):
    """
    Simple finite scheduler that reads CONFIG['jobs'] and fires them.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name)
        self.config = config
        self._task: Optional[asyncio.Task] = None

    async def on_start(self):
        await self.emit("logger", "log", {"scheduler": "online"})
        self._task = asyncio.create_task(self._run_jobs())

    async def on_message(self, msg: Message):
        pass

    async def _run_jobs(self):
        jobs = self.config.get("jobs", [])
        await asyncio.sleep(0.3)

        for job in jobs:
            jtype = job.get("type")
            delay_s = float(job.get("delay_s", 0.0))
            repeat = int(job.get("repeat", 1))
            interval_s = float(job.get("interval_s", 0.0))

            await self.emit("logger", "log", {
                "scheduler": "job_starting",
                "job_id": job.get("id"),
                "type": jtype,
                "repeat": repeat,
            })

            if delay_s > 0:
                await asyncio.sleep(delay_s)

            for i in range(repeat):
                if jtype == "control_run_demo":
                    await self._job_control_run_demo(job)

                if i < repeat - 1 and interval_s > 0:
                    await asyncio.sleep(interval_s)

            await self.emit("logger", "log", {
                "scheduler": "job_finished",
                "job_id": job.get("id"),
                "type": jtype,
            })

        await self.emit("logger", "log", {
            "scheduler": "all_jobs_complete",
        })

    async def _job_control_run_demo(self, job: Dict[str, Any]):
        control_module = job.get("control_module", "control")
        args = job.get("args", {}) or {}

        if control_module not in self.runtime.modules:
            await self.emit("logger", "log", {
                "scheduler": "job_failed",
                "job_id": job.get("id"),
                "reason": "control_module_not_present",
            })
            return

        await self.emit(
            control_module,
            "control",
            {"cmd": "run_demo", "args": args},
        )


# =========================
# External Control Port Module
# =========================

class ControlPortModule(BaseModule):
    """
    Simple TCP JSON-line control port.

    Listens on CONFIG['control_port'] on nodeA, accepts lines like:
        {"cmd": "status", "args": {}}
        {"cmd": "run_demo", "args": {"stim_cycles": 3}}
        {"cmd": "experiment_pattern_classify", "args": {"epochs": 20}}
        {"cmd": "run_all_experiments", "args": {}}

    For each valid line, forwards into local ControlPlaneModule.
    """

    def __init__(self, name: str, host: str, port: int):
        super().__init__(name)
        self.host = host
        self.port = port
        self.server: Optional[asyncio.AbstractServer] = None

    async def on_start(self):
        async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
            addr = writer.get_extra_info("peername")
            try:
                while True:
                    line = await reader.readline()
                    if not line:
                        break
                    line_str = line.decode("utf-8").strip()
                    if not line_str:
                        continue
                    try:
                        obj = json.loads(line_str)
                    except Exception as e:
                        err = {"error": "bad_json", "detail": str(e)}
                        writer.write((json.dumps(err) + "\n").encode("utf-8"))
                        await writer.drain()
                        continue

                    cmd = obj.get("cmd")
                    args = obj.get("args", {}) or {}
                    if not isinstance(cmd, str):
                        err = {"error": "missing_cmd"}
                        writer.write((json.dumps(err) + "\n").encode("utf-8"))
                        await writer.drain()
                        continue

                    await self.emit("control", "control", {"cmd": cmd, "args": args})
                    ok = {"status": "accepted", "cmd": cmd}
                    writer.write((json.dumps(ok) + "\n").encode("utf-8"))
                    await writer.drain()
            finally:
                writer.close()
                await writer.wait_closed()

        self.server = await asyncio.start_server(handle_client, self.host, self.port)
        addr = self.server.sockets[0].getsockname()
        await self.emit("logger", "log", {"control_port": "online", "addr": f"{addr[0]}:{addr[1]}"})

    async def on_message(self, msg: Message):
        # No internal messages yet; this module reacts only to TCP clients
        pass


# =========================
# Peer Map Builder
# =========================

def build_peer_maps(config: Dict[str, Any]) -> Dict[str, Dict[str, tuple]]:
    host = config["listen_host"]
    peers_map: Dict[str, Dict[str, tuple]] = {}
    id_to_port = {n["id"]: n["port"] for n in config["nodes"]}

    for node in config["nodes"]:
        nid = node["id"]
        peers: Dict[str, tuple] = {}
        for nid2, port2 in id_to_port.items():
            if nid2 == nid:
                continue
            peers[nid2] = (host, port2)
        peers_map[nid] = peers
    return peers_map


# =========================
# Cluster Runner
# =========================

async def run_cluster(config: Dict[str, Any]):
    host = config["listen_host"]
    peers_map = build_peer_maps(config)
    cluster_secret = config.get("cluster_secret", "p3p_default_secret")

    nodes: Dict[str, NodeRuntime] = {}
    for node_def in config["nodes"]:
        nid = node_def["id"]
        rt = NodeRuntime(node_id=nid)
        nodes[nid] = rt

    # Determine metrics-root node (the one that hosts "metrics")
    metrics_root_node: Optional[str] = None
    for node_def in config["nodes"]:
        if "metrics" in node_def["modules"]:
            metrics_root_node = node_def["id"]
            break

    # Register modules
    for node_def in config["nodes"]:
        nid = node_def["id"]
        port = node_def["port"]
        mods = node_def["modules"]
        rt = nodes[nid]

        if "logger" in mods:
            rt.register(LoggerModule("logger"))
        if "echo" in mods:
            rt.register(EchoModule("echo"))
        if "cortex7" in mods:
            has_local_metrics = "metrics" in mods
            remote_metrics_node = None
            if not has_local_metrics and metrics_root_node is not None:
                remote_metrics_node = metrics_root_node
            rt.register(
                Cortex7Module(
                    "cortex7",
                    n_neurons=4,
                    has_local_metrics=has_local_metrics,
                    remote_metrics_node=remote_metrics_node,
                )
            )
        if "mesh" in mods:
            rt.register(NetworkMeshModule("mesh", host, port, peers_map[nid], cluster_secret))
        if "metrics" in mods:
            rt.register(MetricsModule("metrics"))
        if "control" in mods:
            rt.register(ControlPlaneModule("control", config))
        if "persist" in mods:
            rt.register(PersistenceModule("persist", config))
        if "scheduler" in mods:
            rt.register(JobSchedulerModule("scheduler", config))
        if "control_port" in mods:
            control_port = config.get("control_port", 10080)
            rt.register(ControlPortModule("control_port", host, control_port))
        # Mesocortex controllers are registered after this loop

    # Register Mesocortex controllers
    for ctrl_def in config.get("mesocortex_controllers", []):
        nid = ctrl_def["node"]
        module_name = ctrl_def["module_name"]
        region_names = ctrl_def["regions"]
        topo_slice: Dict[str, List[Dict[str, Any]]] = {}
        for rname in region_names:
            topo_slice[rname] = config["regions"][rname]
        nodes[nid].register(MesocortexModule(module_name, topo_slice))

    # Start all runtimes
    tasks = [asyncio.create_task(rt.start()) for rt in nodes.values()]
    await asyncio.sleep(0.3)

    # Sanity: nodeA pings nodeB.echo
    if "nodeA" in nodes and "nodeB" in nodes:
        print("\n=== Sanity: nodeA ping -> nodeB.echo ===")
        await nodes["nodeA"].emit(
            src="testerA",
            dest="mesh",
            kind="send_remote",
            payload={
                "target_node": "nodeB",
                "dest_module": "echo",
                "remote_kind": "ping",
                "payload": {"hello": "from nodeA"},
            },
        )
        await asyncio.sleep(0.2)

    # Ask control-plane for status once (scheduler handles demos)
    if "nodeA" in nodes and "control" in nodes["nodeA"].modules:
        print("\n=== Control: status (scheduler will trigger demos) ===")
        await nodes["nodeA"].emit(
            src="tester_control",
            dest="control",
            kind="control",
            payload={"cmd": "status", "args": {}},
        )

    # Allow scheduler + jobs + snapshots to run
    await asyncio.sleep(2.0)

    # After snapshot, ask control to load latest snapshot (replay)
    if "nodeA" in nodes and "control" in nodes["nodeA"].modules:
        print("\n=== Control: load_snapshot (latest) on nodeA ===")
        await nodes["nodeA"].emit(
            src="tester_control",
            dest="control",
            kind="control",
            payload={"cmd": "load_snapshot", "args": {"path": "latest"}},
        )
        await asyncio.sleep(0.3)

    # Run the full experiment suite once as part of core boot
    if "nodeA" in nodes and "control" in nodes["nodeA"].modules:
        print("\n=== Control: run_all_experiments on nodeA ===")
        await nodes["nodeA"].emit(
            src="tester_control",
            dest="control",
            kind="control",
            payload={"cmd": "run_all_experiments", "args": {}},
        )
        # Give experiments time to complete
        await asyncio.sleep(5.0)

    # Gather metrics summary and dict
    summary_text = None
    metrics_dict: Optional[Dict[str, Any]] = None
    if "nodeA" in nodes and "metrics" in nodes["nodeA"].modules:
        metrics_mod: MetricsModule = nodes["nodeA"].modules["metrics"]  # type: ignore
        summary_text = metrics_mod.summary()
        metrics_dict = metrics_mod.as_dict()

    # Clean shutdown of all runtimes
    for rt in nodes.values():
        rt.stop()
    await asyncio.gather(*tasks, return_exceptions=True)

    # Print metrics summary
    if summary_text:
        print(summary_text)

    # Dump JSON metrics to disk
    if metrics_dict is not None:
        path = config.get("metrics_dump_path", "metrics_dump_core.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(metrics_dict, f, indent=2)
            print(f"[cluster] metrics written to {os.path.abspath(path)}")
        except Exception as e:
            print(f"[cluster] failed to write metrics JSON: {e}")


# =========================
# Entry
# =========================

if __name__ == "__main__":
    asyncio.run(run_cluster(CONFIG))
