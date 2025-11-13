#!/usr/bin/env python3
"""
Network Mesh + CORTEX-7 Core v1

Builds on the previous fragments:

- NodeRuntime         — single-node event loop + message router
- BaseModule          — pluggable modules
- LoggerModule        — logging
- NetworkMeshModule   — TCP mesh between nodes

New in this fragment:

- Cortex7Core         — tiny multi-neuron spiking network with synapses + STDP-like rule
- Cortex7Module       — wraps Cortex7Core as a module that can be driven locally or remotely

Demo:
    python network_cortex7_substrate_v1.py

You should see:
- nodeA and nodeB online
- TCP mesh listening on two ports
- nodeA sending remote 'inject' commands to nodeB.cortex7
- nodeB logger printing spikes from multiple neurons in the CORTEX-7 core
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple
import asyncio
import time
import uuid
import json
import math


# =========================
# Message Envelope
# =========================

@dataclass
class Message:
    msg_id: str
    src: str          # module name on the sending node
    dest: str         # module name on the receiving node OR "broadcast"
    kind: str
    ts: float
    payload: Dict[str, Any] = field(default_factory=dict)


# =========================
# Base Module
# =========================

class BaseModule:
    """
    Base class for any module that lives in a NodeRuntime.
    """
    def __init__(self, name: str):
        self.name = name
        self.runtime: "NodeRuntime" = None  # type: ignore

    async def on_start(self):
        """Called once when the runtime starts."""
        pass

    async def on_message(self, msg: Message):
        """Handle incoming messages."""
        pass

    async def emit(self, dest: str, kind: str, payload: Dict[str, Any]):
        """Send a message via the shared runtime."""
        await self.runtime.emit(self.name, dest, kind, payload)


# =========================
# Node Runtime (Single Node)
# =========================

class NodeRuntime:
    """
    Single-node event runtime.

    - Holds modules
    - Routes messages via an asyncio.Queue
    - Does NOT know about the network; network is a module (NetworkMeshModule).
    """
    def __init__(self, node_id: Optional[str] = None):
        self.node_id = node_id or uuid.uuid4().hex
        self.modules: Dict[str, BaseModule] = {}
        self.queue: "asyncio.Queue[Message]" = asyncio.Queue()
        self._running = False

    def register(self, module: BaseModule):
        """Attach a module to this node."""
        if module.name in self.modules:
            raise ValueError(f"Module '{module.name}' already registered on node {self.node_id}")
        module.runtime = self
        self.modules[module.name] = module

    async def emit(self, src: str, dest: str, kind: str, payload: Dict[str, Any]):
        """Create and enqueue a message."""
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
        """Deliver a message to the target module(s)."""
        if msg.dest == "broadcast":
            for m in self.modules.values():
                if m.name != msg.src:
                    await m.on_message(msg)
        else:
            m = self.modules.get(msg.dest)
            if m:
                await m.on_message(msg)

    async def start(self):
        """Start all modules and run the event loop."""
        self._running = True
        # Call on_start on all modules
        for m in self.modules.values():
            await m.on_start()

        # Main dispatch loop
        while self._running:
            msg = await self.queue.get()
            await self._dispatch(msg)

    def stop(self):
        """Signal the loop to end."""
        self._running = False


# =========================
# Example Modules (Logger)
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
# Network Mesh Module (TCP-based)
# =========================

class NetworkMeshModule(BaseModule):
    """
    Mesh adapter for a node over TCP.

    Local modules send messages to us with:
      dest='mesh', kind='send_remote', payload={
          'target_node': <node_id>,
          'dest_module': <module_name>,
          'remote_kind': <remote_kind>,
          'payload': {...}
      }

    We forward that using TCP to the target node's listening address.

    On the receiving side:
      - We run an asyncio TCP server
      - For each incoming JSON-line:
          * Decode envelope
          * Inject it into the local NodeRuntime as a message from 'mesh'
    """

    def __init__(self, name: str, listen_host: str, listen_port: int, peers: Dict[str, tuple]):
        super().__init__(name)
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.peers = peers  # node_id -> (host, port)
        self.server: Optional[asyncio.AbstractServer] = None

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

                    # Envelope: {from_node, src_module, dest_module, kind, payload}
                    payload = dict(obj.get("payload", {}))
                    payload.setdefault("_remote_from_node", obj.get("from_node"))
                    payload.setdefault("_remote_from_module", obj.get("src_module"))

                    # Inject into local runtime: from module 'mesh' to dest_module
                    await self.emit(obj["dest_module"], obj["kind"], payload)
            finally:
                writer.close()
                await writer.wait_closed()

        # Start TCP server
        self.server = await asyncio.start_server(
            handle_conn, self.listen_host, self.listen_port
        )
        addr = self.server.sockets[0].getsockname()
        print(f"[{self.runtime.node_id}::mesh] listening on {addr}")
        await self.emit("logger", "log", {"mesh": "online", "addr": f"{addr[0]}:{addr[1]}"})

    async def on_message(self, msg: Message):
        # Handle local "send_remote" commands
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

            envelope = {
                "from_node": self.runtime.node_id,
                "src_module": msg.src,
                "dest_module": dest_module,
                "kind": remote_kind,
                "payload": remote_payload,
            }

            data = (json.dumps(envelope) + "\n").encode("utf-8")
            writer.write(data)
            await writer.drain()
            writer.close()
            await writer.wait_closed()


# =========================
# CORTEX-7 Style Core
# =========================

@dataclass
class Neuron:
    v: float = 0.0            # membrane potential
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
    """
    Minimal CORTEX-7 style core:

    - Discrete time t
    - N neurons with leaky integration
    - Synapses with simple STDP-like plasticity
    """

    def __init__(self, n_neurons: int):
        self.t: int = 0
        self.neurons: Dict[int, Neuron] = {i: Neuron() for i in range(n_neurons)}
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
        self.outgoing[pre].append(s)
        self.incoming[post].append(s)

    def add_current(self, n: int, current: float):
        """External input current added to neuron n."""
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
        """
        Advance by one tick:
        - leak potentials
        - check threshold
        - propagate spikes via synapses
        - update synaptic weights via STDP
        Returns list of neuron indices that spiked this tick.
        """
        # leak
        for n in self.neurons.values():
            n.v *= self.leak

        spiked: List[int] = []

        # threshold & spike
        for idx, n in self.neurons.items():
            if n.v >= n.threshold:
                n.v = 0.0
                n.spikes.append(self.t)
                n.last_spike_t = self.t
                spiked.append(idx)

        # propagate + STDP
        for pre_idx in spiked:
            for s in self.outgoing[pre_idx]:
                self.neurons[s.post].v += s.w
                self._stdp_update(s)

        self.t += 1
        return spiked

    def advance(self, ticks: int) -> List[Tuple[int, int]]:
        """
        Advance multiple ticks.
        Returns list of (neuron_index, tick) for spikes in this interval.
        """
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


# =========================
# CORTEX-7 Module
# =========================

class Cortex7Module(BaseModule):
    """
    Wraps a Cortex7Core as a runtime module.

    Incoming messages it understands:

    - kind='inject'
        payload = { 'neuron': int, 'current': float }

      Adds current to that neuron, advances 1 tick,
      emits 'cortex_spike' events for any spikes.

    - kind='advance'
        payload = { 'ticks': int }

      Advances by N ticks without external input,
      emits 'cortex_spike' events.

    - kind='dump'
        payload = {}

      Emits 'cortex_state' to logger with a compact snapshot.
    """

    def __init__(self, name: str, n_neurons: int = 4):
        super().__init__(name)
        self.core = Cortex7Core(n_neurons=n_neurons)
        # simple topology: chain 0->1, 1->2, 2->3
        self.core.set_threshold(0, 1.0)
        self.core.set_threshold(1, 1.2)
        self.core.set_threshold(2, 1.3)
        self.core.set_threshold(3, 1.5)
        self.core.link(0, 1, 0.6)
        self.core.link(1, 2, 0.5)
        self.core.link(2, 3, 0.4)

    async def on_start(self):
        await self.emit("logger", "log", {"cortex7": "online", "neurons": len(self.core.neurons)})

    async def _emit_spikes(self, spikes: List[Tuple[int, int]], remote_source: Optional[str]):
        if not spikes:
            return
        for n_idx, tick in spikes:
            await self.emit(
                "logger",
                "cortex_spike",
                {
                    "module": self.name,
                    "neuron": n_idx,
                    "tick": tick,
                    "t_global": self.core.t,
                    "source": remote_source,
                },
            )

    async def on_message(self, msg: Message):
        if msg.kind == "inject":
            neuron = int(msg.payload.get("neuron", 0))
            current = float(msg.payload.get("current", 0.0))
            remote_source = msg.payload.get("_remote_from_node")
            self.core.add_current(neuron, current)
            spikes = self.core.advance(1)
            await self._emit_spikes(spikes, remote_source)

        elif msg.kind == "advance":
            ticks = int(msg.payload.get("ticks", 1))
            remote_source = msg.payload.get("_remote_from_node")
            spikes = self.core.advance(ticks)
            await self._emit_spikes(spikes, remote_source)

        elif msg.kind == "dump":
            state = self.core.dump_state()
            # keep it compact-ish for logging
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


# =========================
# Demo / Smoke Test
# =========================

async def run_demo():
    """
    Spin up two nodes (nodeA, nodeB) in a single process, each with:

    - logger
    - echo
    - cortex7
    - mesh (NetworkMeshModule)

    nodeA drives nodeB's cortex7 remotely over TCP.
    """

    # Static peer maps for this demo
    peersA = {"nodeB": ("127.0.0.1", 9202)}
    peersB = {"nodeA": ("127.0.0.1", 9201)}

    # Create runtimes
    nodeA = NodeRuntime(node_id="nodeA")
    nodeB = NodeRuntime(node_id="nodeB")

    # Register modules on nodeA
    nodeA.register(LoggerModule("logger"))
    nodeA.register(EchoModule("echo"))
    nodeA.register(Cortex7Module("cortex7", n_neurons=4))
    nodeA.register(NetworkMeshModule("mesh", "127.0.0.1", 9201, peersA))

    # Register modules on nodeB
    nodeB.register(LoggerModule("logger"))
    nodeB.register(EchoModule("echo"))
    nodeB.register(Cortex7Module("cortex7", n_neurons=4))
    nodeB.register(NetworkMeshModule("mesh", "127.0.0.1", 9202, peersB))

    # Start both runtimes
    taskA = asyncio.create_task(nodeA.start())
    taskB = asyncio.create_task(nodeB.start())

    # Give modules a moment to start
    await asyncio.sleep(0.1)

    # 1) Remote ping: nodeA -> nodeB.echo (sanity)
    print("\n=== Remote ping: nodeA -> nodeB.echo via TCP mesh ===")
    await nodeA.emit(
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

    await asyncio.sleep(0.1)

    # 2) Remote cortex stimulation: nodeA -> nodeB.cortex7
    print("\n=== Remote cortex injections: nodeA -> nodeB.cortex7 ===")
    # Drive neuron 0 on nodeB several times to trigger propagation
    for i in range(8):
        await nodeA.emit(
            src="testerA",
            dest="mesh",
            kind="send_remote",
            payload={
                "target_node": "nodeB",
                "dest_module": "cortex7",
                "remote_kind": "inject",
                "payload": {"neuron": 0, "current": 0.7},
            },
        )
        await asyncio.sleep(0.03)

    # Ask nodeB cortex7 to dump state
    await nodeA.emit(
        src="testerA",
        dest="mesh",
        kind="send_remote",
        payload={
            "target_node": "nodeB",
            "dest_module": "cortex7",
            "remote_kind": "dump",
            "payload": {},
        },
    )

    # Let everything flush
    await asyncio.sleep(0.5)

    # Stop both nodes
    nodeA.stop()
    nodeB.stop()
    await asyncio.sleep(0.05)
    taskA.cancel()
    taskB.cancel()


if __name__ == "__main__":
    asyncio.run(run_demo())
