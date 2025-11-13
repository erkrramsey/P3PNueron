#!/usr/bin/env python3
"""
Mesocortex Cluster v1 — Config-Driven Multi-Node Mesh + Cortex

This is the next step toward scaling:

- You define nodes, ports, and regions in CONFIG
- The launcher:
    * builds NodeRuntime instances for each node
    * attaches logger/echo/cortex7/mesh/mesocortex modules as requested
    * derives mesh peer maps from CONFIG
    * runs a stimulation pass across regions

Everything builds on the same primitives you've already tested.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple
import asyncio
import time
import uuid
import json
import math

# =========================
# CONFIGURATION
# =========================

CONFIG = {
    "listen_host": "127.0.0.1",
    "nodes": [
        {
            "id": "nodeA",
            "port": 9401,
            "modules": ["logger", "echo", "cortex7", "mesocortex", "mesh"],
        },
        {
            "id": "nodeB",
            "port": 9402,
            "modules": ["logger", "echo", "cortex7", "mesh"],
        },
        {
            "id": "nodeC",
            "port": 9403,
            "modules": ["logger", "echo", "cortex7", "mesh"],
        },
    ],
    "regions": {
        # Region spanning A + B
        "Region_AB": [
            {"id": "col_A1", "node": "nodeA", "module": "cortex7"},
            {"id": "col_B1", "node": "nodeB", "module": "cortex7"},
        ],
        # Region spanning B + C
        "Region_BC": [
            {"id": "col_B2", "node": "nodeB", "module": "cortex7"},
            {"id": "col_C1", "node": "nodeC", "module": "cortex7"},
        ],
    },
    # Which node(s) host a Mesocortex controller, and which regions they manage
    "mesocortex_controllers": [
        {
            "node": "nodeA",
            "module_name": "mesocortex_main",
            "regions": ["Region_AB", "Region_BC"],
        }
    ],
    # Simple demo stimulation pattern
    "demo": {
        "stim_cycles": 5,
        "advance_ticks": 5,
    },
}

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
# Node Runtime
# =========================

class NodeRuntime:
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
        for m in self.modules.values():
            await m.on_start()
        while self._running:
            msg = await self.queue.get()
            await self._dispatch(msg)

    def stop(self):
        self._running = False


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
# Network Mesh Module (TCP-based)
# =========================

class NetworkMeshModule(BaseModule):
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
            for s in self.outgoing[pre_idx]:
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


# =========================
# Cortex7 Module
# =========================

class Cortex7Module(BaseModule):
    def __init__(self, name: str, n_neurons: int = 4):
        super().__init__(name)
        self.core = Cortex7Core(n_neurons=n_neurons)
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
# Mesocortex Module
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


# =========================
# Cluster Launcher
# =========================

def build_peer_maps(config: Dict[str, Any]) -> Dict[str, Dict[str, tuple]]:
    """
    Returns node_id -> (peer_node_id -> (host, port)) maps.
    Each node peers with all others by default (fully connected).
    """
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


async def run_cluster(config: Dict[str, Any]):
    host = config["listen_host"]
    peers_map = build_peer_maps(config)

    # Build NodeRuntime instances
    nodes: Dict[str, NodeRuntime] = {}
    for node_def in config["nodes"]:
        nid = node_def["id"]
        rt = NodeRuntime(node_id=nid)
        nodes[nid] = rt

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
            rt.register(Cortex7Module("cortex7", n_neurons=4))
        if "mesh" in mods:
            rt.register(NetworkMeshModule("mesh", host, port, peers_map[nid]))

    # Register Mesocortex controllers
    for ctrl_def in config.get("mesocortex_controllers", []):
        nid = ctrl_def["node"]
        module_name = ctrl_def["module_name"]
        region_names = ctrl_def["regions"]

        # Build topology slice for this controller
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

    # Use first mesocortex controller in config for demo
    demo = config["demo"]
    if config.get("mesocortex_controllers"):
        ctrl = config["mesocortex_controllers"][0]
        ctrl_node = ctrl["node"]
        ctrl_module = ctrl["module_name"]

        stim_cycles = demo.get("stim_cycles", 5)
        advance_ticks = demo.get("advance_ticks", 5)

        print("\n=== Mesocortex demo: stimulate + advance + dump ===")
        # Stimulate both regions
        for i in range(stim_cycles):
            for region in ctrl["regions"]:
                await nodes[ctrl_node].emit(
                    src="tester_cluster",
                    dest=ctrl_module,
                    kind="stimulate_region",
                    payload={
                        "region": region,
                        "pattern": [{"neuron": 0, "current": 0.7}],
                    },
                )
            await asyncio.sleep(0.05)

        # Advance all columns
        await nodes[ctrl_node].emit(
            src="tester_cluster",
            dest=ctrl_module,
            kind="advance_all",
            payload={"ticks": advance_ticks},
        )
        await asyncio.sleep(0.2)

        # Dump all states
        await nodes[ctrl_node].emit(
            src="tester_cluster",
            dest=ctrl_module,
            kind="dump_all",
            payload={},
        )
        await asyncio.sleep(0.5)

    # Shut down
    for rt in nodes.values():
        rt.stop()
    await asyncio.sleep(0.05)
    for t in tasks:
        t.cancel()


# =========================
# Entry
# =========================

if __name__ == "__main__":
    asyncio.run(run_cluster(CONFIG))
