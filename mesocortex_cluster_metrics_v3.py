#!/usr/bin/env python3
"""
Mesocortex Cluster + Metrics v3 (Region-Aware + JSON Dump, Clean Shutdown)

- Config-driven multi-node mesh + cortex + mesocortex
- Central MetricsModule that aggregates spikes and state
- Region-aware spike counts (Mesocortex tags injects with region)
- Clean shutdown via shutdown sentinel
- Writes metrics_dump.json at the end
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple
import asyncio
import time
import uuid
import json
import math
import os

# =========================
# CONFIGURATION
# =========================

CONFIG = {
    "listen_host": "127.0.0.1",
    "nodes": [
        {
            "id": "nodeA",
            "port": 9501,
            "modules": ["logger", "echo", "cortex7", "mesocortex", "mesh", "metrics"],
        },
        {
            "id": "nodeB",
            "port": 9502,
            "modules": ["logger", "echo", "cortex7", "mesh"],
        },
        {
            "id": "nodeC",
            "port": 9503,
            "modules": ["logger", "echo", "cortex7", "mesh"],
        },
    ],
    "regions": {
        "Region_AB": [
            {"id": "col_A1", "node": "nodeA", "module": "cortex7"},
            {"id": "col_B1", "node": "nodeB", "module": "cortex7"},
        ],
        "Region_BC": [
            {"id": "col_B2", "node": "nodeB", "module": "cortex7"},
            {"id": "col_C1", "node": "nodeC", "module": "cortex7"},
        ],
    },
    "mesocortex_controllers": [
        {
            "node": "nodeA",
            "module_name": "mesocortex_main",
            "regions": ["Region_AB", "Region_BC"],
        }
    ],
    "demo": {
        "stim_cycles": 5,
        "advance_ticks": 5,
    },
    # where to write JSON metrics
    "metrics_dump_path": "metrics_dump.json",
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

    Receives:

    - 'metrics_spike':
        payload = {
          'node': str,
          'module': str,
          'neuron': int,
          'tick': int,
          't_global': int,
          'region': Optional[str]
        }

    - 'metrics_state':
        payload = {
          'node': str,
          'module': str,
          'state': {...}
        }
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
        """
        Structured form suitable for JSON dumping.
        """
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
# Network Mesh Module (TCP-based)
# =========================

class NetworkMeshModule(BaseModule):
    def __init__(self, name: str, listen_host: str, listen_port: int, peers: Dict[str, tuple]):
        super().__init__(name)
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.peers = peers
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
# CORTEX-7 Core
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
# Cortex7 Module (metrics + region-aware)
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
            if "metrics" in self.runtime.modules:
                mp = {
                    "node": node_id,
                    "module": self.name,
                    "neuron": n_idx,
                    "tick": tick,
                    "t_global": self.core.t,
                    "region": region,
                }
                await self.emit("metrics", "metrics_spike", mp)

    async def on_message(self, msg: Message):
        # Region context is propagated via msg.payload.get("region")
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
            if "metrics" in self.runtime.modules:
                await self.emit(
                    "metrics",
                    "metrics_state",
                    {
                        "node": self.runtime.node_id,
                        "module": self.name,
                        "state": compact,
                    },
                )


# =========================
# Mesocortex Module (now tagging region)
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

            # Tag inject payloads with region
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
            # For now, advance events are not tied to a specific region
            # (to avoid double-counting spikes when a column is in multiple regions).
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
        if "metrics" in mods:
            rt.register(MetricsModule("metrics"))

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

    demo = config["demo"]
    if config.get("mesocortex_controllers"):
        ctrl = config["mesocortex_controllers"][0]
        ctrl_node = ctrl["node"]
        ctrl_module = ctrl["module_name"]

        stim_cycles = demo.get("stim_cycles", 5)
        advance_ticks = demo.get("advance_ticks", 5)

        print("\n=== Mesocortex demo: stimulate + advance + dump ===")
        # Stimulate all regions this controller manages
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

        # Advance all
        await nodes[ctrl_node].emit(
            src="tester_cluster",
            dest=ctrl_module,
            kind="advance_all",
            payload={"ticks": advance_ticks},
        )
        await asyncio.sleep(0.2)

        # Dump all
        await nodes[ctrl_node].emit(
            src="tester_cluster",
            dest=ctrl_module,
            kind="dump_all",
            payload={},
        )
        await asyncio.sleep(0.5)

    # Get metrics summary and dict before shutdown
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
        path = config.get("metrics_dump_path", "metrics_dump.json")
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
# Metrics Module
# =========================

class MetricsModule(BaseModule):
    """
    Aggregates global metrics:

    - spike_counts[(node, module, neuron)] -> int
    - region_spike_counts[region] -> int
    - last_t[node] -> int

    Receives:

    - 'metrics_spike':
        payload = {
          'node': str,
          'module': str,
          'neuron': int,
          'tick': int,
          't_global': int,
          'region': Optional[str]
        }

    - 'metrics_state':
        payload = {
          'node': str,
          'module': str,
          'state': {...}
        }
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
            tick = int(msg.payload["tick"])
            t_global = int(msg.payload["t_global"])
            region = msg.payload.get("region")

            key = (node, module, neuron)
            self.spike_counts[key] = self.spike_counts.get(key, 0) + 1
            self.last_t[node] = max(self.last_t.get(node, 0), t_global)

            if region:
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


# =========================
# Network Mesh Module (TCP-based)
# =========================

class NetworkMeshModule(BaseModule):
    def __init__(self, name: str, listen_host: str, listen_port: int, peers: Dict[str, tuple]):
        super().__init__(name)
        self.listen_host = listen_host
        self.listen_port = listen_port
        self.peers = peers
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
# CORTEX-7 Core
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
# Cortex7 Module (metrics-aware)
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
        node_id = self.runtime.node_id
        for n_idx, tick in spikes:
            payload = {
                "module": self.name,
                "neuron": n_idx,
                "tick": tick,
                "t_global": self.core.t,
                "source": remote_source,
            }
            await self.emit("logger", "cortex_spike", payload)
            # If metrics module exists locally, notify it too
            if "metrics" in self.runtime.modules:
                mp = {
                    "node": node_id,
                    "module": self.name,
                    "neuron": n_idx,
                    "tick": tick,
                    "t_global": self.core.t,
                    "region": None,  # mesocortex will add region context in its own events
                }
                await self.emit("metrics", "metrics_spike", mp)

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
            if "metrics" in self.runtime.modules:
                await self.emit(
                    "metrics",
                    "metrics_state",
                    {
                        "node": self.runtime.node_id,
                        "module": self.name,
                        "state": compact,
                    },
                )


# =========================
# Mesocortex Module (metrics-aware)
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

            # If metrics exists, mark region-level stim (we piggy-back off cortex spikes,
            # but can also send synthetic "region" spike events if you want)
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
        if "metrics" in mods:
            rt.register(MetricsModule("metrics"))

    # Register Mesocortex controllers
    for ctrl_def in config.get("mesocortex_controllers", []):
        nid = ctrl_def["node"]
        module_name = ctrl_def["module_name"]
        region_names = ctrl_def["regions"]
        topo_slice: Dict[str, List[Dict[str, Any]]] = {}
        for rname in region_names:
            topo_slice[rname] = config["regions"][rname]
        nodes[nid].register(MesocortexModule(module_name, topo_slice))

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

    demo = config["demo"]
    if config.get("mesocortex_controllers"):
        ctrl = config["mesocortex_controllers"][0]
        ctrl_node = ctrl["node"]
        ctrl_module = ctrl["module_name"]

        stim_cycles = demo.get("stim_cycles", 5)
        advance_ticks = demo.get("advance_ticks", 5)

        print("\n=== Mesocortex demo: stimulate + advance + dump ===")
        # Stimulate all regions this controller manages
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

        # Advance all
        await nodes[ctrl_node].emit(
            src="tester_cluster",
            dest=ctrl_module,
            kind="advance_all",
            payload={"ticks": advance_ticks},
        )
        await asyncio.sleep(0.2)

        # Dump all
        await nodes[ctrl_node].emit(
            src="tester_cluster",
            dest=ctrl_module,
            kind="dump_all",
            payload={},
        )
        await asyncio.sleep(0.5)

    # Grab metrics summary if present on nodeA
    summary_text = None
    if "nodeA" in nodes and "metrics" in nodes["nodeA"].modules:
        metrics_mod: MetricsModule = nodes["nodeA"].modules["metrics"]  # type: ignore
        summary_text = metrics_mod.summary()

    # Shut down
    for rt in nodes.values():
        rt.stop()
    await asyncio.sleep(0.05)
    for t in tasks:
        t.cancel()

    if summary_text:
        print(summary_text)


# =========================
# Entry
# =========================

if __name__ == "__main__":
    asyncio.run(run_cluster(CONFIG))
