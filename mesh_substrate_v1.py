#!/usr/bin/env python3
"""
Mesh Substrate v1 — Multi-Node Local Mesh Simulator

Builds directly on Root Substrate v0:

- NodeRuntime: single-node event loop + message router
- BaseModule: pluggable modules

New in this fragment:

- LocalMeshBus: in-process "network" that connects multiple NodeRuntime instances
- MeshModule: per-node adapter that:
    * accepts local "send_remote" commands
    * forwards them via LocalMeshBus to another node
    * delivers incoming remote messages back into the local runtime

We spin up two nodes (nodeA, nodeB), each with:
- logger
- echo
- neuron1
- mesh

Then:
- nodeA sends a remote ping to nodeB.echo
- nodeA sends remote current injections to nodeB.neuron1 until it spikes

Everything still uses the same Message envelope and module API.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
import asyncio
import time
import uuid

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
    - Does NOT know about the mesh; mesh is a module (MeshModule).
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
# Example Modules (Logger, Echo, Neuron)
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
            # respond locally to whoever pinged us
            await self.emit(msg.src, "pong", {"echo": msg.payload})


class NeuronModule(BaseModule):
    """
    Toy single 'neuron' that:
    - integrates input 'current'
    - leaks over time
    - fires spike when v >= threshold

    Incoming messages:
      - kind='inject', payload={'current': float}
      - kind='reset'
    """

    def __init__(self, name: str, threshold: float = 1.0, leak: float = 0.9):
        super().__init__(name)
        self.v = 0.0
        self.threshold = threshold
        self.leak = leak

    async def on_start(self):
        await self.emit("logger", "log", {"neuron": self.name, "event": "online"})

    async def on_message(self, msg: Message):
        if msg.kind == "inject":
            current = float(msg.payload.get("current", 0.0))
            self.v = self.v * self.leak + current
            if self.v >= self.threshold:
                self.v = 0.0
                # spike event
                await self.emit("logger", "spike", {
                    "neuron": self.name,
                    "ts": msg.ts,
                    "source": msg.payload.get("_remote_from_node")  # might be None if local
                })
        elif msg.kind == "reset":
            self.v = 0.0
            await self.emit("logger", "log", {"neuron": self.name, "event": "reset"})


# =========================
# Local Mesh Bus
# =========================

class LocalMeshBus:
    """
    In-process "network fabric" that connects multiple nodes.

    It does not know about modules, only:
      - node_ids
      - each node's MeshModule

    MeshModule uses:
      bus.send_remote(from_node, to_node, src_module, dest_module, kind, payload)
    """

    def __init__(self):
        self.nodes: Dict[str, "MeshModule"] = {}

    def attach_node(self, node_id: str, mesh_module: "MeshModule"):
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already attached to mesh")
        self.nodes[node_id] = mesh_module

    async def send_remote(
        self,
        from_node: str,
        to_node: str,
        src_module: str,
        dest_module: str,
        kind: str,
        payload: Dict[str, Any],
    ):
        """
        Simulate sending a message from one node to another over the mesh.
        """
        if to_node not in self.nodes:
            # drop silently (or log)
            return

        dest_mesh = self.nodes[to_node]
        await dest_mesh.deliver_from_bus(
            from_node=from_node,
            src_module=src_module,
            dest_module=dest_module,
            kind=kind,
            payload=payload,
        )


# =========================
# Mesh Module (per node)
# =========================

class MeshModule(BaseModule):
    """
    Mesh adapter for a node.

    Local modules send messages to us with:
      dest='mesh', kind='send_remote', payload={
          'target_node': <node_id>,
          'dest_module': <module_name>,
          'kind': <remote_kind>,
          'payload': {...}
      }

    We forward that over LocalMeshBus.
    When LocalMeshBus delivers to us, we re-emit inside our node
    toward the real dest_module.
    """

    def __init__(self, name: str, bus: LocalMeshBus):
        super().__init__(name)
        self.bus = bus

    async def on_start(self):
        # Register this node with the bus
        self.bus.attach_node(self.runtime.node_id, self)
        await self.emit("logger", "log", {"mesh": "online", "node": self.runtime.node_id})

    async def on_message(self, msg: Message):
        # Handle local "send_remote" commands
        if msg.kind == "send_remote":
            target_node = msg.payload["target_node"]
            dest_module = msg.payload["dest_module"]
            remote_kind = msg.payload["remote_kind"]
            remote_payload = dict(msg.payload.get("payload", {}))

            # Optionally include metadata
            remote_payload["_remote_from_node"] = self.runtime.node_id
            remote_payload["_remote_from_module"] = msg.src

            await self.bus.send_remote(
                from_node=self.runtime.node_id,
                to_node=target_node,
                src_module=msg.src,
                dest_module=dest_module,
                kind=remote_kind,
                payload=remote_payload,
            )

    async def deliver_from_bus(
        self,
        from_node: str,
        src_module: str,
        dest_module: str,
        kind: str,
        payload: Dict[str, Any],
    ):
        """
        Called by LocalMeshBus when a remote node sends us something.
        We inject it back into our local runtime as a message from 'mesh'
        to dest_module, carrying metadata about the remote sender.
        """
        enriched_payload = dict(payload)
        enriched_payload["_remote_from_node"] = from_node
        enriched_payload["_remote_from_module"] = src_module
        await self.emit(dest_module, kind, enriched_payload)


# =========================
# Demo / Smoke Test
# =========================

async def run_demo():
    # Global in-process mesh bus
    bus = LocalMeshBus()

    # Create two nodes
    nodeA = NodeRuntime(node_id="nodeA")
    nodeB = NodeRuntime(node_id="nodeB")

    # Register modules on nodeA
    nodeA.register(LoggerModule("logger"))
    nodeA.register(EchoModule("echo"))
    nodeA.register(NeuronModule("neuron1", threshold=1.5, leak=0.8))
    nodeA.register(MeshModule("mesh", bus))

    # Register modules on nodeB
    nodeB.register(LoggerModule("logger"))
    nodeB.register(EchoModule("echo"))
    nodeB.register(NeuronModule("neuron1", threshold=1.2, leak=0.85))
    nodeB.register(MeshModule("mesh", bus))

    # Start both runtimes
    taskA = asyncio.create_task(nodeA.start())
    taskB = asyncio.create_task(nodeB.start())

    # Give modules a moment to start
    await asyncio.sleep(0.05)

    # 1) Remote ping: nodeA -> nodeB.echo
    print("\n=== Remote ping: nodeA.echo -> nodeB.echo via mesh ===")
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

    await asyncio.sleep(0.05)

    # 2) Remote neuron drive: nodeA -> nodeB.neuron1
    print("\n=== Remote neuron injections: nodeA -> nodeB.neuron1 ===")
    for i in range(5):
        await nodeA.emit(
            src="testerA",
            dest="mesh",
            kind="send_remote",
            payload={
                "target_node": "nodeB",
                "dest_module": "neuron1",
                "remote_kind": "inject",
                "payload": {"current": 0.6},
            },
        )
        await asyncio.sleep(0.01)

    # Let everything flush
    await asyncio.sleep(0.2)

    # Stop both nodes
    nodeA.stop()
    nodeB.stop()
    await asyncio.sleep(0.05)
    taskA.cancel()
    taskB.cancel()


if __name__ == "__main__":
    asyncio.run(run_demo())
