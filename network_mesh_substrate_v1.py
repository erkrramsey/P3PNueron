#!/usr/bin/env python3
"""
Network Mesh Substrate v1 — TCP-based P3P-lite

Builds on the previous fragments:

- NodeRuntime: single-node event loop + message router
- BaseModule: pluggable modules
- LoggerModule, EchoModule, NeuronModule: example behaviors

New in this fragment:

- NetworkMeshModule:
    * Listens on a TCP port for incoming messages from other nodes
    * Sends JSON-line messages to peers for remote delivery
    * Injects remote messages back into the local NodeRuntime

This gives you:
- Real network behavior (TCP), but still easy to test in a single script.
- A direct stepping stone to multi-machine P3P nodes.

Run the built-in demo:

    python network_mesh_substrate_v1.py

You should see:
- nodeA and nodeB loggers, echoes, neurons, and meshes coming online
- a remote ping from nodeA -> nodeB.echo
- remote current injections from nodeA -> nodeB.neuron1, causing a spike
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import asyncio
import time
import uuid
import json


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
# Example Modules
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
                    "source": msg.payload.get("_remote_from_node")
                })
        elif msg.kind == "reset":
            self.v = 0.0
            await self.emit("logger", "log", {"neuron": self.name, "event": "reset"})


# =========================
# Network Mesh Module (per node, TCP-based)
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
            # print(f"[{self.runtime.node_id}::mesh] incoming connection from {addr}")
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
                # Unknown target, drop or log
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
# Demo / Smoke Test
# =========================

async def run_demo():
    """
    Spin up two nodes (nodeA, nodeB) in a single process, each with:

    - logger
    - echo
    - neuron1
    - mesh (NetworkMeshModule)

    They talk to each other via TCP sockets on localhost.
    """

    # Static peer maps for this demo
    peersA = {"nodeB": ("127.0.0.1", 9102)}
    peersB = {"nodeA": ("127.0.0.1", 9101)}

    # Create runtimes
    nodeA = NodeRuntime(node_id="nodeA")
    nodeB = NodeRuntime(node_id="nodeB")

    # Register modules on nodeA
    nodeA.register(LoggerModule("logger"))
    nodeA.register(EchoModule("echo"))
    nodeA.register(NeuronModule("neuron1", threshold=1.5, leak=0.8))
    nodeA.register(NetworkMeshModule("mesh", "127.0.0.1", 9101, peersA))

    # Register modules on nodeB
    nodeB.register(LoggerModule("logger"))
    nodeB.register(EchoModule("echo"))
    nodeB.register(NeuronModule("neuron1", threshold=1.2, leak=0.85))
    nodeB.register(NetworkMeshModule("mesh", "127.0.0.1", 9102, peersB))

    # Start both runtimes
    taskA = asyncio.create_task(nodeA.start())
    taskB = asyncio.create_task(nodeB.start())

    # Give modules a moment to start
    await asyncio.sleep(0.1)

    # 1) Remote ping: nodeA -> nodeB.echo
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
        await asyncio.sleep(0.02)

    # Let everything flush
    await asyncio.sleep(0.3)

    # Stop both nodes
    nodeA.stop()
    nodeB.stop()
    await asyncio.sleep(0.05)
    taskA.cancel()
    taskB.cancel()


if __name__ == "__main__":
    asyncio.run(run_demo())
