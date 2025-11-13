#!/usr/bin/env python3
"""
Root Substrate v0 — Single-Node Event Runtime

This is the absolute base layer:

- NodeRuntime: in-process event loop + message router
- BaseModule: pluggable modules that send/receive messages
- LoggerModule: observes the system
- EchoModule: simple request/response behavior
- NeuronModule: tiny leaky-integrate-and-fire neuron on the same substrate

Everything else (mesh, network, cortex, mesocortex) will be built
by adding modules and swapping the transport, NOT changing this shape.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import asyncio
import time
import uuid


# =========================
# Message Envelope
# =========================

@dataclass
class Message:
    msg_id: str
    src: str
    dest: str   # module name or "broadcast"
    kind: str
    ts: float
    payload: Dict[str, Any] = field(default_factory=dict)


# =========================
# Base Module
# =========================

class BaseModule:
    """
    Base class for any module that lives in the runtime.
    Later:
      - networking module
      - CORTEX-7 simulator
      - economic layer
      - routing, etc.
    All will subclass this.
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
# Node Runtime (Root)
# =========================

class NodeRuntime:
    """
    Single-node event runtime.

    - Holds modules
    - Routes messages via an asyncio.Queue
    - Will later be extended with network I/O, persistence, etc.
    """
    def __init__(self, node_id: Optional[str] = None):
        self.node_id = node_id or uuid.uuid4().hex
        self.modules: Dict[str, BaseModule] = {}
        self.queue: "asyncio.Queue[Message]" = asyncio.Queue()
        self._running = False

    def register(self, module: BaseModule):
        """Attach a module to this node."""
        if module.name in self.modules:
            raise ValueError(f"Module '{module.name}' already registered")
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
        print(f"[{self.name}] logger online")

    async def on_message(self, msg: Message):
        print(f"[{self.name}] got {msg.kind} from {msg.src}: {msg.payload}")


class EchoModule(BaseModule):
    async def on_start(self):
        print(f"[{self.name}] echo online")
        # Announce ourselves
        await self.emit("logger", "log", {"status": "echo online"})

    async def on_message(self, msg: Message):
        if msg.kind == "ping":
            # bounce it back with a 'pong'
            await self.emit(msg.src, "pong", {"echo": msg.payload})


class NeuronModule(BaseModule):
    """
    Toy single 'neuron' that:
    - integrates input 'current'
    - leaks over time
    - fires spike when v >= threshold

    Incoming messages it understands:
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
            # simple leaky integrate-and-fire
            self.v = self.v * self.leak + current
            if self.v >= self.threshold:
                self.v = 0.0
                await self.emit("logger", "spike", {"neuron": self.name, "ts": msg.ts})
        elif msg.kind == "reset":
            self.v = 0.0
            await self.emit("logger", "log", {"neuron": self.name, "event": "reset"})


# =========================
# Demo / Smoke Test
# =========================

async def main():
    # Create runtime and modules
    rt = NodeRuntime()
    logger = LoggerModule("logger")
    echo = EchoModule("echo")
    neuron = NeuronModule("neuron1", threshold=1.5, leak=0.8)

    rt.register(logger)
    rt.register(echo)
    rt.register(neuron)

    # Start runtime in background
    task = asyncio.create_task(rt.start())

    # 1) Test echo behavior
    await rt.emit("tester", "echo", "ping", {"x": 42})

    # 2) Drive neuron with repeated injections until it spikes
    for i in range(5):
        await rt.emit("tester", "neuron1", "inject", {"current": 0.6})
        await asyncio.sleep(0.01)

    # Let messages flush
    await asyncio.sleep(0.1)

    # Stop runtime
    rt.stop()
    await asyncio.sleep(0.05)
    task.cancel()

if __name__ == "__main__":
    asyncio.run(main())
