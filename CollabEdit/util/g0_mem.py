# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Sota Sugawara, Yukihiko Okada
# Builds on CollabEdit / MEMIT / AlphaEdit (MIT); see THIRD_PARTY_NOTICES.md.
"""Instrumentation: phase-wise peak VRAM and wall-time tracking.

Measurement only -- no effect on editing math or evaluation logic.
Records torch.cuda.max_memory_allocated / max_memory_reserved per phase
and rewrites peak_mem.json / timing.json after every phase end, so
partial records survive an OOM or crash (fail-loud friendly).
"""
import json
import time
from pathlib import Path

import torch


class G0MemTracker:
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.phases = {}
        self.timing = {}
        self._start = {}
        self.per_call = {}
        self._retries0 = {}

    @staticmethod
    def _alloc_retries():
        # cumulative count of allocator cache-reclaim retries (the "reclaim
        # path" that fires when an allocation first fails under the cap)
        if torch.cuda.is_available():
            return torch.cuda.memory_stats().get("num_alloc_retries", 0)
        return 0

    def phase_start(self, name):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self._retries0[name] = self._alloc_retries()
        self._start[name] = time.time()

    def phase_end(self, name, detail=None):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            alloc = torch.cuda.max_memory_allocated() / 2**30
            reserved = torch.cuda.max_memory_reserved() / 2**30
        else:
            alloc = reserved = 0.0
        cur = self.phases.setdefault(
            name, {"peak_alloc_gb": 0.0, "peak_reserved_gb": 0.0, "calls": 0}
        )
        cur["peak_alloc_gb"] = max(cur["peak_alloc_gb"], round(alloc, 4))
        cur["peak_reserved_gb"] = max(cur["peak_reserved_gb"], round(reserved, 4))
        cur["calls"] += 1
        retries = self._alloc_retries() - self._retries0.pop(name, 0)
        cur["alloc_retries"] = cur.get("alloc_retries", 0) + retries
        dt = time.time() - self._start.pop(name, self.t0)
        self.timing[name] = round(self.timing.get(name, 0.0) + dt, 2)
        if detail is not None:
            self.per_call.setdefault(name, []).append(
                {
                    "n": detail,
                    "alloc_gb": round(alloc, 4),
                    "reserved_gb": round(reserved, 4),
                    "seconds": round(dt, 2),
                    "alloc_retries": retries,
                }
            )
        self.save()

    def save(self):
        self.timing["total_wall_seconds"] = round(time.time() - self.t0, 2)
        with open(self.out_dir / "peak_mem.json", "w") as f:
            json.dump(self.phases, f, indent=2)
        with open(self.out_dir / "timing.json", "w") as f:
            json.dump(self.timing, f, indent=2)
        if self.per_call:
            with open(self.out_dir / "per_batch_mem.json", "w") as f:
                json.dump(self.per_call, f, indent=2)
