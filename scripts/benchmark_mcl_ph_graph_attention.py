#!/usr/bin/env python3
"""Short synthetic CUDA benchmark against the original chunked batch attention."""
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.modules.mcl_ph import FusionCrossAttention, FUSION_SCALE


def original(module, hidden, tokens, graph):
    count = len(hidden)
    q = module.query(module.norm2(hidden)).reshape(count, module.heads, module.dim)
    k = module.key(module.norm3(tokens)).reshape(count, module.heads, module.dim)
    v = module.value(module.norm3(tokens)).reshape(count, module.heads, module.dim)
    same = graph[:, None] == graph[None, :]
    output = hidden.new_zeros((count, module.heads, module.dim), dtype=torch.float32)
    for start in range(0, count, module.chunk):
        score = torch.einsum('ihd,jhd->hij', q[start:start + module.chunk].float(), k.float())
        score = (score * module.scale).masked_fill(~same[start:start + module.chunk][None], float('-inf'))
        output[start:start + module.chunk] = torch.einsum('hij,jhd->ihd', score.softmax(-1), v.float())
    return hidden + FUSION_SCALE * module.out(output.to(hidden.dtype).reshape(count, module.heads * module.dim))


def benchmark(report, graphs=24):
    if report.exists():
        raise FileExistsError(report)
    torch.manual_seed(44)
    module = FusionCrossAttention().cuda().eval()
    sizes = [8, 16, 24, 32, 64, 128] * (graphs // 6)
    graph = torch.repeat_interleave(torch.arange(graphs, device='cuda'),
                                    torch.tensor(sizes, device='cuda'))
    hidden = torch.randn(sum(sizes), 512, device='cuda')
    tokens = torch.randn(sum(sizes), 128, device='cuda')
    rows = []
    started = time.perf_counter()
    with torch.no_grad():
        for dtype in (torch.float32, torch.bfloat16):
            for name, call in [('original', lambda: original(module, hidden, tokens, graph)),
                               ('graph_local', lambda: module(hidden, tokens, graph))]:
                with torch.autocast('cuda', enabled=dtype != torch.float32, dtype=dtype):
                    for _ in range(5):
                        call()
                    torch.cuda.synchronize()
                    baseline = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(10):
                        call()
                    end.record()
                    torch.cuda.synchronize()
                    rows.append({'variant': name, 'dtype': str(dtype),
                                 'milliseconds_per_forward': begin.elapsed_time(end) / 10,
                                 'peak_extra_bytes': torch.cuda.max_memory_allocated() - baseline})
    result = {'scope': 'synthetic inference only', 'gpu': torch.cuda.get_device_name(),
              'graph_sizes': sizes, 'warmups': 5, 'repeats': 10,
              'results': rows, 'wall_seconds': time.perf_counter() - started}
    report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--graphs', type=int, choices=(24, 84), default=24)
    args = parser.parse_args()
    benchmark(args.report, args.graphs)
