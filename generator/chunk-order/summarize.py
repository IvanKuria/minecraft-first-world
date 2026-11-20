"""Summarize actual preserved-class execution, without treating it as video recovery."""
import json
from pathlib import Path

p = Path(__file__).resolve().parent
rows = [json.loads(line) for line in (p / "trace.jsonl").read_text().splitlines()]
compiled = [r for r in rows if r["type"] == "compile"]
result = {
    "actual_unchanged_game_classes": True,
    "headless_GL_adapter_not_rasterized": True,
    "compiled_chunks": len({r["chunk_index"] for r in compiled}),
    "first_front_facing_list": next(r for r in compiled if r["front_facing_vertices"]),
    "all_zero_vertex_lists": sum(r["vertices"] == 0 for r in compiled),
    "nonzero_lists_but_all_faces_backfacing": sum(r["vertices"] > 0 and r["front_facing_vertices"] == 0 for r in compiled),
    "rng_control": next(r for r in rows if r["type"] == "rng_control"),
    "translated_order_grid_test": next(r for r in rows if r["type"] == "translated_order_grid_test"),
}
(p / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
