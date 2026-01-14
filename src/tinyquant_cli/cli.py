import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import typer
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

from tinyquant.quantizer import quantize

app = typer.Typer(add_completion=False, no_args_is_help=True)


DEFAULT_LLAMA_LINEAR_PATTERN = (
    r"^model\.layers\.\d+\.(?:"
    r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(?:gate_proj|up_proj|down_proj)"
    r")\.weight$"
)


def to_cpu_contig(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu").contiguous()


def resolve_model_source(model: str, revision: Optional[str]) -> Path:
    p = Path(model)
    if p.exists():
        return (p.parent if p.is_file() else p).resolve()
    local_dir = snapshot_download(repo_id=model, revision=revision)
    return Path(local_dir).resolve()


def find_weight_files(model_dir: Path) -> Tuple[List[Path], Optional[Path], Optional[Dict]]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        idx = json.loads(index_path.read_text())
        shard_names = sorted(set(idx["weight_map"].values()))
        shard_files = [(model_dir / s) for s in shard_names]
        return shard_files, index_path, idx

    st_files = sorted(model_dir.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No .safetensors found in: {model_dir}")

    ms = model_dir / "model.safetensors"
    if ms.exists():
        return [ms], None, None

    return st_files, None, None


def default_out_path(model: str, method: str) -> Path:
    p = Path(model)

    if p.exists() and p.is_file() and p.suffix == ".safetensors":
        return p.with_name(f"{p.stem}.{method}.tq{p.suffix}")

    if p.exists() and p.is_dir():
        base = p.name
    else:
        base = model.split("/")[-1]

    return Path.cwd() / f"{base}.{method}.tq"


def process_one_safetensors(
    in_path: Path,
    method: str,
    is_target_fn,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    with safe_open(str(in_path), framework="pt", device="cpu") as f:
        for name in f.keys():
            if not is_target_fn(name):
                out[name] = f.get_tensor(name)
                continue

            w = f.get_tensor(name)
            q = quantize(method_name=method, weight=w, bias=None)

            prefix = name[:-len(".weight")]

            for k, v in q.tq_tensors.items():
                out[f"{prefix}.tq_tensors.{k}"] = to_cpu_contig(v)

    return out


def build_new_index(weight_map: Dict[str, str], total_size: int) -> Dict:
    return {"weight_map": weight_map, "metadata": {"total_size": total_size}}


@app.command()
def quantize_to_safetensors(
    model: str = typer.Argument(..., help="HF repo id (e.g. unsloth/Llama-3.2-1B) or local path (dir or .safetensors)"),
    out: Optional[str] = typer.Argument(
        None,
        help="Output dir or .safetensors path. Default: <input>.<method>.tq"
    ),
    method: str = typer.Option("nf4", "--method", "-m", help="tinyquant quantization method name"),
    revision: Optional[str] = typer.Option(None, "--revision", help="HF revision (branch/tag/commit) if model is a repo id"),
    pattern: str = typer.Option(
        DEFAULT_LLAMA_LINEAR_PATTERN,
        "--pattern",
        help="Regex for selecting tensors by name (default: Llama linear weights: q/k/v/o + gate/up/down).",
    ),
):

    model_dir = resolve_model_source(model, revision)
    shard_files, index_path, _index_obj = find_weight_files(model_dir)
    typer.echo()

    out_path = Path(out) if out is not None else default_out_path(model, method)
    if out_path.suffix == ".safetensors":
        if len(shard_files) != 1:
            raise typer.BadParameter("Output .safetensors path is only supported for single-file (non-sharded) input. Use an output directory.")
        out_dir = out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_single_file = out_path
    else:
        out_dir = out_path
        out_dir.mkdir(parents=True, exist_ok=True)
        out_single_file = None

    try:
        rx = re.compile(pattern)
    except re.error as e:
        raise typer.BadParameter(f"Invalid --pattern regex: {e}")
    is_target_fn = lambda n: bool(rx.search(n))

    if len(shard_files) == 1 and (out_single_file is not None or index_path is None):
        in_file = shard_files[0]
        out_file = out_single_file if out_single_file is not None else (out_dir / "model.safetensors")

        typer.echo(f"[+] Processing single safetensors: {in_file.name}")
        out_sd = process_one_safetensors(in_file, method, is_target_fn)
        save_file(out_sd, str(out_file), metadata={"format": "pt", "tinyquant": method})
        typer.echo(f"[✓] Saved: {out_file}")
        return

    typer.echo(f"[+] Detected sharded weights: {len(shard_files)} files")
    new_weight_map: Dict[str, str] = {}
    total_size_acc = 0

    for in_file in shard_files:
        typer.echo(f"[+] Processing shard: {in_file.name}")
        out_sd = process_one_safetensors(in_file, method, is_target_fn)

        out_file = out_dir / in_file.name
        save_file(out_sd, str(out_file), metadata={"format": "pt", "tinyquant": method})

        for k in out_sd.keys():
            new_weight_map[k] = out_file.name

        shard_bytes = sum(t.numel() * t.element_size() for t in out_sd.values())
        total_size_acc += shard_bytes

        typer.echo(f"[✓] Saved shard: {out_file.name}")

    out_index = build_new_index(new_weight_map, total_size_acc)
    out_index_path = out_dir / "model.safetensors.index.json"
    out_index_path.write_text(json.dumps(out_index, indent=2))
    typer.echo(f"[✓] Saved index: {out_index_path}")


if __name__ == "__main__":
    app()
