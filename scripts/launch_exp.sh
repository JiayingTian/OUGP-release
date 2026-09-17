#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/launch_exp.sh configs/experiment.yaml [--dry-run] [--foreground]

The YAML file defines the experiment matrix, Python entrypoint, arguments,
output layout, skip result file, GPU policy, and optional summary command.
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_FILE=""
DRY_RUN=0
FOREGROUND=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --foreground) FOREGROUND=1 ;;
    -h|--help) usage; exit 0 ;;
    *)
      if [[ -z "$CONFIG_FILE" ]]; then
        CONFIG_FILE="$arg"
      else
        echo "Unexpected argument: $arg" >&2
        usage >&2
        exit 2
      fi
      ;;
  esac
done

if [[ -z "$CONFIG_FILE" ]]; then
  usage >&2
  exit 2
fi
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Config file not found: $CONFIG_FILE" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIG_PROBE="$("$PYTHON_BIN" - "$CONFIG_FILE" <<'PY'
from __future__ import annotations

import ast
import sys
from pathlib import Path


def strip_comment(line: str) -> str:
    quote = None
    out = []
    for char in line:
        if char in {"'", '"'}:
            quote = None if quote == char else char if quote is None else quote
        if char == "#" and quote is None:
            break
        out.append(char)
    return "".join(out).rstrip()


def parse_scalar(text: str):
    text = text.strip()
    if text == "":
        return ""
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if text.startswith("[") and text.endswith("]"):
        try:
            return ast.literal_eval(text)
        except Exception:
            body = text[1:-1].strip()
            return [] if not body else [item.strip().strip("'\"") for item in body.split(",")]
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return ast.literal_eval(text)
    try:
        if any(ch in text for ch in ".eE"):
            return float(text)
        return int(text)
    except ValueError:
        return text


def parse_yaml(path: str):
    root = {}
    stack = [(-1, root)]
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = strip_comment(raw)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        text = line.strip()
        if text.startswith("- "):
            raise ValueError("Top-level YAML lists are not supported by launch_exp.sh.")
        if ":" not in text:
            raise ValueError(f"Expected key/value YAML line: {raw}")
        key, value = text.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_scalar(value)
    return root


cfg = parse_yaml(sys.argv[1])
experiment = cfg.get("experiment", {})
runtime = cfg.get("runtime", {})
name = str(experiment.get("name") or Path(sys.argv[1]).stem)
out_root = str(experiment.get("out_root") or f"experiments/{name}")
screen = bool(runtime.get("screen", True))
screen_name = str(runtime.get("screen_name") or name).replace("/", "_")
print(out_root)
print(screen_name)
print("1" if screen else "0")
PY
)"
OUT_ROOT="$(printf "%s\n" "$CONFIG_PROBE" | sed -n '1p')"
SCREEN_NAME="$(printf "%s\n" "$CONFIG_PROBE" | sed -n '2p')"
SCREEN_ENABLED="$(printf "%s\n" "$CONFIG_PROBE" | sed -n '3p')"

mkdir -p "$OUT_ROOT"

if [[ "$DRY_RUN" -eq 0 && "$FOREGROUND" -eq 0 && "${LAUNCH_EXP_IN_SCREEN:-0}" != "1" && "$SCREEN_ENABLED" == "1" ]]; then
  screen -dmS "$SCREEN_NAME" bash -lc "cd $(printf '%q' "$ROOT_DIR") && LAUNCH_EXP_IN_SCREEN=1 PYTHON_BIN=$(printf '%q' "$PYTHON_BIN") bash scripts/launch_exp.sh $(printf '%q' "$CONFIG_FILE") --foreground"
  echo "Launched screen session: $SCREEN_NAME"
  echo "Experiment root: $OUT_ROOT"
  echo "Launcher log: $OUT_ROOT/launcher.log"
  exit 0
fi

mkdir -p "$OUT_ROOT/logs"
exec > >(tee -a "$OUT_ROOT/launcher.log") 2>&1

BUILD_DIR="$OUT_ROOT/.launcher"
mkdir -p "$BUILD_DIR"
JOBS_TSV="$BUILD_DIR/jobs.tsv"
SUMMARY_SH="$BUILD_DIR/summary.sh"
META_JSON="$OUT_ROOT/meta.json"
STATUS_FILE="$OUT_ROOT/status.tsv"
RUNTIME_ENV="$BUILD_DIR/runtime.env"

if ! "$PYTHON_BIN" - "$CONFIG_FILE" "$JOBS_TSV" "$SUMMARY_SH" "$META_JSON" "$RUNTIME_ENV" <<'PY'
from __future__ import annotations

import ast
import itertools
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


CONFIG_FILE, JOBS_TSV, SUMMARY_SH, META_JSON, RUNTIME_ENV = sys.argv[1:6]
PYTHON_BIN = os.environ.get("PYTHON_BIN", sys.executable)


def strip_comment(line: str) -> str:
    quote = None
    out = []
    for char in line:
        if char in {"'", '"'}:
            quote = None if quote == char else char if quote is None else quote
        if char == "#" and quote is None:
            break
        out.append(char)
    return "".join(out).rstrip()


def parse_scalar(text: str):
    text = text.strip()
    if text == "":
        return ""
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if text.startswith("[") and text.endswith("]"):
        try:
            return ast.literal_eval(text)
        except Exception:
            body = text[1:-1].strip()
            return [] if not body else [item.strip().strip("'\"") for item in body.split(",")]
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return ast.literal_eval(text)
    try:
        if any(ch in text for ch in ".eE"):
            return float(text)
        return int(text)
    except ValueError:
        return text


def parse_yaml(path: str) -> dict:
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = strip_comment(raw)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        text = line.strip()
        if text.startswith("- "):
            raise ValueError("Only inline lists like [0, 1, 2] are supported.")
        if ":" not in text:
            raise ValueError(f"Expected key/value YAML line: {raw}")
        key, value = text.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_scalar(value)
    return root


def run_text(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def render(value, context: dict[str, object]):
    if isinstance(value, str):
        return value.format(**context)
    if isinstance(value, list):
        return [render(item, context) for item in value]
    if isinstance(value, dict):
        return {key: render(item, context) for key, item in value.items()}
    return value


def ensure_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def command_from_args(entrypoint: str, args: dict, context: dict[str, object], extra_args: list | None = None) -> list[str]:
    cmd = [PYTHON_BIN, entrypoint]
    for key, raw_value in args.items():
        value = render(raw_value, context)
        flag = "--" + str(key).replace("_", "-")
        if value is None or value is False:
            continue
        if value is True:
            cmd.append(flag)
        elif isinstance(value, list):
            if value:
                cmd.append(flag)
                cmd.extend(fmt(item) for item in value)
        else:
            cmd.extend([flag, fmt(value)])
    for item in ensure_list(extra_args):
        cmd.append(fmt(render(item, context)))
    return cmd


cfg = parse_yaml(CONFIG_FILE)
experiment = cfg.get("experiment", {})
runtime = cfg.get("runtime", {})
matrix = cfg.get("matrix", {})
paths = cfg.get("paths", {})
job_cfg = cfg.get("job", {})
summary = cfg.get("summary", {})

name = str(experiment.get("name") or Path(CONFIG_FILE).stem)
out_root = str(experiment.get("out_root") or f"experiments/{name}")
entrypoint = str(job_cfg.get("entrypoint"))
if not entrypoint or entrypoint == "None":
    raise ValueError("YAML must set job.entrypoint.")
args = job_cfg.get("args", {})
if not isinstance(args, dict):
    raise TypeError("job.args must be a mapping.")

matrix_values: dict[str, list] = {key: ensure_list(value) for key, value in matrix.items()}
if not matrix_values:
    matrix_values = {"job": [name]}
keys = list(matrix_values)
combinations = [dict(zip(keys, values)) for values in itertools.product(*(matrix_values[key] for key in keys))]

out_template = str(paths.get("out_dir_template") or f"{out_root}/seed{{seed}}")
log_template = str(paths.get("log_file_template") or f"{out_root}/logs/{{job_id}}.log")
result_template = str(paths.get("result_file") or paths.get("result_file_template") or "result.json")

rows = []
for index, combo in enumerate(combinations):
    context = {
        "experiment": name,
        "out_root": out_root,
        "index": index,
        **combo,
    }
    for context_key in ("dataset", "backbone"):
        if context_key not in context and context_key in args:
            context[context_key] = render(args[context_key], context)
    if "route" not in context and "node_sample_mode" in args:
        context["route"] = render(args["node_sample_mode"], context)
    if "dataset" in context:
        context["dataset_key"] = str(context["dataset"]).replace("-", "_")
    if "route" in context:
        route = str(context["route"])
        context["route_dir"] = route if route.endswith("_subgraph") else f"{route}_subgraph"
    if "seed" in context:
        context["seed"] = str(context["seed"])
    job_id_template = str(paths.get("job_id_template") or "{index}")
    job_id = render(job_id_template, context)
    context["job_id"] = job_id
    out_dir = render(out_template, context)
    context["out_dir"] = out_dir
    log_file = render(log_template, context)
    result_file = render(result_template, context)
    if not os.path.isabs(result_file):
        result_file = str(Path(out_dir) / result_file)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    cmd = command_from_args(entrypoint, args, context, job_cfg.get("extra_args"))
    command_text = " ".join(shlex.quote(part) for part in cmd)
    Path(out_dir, "command.txt").write_text(command_text + "\n", encoding="utf-8")
    rows.append(
        {
            "job_id": str(job_id),
            "out_dir": str(out_dir),
            "log_file": str(log_file),
            "result_file": str(result_file),
            "command": command_text,
            "context": context,
        }
    )

with Path(JOBS_TSV).open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(
            "\t".join(
                [
                    row["job_id"],
                    row["out_dir"],
                    row["log_file"],
                    row["result_file"],
                    row["command"],
                ]
            )
            + "\n"
        )

summary_entrypoint = summary.get("entrypoint")
if summary_entrypoint:
    summary_args = summary.get("args", {})
    summary_context = {"experiment": name, "out_root": out_root}
    summary_cmd = command_from_args(str(summary_entrypoint), summary_args, summary_context, summary.get("extra_args"))
    summary_text = " ".join(shlex.quote(part) for part in summary_cmd)
    Path(SUMMARY_SH).write_text(
        "#!/usr/bin/env bash\nset -uo pipefail\n" + summary_text + "\n",
        encoding="utf-8",
    )
    os.chmod(SUMMARY_SH, 0o755)
else:
    Path(SUMMARY_SH).write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    os.chmod(SUMMARY_SH, 0o755)

meta = {
    "experiment": name,
    "config_file": CONFIG_FILE,
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "out_root": out_root,
    "python_bin": PYTHON_BIN,
    "python_version": platform.python_version(),
    "platform": platform.platform(),
    "git_commit": run_text(["git", "rev-parse", "HEAD"]),
    "git_status_short": run_text(["git", "status", "--short"]),
    "nvidia_smi": run_text(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader"]),
    "config": cfg,
    "jobs": [
        {
            "job_id": row["job_id"],
            "out_dir": row["out_dir"],
            "log_file": row["log_file"],
            "result_file": row["result_file"],
            "command": row["command"],
            "context": row["context"],
        }
        for row in rows
    ],
}
Path(META_JSON).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
gpu_ids = runtime.get("gpus", [])
if not isinstance(gpu_ids, list):
    gpu_ids = [gpu_ids]
wait_for_result_roots = runtime.get("wait_for_result_roots", [])
if not isinstance(wait_for_result_roots, list):
    wait_for_result_roots = [wait_for_result_roots]
wait_for_paths = runtime.get("wait_for_paths", [])
if not isinstance(wait_for_paths, list):
    wait_for_paths = [wait_for_paths]
Path(RUNTIME_ENV).write_text(
    "\n".join(
        [
            f"FREE_MEM_MIB={shlex.quote(str(runtime.get('free_mem_mib', 6000)))}",
            f"MAX_PARALLEL={shlex.quote(str(runtime.get('max_parallel', 1)))}",
            f"POLL_SECONDS={shlex.quote(str(runtime.get('poll_seconds', 120)))}",
            f"GPU_IDS={shlex.quote(' '.join(str(item) for item in gpu_ids))}",
            f"WAIT_FOR_RESULT_ROOTS={shlex.quote(' '.join(str(item) for item in wait_for_result_roots))}",
            f"WAIT_FOR_PATHS={shlex.quote(' '.join(str(item) for item in wait_for_paths))}",
        ]
    )
    + "\n",
    encoding="utf-8",
)
print(f"Generated {len(rows)} jobs under {out_root}")
PY
then
  echo "Failed to parse config or generate jobs: $CONFIG_FILE" >&2
  exit 1
fi

source "$RUNTIME_ENV"
GPU_LOCK_DIR="${TMPDIR:-/tmp}/ougp_launch_exp_gpu_locks"
mkdir -p "$GPU_LOCK_DIR"

wait_for_dependency_root() {
  local root="$1"
  local jobs_file="$root/.launcher/jobs.tsv"
  local dependency_status="$root/status.tsv"
  while true; do
    if [[ ! -f "$jobs_file" ]]; then
      echo "[$(date '+%F %T')] dependency jobs file not ready: $jobs_file; waiting ${POLL_SECONDS}s"
      sleep "$POLL_SECONDS"
      continue
    fi
    local expected completed failed
    expected="$(( $(wc -l < "$jobs_file") ))"
    completed=0
    failed=0
    if [[ -f "$dependency_status" ]]; then
      completed="$(awk -F '\t' 'NR > 1 && ($4 == "complete" || $4 == "skipped_existing") { count++ } END { print count + 0 }' "$dependency_status")"
      failed="$(awk -F '\t' 'NR > 1 && $4 == "failed" { count++ } END { print count + 0 }' "$dependency_status")"
    fi
    if [[ "$failed" -gt 0 ]]; then
      echo "Dependency experiment has $failed failed job(s): $root" >&2
      exit 1
    fi
    if [[ "$completed" -ge "$expected" ]]; then
      echo "[$(date '+%F %T')] dependency complete: $root ($completed/$expected)"
      return 0
    fi
    echo "[$(date '+%F %T')] waiting for dependency: $root ($completed/$expected complete)"
    sleep "$POLL_SECONDS"
  done
}

wait_for_paths() {
  local missing path
  while true; do
    missing=0
    for path in $WAIT_FOR_PATHS; do
      if [[ ! -f "$path" ]]; then
        echo "[$(date '+%F %T')] waiting for required input: $path"
        missing=1
      fi
    done
    if [[ "$missing" -eq 0 ]]; then
      return 0
    fi
    sleep "$POLL_SECONDS"
  done
}

if [[ "$DRY_RUN" -eq 0 ]]; then
  wait_for_paths
  for dependency_root in $WAIT_FOR_RESULT_ROOTS; do
    wait_for_dependency_root "$dependency_root"
  done
fi

if [[ ! -f "$STATUS_FILE" ]]; then
  printf "time\tjob_id\tgpu\tstatus\tout_dir\tlog\tresult\n" > "$STATUS_FILE"
fi

running_jobs=()

gpu_memory_table() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits
}

gpu_allowed() {
  local gpu="$1"
  if [[ -z "$GPU_IDS" ]]; then
    return 0
  fi
  local allowed
  for allowed in $GPU_IDS; do
    if [[ "$allowed" == "$gpu" ]]; then
      return 0
    fi
  done
  return 1
}

gpu_is_reserved() {
  local gpu="$1"
  local entry entry_gpu
  for entry in "${running_jobs[@]:-}"; do
    entry_gpu="${entry#*:}"
    if [[ "$entry_gpu" == "$gpu" ]]; then
      return 0
    fi
  done
  return 1
}

process_start_time() {
  local pid="$1"
  awk '{print $22}' "/proc/$pid/stat" 2>/dev/null
}

gpu_is_globally_locked() {
  local gpu="$1"
  local lock_dir="$GPU_LOCK_DIR/gpu_$gpu"
  local owner_file="$lock_dir/owner"
  local owner_pid owner_start current_start

  [[ -d "$lock_dir" ]] || return 1

  # Legacy empty lock directories cannot be attributed safely. They remain
  # locked until manually removed; all newly-created locks carry an owner.
  [[ -f "$owner_file" ]] || return 0
  if ! read -r owner_pid owner_start < "$owner_file"; then
    return 0
  fi
  current_start="$(process_start_time "$owner_pid")"
  if [[ -n "$current_start" && "$current_start" == "$owner_start" ]]; then
    return 0
  fi

  # The launcher that owned this lock no longer exists. Reclaim the stale
  # directory so a new launcher can use the otherwise-free GPU.
  rm -f "$owner_file"
  rmdir "$lock_dir" 2>/dev/null || true
  [[ -d "$lock_dir" ]]
}

acquire_gpu_lock() {
  local gpu="$1"
  local lock_dir="$GPU_LOCK_DIR/gpu_$gpu"
  local owner_start

  mkdir "$lock_dir" 2>/dev/null || return 1
  owner_start="$(process_start_time "$$")"
  if [[ -z "$owner_start" ]] || ! printf '%s %s\n' "$$" "$owner_start" > "$lock_dir/owner"; then
    rm -f "$lock_dir/owner"
    rmdir "$lock_dir" 2>/dev/null || true
    return 1
  fi
}

release_gpu_lock() {
  local gpu="$1"
  local lock_dir="$GPU_LOCK_DIR/gpu_$gpu"
  local owner_file="$lock_dir/owner"
  local owner_pid owner_start current_start

  [[ -d "$lock_dir" ]] || return 0
  [[ -f "$owner_file" ]] || return 0
  read -r owner_pid owner_start < "$owner_file" || return 0
  current_start="$(process_start_time "$$")"
  if [[ "$owner_pid" != "$$" || -z "$current_start" || "$owner_start" != "$current_start" ]]; then
    return 0
  fi
  rm -f "$owner_file"
  rmdir "$lock_dir" 2>/dev/null || true
}

find_free_gpu() {
  gpu_memory_table | while IFS=',' read -r raw_idx raw_free; do
    idx="$(echo "$raw_idx" | tr -d ' ')"
    free="$(echo "$raw_free" | tr -d ' ')"
    if [[ -n "$idx" && -n "$free" && "$free" -ge "$FREE_MEM_MIB" ]] && gpu_allowed "$idx" && ! gpu_is_reserved "$idx" && ! gpu_is_globally_locked "$idx"; then
      echo "$idx"
      return 0
    fi
  done
}

reap_finished_jobs() {
  if [[ "${#running_jobs[@]}" -eq 0 ]]; then
    return 0
  fi
  local alive=()
  local entry pid
  for entry in "${running_jobs[@]}"; do
    pid="${entry%%:*}"
    if kill -0 "$pid" 2>/dev/null; then
      alive+=("$entry")
    else
      wait "$pid" || true
      release_gpu_lock "${entry#*:}"
    fi
  done
  running_jobs=("${alive[@]}")
}

wait_for_slot() {
  while true; do
    reap_finished_jobs
    if [[ "${#running_jobs[@]}" -lt "$MAX_PARALLEL" ]]; then
      return 0
    fi
    echo "[$(date '+%F %T')] waiting for job slot; active=${#running_jobs[@]}"
    sleep "$POLL_SECONDS"
  done
}

wait_for_gpu() {
  local gpu
  while true; do
    reap_finished_jobs
    gpu="$(find_free_gpu || true)"
    if [[ -n "$gpu" ]] && acquire_gpu_lock "$gpu"; then
      echo "$gpu"
      return 0
    fi
    echo "[$(date '+%F %T')] no eligible unlocked GPU with at least ${FREE_MEM_MIB} MiB free; waiting ${POLL_SECONDS}s" >&2
    sleep "$POLL_SECONDS"
  done
}

echo "[$(date '+%F %T')] Generic experiment launcher started."
echo "CONFIG=$CONFIG_FILE OUT_ROOT=$OUT_ROOT MAX_PARALLEL=$MAX_PARALLEL FREE_MEM_MIB=$FREE_MEM_MIB GPU_IDS=${GPU_IDS:-all}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run: generated jobs only."
  cat "$JOBS_TSV"
  exit 0
fi

while IFS=$'\t' read -r job_id out_dir log_file result_file command_text; do
  if [[ -f "$result_file" ]]; then
    echo "[$(date '+%F %T')] skip job=$job_id; found $result_file"
    printf "%s\t%s\t%s\tskipped_existing\t%s\t%s\t%s\n" "$(date '+%F %T')" "$job_id" "-" "$out_dir" "$log_file" "$result_file" >> "$STATUS_FILE"
    continue
  fi
  wait_for_slot
  gpu="$(wait_for_gpu)"
  echo "[$(date '+%F %T')] launch job=$job_id gpu=$gpu"
  (
    set +e
    export PYTHONPATH=".:src:scripts${PYTHONPATH:+:$PYTHONPATH}"
    CUDA_VISIBLE_DEVICES="$gpu" bash -lc "$command_text" > "$log_file" 2>&1
    status="$?"
    if [[ "$status" -eq 0 ]]; then
      printf "%s\t%s\t%s\tcomplete\t%s\t%s\t%s\n" "$(date '+%F %T')" "$job_id" "$gpu" "$out_dir" "$log_file" "$result_file" >> "$STATUS_FILE"
    else
      printf "%s\t%s\t%s\tfailed_%s\t%s\t%s\t%s\n" "$(date '+%F %T')" "$job_id" "$gpu" "$status" "$out_dir" "$log_file" "$result_file" >> "$STATUS_FILE"
    fi
    exit "$status"
  ) &
  running_jobs+=("$!:$gpu")
  sleep 5
done < "$JOBS_TSV"

echo "[$(date '+%F %T')] all jobs launched; waiting."
while [[ "${#running_jobs[@]}" -gt 0 ]]; do
  reap_finished_jobs
  sleep "$POLL_SECONDS"
done

if [[ -s "$SUMMARY_SH" ]]; then
  echo "[$(date '+%F %T')] running summary command."
  bash "$SUMMARY_SH" > "$OUT_ROOT/summary.log" 2>&1
fi
echo "[$(date '+%F %T')] Generic experiment launcher finished."
