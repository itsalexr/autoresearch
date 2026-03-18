#!/usr/bin/env bash
set -euo pipefail

REMOTE="alex@turtle.local"
REMOTE_DIR="~/autoresearch"
LOCAL_DIR="/Users/alexreznik/code/autoresearch"
CACHE_DIR="/tmp/inductor_cache"
UV="~/.local/bin/uv"

echo "=== Syncing repo to turtle ==="
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='.venv' \
    "$LOCAL_DIR/" "$REMOTE:$REMOTE_DIR/"

echo "=== Setting up environment on turtle ==="
ssh "$REMOTE" "cd $REMOTE_DIR && $UV sync --quiet"

echo "=== Checking data cache ==="
ssh "$REMOTE" "ls ~/.cache/autoresearch/data/*.parquet 2>/dev/null | wc -l | grep -qv '^0$' \
    || (echo 'Running prepare.py...' && cd $REMOTE_DIR && $UV run prepare.py)"

echo "=== Killing any leftover GPU processes and clearing old logs ==="
ssh "$REMOTE" '
    # Kill any python/uv processes that might be holding GPU memory
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d " ")
    if [ -n "$pids" ]; then
        echo "Killing GPU processes: $pids"
        echo "$pids" | xargs -r kill -9 2>/dev/null || true
        sleep 3
    fi
    # Also kill any leftover train_gpu script processes
    pkill -9 -f "train_gpu" 2>/dev/null || true
    rm -f ~/autoresearch/run_gpu*.log
    echo "cleared"
'

echo "=== Launching GPU 0 (will compile torch.compile graph) ==="
ssh "$REMOTE" "cd $REMOTE_DIR && \
    CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=$CACHE_DIR \
    nohup $UV run train_gpu0.py > run_gpu0.log 2>&1 &"

sleep 2  # give nohup time to create/truncate the log file

echo "=== Waiting for compilation to finish (watching for first training step) ==="
until ssh "$REMOTE" "grep -q 'step' $REMOTE_DIR/run_gpu0.log 2>/dev/null"; do
    sleep 5
    echo -n "."
done
echo " Compilation done!"

echo "=== Launching GPUs 1-5 (reading from warm compile cache) ==="
for i in 1 2 3 4 5; do
    ssh "$REMOTE" "cd $REMOTE_DIR && \
        CUDA_VISIBLE_DEVICES=$i TORCHINDUCTOR_CACHE_DIR=$CACHE_DIR \
        nohup $UV run train_gpu${i}.py > run_gpu${i}.log 2>&1 &"
    echo "GPU $i launched"
done

echo "=== Waiting for all runs to complete (~5 min) ==="
ssh "$REMOTE" "while pgrep -f 'train_gpu' > /dev/null; do sleep 10; done"

echo ""
echo "=== RESULTS ==="
for i in 0 1 2 3 4 5; do
    echo -n "GPU $i (train_gpu${i}.py): "
    ssh "$REMOTE" "grep '^val_bpb:\|^peak_vram_mb:' $REMOTE_DIR/run_gpu${i}.log 2>/dev/null \
        | tr '\n' '  ' || echo 'CRASHED/NO OUTPUT'"
done
