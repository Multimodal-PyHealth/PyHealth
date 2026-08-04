#!/bin/bash
# Hard-stop all full-pretrain DDP on this box, top-down (driver -> launcher ->
# torchrun -> workers), so torchrun can't restart workers mid-kill.
for pat in run_full_pretrain_local.sh run_full_pretrain.py "torch.distributed.run" torchrun "scripts/pretrain_ssl.py"; do
  pkill -9 -f "$pat"
done
sleep 3
# anything still holding a GPU, by PID
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do
  kill -9 "$p" 2>/dev/null
done
sleep 20
echo "=== after kill ==="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "remaining workers: $(pgrep -f 'scripts/pretrain_ssl.py' | wc -l)"
