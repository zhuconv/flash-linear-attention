#!/bin/bash
#SBATCH -J fla_mamba3_test
#SBATCH -o fla_mamba3_test_%j.out
#SBATCH -e fla_mamba3_test_%j.err
#SBATCH -p gh-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -A ASC26009

set -e
export PYTHONUNBUFFERED=1

module purge
module load gcc/13.2.0 cuda/12.5
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v DLMBench | tr '\n' ':' | sed 's/:$//')
source /scratch/11012/jiajunzhu2002/mamba3-env/bin/activate
export MAMBA_SSM_PATH=/work/11012/jiajunzhu2002/mamba3

nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "nvidia-smi not in PATH"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

cd /work/11012/jiajunzhu2002/flash-linear-attention

echo ""
echo "=== Quick tests (shape, fwd/bwd, cache, mask, numerical) ==="
python tests/test_mamba3.py

echo ""
echo "=== 1k-step training loss test (FLA vs native) ==="
python -c "
import sys, os
sys.path.insert(0, os.environ.get('MAMBA_SSM_PATH', '/work/11012/jiajunzhu2002/mamba3'))
from tests.test_mamba3 import TestTrainingLoss
t = TestTrainingLoss()
t.test_1k_steps_loss_close()
print('  1k-step loss test: PASS')
"

echo ""
echo "=== All tests passed ==="
