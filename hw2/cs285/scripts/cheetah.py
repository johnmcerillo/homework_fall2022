import subprocess
import itertools

CMD = """python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
--discount 0.95 -n 100 -l 2 -s 32 -b <b> -lr <r> -rtg --nn_baseline \
--exp_name q4_search_b<b>_lr<r>_rtg_nnbaseline"""

B = [10_000, 30_000, 50_000]
LR = [0.005, 0.01, 0.02]

for b, lr in itertools.product(B, LR):
    cmd = CMD.replace("<b>", str(b)).replace("<r>", str(lr))
    print(cmd)
    subprocess.run(cmd.split())
