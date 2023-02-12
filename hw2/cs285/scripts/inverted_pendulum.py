import subprocess
import itertools

CMD = """python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 \
--ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b <b*> -lr <r*> -rtg \
--exp_name q2_b<b*>_r<r*>"""

B = [500, 1_000, 2_000]
LR = [0.005, 0.01, 0.02]

for b, lr in itertools.product(B, LR):
    cmd = CMD.replace("<b*>", str(b)).replace("<r*>", str(lr))
    subprocess.run(cmd.split())
