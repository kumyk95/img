import sys, os, time
log = open('/tmp/cf_result.log', 'w')
def p(msg):
    log.write(msg + '\n')
    log.flush()

p('1. Importing torch...')
import torch
p(f'  CUDA: {torch.cuda.is_available()}')

p('2. Importing CodeFormer...')
from codeformer import CodeFormer
p('  OK')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

p('3. Loading model weights...')
net = CodeFormer(
    dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
    connect_list=['32', '64', '128', '256']
).to(device)
ckpt = torch.load('/root/site-img/backend/models/codeformer.pth', map_location=device)['params_ema']
net.load_state_dict(ckpt)
net.eval()
p('  Model loaded!')

p('4. Simple inference test...')
import numpy as np
fake_input = torch.randn(1, 3, 512, 512).to(device)
t0 = time.time()
with torch.no_grad():
    output = net(fake_input, w=0.7, adain=True)[0]
t1 = time.time()
p(f'  Output shape: {output.shape}')
p(f'  Inference time: {(t1-t0)*1000:.0f} ms')
p('SUCCESS')
log.close()
