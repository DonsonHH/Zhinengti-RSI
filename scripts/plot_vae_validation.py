"""Offline VAE validation figure in a scheduled allocation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from patrol_rsi.environment import synthetic_field
from patrol_rsi.experiment import room,split_manifest
from patrol_rsi.models import ConditionalVAEUNet

p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4);torch.manual_seed(98765)
model=ConditionalVAEUNet()
model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True)['model']);model.eval()
seeds=split_manifest()['vae_validation_field_seeds'][:3]
truth=np.stack([synthetic_field((24,24),s) for s in seeds])
target=np.stack([room(2000+i).targets for i in range(3)])
rng=np.random.default_rng(42)
mask=target&(rng.random(truth.shape)<.5)
x=torch.as_tensor(np.stack([truth*mask,mask],axis=1),dtype=torch.float32)
with torch.no_grad():estimated=model.predict(x,4)[:,0].numpy()
fig,axes=plt.subplots(3,3,figsize=(8,7.5),layout='constrained')
for row in range(3):
    for col,array in enumerate([truth[row],np.ma.masked_where(~mask[row],truth[row]),estimated[row]]):
        im=axes[row,col].imshow(array,vmin=0,vmax=1,cmap='viridis');axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
    axes[row,0].set_ylabel(f'Validation field {seeds[row]}')
for col,label in enumerate(['Simulator truth\n(evaluation only)','Available observations','Prior prediction\n(no posterior at inference)']):axes[0,col].set_title(label,fontsize=10)
fig.colorbar(im,ax=axes,shrink=.6,label='Synthetic importance [0,1], not temperature')
fig.savefig(a.output/'vae-validation.png',dpi=170);fig.savefig(a.output/'vae-validation.svg');plt.close(fig)
(a.output/'vae-visualization.json').write_text(json.dumps({'field_seeds':seeds,'mask_seed':42,'samples':4,'mse':float(((estimated-truth)**2).mean()),'purpose':'illustration_on_validation_only_not_test_selection'},indent=2))
