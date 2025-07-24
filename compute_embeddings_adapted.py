import torch
import models
import numpy as np
import umap
from tqdm import tqdm
import pandas as pd
import utils as u

#CPU instead of GPU
device = torch.device('cpu')

# Model update
modelname = 'bengalese_finch1/weights/bengalese_finch1_16_logMel128_sparrow_encoder_decod2_BN_nomaxPool.stdc'

# Using hardcoded parameters
sr, nfft, sampleDur = 32000, 256, 0.5

# models.get function referred to in the original script doesn't exist, hence defining architecture explicitly
frontend = models.frontend['logMel'](sr, nfft, sampleDur, 128)
encoder = models.sparrow_encoder(1, (4, 4))
decoder = models.sparrow_decoder(16, (4, 4))
model = torch.nn.Sequential(frontend, encoder, decoder).eval().to(device)

# map_location added to use CPU instead of GPU
model.load_state_dict(torch.load(modelname, map_location=device))

# Changing the detections format to CSV files as no pickle files are available in the repo
df = pd.read_csv('bengalese_finch1/bengalese_finch1.csv')

print('computing encodings')
# Added  audiopath and sr parameters for dataset class were needed:
audiopath = 'bengalese_finch1/finch1/'
loader = torch.utils.data.DataLoader(u.Dataset(df, audiopath, sr, sampleDur), batch_size=16, shuffle=False, num_workers=16, prefetch_factor=8)

with torch.no_grad():
    encodings, idxs = [], []
    for i, (x, idx) in enumerate(tqdm(loader)):
        encoding = model[:2](x.to(device))
        idxs.extend(idx)
        encodings.extend(encoding.cpu().detach())
        # Verification of encoding dimensions:
        if i == 0:
            print(f"First batch encoding shape: {encoding.shape}")

idxs = np.array(idxs)
encodings = np.stack(encodings)

print(f"Final embeddings shape: {encodings.shape}") 

print('running umap...')
X = umap.UMAP(n_jobs=-1).fit_transform(encodings)

output_file = 'encodings_bengalese_finch1_16_adapted.npy'
np.save(output_file, {'encodings':encodings, 'idx':idxs, 'umap':X})

# Confirmation of path
print("Saved to:", output_file)
