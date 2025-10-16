import torch

### Please download the two VAE checkpoints first:
### 1. 768-v-ema-first-stage-VAE.ckpt from https://github.com/King-HAW/GMS/tree/main/SD-VAE-weights
### 2. 768-v-ema.ckpt from https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/768-v-ema.ckpt

### GMS VAE
ckpt = torch.load('768-v-ema-first-stage-VAE.ckpt', map_location='cpu', weights_only=False)
gms_vae_model = ckpt['state_dict']
print(gms_vae_model.keys())
print(len(gms_vae_model.keys()))

### SD 2.0 VAE
ckpt = torch.load('768-v-ema.ckpt', map_location='cpu', weights_only=False)
sd_model = ckpt['state_dict']
sd_vae_model = {}
for k, v in sd_model.items():
    if k.startswith('first_stage_model.'):
        sd_vae_model[k.replace('first_stage_model.', '')] = v
print(sd_vae_model.keys())
print(len(sd_vae_model.keys()))

equal_flag = True
for (k1, v1), (k2, v2) in zip(gms_vae_model.items(), sd_vae_model.items()):
    if k1 != k2:
        print(f'Key not equal: {k1} != {k2}')
        equal_flag = False
        break
    if v1.shape != v2.shape:
        print(f'Shape not equal: {k1}: {v1.shape} != {v2.shape}')
        equal_flag = False
        break
    if not torch.equal(v1, v2):
        print(f'Value not equal: {k1}')
        equal_flag = False
        break
if equal_flag:
    print('The two models are identical!')
else:
    print('The two models are different!')
