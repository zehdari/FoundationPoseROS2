import torch
print("Torch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)

import torchvision
print("TorchVision Version:", torchvision.__version__)

import torchaudio
print("TorchAudio Version:", torchaudio.__version__)
from pytorch3d.renderer import FoVPerspectiveCameras
