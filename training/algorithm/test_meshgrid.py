import sys
import torch

print("py:", sys.version, file=sys.stderr)
print("Torch:", torch.__version__, file=sys.stderr)
torch.meshgrid(torch.rand(3), torch.rand(5), indexing="ij")
print("meshgrid is OK", file=sys.stderr)
