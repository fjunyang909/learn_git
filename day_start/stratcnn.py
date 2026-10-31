import sys
import torch

print("当前解释器：", sys.executable)
print("PyTorch 版本：", torch.__version__)
print("GPU 可用：", torch.cuda.is_available())