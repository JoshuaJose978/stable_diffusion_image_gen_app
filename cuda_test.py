import torch
print("Is CUDA available?:", torch.cuda.is_available())
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda if torch.cuda.is_available() else "Not available")
if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name(0))