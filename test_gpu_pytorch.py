import torch

flag_gpu = torch.cuda.is_available()

print(f'flag_gpu: {flag_gpu}')

gpu_name = torch.cuda.get_device_name(0)

print(f'gpu_name: {gpu_name}')
