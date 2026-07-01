import torch
import torch.nn.functional as F

print("Testing RTX 5090 compatibility...")

# Apply patch
original_conv3d = F.conv3d

def patched_conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    try:
        return original_conv3d(input, weight, bias, stride, padding, dilation, groups)
    except RuntimeError as e:
        if "no kernel image" in str(e):
            print("GPU failed, using CPU fallback")
            device = input.device
            result = original_conv3d(input.cpu(), weight.cpu(), 
                                   bias.cpu() if bias else None,
                                   stride, padding, dilation, groups)
            return result.to(device)
        raise

F.conv3d = patched_conv3d

# Test
x = torch.randn(1, 1, 12, 256, 256).cuda()
w = torch.randn(32, 1, 3, 3, 3).cuda()

try:
    y = F.conv3d(x, w)
    print("SUCCESS: Conv3d works (with CPU fallback if needed)")
except Exception as e:
    print(f"FAILED: {e}")
