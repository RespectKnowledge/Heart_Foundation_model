
# Fix all custom architecture files at once
cd /home/aqayyum/Scar_Segmentation_models

# Create a Python script to fix the files
cat > fix_all_architectures.py << 'EOF'
import os
import re

# Files to fix
files = [
    'nnunetv2/nets/UxLSTMEnc_3d.py',
    'nnunetv2/nets/UxLSTMEnc_2d.py',
    'nnunetv2/nets/UxLSTMBot_3d.py',
    'nnunetv2/nets/UxLSTMBot_2d.py'
]

# Common attributes that need fixing
attributes = [
    'conv_kernel_sizes',
    'pool_op_kernel_sizes', 
    'patch_size',
    'base_num_features',
    'num_blocks_per_stage',
    'feat_map_mul_on_downscale',
    'lstm_hidden_size',
    'lstm_num_layers',
    'lstm_bidirectional',
    'lstm_dropout',
    'use_attention'
]

for file_path in files:
    if os.path.exists(file_path):
        print(f"Fixing {file_path}...")
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Add helper function to the beginning of each from_plans function
        helper_code = '''
    # Compatibility helper for nnU-Net v2.2+
    def _get_param(param_name, default=None):
        # Try direct attribute
        if hasattr(configuration_manager, param_name):
            return getattr(configuration_manager, param_name)
        # Try in network_arch_kwargs
        if hasattr(configuration_manager, 'network_arch_kwargs'):
            if param_name in configuration_manager.network_arch_kwargs:
                return configuration_manager.network_arch_kwargs[param_name]
        # Try in architecture_kwargs
        if hasattr(configuration_manager, 'architecture_kwargs'):
            if param_name in configuration_manager.architecture_kwargs:
                return configuration_manager.architecture_kwargs[param_name]
        # Try in kwargs
        if hasattr(configuration_manager, 'kwargs'):
            if param_name in configuration_manager.kwargs:
                return configuration_manager.kwargs[param_name]
        return default
'''
        
        # Insert helper after function definition
        pattern = r'(def get_.*from_plans\([^)]+\):)'
        replacement = r'\1' + helper_code
        content = re.sub(pattern, replacement, content, count=1)
        
        # Replace all configuration_manager.attribute references
        for attr in attributes:
            pattern = rf'configuration_manager\.{attr}'
            replacement = f"_get_param('{attr}')"
            content = re.sub(pattern, replacement, content)
        
        # Fix specific line 518 in UxLSTMEnc_3d.py
        if 'UxLSTMEnc_3d.py' in file_path:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'num_stages = len(configuration_manager.conv_kernel_sizes)' in line:
                    lines[i] = '    num_stages = len(_get_param(\'conv_kernel_sizes\'))'
            content = '\n'.join(lines)
        
        # Fix deprecated autocast
        content = content.replace('torch.cuda.amp.autocast', "torch.amp.autocast('cuda')")
        content = content.replace('@autocast(enabled=False)', "@torch.amp.autocast('cuda', enabled=False)")
        
        # Write back
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"  ✓ Fixed {file_path}")

print("\nAll files have been updated!")
print("Now run: python3 predict_scar.py --input_dir /path/to/input --output_dir /path/to/output")
EOF
#
## Run the fix script
python3 fix_all_architectures.py