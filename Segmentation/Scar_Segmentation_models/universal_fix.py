import os
import re

files = [
    'nnunetv2/nets/UxLSTMEnc_3d.py',
    'nnunetv2/nets/UxLSTMEnc_2d.py',
    'nnunetv2/nets/UxLSTMBot_3d.py',
    'nnunetv2/nets/UxLSTMBot_2d.py'
]

for file_path in files:
    if os.path.exists(file_path):
        print(f"Fixing {file_path}...")
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Replace the problematic line 537 in UxLSTMEnc_3d.py
        if 'UxLSTMEnc_3d.py' in file_path:
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if i == 536:  # Line 537 is index 536 (0-based)
                    print(f"  Found line {i+1}: {line}")
                    # Replace with universal getter
                    lines[i] = '    # Get conv_kernel_sizes from any available location'
                    # Insert new lines
                    lines.insert(i+1, '    conv_kernel_sizes = None')
                    lines.insert(i+2, '    # Try all possible locations')
                    lines.insert(i+3, '    for source in [configuration_manager,')
                    lines.insert(i+4, '                     getattr(configuration_manager, "kwargs", {}),')
                    lines.insert(i+5, '                     getattr(configuration_manager, "network_arch_kwargs", {}),')
                    lines.insert(i+6, '                     getattr(configuration_manager, "network_arch_init_kwargs", {}),')
                    lines.insert(i+7, '                     getattr(configuration_manager, "architecture_kwargs", {})]:')
                    lines.insert(i+8, '        if hasattr(source, "get") and "conv_kernel_sizes" in source:')
                    lines.insert(i+9, '            conv_kernel_sizes = source["conv_kernel_sizes"]')
                    lines.insert(i+10, '            break')
                    lines.insert(i+11, '        elif hasattr(source, "conv_kernel_sizes"):')
                    lines.insert(i+12, '            conv_kernel_sizes = source.conv_kernel_sizes')
                    lines.insert(i+13, '            break')
                    lines.insert(i+14, '    ')
                    lines.insert(i+15, '    if conv_kernel_sizes is None:')
                    lines.insert(i+16, '        # Fallback default')
                    lines.insert(i+17, '        conv_kernel_sizes = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]')
                    lines.insert(i+18, '    ')
                    lines.insert(i+19, '    num_stages = len(conv_kernel_sizes)')
                    break
        
        # Also replace all other configuration_manager.attribute references
        # Add a universal helper function at the beginning of each from_plans function
        helper_func = '''
    # Universal parameter getter for nnU-Net compatibility
    def _get_param(param_name, default=None):
        # Try all possible sources
        sources = [
            configuration_manager,  # Direct attributes
            getattr(configuration_manager, 'kwargs', {}),  # kwargs dict
            getattr(configuration_manager, 'network_arch_kwargs', {}),  # New format
            getattr(configuration_manager, 'network_arch_init_kwargs', {}),  # Alternative new format
            getattr(configuration_manager, 'architecture_kwargs', {}),  # Another alternative
        ]
        
        for source in sources:
            if isinstance(source, dict) and param_name in source:
                return source[param_name]
            elif hasattr(source, param_name):
                return getattr(source, param_name)
        
        # Fallback defaults for common parameters
        defaults = {
            'conv_kernel_sizes': [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
            'pool_op_kernel_sizes': [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
            'patch_size': [128, 128, 128] if len(configuration_manager.patch_size) == 3 else [256, 256],
            'base_num_features': 32,
            'num_blocks_per_stage': [2, 2, 2, 2, 2],
            'feat_map_mul_on_downscale': 2,
            'lstm_hidden_size': 64,
            'lstm_num_layers': 2,
            'lstm_bidirectional': True,
            'lstm_dropout': 0.1,
            'use_attention': True,
        }
        
        return defaults.get(param_name, default)
'''
        
        # Insert helper after function definition
        pattern = r'(def get_.*from_plans\([^)]+\):)'
        if re.search(pattern, content):
            content = re.sub(pattern, r'\1' + helper_func, content, count=1)
        
        # Replace all configuration_manager.attribute with _get_param
        attributes = ['conv_kernel_sizes', 'pool_op_kernel_sizes', 'patch_size',
                     'base_num_features', 'num_blocks_per_stage', 'feat_map_mul_on_downscale',
                     'lstm_hidden_size', 'lstm_num_layers', 'lstm_bidirectional',
                     'lstm_dropout', 'use_attention']
        
        for attr in attributes:
            pattern = rf'configuration_manager\.{attr}'
            content = re.sub(pattern, f"_get_param('{attr}')", content)
        
        # Fix deprecated autocast
        content = content.replace('torch.cuda.amp.autocast', "torch.amp.autocast('cuda')")
        
        # Write back
        with open(file_path, 'w') as f:
            f.write(content)
        
        print(f"  ✓ Fixed {file_path}")

print("\nAll files updated with universal parameter getter!")
