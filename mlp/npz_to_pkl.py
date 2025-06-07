import numpy as np
import pickle
import os
from collections import defaultdict

# Load your NPZ file
data = np.load('../demos.npz')

# Group data by demo number
demos = defaultdict(dict)
for key in data.files:
    if key.startswith('demo_'):
        parts = key.split('_', 2)
        demo_num = parts[1]  # Extract demo number
        field_name = '_'.join(parts[2:])  # Rest of the field name
        demos[demo_num][key] = data[key]  # Keep original key format

# Create demos directory
os.makedirs('demos', exist_ok=True)

# Save each demo as separate pickle file
for demo_num, demo_data in demos.items():
    filename = f'demos/demo_{demo_num}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(demo_data, f)
    print(f"Saved {filename} with {len(demo_data)} fields")

print(f"Converted {len(demos)} demonstrations to demos/ directory")
