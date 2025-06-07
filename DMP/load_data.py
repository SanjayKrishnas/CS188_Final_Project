import numpy as np
import sys
from collections import defaultdict

def reconstruct_from_npz(npz_path):
    try:
        flat_data = np.load(npz_path)
        demos = defaultdict(lambda: {})  
        for key in flat_data.files:
            parts = key.split('_', 2)  
            if len(parts) < 3:
                print(f"Skipping malformed key: {key}")
                continue
            demo_id = f"{parts[0]}_{parts[1]}" 
            field_name = parts[2]  
            demos[demo_id][field_name] = flat_data[key]

        return demos
    except Exception as e:
        print(f"Error reconstructing data: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python reconstruct_npz.py <file.npz>")
    else:
        reconstruct_from_npz(sys.argv[1])