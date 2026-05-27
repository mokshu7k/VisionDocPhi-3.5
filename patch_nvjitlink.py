import json
import os

notebook_path = r"d:\Projects\VisionDocPhi-3.5\notebooks\DocVQA_Colab_Production.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the model initialization cell to insert before it
insert_idx = -1
for i, cell in enumerate(nb.get('cells', [])):
    if cell.get('cell_type') == 'markdown' and "Step 8: Initialize Model" in "".join(cell.get('source', [])):
        insert_idx = i
        break

if insert_idx != -1:
    fix_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ============ FIX FOR BITSANDBYTES ON COLAB ============\n",
            "# Colab's CUDA 13 environment sometimes hides the nvJitLink library\n",
            "# which bitsandbytes needs. This symlink fixes the library load error.\n",
            "import os\n",
            "print(\"🔧 Fixing bitsandbytes NVIDIA library paths...\")\n",
            "!ln -sf /usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib/libnvJitLink.so.13 /usr/lib/libnvJitLink.so.13\n",
            "!ln -sf /usr/local/lib/python3.12/dist-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12 /usr/lib/libnvJitLink.so.12 2>/dev/null || true\n",
            "print(\"✅ Fix applied!\")\n"
        ]
    }
    nb['cells'].insert(insert_idx + 1, fix_cell)
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook patched with nvJitLink fix!")
else:
    print("Could not find Step 8 cell.")
