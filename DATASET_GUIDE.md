# Dataset Management for Colab

## 📊 Why Images Aren't on GitHub

### Problem with GitHub for Large Files:
```
❌ File size limits (100MB per file, repo becomes slow)
❌ Not designed for binary data
❌ Bandwidth costs for downloads
❌ Inefficient for ML workflows
```

### Solution: Download in Colab
```
✅ Colab provides 100GB storage (/tmp)
✅ Fast downloads within Google infrastructure
✅ Only download once per session
✅ Standard practice for ML projects
```

---

## 📥 How to Get Dataset in Colab

### Scenario 1: You Have Images Locally (Windows)

**Step 1: Compress locally**
```bash
# On Windows PowerShell
cd d:\Projects\VisionDocPhi-3.5\data\raw
Compress-Archive -Path spdocvqa_images -DestinationPath images.zip
```

**Step 2: Upload to Google Drive**
- Upload `images.zip` to your Google Drive
- Note the file ID from the URL: `https://drive.google.com/file/d/{FILE_ID}/view`

**Step 3: Download in Colab**
```python
!pip install -q gdown
!gdown 1YourGoogleDriveFileIDHere -O /content/VisionDocPhi-3.5/data/raw/images.zip
!cd /content/VisionDocPhi-3.5/data/raw && unzip -q images.zip
```

---

### Scenario 2: Download from Official Source

**Option A: Official SPDocVQA Dataset**

1. Go to: https://rrc.cvc.uab.es/?ch=17
2. Register with your email
3. Accept the terms and download links
4. Use `gdown` with the shared links (if available)

**Option B: Use wget/curl**
```python
# In Colab:
import os
os.chdir('/content/VisionDocPhi-3.5/data/raw')

# Download annotations first (smaller, faster)
!wget -q https://your-dataset-url/train_v1.0_withQT.json -P spdocvqa_qas/
!wget -q https://your-dataset-url/val_v1.0_withQT.json -P spdocvqa_qas/
!wget -q https://your-dataset-url/test_v1.0.json -P spdocvqa_qas/

# Download images (large, may take time)
!wget -q https://your-dataset-url/images.tar.gz
!tar -xzf images.tar.gz
```

---

### Scenario 3: Use Alternative Public Dataset

Some alternatives to consider:
- **DocVQA** - Official Microsoft dataset
- **VisualMRC** - Document understanding
- **FUNSD** - Form understanding

Each has its own download process - check their GitHub repos for scripts.

---

## 💾 Storage Structure in Colab

```
/content/VisionDocPhi-3.5/
├── data/
│   ├── raw/
│   │   ├── spdocvqa_images/
│   │   │   ├── image_1.jpg
│   │   │   ├── image_2.jpg
│   │   │   └── ...
│   │   └── spdocvqa_qas/
│   │       ├── train_v1.0_withQT.json
│   │       ├── val_v1.0_withQT.json
│   │       └── test_v1.0.json
│   └── outputs/
│       ├── predictions_zeroshot.json
│       └── results_zeroshot.json
└── ...
```

---

## ⚡ Quick Download Code (Copy-Paste)

### If images are on Google Drive:
```python
# In Colab notebook
!pip install -q gdown

# Replace with your actual file ID
FILE_ID = "1YourGoogleDriveFileIDHere"
OUTPUT_PATH = "/content/VisionDocPhi-3.5/data/raw/images.zip"

!gdown {FILE_ID} -O {OUTPUT_PATH}
!cd /content/VisionDocPhi-3.5/data/raw && unzip -q images.zip && rm images.zip

print("✅ Dataset downloaded and extracted!")
```

### Manual upload:
```python
from google.colab import files
import os

print("📁 Select images.zip from your computer:")
uploaded = files.upload()

# Extract
os.system("cd /content/VisionDocPhi-3.5/data/raw && unzip -q images.zip && rm images.zip")
print("✅ Dataset uploaded and extracted!")
```

---

## 🔑 Key Points

| Aspect | Details |
|--------|---------|
| **Dataset Location** | GitHub: NO (too large) → Colab: YES (100GB available) |
| **Annotations** | Always in repo (small JSON files) |
| **Images** | Download/upload to Colab per session |
| **Download Time** | 5-20 minutes depending on size |
| **Storage Cost** | 0 (Colab provides 100GB) |
| **Persistence** | Session-specific (deleted when Colab ends) |

---

## ❓ FAQ

**Q: What happens when my Colab session ends?**
A: Everything is deleted. You'll need to re-download next time, but it's fast.

**Q: Can I keep dataset in Google Drive permanently?**
A: Yes, but then mount Drive: `drive.mount('/content/drive')` and access from there.

**Q: How much storage do images take?**
A: Typically 2-5 GB for full DocVQA dataset. Colab has 100GB, so plenty of space.

**Q: Can I download to a specific path?**
A: Yes, use full paths: `/content/VisionDocPhi-3.5/data/raw/spdocvqa_images/`

**Q: What if download fails?**
A: Check the URL/file ID, ensure internet is stable, try again.

**Q: Do I need all three annotation files?**
A: No, download only what you need (usually just `val_v1.0_withQT.json` for testing).

---

## 🎯 Recommended Workflow

```
1️⃣  Setup project from GitHub (Step 1-3)
    ↓
2️⃣  Install dependencies (Step 3)
    ↓
3️⃣  Download dataset (Step 3.5)
    ↓
4️⃣  Test inference (Step 9) - Quick test with 5 samples
    ↓
5️⃣  Full evaluation (Step 10) - If step 9 works
    ↓
6️⃣  Download results (Step 12)
```

---

## 🚀 Next Steps

1. **Get dataset ready:**
   - Compress images locally OR
   - Use official download link

2. **Upload to Colab:**
   - Use gdown with Google Drive link OR
   - Manual upload in Colab

3. **Run notebook:**
   - Uncomment the download code in Step 3.5
   - Execute and wait for extraction

4. **Run inference:**
   - Step 9 for quick test
   - Step 10 for full evaluation

---

*Last Updated: May 19, 2026*
