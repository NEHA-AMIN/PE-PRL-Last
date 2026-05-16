# Google Colab Setup Instructions for Informer2020 Baseline Experiments

## Overview
This guide will help you run the baseline Phase 1 experiments on Google Colab using your uploaded Informer2020 folder.

---

## Prerequisites
- Google account with Google Drive
- Upload the entire `Informer2020` folder to your Google Drive at: `/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1/`

---

## Step-by-Step Instructions

### Step 1: Create New Colab Notebook
1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook: `File → New notebook`
3. Enable GPU: `Runtime → Change runtime type → Hardware accelerator → GPU → Save`

---

### Step 2: Mount Google Drive
Run this in the first cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Click the link, authorize, and paste the code back.

---

### Step 3: Verify Folder Structure
Run this to check your folder exists:

```bash
%%bash
ls -la /content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1/Informer2020/
```

You should see:
- `baseline_phase1.sh`
- `setup_dataset.sh`
- `main_informer.py`
- `data/`, `exp/`, `models/`, `utils/` folders

---

### Step 4: Install Dependencies
```bash
!pip install matplotlib==3.1.1 numpy==1.19.4 pandas==0.25.1 scikit-learn==0.21.3 torch==1.8.0
```

---

### Step 5: Download ETTh1 Dataset
```bash
%%bash
cd /content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1/Informer2020
chmod +x setup_dataset.sh
./setup_dataset.sh
```

Expected output:
```
✓ ETTh1.csv downloaded successfully
File size: 1.5M
Lines: 17421
```

---

### Step 6: Run Baseline Phase 1 Experiments
```bash
%%bash
cd /content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1/Informer2020
chmod +x baseline_phase1.sh
./baseline_phase1.sh
```

---

### Step 7: Monitor Progress
The script will:
- Run 2 experiments (pred_len 96 and 192)
- Save logs to `/content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1/logs/baseline_phase1/`
- Display progress in real-time
- Show summary at the end

Expected runtime: **1-2 hours total** (~30-60 min per experiment)

---

### Step 8: View Results
After completion, check the master log:

```bash
%%bash
cat /content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1/logs/baseline_phase1/master_run.log
```

Or view individual run logs:

```bash
%%bash
cat /content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1/logs/baseline_phase1/baseline_ph1_ETTh1_pred96_seed2021.log
```

---

## Troubleshooting

### Issue: "Cannot cd to directory"
**Solution:** Check your folder path. Adjust `PROJECT_ROOT` in `baseline_phase1.sh` if needed.

### Issue: "No module named 'exp'"
**Solution:** Make sure you're running from the Informer2020 directory:
```bash
cd /content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1/Informer2020
```

### Issue: "ETTh1.csv not found"
**Solution:** Run the setup_dataset.sh script again:
```bash
cd /content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1/Informer2020
./setup_dataset.sh
```

### Issue: Colab disconnects
**Solution:** The script has resume capability. Just re-run Step 6. It will skip completed experiments.

---

## File Structure After Setup

```
/content/drive/MyDrive/Dist-Abl-PRL-All-Exs-ETTH1/
├── Informer2020/
│   ├── baseline_phase1.sh          ← Main experiment script
│   ├── setup_dataset.sh            ← Dataset download script
│   ├── main_informer.py
│   ├── data/
│   │   └── ETT/
│   │       └── ETTh1.csv           ← Downloaded dataset
│   ├── exp/
│   ├── models/
│   └── utils/
├── logs/
│   └── baseline_phase1/
│       ├── master_run.log          ← Summary log
│       ├── baseline_ph1_ETTh1_pred96_seed2021.log
│       └── baseline_ph1_ETTh1_pred192_seed2021.log
└── results/
    ├── baseline_ph1_ETTh1_pred96_seed2021/
    └── baseline_ph1_ETTh1_pred192_seed2021/
```

---

## Expected Output Format

```
=============================================
PHASE 1 COMPLETE — BASELINE SUMMARY
Total: 2 | Done: 2 | Skipped: 0 | Failed: 0
End time: Sat May 16 14:30:45 UTC 2026
=============================================

MSE results extracted from logs:
PredLen | MSE      | MAE      | Status
--------|----------|----------|-------
96      | 0.4523   | 0.4321   | COMPLETED
192     | 0.5234   | 0.4876   | COMPLETED
```

---

## Tips for Success

1. **Keep Colab Active:** Move your mouse occasionally to prevent disconnection
2. **Use Colab Pro:** For longer uninterrupted sessions
3. **Check GPU Usage:** Run `!nvidia-smi` to verify GPU is being used
4. **Save Checkpoints:** All results are saved to Drive automatically
5. **Resume Capability:** Script automatically skips completed runs

---

## Next Steps

After Phase 1 completes:
1. Review the results in the master log
2. Analyze individual run logs for detailed metrics
3. Check the results directories for model checkpoints
4. Proceed to Phase 2 experiments (if applicable)

---

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all paths are correct
3. Ensure GPU is enabled in Colab
4. Check that all dependencies are installed

Good luck with your experiments! 🚀