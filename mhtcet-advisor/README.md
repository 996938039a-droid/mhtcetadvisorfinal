# MHT-CET College Preference Advisor

A Streamlit web application that helps Maharashtra engineering students create optimal college preference lists for CAP rounds using 3 years of historical cutoff data.

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Place Your Data Files

```
mhtcet-advisor/
├── data/
│   ├── cutoffs/
│   │   ├── 2022_CAP1_MH.xlsx      ← rename your files to this format
│   │   ├── 2022_CAP2_MH.xlsx
│   │   ├── 2022_CAP3_MH.xlsx
│   │   ├── 2023_CAP1_MH.xlsx
│   │   ├── 2023_CAP2_MH.xlsx
│   │   ├── 2023_CAP3_MH.xlsx
│   │   ├── 2024_CAP1_MH.xlsx
│   │   ├── 2024_CAP2_MH.xlsx
│   │   └── 2024_CAP3_MH.xlsx
│   └── seat_matrix/
│       └── seat_matrix_2024.xlsx  ← latest seat matrix file
```

### File Naming Convention

Cutoff files must follow: `YYYY_CAPX_MH.xlsx`
- `YYYY` = year (2022, 2023, 2024)
- `X` = round number (1, 2, 3)

The app auto-detects year and round from the filename.
Your current file `2022ENGG_CAP1_CutOff.xlsx` will also be detected — the parser handles flexible naming as long as the year and "CAP" + round number appear in the filename.

### 3. Run the App

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

---

## Features

| Feature | Description |
|---------|-------------|
| 📋 Preference List | Optimised Dream/Target/Safe preference list for CAP |
| 🔍 All Options | Full filterable table of all matching options |
| 📊 Round Analysis | How probability changes across CAP Rounds 1-3 + heatmap |
| ⚖️ Float/Freeze | Advisor for after getting an allocation |
| 🏛️ ACAP Guide | Complete guide to Autonomous College Admission Process |
| 📤 Export | PDF + CSV download of your preference list |

---

## Category Codes in the Data

The cutoff files use codes like `GOPENS`, `LOBCS`, `GSCH`:
- First letter: `G` = General/Male, `L` = Ladies/Female
- Middle: `OPEN`, `SC`, `ST`, `VJ`, `NT1/2/3`, `OBC`, `SEBC`, `EWS`
- Last letter: `S` = State Level, `H` = Home University, `O` = Other Than HU

---

## Probability Formula

For each college-branch-category:
1. Weighted average of historical cutoffs (40% most recent year, 30%, 20%, 10%)
2. Round adjustment (Round 2 = -1.5 pts, Round 3 = -3.0 pts)
3. User trend adjustment (-5 to +5 pts)
4. Sigmoid function → probability (0-100%)

Classification: Reach <10%, Dream 10-30%, Target 30-70%, Safe 70-90%, Assured >90%
