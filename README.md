# Trend & Anomaly Detection Module

This module is designed for **time-series analysis of sensors/components** and provides:
- Trend detection
- Anomaly scoring
- Region labeling (low, medium, high)
- Sharp change detection
- Consistency check of data timestamps

---

## Main Functions

### 1. `compute_region_labels(values)`
Clusters numerical values into 3 regions: **low, medium, high**.
- If fewer than 3 unique values exist → artificial centers are created.
- Otherwise → `KMeans (n_clusters=3)` is applied.
- Each value gets a **label**:
  - `0` = low  
  - `1` = medium  
  - `2` = high  
  - `-1` = missing value (`NaN`)

**Returns:**
- `labels`: list of region labels for each input value  
- `centers`: the 3 cluster centers  

---

### 2. `check_trend_refactored(df, col_names)`
Performs multiple analyses for each numeric column in `col_names` over grouped time-series data.

#### Steps:
1. **Sorting & Grouping:** groups data by `(Comp_Number, Point, Direction, Timestamp)`.
2. **Trend detection:**
   - Rolling mean (`window=3`) of past values.
   - Relative change compared to previous average.
   - Labels: `1` (upward), `-1` (downward), `0` (stable) if threshold (±10%) is exceeded.
3. **Anomaly scoring:**
   - Z-score against historical values.
   - Normalized to `[0,1]` (capped at `Z_MAX=3.5`).
   - Conditions: `"Normal"`, `"Alarm"`, `"Danger"`.
4. **Region labeling:**
   - Uses `compute_region_labels` to assign values into regions 0/1/2.
5. **Sharp change detection:**
   - Jump two regions (0→2) within ≤7 days → `1.0`
   - Jump one region (0→1 or 1→2) within ≤7 days → `0.5`
   - Otherwise → `0.0`
6. **Consistency check:**
   - Checks time gaps between consecutive timestamps.
   - If ≤70 days → `consistent=1`, else `0`.

#### Returns:
A `DataFrame` with added columns:
- `<col>_trend`
- `<col>_trend_score`
- `<col>_trend_condition`
- `<col>_sharp_score`
- `<col>_region`
- `consistent`

---

## Example

```python
import pandas as pd
from datetime import datetime

data = {
    "Comp_Number": [1, 1, 1, 1],
    "Point": ["P1"]*4,
    "Direction": ["X"]*4,
    "Timestamp": pd.to_datetime([
        "2025-01-01", "2025-01-10", "2025-01-20", "2025-02-01"
    ]),
    "Value": [10, 11, 15, 30],
}

df = pd.DataFrame(data)

result = check_trend_refactored(df, ["Value"])
print(result)
```

---

## Dependencies

Install required packages:

```bash
pip install numpy pandas scikit-learn
```

---

## Notes
- Mandatory DataFrame columns:
  - `Comp_Number`, `Point`, `Direction`, `Timestamp`
- Analytical columns must be provided in `col_names`.
- `Timestamp` must be in datetime format.
