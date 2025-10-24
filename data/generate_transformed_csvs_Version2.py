#!/usr/bin/env python3
"""
generate_transformed_csvs.py

Reads Sleep_health_and_lifestyle_dataset.csv (placed in same directory),
extends it to 1000 rows using deterministic augmentation, creates transformed
feature and target CSV files:

- extended_dataset_1000.csv      : extended raw dataset (1000 rows)
- transformed_features.csv       : features transformed (label-encoded cats, numeric scaled)
- transformed_targets.csv        : targets (binary and multiclass) with original disorder label

Usage:
  pip install pandas numpy scikit-learn
  python generate_transformed_csvs.py

Outputs are written to the current directory.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

INPUT = "Sleep_health_and_lifestyle_dataset.csv"
EXTENDED_OUT = "extended_dataset_1000.csv"
FEATURES_OUT = "transformed_features.csv"
TARGETS_OUT = "transformed_targets.csv"
N_TARGET = 1000
SEED = 42

def parse_bp(bp_str):
    try:
        s, d = bp_str.split('/')
        return int(s), int(d)
    except Exception:
        return np.nan, np.nan

def main():
    df = pd.read_csv(INPUT)
    rng = np.random.default_rng(SEED)

    base_n = len(df)
    rows = []
    for i in range(1, N_TARGET + 1):
        src_idx = (i - 1) % base_n
        cycle = (i - 1) // base_n  # how many full repeats we've done
        row = df.iloc[src_idx].copy()

        # assign new Person ID
        row['Person ID'] = i

        # parse and augment blood pressure
        s, d = parse_bp(str(row['Blood Pressure']))
        if not np.isnan(s):
            s = int(s) + cycle  # small deterministic increase per cycle
            d = int(d) + cycle
            row['Blood Pressure'] = f"{s}/{d}"

        # deterministic noise for numeric columns using rng
        row['Sleep Duration'] = round(float(row['Sleep Duration']) + 0.02 * cycle + float(rng.normal(0, 0.05)), 2)
        row['Quality of Sleep'] = int(row['Quality of Sleep'])  # keep quality stable
        row['Physical Activity Level'] = int(min(100, max(0, int(row['Physical Activity Level']) + cycle * 1 + int(rng.integers(-3, 4)))))
        row['Stress Level'] = int(row['Stress Level'])  # keep stable
        row['Heart Rate'] = int(max(40, int(row['Heart Rate']) + int(rng.integers(-3, 4))))
        row['Daily Steps'] = int(max(0, int(row['Daily Steps']) + cycle * 50 + int(rng.integers(-300, 301))))

        # occasionally flip disorder to introduce variation (low prob deterministic by RNG)
        if rng.random() < 0.02:
            # choose between None, Sleep Apnea, Insomnia
            choices = ['None', 'Sleep Apnea', 'Insomnia']
            row['Sleep Disorder'] = choices[int(rng.integers(0, len(choices)))]
        # append
        rows.append(row)

    extended = pd.DataFrame(rows)

    # Ensure types
    extended['Person ID'] = extended['Person ID'].astype(int)
    # Extract Systolic/Diastolic
    bp_tups = extended['Blood Pressure'].astype(str).apply(parse_bp)
    extended['Systolic'] = bp_tups.apply(lambda x: x[0])
    extended['Diastolic'] = bp_tups.apply(lambda x: x[1])

    # --- Transform features ---
    feat_df = extended.copy()

    # Label encode categorical fields
    le_gender = LabelEncoder()
    feat_df['Gender_encoded'] = le_gender.fit_transform(feat_df['Gender'].astype(str))

    le_occ = LabelEncoder()
    feat_df['Occupation_encoded'] = le_occ.fit_transform(feat_df['Occupation'].astype(str))

    le_bmi = LabelEncoder()
    feat_df['BMI_encoded'] = le_bmi.fit_transform(feat_df['BMI Category'].astype(str))

    # Numeric columns to scale
    numeric_cols = ['Sleep Duration', 'Quality of Sleep', 'Physical Activity Level',
                    'Stress Level', 'Systolic', 'Diastolic', 'Heart Rate', 'Daily Steps']

    # Convert to numeric
    for c in numeric_cols:
        feat_df[c] = pd.to_numeric(feat_df[c], errors='coerce').fillna(0)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(feat_df[numeric_cols])
    scaled_cols = [c + "_scaled" for c in numeric_cols]
    scaled_df = pd.DataFrame(scaled, columns=scaled_cols, index=feat_df.index)

    # Build transformed features DataFrame
    transformed_features = pd.concat([
        feat_df[['Person ID', 'Age', 'Gender_encoded', 'Occupation_encoded', 'BMI_encoded']],
        feat_df[numeric_cols],
        scaled_df
    ], axis=1)

    # --- Build targets ---
    target_df = extended[['Person ID', 'Sleep Disorder']].copy()
    target_df['target_binary'] = target_df['Sleep Disorder'].apply(lambda x: 0 if str(x).strip().lower() == 'none' else 1)
    def multiclass_map(x):
        x = str(x).strip().lower()
        if x == 'none':
            return 0
        if 'sleep apnea' in x:
            return 1
        if 'insomnia' in x:
            return 2
        return 3
    target_df['target_multiclass'] = target_df['Sleep Disorder'].apply(multiclass_map)

    # --- Save outputs ---
    extended.to_csv(EXTENDED_OUT, index=False)
    transformed_features.to_csv(FEATURES_OUT, index=False)
    target_df.to_csv(TARGETS_OUT, index=False)

    # Print some summary
    print(f"Wrote {EXTENDED_OUT} ({len(extended)} rows)")
    print(f"Wrote {FEATURES_OUT} ({len(transformed_features)} rows, {len(transformed_features.columns)} columns)")
    print(f"Wrote {TARGETS_OUT} ({len(target_df)} rows)")

if __name__ == "__main__":
    main()