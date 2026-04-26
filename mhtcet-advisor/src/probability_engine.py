"""
probability_engine.py — Core probability and prediction logic.

Key concepts:
- Uses weighted average of historical cutoffs (most recent year = highest weight)
- Applies round adjustment (each successive round, cutoffs drop ~1-2 percentile pts)
- Applies user's cutoff trend adjustment (slider: -5 to +5 percentile pts)
- Uses a sigmoid function to convert gap between student percentile and predicted
  cutoff into an admission probability (0-100%)
"""

import numpy as np
import pandas as pd
from typing import Optional


# Year weights: 40% most recent, 30% previous, 20% before, 10% oldest
YEAR_WEIGHTS = [0.40, 0.30, 0.20, 0.10]

# How much cutoffs drop per successive CAP round (in percentile points)
ROUND_DROP = {1: 0.0, 2: 1.5, 3: 3.0, 4: 4.5}

# Sigmoid steepness: higher = sharper transition between safe and unsafe
SIGMOID_K = 0.8


def sigmoid(x: float, k: float = SIGMOID_K) -> float:
    """Smooth probability transition. x = (student_pct - cutoff_pct)."""
    return 100.0 / (1.0 + np.exp(-k * x))


def predict_cutoff(
    historical: list[float],
    years: list[int],
    cap_round: int,
    trend_adjustment: float = 0.0,
) -> Optional[float]:
    """
    Predict the closing percentile cutoff for the given round.

    Args:
        historical: list of closing percentiles, ordered oldest → newest
        years: corresponding years
        cap_round: 1, 2, or 3
        trend_adjustment: user slider value (-5 to +5 percentile points)
                          positive = cutoffs expected to rise (harder),
                          negative = cutoffs expected to fall (easier)

    Returns:
        Predicted closing percentile (float) or None if insufficient data.
    """
    if not historical:
        return None

    n = len(historical)
    weights = YEAR_WEIGHTS[:n][::-1]  # most recent gets highest weight
    weights = [w / sum(weights) for w in weights]  # renormalise

    weighted_avg = sum(h * w for h, w in zip(historical, weights))

    # Apply round drop
    round_drop = ROUND_DROP.get(cap_round, 0.0)
    predicted = weighted_avg - round_drop

    # Apply user's trend adjustment
    predicted += trend_adjustment

    return round(predicted, 4)


def compute_probability(
    student_pct: float,
    predicted_cutoff: float,
    historical: list[float],
) -> float:
    """
    Compute admission probability (0-100) using sigmoid.

    The steepness adapts to historical volatility — if cutoffs fluctuate a lot,
    the transition is gentler (more uncertainty).
    """
    gap = student_pct - predicted_cutoff

    # Adjust sigmoid steepness based on historical consistency
    if len(historical) >= 2:
        std = np.std(historical)
        # More volatile history → gentler curve (more uncertainty)
        k = max(0.3, SIGMOID_K - std * 0.05)
    else:
        k = SIGMOID_K

    prob = sigmoid(gap, k)
    return round(float(prob), 1)


def classify(probability: float) -> dict:
    """Return classification dict based on probability."""
    if probability < 10:
        return {"label": "Reach", "color": "#EF4444", "emoji": "🎯", "order": 0}
    elif probability < 30:
        return {"label": "Dream", "color": "#F97316", "emoji": "⭐", "order": 1}
    elif probability < 70:
        return {"label": "Target", "color": "#3B82F6", "emoji": "✅", "order": 2}
    elif probability < 90:
        return {"label": "Safe", "color": "#22C55E", "emoji": "🛡️", "order": 3}
    else:
        return {"label": "Assured", "color": "#6B7280", "emoji": "🔒", "order": 4}


def build_category_columns(
    base_category: str,
    gender: str,
    seat_level: str,  # 'S' = State, 'H' = Home University, 'O' = Other Than HU
) -> list[str]:
    """
    Build the list of category column prefixes to look up in cutoff data.
    e.g., base_category='OBC', gender='female', seat_level='S'
         → ['LOBCS']  (Ladies OBC State)
    Also returns the general gender+cat without seat suffix for TFWS/special.
    """
    g = "L" if gender == "female" else "G"

    # Map base category to data column prefix
    cat_map = {
        "OPEN": "OPEN",
        "SC": "SC",
        "ST": "ST",
        "VJ": "VJ",
        "NT1": "NT1",
        "NT2": "NT2",
        "NT3": "NT3",
        "OBC": "OBC",
        "SEBC": "SEBC",
        "EWS": "EWS",
    }
    cat_key = cat_map.get(base_category, base_category)

    # EWS has no gender prefix in the data
    if base_category == "EWS":
        return [f"EWS{seat_level}"]

    return [f"{g}{cat_key}{seat_level}"]


def get_relevant_categories(
    base_category: str,
    gender: str,
    home_university: str,
    college_university: str,
    special_quotas: list[str],
) -> list[str]:
    """
    Return all category column prefixes the student is eligible for,
    given their base category, gender, and home university vs college university.
    """
    is_home = home_university and college_university and (
        home_university.lower() == college_university.lower()
    )

    cols = []

    # State Level always applicable
    cols += build_category_columns(base_category, gender, "S")

    # Home vs Other-Than-Home
    if is_home:
        cols += build_category_columns(base_category, gender, "H")
    else:
        cols += build_category_columns(base_category, gender, "O")

    # Special quotas
    for sq in special_quotas:
        if sq == "TFWS":
            cols.append("TFWS")
        elif sq == "DEF":
            cols.append("DEFOPEN")
            cols.append(f"DEF{base_category[:2].upper()}")
        elif sq == "PWD":
            cols.append("PWDOPEN")
        elif sq == "ORPHAN":
            cols.append("ORPHAN")

    return list(dict.fromkeys(cols))  # deduplicate, preserve order


def analyse_college_branch(
    cutoff_df: pd.DataFrame,
    college_name: str,
    course_name: str,
    eligible_categories: list[str],
    student_percentile: float,
    target_round: int,
    trend_adjustment: float,
) -> Optional[dict]:
    """
    Analyse a single college-branch combination and return prediction dict.
    Returns None if no matching data found.
    """
    mask = (
        (cutoff_df['college_name'] == college_name) &
        (cutoff_df['course_name'] == course_name)
    )
    sub = cutoff_df[mask]
    if sub.empty:
        return None

    best_result = None
    best_prob = -1

    for cat in eligible_categories:
        cat_data = sub[sub['category'] == cat].copy()
        if cat_data.empty:
            continue

        # Group by year, get most recent cutoff per year (use lowest percentile = harder to get)
        yearly = (
            cat_data.sort_values('percentile')
            .groupby(['year', 'cap_round'])
            .agg(percentile=('percentile', 'min'), merit=('merit', 'min'))
            .reset_index()
        )

        # Use only round 1 data for historical trend (most consistent)
        r1 = yearly[yearly['cap_round'] == 1].sort_values('year')
        if r1.empty:
            r1 = yearly.sort_values(['year', 'cap_round'])

        historical = r1['percentile'].tolist()
        years = r1['year'].tolist()

        if not historical:
            continue

        predicted = predict_cutoff(historical, years, target_round, trend_adjustment)
        if predicted is None:
            continue

        prob = compute_probability(student_percentile, predicted, historical)
        classification = classify(prob)

        if prob > best_prob:
            best_prob = prob
            best_result = {
                "college_name": college_name,
                "course_name": course_name,
                "best_category": cat,
                "predicted_cutoff": predicted,
                "historical_cutoffs": historical,
                "years": years,
                "probability": prob,
                "classification": classification,
                "gap": round(student_percentile - predicted, 2),
                "trend": _detect_trend(historical),
                "data_years": len(historical),
            }

    return best_result


def _detect_trend(historical: list[float]) -> str:
    """Detect if cutoffs are rising, falling, or stable."""
    if len(historical) < 2:
        return "stable"
    delta = historical[-1] - historical[0]
    if delta > 1.5:
        return "rising"
    elif delta < -1.5:
        return "falling"
    return "stable"


def generate_all_predictions(
    cutoff_df: pd.DataFrame,
    seat_matrix_df: pd.DataFrame,
    student_percentile: float,
    base_category: str,
    gender: str,
    home_university: str,
    special_quotas: list[str],
    preferred_branches: list[str],
    college_type_filter: list[str],
    target_round: int,
    trend_adjustment: float,
    branch_priority: bool,
    university_map: dict,
) -> pd.DataFrame:
    """
    Main engine: iterate all college-branch combos and build prediction table.

    Args:
        branch_priority: if True, preferred branches ranked first
        university_map: {college_name: home_university}
    """
    if cutoff_df.empty:
        return pd.DataFrame()

    # Filter by college type
    if college_type_filter:
        # Use partial match since status strings can be verbose
        def matches_type(status):
            if not status:
                return False
            s = str(status).lower()
            for ct in college_type_filter:
                if ct.lower() in s:
                    return True
            return False
        filtered_df = cutoff_df[cutoff_df['status'].apply(matches_type)]
    else:
        filtered_df = cutoff_df

    if filtered_df.empty:
        return pd.DataFrame()

    # Filter by preferred branches if any selected
    if preferred_branches:
        filtered_df = filtered_df[filtered_df['course_name'].isin(preferred_branches)]

    if filtered_df.empty:
        return pd.DataFrame()

    # Get unique college-branch combos
    combos = filtered_df[['college_name', 'course_name', 'status']].drop_duplicates()

    results = []
    for _, row in combos.iterrows():
        college = row['college_name']
        branch = row['course_name']

        # Determine if this college is HU or OHU for this student
        college_univ = university_map.get(college, "")
        eligible_cats = get_relevant_categories(
            base_category, gender, home_university, college_univ, special_quotas
        )

        result = analyse_college_branch(
            filtered_df, college, branch, eligible_cats,
            student_percentile, target_round, trend_adjustment
        )
        if result:
            result['status'] = row['status']
            result['college_university'] = college_univ
            result['is_home_university'] = (
                home_university and college_univ and
                home_university.lower() == college_univ.lower()
            )
            results.append(result)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Branch priority rank (0 = most preferred)
    if preferred_branches:
        branch_rank = {b: i for i, b in enumerate(preferred_branches)}
        df['branch_rank'] = df['course_name'].map(branch_rank).fillna(len(preferred_branches))
    else:
        df['branch_rank'] = 0

    df['classification_order'] = df['classification'].apply(lambda x: x['order'])

    # Sort: branch_priority flag determines primary sort key
    if branch_priority:
        df = df.sort_values(
            ['branch_rank', 'classification_order', 'probability'],
            ascending=[True, True, False]
        )
    else:
        df = df.sort_values(
            ['classification_order', 'branch_rank', 'probability'],
            ascending=[True, True, False]
        )

    df = df.reset_index(drop=True)
    return df


def generate_preference_list(predictions_df: pd.DataFrame, max_list: int = 10) -> pd.DataFrame:
    """
    Generate the optimised CAP preference list from predictions.
    Distribution: ~20% Dream, ~50% Target, ~30% Safe/Assured
    Always include at least 3 Safe/Assured options.
    """
    if predictions_df.empty:
        return pd.DataFrame()

    df = predictions_df.copy()
    labels = df['classification'].apply(lambda x: x['label'])

    dream = df[labels.isin(['Dream', 'Reach'])].head(max(1, int(max_list * 0.20)))
    target = df[labels == 'Target'].head(max(1, int(max_list * 0.50)))
    safe = df[labels.isin(['Safe', 'Assured'])].head(max(3, int(max_list * 0.30)))

    pref = pd.concat([dream, target, safe]).drop_duplicates(
        subset=['college_name', 'course_name']
    ).head(max_list).reset_index(drop=True)

    pref.index = pref.index + 1  # 1-based preference numbers
    return pref


def float_freeze_advice(
    current_college: str,
    current_branch: str,
    current_probability: float,
    predictions_df: pd.DataFrame,
    next_round: int,
) -> dict:
    """
    Advise whether to Float, Freeze, or Slide after getting an allocation.
    """
    if predictions_df.empty:
        return {"advice": "FREEZE", "reason": "No data available for comparison."}

    current_class = classify(current_probability)

    # Find better options in next round
    better = predictions_df[
        (predictions_df['probability'] > current_probability + 5) &
        ~(
            (predictions_df['college_name'] == current_college) &
            (predictions_df['course_name'] == current_branch)
        )
    ]

    # Find better branch at same college
    same_college_better = predictions_df[
        (predictions_df['college_name'] == current_college) &
        (predictions_df['course_name'] != current_branch) &
        (predictions_df['probability'] > 40)
    ]

    if current_class['label'] in ['Assured', 'Safe'] and better.empty:
        return {
            "advice": "FREEZE",
            "reason": f"Your current allocation at {current_college} ({current_branch}) is strong. "
                      f"No significantly better options are likely in the next round.",
            "better_options": 0,
        }
    elif not same_college_better.empty and better.empty:
        return {
            "advice": "SLIDE",
            "reason": f"Stay at {current_college} but try for a better branch. "
                      f"{len(same_college_better)} better branches available at this college.",
            "better_options": len(same_college_better),
            "slide_options": same_college_better[['course_name', 'probability']].head(3).to_dict('records'),
        }
    elif not better.empty:
        top = better.head(3)[['college_name', 'course_name', 'probability']].to_dict('records')
        return {
            "advice": "FLOAT",
            "reason": f"{len(better)} better options may open up in Round {next_round}. "
                      f"You keep your current seat while trying for better.",
            "better_options": len(better),
            "top_options": top,
        }
    else:
        return {
            "advice": "FREEZE",
            "reason": "Your current allocation is reasonable. Risk of losing it outweighs potential gains.",
            "better_options": len(better),
        }
