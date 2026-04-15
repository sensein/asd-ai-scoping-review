#Imports 
import pandas as pd
import re 
import numpy as np

#Reading The Data 
df = pd.read_csv("paper_annotation_final.csv - data.csv") # df = total rows which have paper annotation data 

#-----------------DATA SUBSET-------------------------#
df_subset = df.iloc[2:143].reset_index(drop=True) # df_subset = rows 2 to 142 which have paper annotation data. 

#------------TOTAL PAPERS ANNOTATED------------#
cols_subset_Total_Paper = df_subset.iloc[:, 4:66] # includes all rows from 2->142and col subset E[4] TO BN[66] contain ALL cols related to information about paper annotation 

#-------ASD Coloumns----------#
cols_subset_ASD = df_subset.iloc[:, 6:15] # includes all rows 2->142 and col subset G[6] to O[14] contain information about ASD participants 

#-----------NEUROTYPICALS Coloumns---------------#
cols_subset_Neur = df_subset.iloc[:, 15:24] # coloumns P[15] to X[23] contain information about Neurotypical participants 

#-----------Other Diagnosis Coloumns------------#
cols_subset_Other = df_subset.iloc[:, 24:34] # coloumns Y[24] to AH[33] contain information about Neurotypical participants 

#---------------- CLEAN DATA CONSISTENTLY ----------------#

invalid_values = [
    "", "-", "not specified", "n/a", "n.a", "na", "not applicable"
]

invalid_values = [v.lower() for v in invalid_values]

# Replace invalid values with NA for each group
cols_clean_Total_Papers = cols_subset_Total_Paper.replace(invalid_values, pd.NA)
cols_clean_ASD = cols_subset_ASD.replace(invalid_values, pd.NA)
cols_clean_Neur = cols_subset_Neur.replace(invalid_values, pd.NA)
cols_clean_Other = cols_subset_Other.replace(invalid_values, pd.NA)
#All invalid values are now NA

#---------------- VALID PAPER MASKS ----------------#

empty_rows_Total_Paper = cols_clean_Total_Papers.isna().all(axis=1)
empty_rows_ASD = cols_clean_ASD.isna().all(axis=1)
empty_rows_Neur = cols_clean_Neur.isna().all(axis=1)
empty_rows_Other = cols_clean_Other.isna().all(axis=1)

valid_total = ~empty_rows_Total_Paper
valid_ASD   = ~empty_rows_ASD
valid_Neur  = ~empty_rows_Neur
valid_Other = ~empty_rows_Other

valid_papers_Total = (~empty_rows_Total_Paper).sum()
valid_papers_ASD = (~empty_rows_ASD).sum()
valid_papers_Neur = (~empty_rows_Neur).sum()
valid_papers_Other = (~empty_rows_Other).sum()

#---------------- PRINT TOTAL PAPERS ----------------#

print("=============TOTAL PAPERS=================")

print("TOTAL PAPERS:")
print("Empty rows:", empty_rows_Total_Paper.sum())
print("Valid papers:", valid_papers_Total)

print("\nASD-Information:")
print("All papers have ASD participants however, some do not have information about them")
print("Empty rows:", empty_rows_ASD.sum())
print("Valid papers:", valid_papers_ASD)
print("Total studies with no information about ASD participants:", valid_papers_Total - valid_papers_ASD)

print("\nNEUROTYPICALS:")
print("Empty rows:", empty_rows_Neur.sum())
print("Valid papers:", valid_papers_Neur)

print("\nOTHER DIAGNOSES:")
print("Empty rows:", empty_rows_Other.sum())
print("Valid papers:", valid_papers_Other)

#---------------- SAMPLE SIZE (CONSISTENT FILTERING) ----------------#

# Extract sample size columns from SAME subset
asd_sample = pd.to_numeric(df_subset.iloc[:, 6], errors="coerce")
neur_sample = pd.to_numeric(df_subset.iloc[:, 15], errors="coerce")
other_sample = pd.to_numeric(df_subset.iloc[:, 24], errors="coerce")

# Apply SAME valid paper filtering
asd_sample = asd_sample[~empty_rows_Total_Paper]
neur_sample = neur_sample[~empty_rows_Neur]
other_sample = other_sample[~empty_rows_Other]
#empty rows includes the subset with invalid papers so null values are not counted 
#---------------- MEAN ----------------#

mean_value_ASD = asd_sample.mean()
mean_value_Neur = neur_sample.mean()
mean_value_Other = other_sample.mean()

print("\n===============MEAN: SAMPLE SIZES==================")
print("ASD:", mean_value_ASD)
print("NEUROTYPICALS:", mean_value_Neur)
print("OTHER DIAGNOSES:", mean_value_Other)

#---------------- STANDARD DEVIATION ----------------#

std_value_ASD = asd_sample.std()
std_value_Neur = neur_sample.std()
std_value_Other = other_sample.std()

print("\n=============STANDARD DEVIATION===========")
print("ASD:", std_value_ASD)
print("NEUROTYPICALS:", std_value_Neur)
print("OTHER DIAGNOSES:", std_value_Other)
#---------------- SAMPLE SIZE: NOT GIVEN ----------------#

print("\n=============SAMPLE SIZE: NOT GIVEN=================")

# Extract sample size columns (same as before)

asd_sample_raw = df_subset.iloc[:, 6]
neur_sample_raw = df_subset.iloc[:, 15]
other_sample_raw = df_subset.iloc[:, 24]

# Function to check invalid
def is_invalid(x):
    if pd.isna(x):
        return True
    return str(x).strip().lower() in invalid_values

# Count missing sample sizes ONLY among valid papers

# ASD
asd_missing = asd_sample_raw[~empty_rows_Total_Paper].apply(is_invalid).sum() # counts the number of rows that are empty excluding the rows which are invalid for given subset 
asd_total_valid = (~empty_rows_Total_Paper).sum()

# Neurotypicals
neur_missing = neur_sample_raw[~empty_rows_Neur].apply(is_invalid).sum()
neur_total_valid = (~empty_rows_Neur).sum()

# Other Diagnoses
other_missing = other_sample_raw[~empty_rows_Other].apply(is_invalid).sum()
other_total_valid = (~empty_rows_Other).sum()

#---------------- PRINT RESULTS ----------------#

print("ASD:")
print("Missing sample size count:", asd_missing)
print("Total valid papers:", asd_total_valid)
print("Proportion missing:", asd_missing / asd_total_valid if asd_total_valid else 0)

print("\nNEUROTYPICALS:")
print("Missing sample size count:", neur_missing)
print("Total valid papers:", neur_total_valid)
print("Proportion missing:", neur_missing / neur_total_valid if neur_total_valid else 0)

print("\nOTHER DIAGNOSES:")
print("Missing sample size count:", other_missing)
print("Total valid papers:", other_total_valid)
print("Proportion missing:", other_missing / other_total_valid if other_total_valid else 0)

#----------------AGE RANGES-----------------#
# ---------------- INVALID VALUES ---------------- #
invalid_values = [
    "", "-", "not specified", "not applicable", "n/a", "na", "n.a"
]
invalid_values = [v.lower() for v in invalid_values]

def is_invalid(x):
    if pd.isna(x):
        return True
    return str(x).strip().lower() in invalid_values


# ---------------- AGE CATEGORIES ---------------- #
categories = {
    "Infants": (0, 1),
    "Toddlers": (1, 3),
    "Pre-schoolers": (3, 6),
    "Grade-schoolers": (7, 12),
    "Teens": (13, 18),
    "Adults": (18, float("inf"))
}


# ---------------- HELPER: ADD RANGE TO COUNTS ---------------- #
def add_range_to_counts(age_min, age_max, counts, categories):
    for cat, (low, high) in categories.items():
        if age_max >= low and age_min <= high:
            counts[cat] += 1


# ---------------- HELPER: EXTRACT NUMBERS ---------------- #
def extract_numbers(text):
    return [float(n) for n in re.findall(r"\d+\.?\d*", text)]


# ---------------- SPECIAL CASE DETECTORS ---------------- #
def has_multiple_ranges(cell):
    cell = str(cell).strip().lower()

    short_dash_count = cell.count("-")
    en_dash_count = cell.count("–")
    em_dash_count = cell.count("—")
    to_count = len(re.findall(r"\bto\b", cell))

    total_dash_count = short_dash_count + en_dash_count + em_dash_count

    if total_dash_count >= 2:
        return True

    if to_count >= 2:
        return True

    return False


def has_plus_minus(cell):
    cell = str(cell).strip().lower()
    return "±" in cell or "+/-" in cell


def has_under_over(cell):
    cell = str(cell).strip().lower()
    return any(word in cell for word in ["under", "below", "over", "above"])


# ---------------- MAIN PARSER: REGULAR CASES ---------------- #
def parse_regular_line(line):
    """
    Handles:
    5 years
    5-7 years
    5 to 7 years
    12 months
    12-24 months
    """
    line = str(line).strip().lower()

    if not line or line in invalid_values:
        return []

    nums = extract_numbers(line)
    if len(nums) == 0:
        return []

    is_months = "month" in line

    if len(nums) == 1:
        age_min = age_max = nums[0]
    else:
        age_min = min(nums)
        age_max = max(nums)

    if is_months:
        age_min /= 12
        age_max /= 12

    return [(age_min, age_max)]


# ---------------- HELPER 1: MULTIPLE RANGES ---------------- #
def parse_multiple_ranges(cell):
    """
    Handles:
    4-6 and 5-7
    4–6, 5–7
    4 to 6 and 5 to 7
    """
    cell = str(cell).strip().lower()

    results = []
    is_months = "month" in cell

    dash_matches = re.findall(r'(\d+\.?\d*)\s*[-–—]\s*(\d+\.?\d*)', cell)
    to_matches = re.findall(r'(\d+\.?\d*)\s+to\s+(\d+\.?\d*)', cell)

    all_matches = dash_matches + to_matches

    for start, end in all_matches:
        age_min = float(start)
        age_max = float(end)

        if is_months:
            age_min /= 12
            age_max /= 12

        results.append((min(age_min, age_max), max(age_min, age_max)))

    return results


# ---------------- HELPER 2: UNDER / OVER / ABOVE / BELOW ---------------- #
def parse_under_over(cell):
    """
    Handles:
    under 12 years
    below 12 years
    over 12 years
    above 12 years
    """
    cell = str(cell).strip().lower()

    nums = extract_numbers(cell)
    if len(nums) != 1:
        return []

    age = nums[0]

    if "month" in cell:
        age /= 12

    if "under" in cell or "below" in cell:
        return [(0, age)]

    if "over" in cell or "above" in cell:
        return [(age, float("inf"))]

    return []


# ---------------- HELPER 3: MEAN ± SD ---------------- #
def parse_plus_minus(cell):
    """
    Handles:
    3.7 ± 1.3 years
    3.7 +/- 1.3 years
    """
    cell = str(cell).strip().lower()

    nums = extract_numbers(cell)
    if len(nums) < 2:
        return []

    mean = nums[0]
    sd = nums[1]

    age_min = max(mean - sd, 0)
    age_max = mean + sd

    if "month" in cell:
        age_min /= 12
        age_max /= 12

    return [(age_min, age_max)]


# ---------------- CELL PARSER WITH TRACKING ---------------- #
def parse_age_cell(cell):
    """
    Returns:
    parsed_ranges, parser_type

    parser_type can be:
    invalid
    regular
    multiple_ranges
    under_over
    plus_minus
    unparsed
    """
    if is_invalid(cell):
        return [], "invalid"

    cell = str(cell).strip().lower()

    if has_multiple_ranges(cell):
        result = parse_multiple_ranges(cell)
        if result:
            return result, "multiple_ranges"

    if has_under_over(cell):
        result = parse_under_over(cell)
        if result:
            return result, "under_over"

    if has_plus_minus(cell):
        result = parse_plus_minus(cell)
        if result:
            return result, "plus_minus"

    result = parse_regular_line(cell)
    if result:
        return result, "regular"

    return [], "unparsed"


# ---------------- ASD AGE RANGE FUNCTION ---------------- #
def compute_age_ranges_asd(age_col, valid_mask, categories):
    print("\n=============AGE RANGE: ASD=============")

    col_filtered = age_col[valid_mask]
    total_valid = valid_mask.sum()

    counts = {key: 0 for key in categories}
    not_given = 0
    unparsed_nonempty = []

    parser_usage = {
        "regular": 0,
        "multiple_ranges": 0,
        "under_over": 0,
        "plus_minus": 0,
        "invalid": 0,
        "unparsed": 0
    }

    debug_examples = {
        "regular": [],
        "multiple_ranges": [],
        "under_over": [],
        "plus_minus": [],
        "unparsed": []
    }

    for val in col_filtered:
        parsed_ranges, parser_type = parse_age_cell(val)
        parser_usage[parser_type] += 1

        if parser_type in debug_examples and len(debug_examples[parser_type]) < 5:
            debug_examples[parser_type].append(val)

        if len(parsed_ranges) == 0:
            not_given += 1
            if not is_invalid(val):
                unparsed_nonempty.append(val)
            continue

        for age_min, age_max in parsed_ranges:
            add_range_to_counts(age_min, age_max, counts, categories)

    counts["Not given"] = not_given

    print("\n=== AGE CATEGORY COUNTS ===")
    for k, v in counts.items():
        pct = (v / total_valid) * 100 if total_valid else 0
        print(f"{k}: {v} ({pct:.2f}%)")

    print(f"\nTotal valid ASD papers: {total_valid}")
    print(f"Unparsed non-empty cells: {len(unparsed_nonempty)}")

    print("\n=== PARSER USAGE ===")
    for k, v in parser_usage.items():
        pct = (v / total_valid) * 100 if total_valid else 0
        print(f"{k}: {v} ({pct:.2f}%)")

    print("\n=== EXAMPLE CELLS BY PARSER TYPE ===")
    for parser_type, examples in debug_examples.items():
        print(f"\n{parser_type.upper()}:")
        if len(examples) == 0:
            print("None")
        else:
            for ex in examples:
                print("-", repr(ex))

    return counts, parser_usage, debug_examples, unparsed_nonempty


# ---------------- RUN FOR ASD ONLY ---------------- #
# Use df_subset, not df, so the indices match empty_rows_ASD
valid_mask_ASD = ~empty_rows_ASD
age_col_asd = df_subset.iloc[:, 8]

asd_counts, asd_parser_usage, asd_debug_examples, asd_unparsed = compute_age_ranges_asd(
    age_col_asd,
    valid_mask_ASD,
    categories
)

#-------------------------------AGE RANGE: NEUROTYPICALS--------------------------#
# ---------------- INVALID VALUES ---------------- #
invalid_values = [
    "", "-", "not specified", "not applicable", "n/a", "na", "n.a"
]
invalid_values = [v.lower() for v in invalid_values]

def is_invalid(x):
    if pd.isna(x):
        return True
    return str(x).strip().lower() in invalid_values


# ---------------- AGE CATEGORIES ---------------- #
categories = {
    "Infants": (0, 1),
    "Toddlers": (1, 3),
    "Pre-schoolers": (3, 6),
    "Grade-schoolers": (7, 12),
    "Teens": (13, 18),
    "Adults": (18, float("inf"))
}


# ---------------- HELPER: ADD RANGE TO COUNTS ---------------- #
def add_range_to_counts(age_min, age_max, counts, categories):
    for cat, (low, high) in categories.items():
        if age_max >= low and age_min <= high:
            counts[cat] += 1


# ---------------- HELPER: EXTRACT NUMBERS ---------------- #
def extract_numbers(text):
    return [float(n) for n in re.findall(r"\d+\.?\d*", text)]


# ---------------- SPECIAL CASE DETECTORS ---------------- #
def has_multiple_ranges(cell):
    cell = str(cell).strip().lower()

    short_dash_count = cell.count("-")
    en_dash_count = cell.count("–")
    em_dash_count = cell.count("—")
    to_count = len(re.findall(r"\bto\b", cell))

    total_dash_count = short_dash_count + en_dash_count + em_dash_count

    if total_dash_count >= 2:
        return True

    if to_count >= 2:
        return True

    return False


def has_plus_minus(cell):
    cell = str(cell).strip().lower()
    return "±" in cell or "+/-" in cell


def has_under_over(cell):
    cell = str(cell).strip().lower()
    return any(word in cell for word in ["under", "below", "over", "above"])


# ---------------- MAIN PARSER: REGULAR CASES ---------------- #
def parse_regular_line(line):
    """
    Handles:
    5 years
    5-7 years
    5 to 7 years
    12 months
    12-24 months
    """
    line = str(line).strip().lower()

    if not line or line in invalid_values:
        return []

    nums = extract_numbers(line)
    if len(nums) == 0:
        return []

    is_months = "month" in line

    if len(nums) == 1:
        age_min = age_max = nums[0]
    else:
        age_min = min(nums)
        age_max = max(nums)

    if is_months:
        age_min /= 12
        age_max /= 12

    return [(age_min, age_max)]


# ---------------- HELPER 1: MULTIPLE RANGES ---------------- #
def parse_multiple_ranges(cell):
    """
    Handles:
    4-6 and 5-7
    4–6, 5–7
    4 to 6 and 5 to 7
    """
    cell = str(cell).strip().lower()

    results = []
    is_months = "month" in cell

    dash_matches = re.findall(r'(\d+\.?\d*)\s*[-–—]\s*(\d+\.?\d*)', cell)
    to_matches = re.findall(r'(\d+\.?\d*)\s+to\s+(\d+\.?\d*)', cell)

    all_matches = dash_matches + to_matches

    for start, end in all_matches:
        age_min = float(start)
        age_max = float(end)

        if is_months:
            age_min /= 12
            age_max /= 12

        results.append((min(age_min, age_max), max(age_min, age_max)))

    return results


# ---------------- HELPER 2: UNDER / OVER / ABOVE / BELOW ---------------- #
def parse_under_over(cell):
    """
    Handles:
    under 12 years
    below 12 years
    over 12 years
    above 12 years
    """
    cell = str(cell).strip().lower()

    nums = extract_numbers(cell)
    if len(nums) != 1:
        return []

    age = nums[0]

    if "month" in cell:
        age /= 12

    if "under" in cell or "below" in cell:
        return [(0, age)]

    if "over" in cell or "above" in cell:
        return [(age, float("inf"))]

    return []


# ---------------- HELPER 3: MEAN ± SD ---------------- #
def parse_plus_minus(cell):
    """
    Handles:
    3.7 ± 1.3 years
    3.7 +/- 1.3 years
    """
    cell = str(cell).strip().lower()

    nums = extract_numbers(cell)
    if len(nums) < 2:
        return []

    mean = nums[0]
    sd = nums[1]

    age_min = max(mean - sd, 0)
    age_max = mean + sd

    if "month" in cell:
        age_min /= 12
        age_max /= 12

    return [(age_min, age_max)]


# ---------------- CELL PARSER WITH TRACKING ---------------- #
def parse_age_cell(cell):
    """
    Returns:
    parsed_ranges, parser_type

    parser_type can be:
    invalid
    regular
    multiple_ranges
    under_over
    plus_minus
    unparsed
    """
    if is_invalid(cell):
        return [], "invalid"

    cell = str(cell).strip().lower()

    if has_multiple_ranges(cell):
        result = parse_multiple_ranges(cell)
        if result:
            return result, "multiple_ranges"

    if has_under_over(cell):
        result = parse_under_over(cell)
        if result:
            return result, "under_over"

    if has_plus_minus(cell):
        result = parse_plus_minus(cell)
        if result:
            return result, "plus_minus"

    result = parse_regular_line(cell)
    if result:
        return result, "regular"

    return [], "unparsed"


# ---------------- NEUROTYPICAL AGE RANGE FUNCTION ---------------- #
def compute_age_ranges_neur(df_subset, categories):
    print("\n=============AGE RANGE: NEUROTYPICALS=============")

    # ---------------- STEP 1: SELECT AGE COLUMN + NEUROTYPICAL SUBSET ---------------- #
    age_col = df_subset.iloc[:, 17]          # neurotypical age column
    neur_diag_cols = df_subset.iloc[:, 15:24]  # columns P to X / neurotypical block

    # ---------------- STEP 2: CLEAN INVALID VALUES ---------------- #
    neur_diag_clean = neur_diag_cols.replace(invalid_values, pd.NA)

    # ---------------- STEP 3: FIND VALID PAPERS ---------------- #
    # valid if at least one neurotypical column has real info
    valid_rows_mask = neur_diag_clean.notna().any(axis=1)
    total_valid_papers = valid_rows_mask.sum()

    print(f"Total valid neurotypical papers: {total_valid_papers}")

    # ---------------- STEP 4: FILTER ONLY VALID PAPERS ---------------- #
    col_filtered = age_col[valid_rows_mask]

    # ---------------- STEP 5: SET UP COUNTS ---------------- #
    counts = {key: 0 for key in categories}
    not_given = 0
    unparsed_nonempty = []

    parser_usage = {
        "regular": 0,
        "multiple_ranges": 0,
        "under_over": 0,
        "plus_minus": 0,
        "invalid": 0,
        "unparsed": 0
    }

    debug_examples = {
        "regular": [],
        "multiple_ranges": [],
        "under_over": [],
        "plus_minus": [],
        "unparsed": []
    }

    # ---------------- STEP 6: LOOP ONLY THROUGH VALID PAPERS ---------------- #
    for val in col_filtered:
        parsed_ranges, parser_type = parse_age_cell(val)
        parser_usage[parser_type] += 1

        if parser_type in debug_examples and len(debug_examples[parser_type]) < 5:
            debug_examples[parser_type].append(val)

        if len(parsed_ranges) == 0:
            not_given += 1
            if not is_invalid(val):
                unparsed_nonempty.append(val)
            continue

        for age_min, age_max in parsed_ranges:
            add_range_to_counts(age_min, age_max, counts, categories)

    counts["Not given"] = not_given

    # ---------------- STEP 7: PRINT COUNTS ---------------- #
    print("\n=== AGE CATEGORY COUNTS (VALID PAPERS ONLY) ===")
    for k, v in counts.items():
        print(f"{k}: {v}")

    # ---------------- STEP 8: PRINT PERCENTAGES ---------------- #
    print("\n=== PERCENTAGES (VALID PAPERS ONLY) ===")
    for k, v in counts.items():
        pct = (v / total_valid_papers) * 100 if total_valid_papers else 0
        print(f"{k}: {pct:.2f}%")

    # ---------------- STEP 9: PRINT PARSER USAGE ---------------- #
    print("\n=== PARSER USAGE ===")
    for k, v in parser_usage.items():
        pct = (v / total_valid_papers) * 100 if total_valid_papers else 0
        print(f"{k}: {v} ({pct:.2f}%)")

    print(f"\nUnparsed non-empty cells: {len(unparsed_nonempty)}")

    print("\n=== EXAMPLE CELLS BY PARSER TYPE ===")
    for parser_type, examples in debug_examples.items():
        print(f"\n{parser_type.upper()}:")
        if len(examples) == 0:
            print("None")
        else:
            for ex in examples:
                print("-", repr(ex))

    return counts, parser_usage, debug_examples, unparsed_nonempty, valid_rows_mask, total_valid_papers


# ---------------- RUN FOR NEUROTYPICALS ONLY ---------------- #
neur_counts, neur_parser_usage, neur_debug_examples, neur_unparsed, neur_valid_rows_mask, neur_total_valid = compute_age_ranges_neur(
    df_subset,
    categories
)

#-----------------------------AGE RANGE: OTHER DIAGNOSES PARTICIPANTS-------------------------#

# ---------------- INVALID VALUES ---------------- #
invalid_values = [
    "", "-", "not specified", "not applicable", "n/a", "na", "n.a"
]
invalid_values = [v.lower() for v in invalid_values]

def is_invalid(x):
    if pd.isna(x):
        return True
    return str(x).strip().lower() in invalid_values


# ---------------- AGE CATEGORIES ---------------- #
categories = {
    "Infants": (0, 1),
    "Toddlers": (1, 3),
    "Pre-schoolers": (3, 6),
    "Grade-schoolers": (7, 12),
    "Teens": (13, 18),
    "Adults": (18, float("inf"))
}


# ---------------- HELPER: ADD RANGE TO COUNTS ---------------- #
def add_range_to_counts(age_min, age_max, counts, categories):
    for cat, (low, high) in categories.items():
        if age_max >= low and age_min <= high:
            counts[cat] += 1


# ---------------- HELPER: EXTRACT NUMBERS ---------------- #
def extract_numbers(text):
    return [float(n) for n in re.findall(r"\d+\.?\d*", text)]


# ---------------- SPECIAL CASE DETECTORS ---------------- #
def has_multiple_ranges(cell):
    cell = str(cell).strip().lower()

    short_dash_count = cell.count("-")
    en_dash_count = cell.count("–")
    em_dash_count = cell.count("—")
    to_count = len(re.findall(r"\bto\b", cell))

    total_dash_count = short_dash_count + en_dash_count + em_dash_count

    if total_dash_count >= 2:
        return True

    if to_count >= 2:
        return True

    return False


def has_plus_minus(cell):
    cell = str(cell).strip().lower()
    return "±" in cell or "+/-" in cell


def has_under_over(cell):
    cell = str(cell).strip().lower()
    return any(word in cell for word in ["under", "below", "over", "above"])


# ---------------- MAIN PARSER: REGULAR CASES ---------------- #
def parse_regular_line(line):
    """
    Handles:
    5 years
    5-7 years
    5 to 7 years
    12 months
    12-24 months
    """
    line = str(line).strip().lower()

    if not line or line in invalid_values:
        return []

    nums = extract_numbers(line)
    if len(nums) == 0:
        return []

    is_months = "month" in line

    if len(nums) == 1:
        age_min = age_max = nums[0]
    else:
        age_min = min(nums)
        age_max = max(nums)

    if is_months:
        age_min /= 12
        age_max /= 12

    return [(age_min, age_max)]


# ---------------- HELPER 1: MULTIPLE RANGES ---------------- #
def parse_multiple_ranges(cell):
    """
    Handles:
    4-6 and 5-7
    4–6, 5–7
    4 to 6 and 5 to 7
    """
    cell = str(cell).strip().lower()

    results = []
    is_months = "month" in cell

    dash_matches = re.findall(r'(\d+\.?\d*)\s*[-–—]\s*(\d+\.?\d*)', cell)
    to_matches = re.findall(r'(\d+\.?\d*)\s+to\s+(\d+\.?\d*)', cell)

    all_matches = dash_matches + to_matches

    for start, end in all_matches:
        age_min = float(start)
        age_max = float(end)

        if is_months:
            age_min /= 12
            age_max /= 12

        results.append((min(age_min, age_max), max(age_min, age_max)))

    return results


# ---------------- HELPER 2: UNDER / OVER / ABOVE / BELOW ---------------- #
def parse_under_over(cell):
    """
    Handles:
    under 12 years
    below 12 years
    over 12 years
    above 12 years
    """
    cell = str(cell).strip().lower()

    nums = extract_numbers(cell)
    if len(nums) != 1:
        return []

    age = nums[0]

    if "month" in cell:
        age /= 12

    if "under" in cell or "below" in cell:
        return [(0, age)]

    if "over" in cell or "above" in cell:
        return [(age, float("inf"))]

    return []


# ---------------- HELPER 3: MEAN ± SD ---------------- #
def parse_plus_minus(cell):
    """
    Handles:
    3.7 ± 1.3 years
    3.7 +/- 1.3 years
    """
    cell = str(cell).strip().lower()

    nums = extract_numbers(cell)
    if len(nums) < 2:
        return []

    mean = nums[0]
    sd = nums[1]

    age_min = max(mean - sd, 0)
    age_max = mean + sd

    if "month" in cell:
        age_min /= 12
        age_max /= 12

    return [(age_min, age_max)]


# ---------------- CELL PARSER WITH TRACKING ---------------- #
def parse_age_cell(cell):
    """
    Returns:
    parsed_ranges, parser_type

    parser_type can be:
    invalid
    regular
    multiple_ranges
    under_over
    plus_minus
    unparsed
    """
    if is_invalid(cell):
        return [], "invalid"

    cell = str(cell).strip().lower()

    if has_multiple_ranges(cell):
        result = parse_multiple_ranges(cell)
        if result:
            return result, "multiple_ranges"

    if has_under_over(cell):
        result = parse_under_over(cell)
        if result:
            return result, "under_over"

    if has_plus_minus(cell):
        result = parse_plus_minus(cell)
        if result:
            return result, "plus_minus"

    result = parse_regular_line(cell)
    if result:
        return result, "regular"

    return [], "unparsed"


# ---------------- OTHER DIAGNOSES AGE RANGE FUNCTION ---------------- #
def compute_age_ranges_other(df_subset, categories):
    print("\n=============AGE RANGE: OTHER DIAGNOSES=============")

    # ---------------- STEP 1: SELECT AGE COLUMN + OTHER DIAGNOSIS SUBSET ---------------- #
    age_col = df_subset.iloc[:, 26]
    other_diag_cols = df_subset.iloc[:, 24:34]

    # ---------------- STEP 2: CLEAN INVALID VALUES ---------------- #
    other_diag_clean = other_diag_cols.replace(invalid_values, pd.NA)

    # ---------------- STEP 3: FIND VALID PAPERS ---------------- #
    # valid if at least one other-diagnosis column has real info
    valid_rows_mask = other_diag_clean.notna().any(axis=1)
    total_valid_papers = valid_rows_mask.sum()

    print(f"Total valid other-diagnosis papers: {total_valid_papers}")

    # ---------------- STEP 4: FILTER ONLY VALID PAPERS ---------------- #
    col_filtered = age_col[valid_rows_mask]

    # ---------------- STEP 5: SET UP COUNTS ---------------- #
    counts = {key: 0 for key in categories}
    not_given = 0
    unparsed_nonempty = []

    parser_usage = {
        "regular": 0,
        "multiple_ranges": 0,
        "under_over": 0,
        "plus_minus": 0,
        "invalid": 0,
        "unparsed": 0
    }

    debug_examples = {
        "regular": [],
        "multiple_ranges": [],
        "under_over": [],
        "plus_minus": [],
        "unparsed": []
    }

    # ---------------- STEP 6: LOOP ONLY THROUGH VALID PAPERS ---------------- #
    for val in col_filtered:
        parsed_ranges, parser_type = parse_age_cell(val)
        parser_usage[parser_type] += 1

        if parser_type in debug_examples and len(debug_examples[parser_type]) < 5:
            debug_examples[parser_type].append(val)

        if len(parsed_ranges) == 0:
            not_given += 1
            if not is_invalid(val):
                unparsed_nonempty.append(val)
            continue

        for age_min, age_max in parsed_ranges:
            add_range_to_counts(age_min, age_max, counts, categories)

    counts["Not given"] = not_given

    # ---------------- STEP 7: PRINT COUNTS ---------------- #
    print("\n=== AGE CATEGORY COUNTS (VALID PAPERS ONLY) ===")
    for k, v in counts.items():
        print(f"{k}: {v}")

    # ---------------- STEP 8: PRINT PERCENTAGES ---------------- #
    print("\n=== PERCENTAGES (VALID PAPERS ONLY) ===")
    for k, v in counts.items():
        pct = (v / total_valid_papers) * 100 if total_valid_papers else 0
        print(f"{k}: {pct:.2f}%")

    # ---------------- STEP 9: PRINT PARSER USAGE ---------------- #
    print("\n=== PARSER USAGE ===")
    for k, v in parser_usage.items():
        pct = (v / total_valid_papers) * 100 if total_valid_papers else 0
        print(f"{k}: {v} ({pct:.2f}%)")

    print(f"\nUnparsed non-empty cells: {len(unparsed_nonempty)}")

    print("\n=== EXAMPLE CELLS BY PARSER TYPE ===")
    for parser_type, examples in debug_examples.items():
        print(f"\n{parser_type.upper()}:")
        if len(examples) == 0:
            print("None")
        else:
            for ex in examples:
                print("-", repr(ex))

    return counts, parser_usage, debug_examples, unparsed_nonempty, valid_rows_mask, total_valid_papers


# ---------------- RUN FOR OTHER DIAGNOSES ONLY ---------------- #
other_counts, other_parser_usage, other_debug_examples, other_unparsed, other_valid_rows_mask, other_total_valid = compute_age_ranges_other(
    df_subset,
    categories
)

#-----------GENDER------------#
import pandas as pd
import numpy as np
import re

# ---------------- INVALID VALUES ---------------- #
invalid_values = [
    "", "-", "not specified", "not applicable", "n/a", "na", "n.a", "n/d"
]
invalid_values = [v.lower() for v in invalid_values]

def is_invalid(x):
    if pd.isna(x):
        return True
    return str(x).strip().lower() in invalid_values


# ---------------- HELPER: EXTRACT ALL NUMBERS ---------------- #
def extract_numbers(text):
    return [float(n) for n in re.findall(r"\d+\.?\d*", str(text))]


# ---------------- HELPER: PARSE TOTAL COUNT ---------------- #
def parse_total_count(x):
    """
    Parses the total participant count.
    If multiple numbers appear, add them together.
    Examples:
    9 + 10 -> 19
    9 10 -> 19
    910 -> 910
    """
    if is_invalid(x):
        return np.nan

    nums = extract_numbers(x)
    if len(nums) == 0:
        return np.nan

    return sum(nums)


# ---------------- HELPER: PARSE MALE COUNT ---------------- #
def parse_male_count(x):
    """
    Parses the male count from messy gender cells.

    Rules:
    1. If invalid -> NaN
    2. If 'female' appears, ignore numbers directly tied to female/females/f
       and use other numbers
    3. If multiple remaining numbers exist, sum them
    4. '9 10' becomes 19
       '910' stays 910
    """
    if is_invalid(x):
        return np.nan

    text = str(x).strip().lower()

    # normalize separators a bit
    text = text.replace("–", "-").replace("—", "-")

    # Case 1: if female/females appears, remove female-labelled chunks
    # examples:
    # "9 male 10 female" -> keep 9
    # "female 10, male 9" -> keep 9
    # "m 9 f 10" -> keep 9
    # "10 females and 9 males" -> keep 9
    if "female" in text or "females" in text or re.search(r"\bf\b", text):
        working = text

        # remove patterns like "female 10", "females 10", "f 10"
        working = re.sub(r"\bfemales?\b\s*[:=]?\s*\d+\.?\d*", " ", working)
        working = re.sub(r"\bf\b\s*[:=]?\s*\d+\.?\d*", " ", working)

        # remove patterns like "10 female", "10 females", "10 f"
        working = re.sub(r"\d+\.?\d*\s*\bfemales?\b", " ", working)
        working = re.sub(r"\d+\.?\d*\s*\bf\b", " ", working)

        # if male-labelled numbers exist, prefer those
        male_labeled = re.findall(r"\bmale[s]?\b\s*[:=]?\s*(\d+\.?\d*)", working)
        male_labeled += re.findall(r"\bm\b\s*[:=]?\s*(\d+\.?\d*)", working)
        male_labeled += re.findall(r"(\d+\.?\d*)\s*\bmale[s]?\b", working)
        male_labeled += re.findall(r"(\d+\.?\d*)\s*\bm\b", working)

        if male_labeled:
            return sum(float(n) for n in male_labeled)

        # otherwise use whatever numbers remain after removing female numbers
        nums = extract_numbers(working)
        if len(nums) == 0:
            return np.nan
        return sum(nums)

    # Case 2: if male-labelled numbers exist, use only those
    male_labeled = re.findall(r"\bmale[s]?\b\s*[:=]?\s*(\d+\.?\d*)", text)
    male_labeled += re.findall(r"\bm\b\s*[:=]?\s*(\d+\.?\d*)", text)
    male_labeled += re.findall(r"(\d+\.?\d*)\s*\bmale[s]?\b", text)
    male_labeled += re.findall(r"(\d+\.?\d*)\s*\bm\b", text)

    if male_labeled:
        return sum(float(n) for n in male_labeled)

    # Case 3: plain multiple numbers -> add them
    nums = extract_numbers(text)
    if len(nums) == 0:
        return np.nan

    return sum(nums)


# ---------------- GENDER FUNCTION ---------------- #
def compute_gender_ratios(df_subset, total_col, male_col,
                          valid_mask, label, extra_filter_cols=None):

    print(f"\n=============GENDER: {label}=============")

    male_ratios, female_ratios = [], []
    missing_gender = 0
    filtered_count = 0

    skipped_invalid_total = 0
    skipped_male_missing = 0
    skipped_male_gt_total = 0

    parser_examples = {
        "male_with_female_text": [],
        "male_labeled": [],
        "multiple_numbers_summed": [],
        "plain_numeric": [],
        "unparsed_male": []
    }

    for idx, row in df_subset.iterrows():

        # Skip invalid papers
        if not valid_mask.iloc[idx]:
            continue

        # Optional subgroup filter
        if extra_filter_cols:
            if all(is_invalid(row.iloc[i]) for i in extra_filter_cols):
                continue
            filtered_count += 1

        total_raw = row.iloc[total_col]
        male_raw = row.iloc[male_col]

        total = parse_total_count(total_raw)
        male = parse_male_count(male_raw)

        # debug examples
        male_text = str(male_raw).strip().lower()

        if not is_invalid(male_raw):
            nums_in_male = extract_numbers(male_raw)

            if ("female" in male_text or "females" in male_text or re.search(r"\bf\b", male_text)) and len(parser_examples["male_with_female_text"]) < 5:
                parser_examples["male_with_female_text"].append(male_raw)

            elif (re.search(r"\bmale[s]?\b", male_text) or re.search(r"\bm\b", male_text)) and len(parser_examples["male_labeled"]) < 5:
                parser_examples["male_labeled"].append(male_raw)

            elif len(nums_in_male) > 1 and len(parser_examples["multiple_numbers_summed"]) < 5:
                parser_examples["multiple_numbers_summed"].append(male_raw)

            elif len(nums_in_male) == 1 and len(parser_examples["plain_numeric"]) < 5:
                parser_examples["plain_numeric"].append(male_raw)

        # Skip invalid totals
        if pd.isna(total) or total == 0:
            skipped_invalid_total += 1
            continue

        # Missing gender info
        if pd.isna(male):
            missing_gender += 1
            skipped_male_missing += 1
            if not is_invalid(male_raw) and len(parser_examples["unparsed_male"]) < 5:
                parser_examples["unparsed_male"].append(male_raw)
            continue

        # Logical check
        if male > total:
            skipped_male_gt_total += 1
            continue

        female = total - male

        male_ratios.append(male / total)
        female_ratios.append(female / total)

    print("Valid papers:", valid_mask.sum())

    if extra_filter_cols:
        print("After subgroup filter:", filtered_count)

    print("Used in ratio calc:", len(male_ratios))

    print("\nMale Ratio → Mean:", np.mean(male_ratios) if len(male_ratios) > 0 else np.nan,
          "SD:", np.std(male_ratios, ddof=1) if len(male_ratios) > 1 else 0)

    print("Female Ratio → Mean:", np.mean(female_ratios) if len(female_ratios) > 0 else np.nan,
          "SD:", np.std(female_ratios, ddof=1) if len(female_ratios) > 1 else 0)

    print("Missing gender:", missing_gender)

    print("\n=== DEBUG BREAKDOWN ===")
    print("Skipped: invalid or zero total:", skipped_invalid_total)
    print("Skipped: missing parsed male count:", skipped_male_missing)
    print("Skipped: male > total:", skipped_male_gt_total)

    print("\n=== PARSER EXAMPLES ===")
    for k, vals in parser_examples.items():
        print(f"\n{k}:")
        if len(vals) == 0:
            print("None")
        else:
            for v in vals:
                print("-", repr(v))


# ---------------- RUN GENDER ---------------- #

# ASD
compute_gender_ratios(
    df_subset=df_subset,
    total_col=6,
    male_col=11,
    valid_mask=valid_ASD,
    label="ASD"
)

# NEUROTYPICALS
compute_gender_ratios(
    df_subset=df_subset,
    total_col=15,
    male_col=19,
    valid_mask=valid_Neur,
    label="NEUROTYPICALS",
    extra_filter_cols=list(range(15, 23))
)

# OTHER DIAGNOSES
compute_gender_ratios(
    df_subset=df_subset,
    total_col=24,
    male_col=28,
    valid_mask=valid_Other,
    label="OTHER DIAGNOSES",
    extra_filter_cols=list(range(24, 33))
)

# ===================== AUTISM DIAGNOSIS METHODS ===================== #

def compute_diagnosis_methods(col, valid_mask, label):
    print(f"\n=============DIAGNOSIS METHODS: {label}=============")

    # Filter to valid papers only
    col_filtered = col[valid_mask].astype(str).str.lower().str.strip()
    total_valid = valid_mask.sum()

    # ===================== PATTERNS ===================== #
    pattern_dsm = (
        r"\bdsm\b"
        r"|\bdsm-\b"
        r"|diagnostic\s*and\s*statistical\s*manual\s*of\s*mental\s*disorders"
    )

    pattern_ados = (
        r"\bados\b"
        r"|\bados-\d+\b"
        r"|autism\s*diagnostic\s*observation\s*schedule"
    )

    pattern_cars = (
        r"\bcars\b"
        r"|\bcars-\d+\b"
        r"|childhood\s*autism\s*rating\s*scale"
    )

    pattern_adi = (
        r"\badi\b"
        r"|\badi-\w+\b"
        r"|autism\s*diagnostic\s*interview"
    )

    pattern_icd = (
        r"\bicd\b"
        r"|\bicd-\w+\b"
        r"|international\s*classification\s*of\s*diseases"
    )

    # ===================== MATCHES ===================== #
    dsm_count = col_filtered.str.contains(pattern_dsm, regex=True, na=False).sum()
    ados_count = col_filtered.str.contains(pattern_ados, regex=True, na=False).sum()
    cars_count = col_filtered.str.contains(pattern_cars, regex=True, na=False).sum()
    adi_count = col_filtered.str.contains(pattern_adi, regex=True, na=False).sum()
    icd_count = col_filtered.str.contains(pattern_icd, regex=True, na=False).sum()

    # ===================== NOT GIVEN ===================== #
    not_given = sum(is_invalid(x) for x in col_filtered)

    # ===================== OUTPUT ===================== #
    print("Total valid papers:", total_valid)

    print("\nDSM:")
    print("Count:", dsm_count)
    print("Percentage:", (dsm_count / total_valid) * 100 if total_valid else 0)

    print("\nICD")
    print("Count:", icd_count)
    print("Percentage:", (icd_count / total_valid) * 100 if total_valid else 0)

    print("\nADOS:")
    print("Count:", ados_count)
    print("Percentage:", (ados_count / total_valid) * 100 if total_valid else 0)

    print("\nCARS:")
    print("Count:", cars_count)
    print("Percentage:", (cars_count / total_valid) * 100 if total_valid else 0)

    print("\nADI:")
    print("Count:", adi_count)
    print("Percentage:", (adi_count / total_valid) * 100 if total_valid else 0)

    print("\nNot Given:")
    print("Count:", not_given)
    print("Percentage:", (not_given / total_valid) * 100 if total_valid else 0)

# ===================== RUN ===================== #

# Column M (index 12) = diagnosis method

compute_diagnosis_methods(df_subset.iloc[:, 12], valid_total, "ASD")