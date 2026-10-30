import re
import math
import numpy as np
import pandas as pd
import statistics
import click
from collections import defaultdict
from utils.utils import RESULT_DIR

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

BENCHMARKS = {
            'blocksworld': ['CA','CI'],
            #'npuzzle': [],
            'linetrack': [''],
            'beluga': ['swap_unsafe'],
            # (!) uncomment when switching to new naming scheme
            'one_way_line': ['det', 'non_det_with_park', 'non_det_no_park'],
            'two_way_line': ['det', 'non_det_with_park', 'non_det_no_park'],
            'mod_linetrack': ['det', 'non_det_icy_drop_park', 'non_det_fail_dec'],
            'race_linetrack': ['det', 'non_det_icy_drop_park', 'non_det_fail_dec'],
            'bouncing_ball': [],
            'cart_pole': [],
            'inverted_pendulum': [],
            'stopping_car': [],
            # 'flight_diversion': [],
        }

# new stuff
BASE_RENAMES = {
    "blocksworld_8_7": "8Blocks",
    "beluga_3_2": "Beluga (3 Jigs)",
    "beluga_4_2": "Beluga (4 Jigs)",
    "beluga_5_2": "Beluga (5 Jigs)",
    "beluga_6_2": "Beluga (6 Jigs)",
    "beluga_8_2": "Beluga (8 Jigs)",
    "one_way_line_17_10": "1 way line (17, 10)",
    "one_way_line_20_10": "1 way line (20, 10)",
    "one_way_line_30_20": "1 way line (30, 20)",
    "one_way_line_40_25": "1 way line (40, 25)",
    "one_way_line": "1 way line (15, 10)",
    "two_way_line_17_10": "2 way line (17, 10)",
    "two_way_line_20_10": "2 way line (20, 10)",
    "two_way_line_30_20": "2 way line (30, 20)",
    "two_way_line_40_25": "2 way line (40, 25)",
    "two_way_line": "2 way line (15, 10)",
    "mod_linetrack_17_10": "1 way line (17, 10)",
    "mod_linetrack_20_10": "1 way line (20, 10)",
    "mod_linetrack_30_20": "1 way line (30, 20)",
    "mod_linetrack_40_25": "1 way line (40, 25)",
    "mod_linetrack": "1 way line (15, 10)",
    "race_linetrack": "2 way line (15, 10)",
    "race_linetrack_17_10": "2 way line (17, 10)",
    "race_linetrack_20_10": "2 way line (20, 10)",
    "race_linetrack_30_20": "2 way line (30, 20)",
    "race_linetrack_40_25": "2 way line (40, 25)",
    "bouncing_ball": "Bouncing Ball",
    "bouncing_ball_0.01_0.01": "Bouncing Ball (0.01, 0.01)",
    "bouncing_ball_0.05_0.05": "Bouncing Ball (0.05, 0.05)",
    "bouncing_ball_0.1_0.1": "Bouncing Ball (0.1, 0.1)",
    "bouncing_ball_0.2_0.2": "Bouncing Ball (0.2, 0.2)",
    "bouncing_ball_0.5_0.5": "Bouncing Ball (0.5, 0.5)",
    "bouncing_ball_0.7_0.7": "Bouncing Ball (0.7, 0.7)",
    "inverted_pendulum": "Inverted Pendulum",
    "inverted_pendulum_0.01_0.01": "Inverted Pendulum (0.01, 0.01)",
    "inverted_pendulum_0.05_0.05": "Inverted Pendulum (0.05, 0.05)",
    "inverted_pendulum_0.1_0.1": "Inverted Pendulum (0.1, 0.1)",
    "inverted_pendulum_0.2_0.2": "Inverted Pendulum (0.2, 0.2)",
    "inverted_pendulum_0.5_0.5": "Inverted Pendulum (0.5, 0.5)",
    # "inverted_pendulum_0.7_0.7": "Inverted Pendulum (0.7, 0.7)",
    "stopping_car": "Follow Car",
    "stopping_car_0.01_0.01": "Follow Car (0.01, 0.01)",
    "stopping_car_0.05_0.05": "Follow Car (0.05, 0.05)",
    "stopping_car_0.1_0.1": "Follow Car (0.1, 0.1)",
    "stopping_car_0.2_0.2": "Follow Car (0.2, 0.2)",
    "stopping_car_0.5_0.5": "Follow Car (0.5, 0.5)",
    "stopping_car_0.7_0.7": "Follow Car (0.7, 0.7)",
    "cart_pole": "Cart Pole",
    "linetrack": "Transport",
    "flight_diversion_10_10": "Flight Diversion (10, 10)",
    "flight_diversion_15_15": "Flight Diversion (15, 15)",
    "flight_diversion_25_25": "Flight Diversion (25, 25)",
    "flight_diversion_50_50": "Flight Diversion (50, 50)",
}

# ICAPS26 stuff
# BASE_RENAMES = {
#     "blocksworld_8_7": "8Blocks",
#     "beluga_3_2": "Beluga (3 Jigs)",
#     "beluga_4_2": "Beluga (4 Jigs)",
#     "beluga_5_2": "Beluga (5 Jigs)",
#     "beluga_6_2": "Beluga (6 Jigs)",
#     "beluga_8_2": "Beluga (8 Jigs)",
#     "mod_linetrack_17_10": "1 way line (17, 10)",
#     "mod_linetrack_20_10": "1 way line (20, 10)",
#     "mod_linetrack_30_20": "1 way line (30, 20)",
#     "mod_linetrack_40_25": "1 way line (40, 25)",
#     "mod_linetrack": "1 way line (15, 10)",
#     "race_linetrack": "2 way line (15, 10)",
#     "race_linetrack_17_10": "2 way line (17, 10)",
#     "race_linetrack_20_10": "2 way line (20, 10)",
#     "race_linetrack_30_20": "2 way line (30, 20)",
#     "race_linetrack_40_25": "2 way line (40, 25)",
#     "bouncing_ball": "Bouncing Ball",
#     "bouncing_ball_0.01_0.01": "Bouncing Ball (0.01, 0.01)",
#     "bouncing_ball_0.05_0.05": "Bouncing Ball (0.05, 0.05)",
#     "bouncing_ball_0.1_0.1": "Bouncing Ball (0.1, 0.1)",
#     "bouncing_ball_0.2_0.2": "Bouncing Ball (0.2, 0.2)",
#     "bouncing_ball_0.5_0.5": "Bouncing Ball (0.5, 0.5)",
#     "bouncing_ball_0.7_0.7": "Bouncing Ball (0.7, 0.7)",
#     "inverted_pendulum": "Inverted Pendulum",
#     "inverted_pendulum_0.01_0.01": "Inverted Pendulum (0.01, 0.01)",
#     "inverted_pendulum_0.05_0.05": "Inverted Pendulum (0.05, 0.05)",
#     "inverted_pendulum_0.1_0.1": "Inverted Pendulum (0.1, 0.1)",
#     "inverted_pendulum_0.2_0.2": "Inverted Pendulum (0.2, 0.2)",
#     "inverted_pendulum_0.5_0.5": "Inverted Pendulum (0.5, 0.5)",
#     # "inverted_pendulum_0.7_0.7": "Inverted Pendulum (0.7, 0.7)",
#     "stopping_car": "Follow Car",
#     "stopping_car_0.01_0.01": "Follow Car (0.01, 0.01)",
#     "stopping_car_0.05_0.05": "Follow Car (0.05, 0.05)",
#     "stopping_car_0.1_0.1": "Follow Car (0.1, 0.1)",
#     "stopping_car_0.2_0.2": "Follow Car (0.2, 0.2)",
#     "stopping_car_0.5_0.5": "Follow Car (0.5, 0.5)",
#     "stopping_car_0.7_0.7": "Follow Car (0.7, 0.7)",
#     "cart_pole": "Cart Pole",
#     "linetrack": "Transport",
# }

PRINT_PREFIXES = {
        'non_det_icy_drop_park': 'Icy + Park',
        'non_det_fail_dec': 'Icy',
        'non_det_with_park': 'Icy + Park',
        'non_det_no_park': 'Icy',
        'det': 'Det',
        'swap_unsafe': 'Next Jig',
        'CA': 'CA',
        'CI': 'CI',
    }

SELECTED_INSTANCES = {
    # blocksworld
    ("8Blocks", "CA"),
    ("8Blocks", "CI"),
    # linetrack
    ("Transport", "Det"),
    # beluga
    ("Beluga (4 Jigs)", "Next Jig"),
    ("Beluga (5 Jigs)", "Next Jig"),
    ("Beluga (6 Jigs)", "Next Jig"),
    # 1 Way Line
    # ("1 way line (17, 10)", "Det"),
    # ("1 way line (40, 25)", "Det"),
    # ("1 way line (17, 10)", "Icy + Park"),
    # ("1 way line (30, 20)", "Icy + Park"),
    # ("1 way line (40, 25)", "Icy + Park"),
    # ("1 way line (15, 10)", "Icy"),
    # ("1 way line (17, 10)", "Icy"),
    ("1 way line (17, 10)", "Det"),
    ("1 way line (30, 20)", "Det"),
    ("1 way line (15, 10)", "Icy + Park"),
    ("1 way line (30, 20)", "Icy + Park"),
    ("1 way line (15, 10)", "Icy"),
    ("1 way line (30, 20)", "Icy"),
    # 2way line
    # ("2 way line (15, 10)", "Det"),
    # ("2 way line (17, 10)", "Det"),
    # ("2 way line (15, 10)", "Icy + Park"),
    # ("2 way line (20, 10)", "Icy + Park"),
    # ("2 way line (15, 10)", "Icy"),
    # ("2 way line (20, 10)", "Icy"),
    ("2 way line (17, 10)", "Det"),
    ("2 way line (30, 20)", "Det"),
    ("2 way line (15, 10)", "Icy + Park"),
    ("2 way line (30, 20)", "Icy + Park"),
    ("2 way line (15, 10)", "Icy"),
    ("2 way line (30, 20)", "Icy"),
    # control
    ("Bouncing Ball (0.01, 0.01)", "Det"),
    ("Inverted Pendulum", "Det"),
    ("Follow Car", "Det"),
    }

PRINT_PREFIX_BENCHMARK = {
    # from pairs of (prefix, base) to a single-columns print ready for latex
    ('CA', '8Blocks'): '8Blocks CA',
    ('CI', '8Blocks'): '8Blocks CI',
    ('', 'Transport'): 'Transport',
    ('Next Jig', 'Beluga (3 Jigs)'): 'Beluga (3 Jigs)',
    ('Next Jig', 'Beluga (4 Jigs)'): 'Beluga (4 Jigs)',
    ('Next Jig', 'Beluga (5 Jigs)'): 'Beluga (5 Jigs)',
    ('Next Jig', 'Beluga (6 Jigs)'): 'Beluga (6 Jigs)',
    ('Det', '1 way line (15, 10)'): '1 way line (15, 10)',
    ('Det', '1 way line (17, 10)'): '1 way line (17, 10)',
    ('Det', '1 way line (20, 10)'): '1 way line (20, 10)',
    ('Det', '1 way line (30, 20)'): '1 way line (30, 20)',
    ('Det', '1 way line (40, 25)'): '1 way line (40, 25)',
    ('Icy + Park', '1 way line (15, 10)'): '1 way line (15, 10) Icy + Park',
    ('Icy + Park', '1 way line (17, 10)'): '1 way line (17, 10) Icy + Park',
    ('Icy + Park', '1 way line (20, 10)'): '1 way line (20, 10) Icy + Park',
    ('Icy + Park', '1 way line (30, 20)'): '1 way line (30, 20) Icy + Park',
    ('Icy + Park', '1 way line (40, 25)'): '1 way line (40, 25) Icy + Park',
    ('Icy', '1 way line (15, 10)'): '1 way line (15, 10) Icy',
    ('Icy', '1 way line (17, 10)'): '1 way line (17, 10) Icy',
    ('Icy', '1 way line (20, 10)'): '1 way line (20, 10) Icy',
    ('Icy', '1 way line (30, 20)'): '1 way line (30, 20) Icy',
    ('Icy', '1 way line (40, 25)'): '1 way line (40, 25) Icy',
    ('Det', '2 way line (15, 10)'): '2 way line (15, 10)',
    ('Det', '2 way line (17, 10)'): '2 way line (17, 10)',
    ('Det', '2 way line (20, 10)'): '2 way line (20, 10)',
    ('Det', '2 way line (30, 20)'): '2 way line (30, 20)',
    ('Det', '2 way line (40, 25)'): '2 way line (40, 25)',
    ('Icy + Park', '2 way line (15, 10)'): '2 way line (15, 10) Icy + Park',
    ('Icy + Park', '2 way line (17, 10)'): '2 way line (17, 10) Icy + Park',
    ('Icy + Park', '2 way line (20, 10)'): '2 way line (20, 10) Icy + Park',
    ('Icy + Park', '2 way line (30, 20)'): '2 way line (30, 20) Icy + Park',
    ('Icy + Park', '2 way line (40, 25)'): '2 way line (40, 25) Icy + Park',
    ('Icy', '2 way line (15, 10)'): '2 way line (15, 10) Icy',
    ('Icy', '2 way line (17, 10)'): '2 way line (17, 10) Icy',
    ('Icy', '2 way line (20, 10)'): '2 way line (20, 10) Icy',
    ('Icy', '2 way line (30, 20)'): '2 way line (30, 20) Icy',
    ('Icy', '2 way line (40, 25)'): '2 way line (40, 25) Icy',
    ('Det', 'Bouncing Ball (0.01, 0.01)'): 'Bouncing Ball (0.01)',
    ('Det', 'Bouncing Ball (0.05, 0.05)'): 'Bouncing Ball (0.05)',
    ('Det', 'Bouncing Ball'): 'Bouncing Ball (0.5)',
    ('Det', 'Inverted Pendulum (0.01, 0.01)'): 'Inverted Pendulum (0.01)',
    ('Det', 'Inverted Pendulum (0.05, 0.05)'): 'Inverted Pendulum (0.05)',
    ('Det', 'Inverted Pendulum'): 'Inverted Pendulum (0.1)',
    ('Det', 'Follow Car (0.01, 0.01)'): 'Follow Car (0.01)',
    ('Det', 'Follow Car (0.05, 0.05)'): 'Follow Car (0.05)',
    ('Det', 'Follow Car'): 'Follow Car (0.4)',
}

SHORT_INSTANCES_NAMES = {
        '8Blocks CI': '8Blocks CI',
        '8Blocks CA': '8Blocks CA',
        'Transport': 'Transport',
        'Beluga (3 Jigs)': 'Beluga 3',
        'Beluga (4 Jigs)': 'Beluga 4',
        'Beluga (5 Jigs)': 'Beluga 5',
        'Beluga (6 Jigs)': 'Beluga 6',
        '1 way line (15, 10)': '1WL (15,10)',
        '1 way line (17, 10)': '1WL (17,10)',
        '1 way line (20, 10)': '1WL (20,10)',
        '1 way line (30, 20)': '1WL (30,20)',
        '1 way line (40, 25)': '1WL (40,25)',
        '1 way line (15, 10) Icy + Park': '1WL (15,10) I+P',
        '1 way line (17, 10) Icy + Park': '1WL (17,10) I+P',
        '1 way line (20, 10) Icy + Park': '1WL (20,10) I+P',
        '1 way line (30, 20) Icy + Park': '1WL (30,20) I+P',
        '1 way line (40, 25) Icy + Park': '1WL (40,25) I+P',
        '1 way line (15, 10) Icy': '1WL (15,10) I',
        '1 way line (17, 10) Icy': '1WL (17,10) I',
        '1 way line (20, 10) Icy': '1WL (20,10) I',
        '1 way line (30, 20) Icy': '1WL (30,20) I',
        '1 way line (40, 25) Icy': '1WL (40,25) I',
        '2 way line (15, 10)': '2WL (15,10)',
        '2 way line (17, 10)': '2WL (17,10)',
        '2 way line (20, 10)': '2WL (20,10)',
        '2 way line (30, 20)': '2WL (30,20)',
        '2 way line (40, 25)': '2WL (40,25)',
        '2 way line (15, 10) Icy + Park': '2WL (15,10) I+P',
        '2 way line (17, 10) Icy + Park': '2WL (17,10) I+P',
        '2 way line (20, 10) Icy + Park': '2WL (20,10) I+P',
        '2 way line (30, 20) Icy + Park': '2WL (30,20) I+P',
        '2 way line (40, 25) Icy + Park': '2WL (40,25) I+P',
        '2 way line (15, 10) Icy': '2WL (15,10) I',
        '2 way line (17, 10) Icy': '2WL (17,10) I',
        '2 way line (20, 10) Icy': '2WL (20,10) I',
        '2 way line (30, 20) Icy': '2WL (30,20) I',
        '2 way line (40, 25) Icy': '2WL (40,25) I',
        'Bouncing Ball (0.01)': 'BouncingBall (0.01)',
        'Bouncing Ball (0.05)': 'BouncingBall (0.05)',
        'Bouncing Ball (0.5)': 'BouncingBall (0.5)',
        'Inverted Pendulum (0.1)': 'InvPendulum (0.1)',
        'Inverted Pendulum (0.01$)': 'InvPendulum (0.01)',
        'Inverted Pendulum (0.05)': 'InvPendulum (0.05)',
        'Follow Car (0.01)': 'Follow Car (0.01)',
        'Follow Car (0.05)': 'Follow Car (0.05)',
        'Follow Car (0.4)': 'Follow Car (0.4)',
    }

def extract_prefix_base(name):
    benchmark_bases = list(BENCHMARKS.keys())
    valid_prefixes = [p for prefixes in BENCHMARKS.values() for p in prefixes]
    # Sort bases by descending length to prioritize more specific matches
    for base in sorted(benchmark_bases, key=len, reverse=True):
        if base in name:
            for prefix in sorted(valid_prefixes, key=len, reverse=True):
                if prefix in name:
                    return prefix, base
            return None, base
    return None, None

def parse_dimensions(base):
    match = re.search(r'\((\d+),\s*(\d+)\)', base)
    if match:
        return int(match.group(1)), int(match.group(2))
    return float('inf'), float('inf')  # fallback for missing dimensions

def parse_float_dimensions(base):
    match = re.search(r'\(([\d\.]+),\s*([\d\.]+)\)', base)
    if match:
        return float(match.group(1)), float(match.group(2))
    return float(-1), float(-1)
    
def create_sort_key(row):
    base = row['Base']
    prefix = row['Prefix']

    if "Blocks" in base or base.startswith("8Blocks"):
        return (0, 0, 0, 0)
    elif base == "Transport":
        return (1, 0, 0, 0)
    elif "Beluga" in base:
        jigs = int(re.search(r'\d+', base).group())
        return (2, jigs, 0, 0)
    elif base.startswith("1 way line"):
        dim1, dim2 = parse_dimensions(base)
        if prefix == "Det":
            return (3, 0, dim1, dim2)
        elif prefix == "Icy + Park":
            return (4, 0, dim1, dim2)
        elif prefix == "Icy":
            return (5, 0, dim1, dim2)
        else:
            return (3, 1, dim1, dim2)  # fallback
    elif base.startswith("2 way line"):
        dim1, dim2 = parse_dimensions(base)
        # return (6, 0, dim1, dim2)
        if prefix == "Det":
            return (6, 0, dim1, dim2)
        elif prefix == "Icy + Park":
            return (7, 0, dim1, dim2)
        elif prefix == "Icy":
            return (8, 0, dim1, dim2)
        else:
            return (6, 1, dim1, dim2)  # fallback
    # elif base == "Bouncing Ball":
    elif base.startswith("Bouncing Ball"):
        dim1, dim2 = parse_float_dimensions(base)
        return (9, 0, dim1, dim2)
    # elif base == "Inverted Pendulum":
    elif base.startswith("Inverted Pendulum"):
        dim1, dim2 = parse_float_dimensions(base)
        return (10, 0, dim1, dim2)
    # elif base == "Follow Car":
    elif base.startswith("Follow Car"):
        dim1, dim2 = parse_float_dimensions(base)
        return (11, 0, dim1, dim2)
    else:
        return (12, 0, 0, 0)
    
def format_runtime(val):
    # skip if string
    if isinstance(val, str):
        return val
    if pd.isna(val):
        return "-"
    val = int(val)
    if val == 0:
        return "$<$1\\si{{s}}"
    elif val < 60:
        return f"{val}\\si{{s}}"
    elif val < 3600:
        return f"{val // 60}\\si{{m}}"
    else:
        return f"{val // 3600}\\si{{h}}"

def _print_benchmark(row: pd.Series) -> str:
    base = str(row['Base']) if pd.notna(row['Base']) else ""
    prefix = str(row['Prefix']) if pd.notna(row['Prefix']) else ""
    return PRINT_PREFIX_BENCHMARK.get((prefix, base), f"{base} {prefix}".strip())

def to4floats(s: str) -> tuple[float, float, float, float]:
    try:
        a,b,c,d = map(float, s.split('_'))
        return a,b,c,d
    except Exception:
        return (float('nan'),)*4
    

# --------------------------
# Build comparison (UNION of rows; fill "-" for missing side)
# --------------------------
def build_comparison_union(
        df_g: pd.DataFrame, df_rl: pd.DataFrame, df_ra: pd.DataFrame, df_p: pd.DataFrame
) -> pd.DataFrame:
    key_cols = ['Base','Prefix']

    # Union of keys present in either dataframe
    union_keys = pd.concat([df_g[key_cols], df_rl[key_cols], df_ra[key_cols], df_p[key_cols]], ignore_index=True)\
        .drop_duplicates()

    # Align both methods to the union of keys
    g = pd.merge(union_keys, df_g, on=key_cols, how='left')
    rl = pd.merge(union_keys, df_rl, on=key_cols, how='left')
    ra = pd.merge(union_keys, df_ra, on=key_cols, how='left')
    p = pd.merge(union_keys, df_p, on=key_cols, how='left')

    keep = key_cols + [
        'Outcome',
        '%Goal Orig.','%Unsafe Orig.',
        # '%Goal Final','%Unsafe Final',
        '%Goal','%Unsafe',
        '%Goal (No Ver.)','%Unsafe (No Ver.)',
        '# Faults',  
        '# AT',
        '# Loop',
        '# Ver.',
        'Runt. FA','Runt. Repair','Runt. Ver.', 'Runt. Tot'
    ]
    g = g[keep].add_prefix('G_')
    rl = rl[keep].add_prefix('RL_')
    ra = ra[keep].add_prefix('RA_')
    p = p[keep].add_prefix('P_')

    comp = pd.concat([g, rl, ra, p], axis=1)
    comp = comp.rename(columns={'G_Base':'Base','G_Prefix':'Prefix'}) \
               .drop(columns=['RL_Base','RL_Prefix','RA_Base','RA_Prefix', 'P_Base', 'P_Prefix'])
    # Unified original metrics: prefer Gurobi if present else Retraining
    comp['%Goal Orig.']   = np.where(comp['G_%Goal Orig.'].notna(),   comp['G_%Goal Orig.'],   comp['RL_%Goal Orig.'])
    comp['%Unsafe Orig.'] = np.where(comp['G_%Unsafe Orig.'].notna(), comp['G_%Unsafe Orig.'], comp['RL_%Unsafe Orig.'])

    # Sorting like results.py
    comp['__sort_key'] = comp.apply(create_sort_key, axis=1)
    comp[['__s0','__s1','__s2','__s3']] = pd.DataFrame(comp['__sort_key'].tolist(), index=comp.index)

    order = [
        'Base','Prefix',
        # All %Unsafe columns first: unified origin before method-specific finals
        '%Unsafe Orig.',
        # 'G_%Unsafe Final','RL_%Unsafe Final','RA_%Unsafe Final',
        'G_%Unsafe','RL_%Unsafe','RA_%Unsafe', 'P_%Unsafe',
        'G_%Unsafe (No Ver.)', 'RL_%Unsafe (No Ver.)', 'RA_%Unsafe (No Ver.)','P_%Unsafe (No Ver.)',
        # Then all %Goal columns: unified origin first, then method-specific finals
        '%Goal Orig.',
        # 'G_%Goal Final','RL_%Goal Final','RA_%Goal Final',
        'G_%Goal','RL_%Goal','RA_%Goal', 'P_%Goal', 
        'G_%Goal (No Ver.)', 'RL_%Goal (No Ver.)', 'RA_%Goal (No Ver.)','P_%Goal (No Ver.)',
        # TMP: only keep performance columns
        'G_# Faults','RL_# Faults','RA_# Faults', 'P_# Faults',
        # #'G_# AT','RL_# AT','RA_# AT',
        # 'G_# Loop','RL_# Loop','RA_# Loop',
        # 'G_Runt. FA','RL_Runt. FA','RA_Runt. FA',
        # 'G_Runt. Repair','RL_Runt. Repair','RA_Runt. Repair',
        # # 'G_Runt. Ver.','RL_Runt. Ver.','RA_Runt. Ver.',
        'G_Outcome','RL_Outcome','RA_Outcome', 'P_Outcome',
    ]

    # 2nd table with remaining columns
    order2 = [
        'Base','Prefix',
        # 'G_# Faults','RL_# Faults','RA_# Faults',
        'G_# AT','RL_# AT','RA_# AT', 'P_# AT',
        'G_# Loop','RL_# Loop','RA_# Loop', 'P_# Loop',
        # 'G_# Its','RL_# Its','RA_# Its',
        'G_# Ver.','RL_# Ver.','RA_# Ver.', 'P_# Ver.',
        'G_Runt. FA','RL_Runt. FA','RA_Runt. FA', 'P_Runt. FA',
        'G_Runt. Repair','RL_Runt. Repair','RA_Runt. Repair', 'P_Runt. Repair',
        'G_Runt. Ver.','RL_Runt. Ver.','RA_Runt. Ver.', 'P_Runt. Ver.',
        'G_Runt. Tot','RL_Runt. Tot','RA_Runt. Tot', 'P_Runt. Tot',
        'G_Outcome','RL_Outcome','RA_Outcome', 'P_Outcome',
    ]

    comp_ = comp.copy()

    comp = comp[order + ['__s0','__s1','__s2','__s3','__sort_key']]
    comp = comp.sort_values(['__s0','__s1','__s2','__s3']) \
               .drop(columns=['__s0','__s1','__s2','__s3','__sort_key']) \
               .reset_index(drop=True)
    
    comp2 = comp_[order2 + ['__s0','__s1','__s2','__s3','__sort_key']]
    comp2 = comp2.sort_values(['__s0','__s1','__s2','__s3']) \
               .drop(columns=['__s0','__s1','__s2','__s3','__sort_key']) \
               .reset_index(drop=True)
    
    merged = comp.merge(
    comp2,
    on=["Base", "Prefix", 'G_Outcome','RL_Outcome','RA_Outcome','P_Outcome'],
    how="inner"   # or "left" / "outer" depending on your needs
    )

    merged["G_# Loop"] += 1
    merged["RL_# Loop"] += 1
    merged["RA_# Loop"] += 1
    merged["P_# Loop"] += 1

    return comp, comp2, merged


def prettify_base(df: pd.DataFrame) -> str:
    df['Benchmark'] = df.apply(_print_benchmark, axis=1)
    # put Benchmark first
    cols = df.columns.tolist()
    cols = ['Benchmark'] + [c for c in cols if c not in ['Benchmark','Base','Prefix']]
    df = df[cols]
    df = df.drop(columns=['Base','Prefix'], errors='ignore')
    return df

# Exclude specific row(s) from both the printed table and averages.
def exclude_benchmark(df: pd.DataFrame, base: str, prefix: str) -> pd.DataFrame:
    if df.empty:
        return df
    mask = df['Base'].astype(str).str.startswith(base) & (df['Prefix'] == prefix)
    if mask.any():
        df = df.loc[~mask].reset_index(drop=True)
    return df

METHOD_MAP = {
    "G": "MILP",
    "RL": "FALoss",
    "RA": "Hard",
    "P": "Penalty", 
    "Orig.": "Input"
}

METRIC_MAP = {
    "%Unsafe": r"\%Unsafe",
    "%Goal": r"\%Goal",
    "%Fixed": r"\%Fixed",
    "# Faults": r"\# Faults",
    "Outcome": "Outcome",
    "# Loop": r"\# Loop",
    "# Ver.": r"\# Ver.",
    "Runt. FA": "Runtime FA",
    "Runt. Repair": "Runtime Repair",
    "Runt. Ver.": "Runtime Ver.",
    "Runt. Tot": "Runtime Total",
    '# AT': r'\# AT'
}

def convert_to_multiindex(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    for col in df.columns:
        if col == "Benchmark":
            new_cols.append(("Benchmark", ""))
            continue

        # Split like: G_%Unsafe, RL_%Goal, RA_Outcome, etc.
        try:
            # if col has "Orig."
            if "Orig." in col:
                prefix = "Orig."
                # assign everything else to rest. Orig. can occur anywhere. Delete trailing spaces.
                rest = col.replace("Orig.", "").strip()
            else:
                prefix, rest = col.split("_", 1)
        except ValueError:
            new_cols.append((col, ""))
            continue

        if prefix in METHOD_MAP:
            method = METHOD_MAP[prefix]

            # Normalize metric names
            metric = METRIC_MAP.get(rest)
            new_cols.append((metric, method))
        else:
            new_cols.append((col, ""))

    df.columns = pd.MultiIndex.from_tuples(new_cols)
    return df