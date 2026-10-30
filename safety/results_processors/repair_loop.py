from utils.repair_methods import resolve_result_path
from results_processors.processor import Processor
from utils.utils import *
from results_processors.helpers.utils import *
from results_processors.helpers.utils import _print_benchmark

# Generates 3 tables reporting debugging loop results for one single repair method
# - Table 1: full results, all benchmarks
# - Table 2:full results, selected benchmarks (+ averages, outcome counts)
# - Table 3: ICAPS26 Table 1, only selected benchmarks and columns (commented out)


TABLE_HEADER = {
        'base': 'Base',
        'prefix': 'Prefix',
        # 'nn_gf': '%Goal NN',
        # tmp
        # 'original_gf_eval': '\%Goal Orig.',
        # 'final_gf_eval': '\%Goal MILP',
        # 'nn_gf': '%Goal NN',
        # end tmp
        'original_bf_eval': '\%Unsafe Orig.',
        'mid_bf_eval': '\%Unsafe (No Ver.)',
        'final_bf_eval': '\%Unsafe MILP',
        'bf_baseline_eval': '\%Unsafe Base',
        'original_gf_eval': '\%Goal Orig.',
        'mid_gf_eval': '\%Goal (No Ver.)',
        'final_gf_eval': '\%Goal MILP',
        # 'nn_gf': '%Goal NN',
        'gf_baseline_eval': '\%Goal Base',
        'num_faults': '\# Faults',
        'added_trees': '\# AT',
        'gurobi_iterations': '\# It Gurobi',
        'verification_iterations': '\# It Ver.',
        'runtime_fa': 'Runtime FA',
        # 'runtime_gur': 'Runtime Repair',
        'runtime_mr': 'Runtime Repair',
        'runtime_ver': 'Runtime Ver.',
        'testing_status': 'Test. Status',
        'verification_status': 'Ver. Status',
        'original_fidelity': 'Orig. Fid',
        'final_fidelity' : 'Final Fid'
    }

def print_summary_stats(df_summary):
    # print averages (only on Safe and Improved)
    filtered = df_summary[
        df_summary['Outcome'].isin(['Safe', 'Improved']) &
        ~((df_summary['Base'].str.startswith("1 way line")) & (df_summary['Prefix'] == "Icy"))
    ]
    print(len(filtered), "records in summary table after filtering")
    columns_to_avg = [
        '\%Goal Orig.', '\%Unsafe Orig.',
        '\%Goal (No Ver.)', '\%Unsafe (No Ver.)',
        '\%Goal MILP', '\%Unsafe MILP',
        '\%Goal Base', '\%Unsafe Base'
    ]
    # Replace "-" with NaN before converting to float, so they are ignored in mean calculation
    filtered[columns_to_avg] = filtered[columns_to_avg].replace("-", np.nan).astype(float)
    # print(filtered)
    # exit()
    averages = filtered[columns_to_avg].mean().round(1)
    print("\n### Averages ###\n")
    print(averages.to_string())

    # count how many outcomes for each status
    status_counts = df_summary['Outcome'].value_counts()
    print("\n### Outcome Counts ###\n")
    for status, count in status_counts.items():
        print(f"{status}: {count}")
        
def create_summary_table(detailed_results):
    df = pd.DataFrame(detailed_results)

    # Flatten runtimes
    runtime_df = df['runtimes'].apply(pd.Series)
    runtime_df = runtime_df.rename(columns={
        'fa': 'runtime_fa',
        # 'milp-l1': 'runtime_gur',
        'repair': 'runtime_mr',
        'ver': 'runtime_ver'
    })
    # df = pd.concat([df.drop(columns=['runtimes']), runtime_df[['runtime_fa', 'runtime_gur', 'runtime_ver']]], axis=1)
    df = pd.concat([df.drop(columns=['runtimes']), runtime_df[['runtime_fa', 'runtime_mr', 'runtime_ver']]], axis=1)

    df['base'] = df.apply(lambda row: row['benchmark'] if not row['prefix'] else row['benchmark'][:-len(row['prefix'])-1], axis=1)
    df['base'] = df['base'].apply(lambda x: '_'.join(x.split('_')[:-2]) if '_' in x else x)
    df['base'] = df['base'].map(BASE_RENAMES).fillna(df['base'])
    df['prefix'] = df['prefix'].map(PRINT_PREFIXES).fillna(df['prefix'])

    # Select + rename columns
    column_order = list(TABLE_HEADER.keys())
    df = df[column_order]
    df = df.rename(columns=TABLE_HEADER)

    # Sort rows
    df['__sort_key'] = df.apply(create_sort_key, axis=1)
    df = df.sort_values(by='__sort_key').drop(columns='__sort_key')

    # merge status columns into a single column
    df['Outcome'] = df.apply(
        lambda row: 'Safe' if row['Ver. Status'] == 'SAFE' else
                    'Unsafe' if row['Ver. Status'] == 'UNSAFE' else
                    # 'MaxIt' if row['Ver. Status'] == 'MAXIT' else
                    'MaxIt' if row['Ver. Status'] == 'MAXIT' or row['Test. Status'] == 'maxit-retraining' else
                    'Improved' if row['Ver. Status'] in ['TIMEOUT', 'OOM', 'N/A'] 
                        and row['Test. Status'] in ['optimal', 'suboptimal'] else
                    'Walltime' if row['Test. Status'] == 'walltime' else
                    'OOM' if row['Test. Status'] == 'oom-repair' else
                    'Gurobi TO' if row['Test. Status'] == 'timeout' else
                    'FA OOM' if row['Test. Status'] == 'oom' else
                    'No faults' if row['Test. Status'] == 'no_faults' else
                    row['Test. Status'], axis=1)
    df.drop(columns=['Test. Status', 'Ver. Status'], inplace=True)

    # escape LaTeX special characters
    df = df.replace('_', r'\_', regex=True)
    df = df.replace('#', r'\#', regex=True)
    df = df.replace('%', r'\%', regex=True)

    return df

def format_final_table(df, format_runtimes=True):
    # Round goal/bug values
    percent_cols = [col for col in df.columns if col.startswith('\%Goal') or col.startswith('\%Unsafe')]
    for col in percent_cols:
        df[col] = df[col].map(lambda x: "-" if pd.isna(x) else f"{x:.1f}")

    # if verification time is >12h, leave TO there
    df.loc[df['Runtime Ver.'] > 43000, 'Runtime Ver.'] = "TO" #43200
    # if verification time is <12h but Outcome is "Improved", set it to "OOM"
    mask = (pd.to_numeric(df['Runtime Ver.'], errors='coerce') < 43000) & (df['Outcome'] == 'Improved')
    df.loc[mask, 'Runtime Ver.'] = "OOM"
    # df.loc[mask, 'Runtime Ver.'] = "-"

    # Format runtimes
    if format_runtimes:
        for col in ['Runtime Repair', 'Runtime Ver.', 'Runtime FA']:
            df[col] = df[col].apply(format_runtime)

    # Replace NaNs with '-'
    df = df.replace("nan", "-").replace(np.nan, "-")

    # if outcome is "FA OOM" or "Gurobi TO", set all final and baseline values to NaN
    df.loc[df['Outcome'].isin(['FA OOM', 'Gurobi TO']),
            [
                '\%Goal MILP', '\%Unsafe MILP',
                '\%Goal Base', '\%Unsafe Base',
                '\%Goal (No Ver.)', '\%Unsafe (No Ver.)',
            ]] = "-"

    # replace columns Base + Prefix with single column
    df['Benchmark'] = df.apply(_print_benchmark, axis=1)
    # reorder columns to have Benchmark first
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('Benchmark')))
    df = df[cols]
    # df.drop(columns=['Base', 'Prefix'], inplace=True)

    return df

def parse_results_file(filename):
    """Parse a repair-loop result file into the list of per-run dicts."""
    with open(resolve_result_path(filename), 'r') as f:
        data = f.readlines()

    detailed_results = []

    for line in data:
        # print(line, "\n")
        # "no data" placeholders contain a space: collapse them so the field count is right
        parts = line.strip().replace("no data", "nan").split()
        if len(parts) < 12:
            continue
        # print(parts)
        # try:
        benchmark_full, __, block_nn, num_faults, added_trees, block_test, block_orig, block_mid, block_final, block_baseline, block_runtime, block_ver, block_fidelity = parts
        if block_ver == "0_None":
            block_ver = "N/A"

        prefix, base = extract_prefix_base(benchmark_full)
        if base is None:
            continue

        tot_its, testing_status = block_test.split("_")
        if block_test == "0_None":
            testing_status = "N/A"
            tot_its = False

        if testing_status=="safe":
            testing_status = "no_faults"

        # parse neural policy performance
        try:
            nn_gf, nn_bf, nn_cf = map(float, block_nn.split("_")[:3])
        except ValueError:
            nn_gf, nn_bf, nn_cf = (float('nan'), float('nan'), float('nan'))

        # parse original model performance
        try:
            orig_gf_train, orig_bf_train, original_gf_eval, original_bf_eval = map(float, block_orig.split("_"))
        except ValueError:
            orig_gf_train, orig_bf_train, original_gf_eval, original_bf_eval = \
                (float('nan'), float('nan'), float('nan'), float('nan'))

        # parse final model performance
        try:
            final_gf_train, final_bf_train, final_gf_eval, final_bf_eval = map(float, block_final.split("_"))
        except ValueError:
            final_gf_train, final_bf_train, final_gf_eval, final_bf_eval = \
                (float('nan'), float('nan'), float('nan'), float('nan'))

        # parse model performance before verification
        try:
            mid_gf_train, mid_bf_train, mid_gf_eval, mid_bf_eval = map(float, block_mid.split("_"))
        except ValueError:
            mid_gf_train, mid_bf_train, mid_gf_eval, mid_bf_eval = \
                (float('nan'), float('nan'), float('nan'), float('nan'))

        # parse baseline model performance
        try:
            gf_baseline_train, bf_baseline_train, gf_baseline_eval, bf_baseline_eval = \
                map(float, block_baseline.split("_"))
        except ValueError:
            gf_baseline_train, bf_baseline_train, gf_baseline_eval, bf_baseline_eval = \
                (float('nan'), float('nan'), float('nan'), float('nan'))
        # parse fidelity
        try:
            orig_fid, final_fid = map(float, block_fidelity.split("_"))
        except ValueError:
            orig_fid, final_fid = (float('nan'), float('nan'))

        # parse verification status
        try:
            verification_status = block_ver.split('_')[1]
            ver_its = block_ver.split('_')[0]
        except IndexError:
            verification_status = "N/A"
            ver_its = 0
        gur_its = int(tot_its) - int(ver_its) if tot_its else 0

        # parse runtimes
        runtime_fields = [
            # 'tot', 'gf_bf_o', 'fa', 'mr', 'gur', 'gf_bf_mid', 'gf_bf_f', 'pen_bl', 'gf_bf_b', 'ver'
            'tot', 'performance_origin',  'fa', 'repair', 'gurobi',
            'performance_fa_only', 'performance_final',
            'penalty_trees', 'performance_penalty_trees', 'ver'

        ]
        runtime_values = block_runtime.split("_")
        runtimes = {}
        for i, key in enumerate(runtime_fields):
            try:
                runtimes[key] = int(runtime_values[i])
            except (IndexError, ValueError):
                runtimes[key] = float('nan')

        detailed_results.append({
            'benchmark': benchmark_full,
            'prefix': prefix,
            'base': base,
            'testing_status': testing_status,
            'verification_status': verification_status,
            'nn_gf': nn_gf,
            'nn_bf': nn_bf,
            'nn_cf': nn_cf,
            'num_faults': int(num_faults),
            'added_trees': int(added_trees) if added_trees.isdigit() else 0,
            'original_gf_train': orig_gf_train,
            'original_bf_train': orig_bf_train,
            'original_gf_eval': original_gf_eval,
            'original_bf_eval': original_bf_eval,
            'mid_gf_train': mid_gf_train,
            'mid_bf_train': mid_bf_train,
            'mid_gf_eval': mid_gf_eval,
            'mid_bf_eval': mid_bf_eval,
            'final_gf_train': final_gf_train,
            'final_bf_train': final_bf_train,
            'final_gf_eval': final_gf_eval,
            'final_bf_eval': final_bf_eval,
            'gf_baseline_train': gf_baseline_train,
            'bf_baseline_train': bf_baseline_train,
            'gf_baseline_eval': gf_baseline_eval,
            'bf_baseline_eval': bf_baseline_eval,
            'runtimes': runtimes,
            'gurobi_iterations': int(gur_its),
            'verification_iterations': int(ver_its),
            'original_fidelity': round(orig_fid, 3),
            'final_fidelity': round(final_fid, 3)
        })

    return detailed_results

class RepairLoopResults(Processor):
    def get_result_file_name(self):        
        lr_string = f"_{self.tree_config['lr']}" if self.tree_config['model_type'] == "xgb" else ""
        tree_string = f"{self.tree_config['model_type']}_{self.tree_config['num_trees']}_"\
                        f"{self.tree_config['max_depth']}{lr_string}"
        
        mr_string = f"{self.mr_config['iterations']}it" \
                        f"_{self.mr_config['repair_method']}" 
                        # f"_flonly{mr_config['fl_only']}" 
        # compact_string = "_compact" if fa_config['compact'] else ""
        verifier_string = f"_{self.fa_config['verifier']}" if self.fa_config['run_verification'] else ""
        fa_string = f"{self.fa_config['num_starts']}N{verifier_string}" #{compact_string}"
        filename = f"{RESULT_DIR}/results-{tree_string}-{mr_string}-{fa_string}.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        return f"{RESULT_DIR}/results-{tree_string}-{mr_string}-{fa_string}.txt"
    
    # verification status used for the snapshot line written before verification starts
    PROVISIONAL_VER_STATUS = "PENDING"

    def write_results(self, logs, provisional=False, **kwargs):
        """Append one results line.

        provisional=True writes a snapshot taken right before verification, with
        verification status PENDING, so that an uncatchable kill during
        verification (cgroup OOM-killer, condor_rm) still leaves a record.
        The next non-provisional write for the same benchmark removes that
        snapshot, since the table parsers keep the first line per benchmark.
        File access is serialized with an exclusive lock: several jobs with the
        same configuration append to the same file.
        """
        import fcntl
        runtime = kwargs.get("runtime")
        prefix = kwargs.get("prefix")
        benchmark = kwargs.get("benchmark")

        fa_total_time = self.fa_config['fa_total_time']
        fa_tq = self.fa_config['fa_time_per_query']
        vtt = self.fa_config['verification_total_time']

        p = f"_{prefix}" if prefix else ""
        benchmark_string = f"{benchmark}{p}"
        ver_status = self.PROVISIONAL_VER_STATUS if provisional else logs['ver_status']

        plaja_string = f"{fa_total_time}_{fa_tq}_{vtt}"
        line = (
            f"{benchmark_string} {plaja_string} "
            f"{'_'.join(map(str, logs['perf_teacher']))} "
            f"{logs['#faults']} {logs['#trees']} "
            f"{logs['#its']}_{logs['repair_status']} "
            f"{'_'.join(map(str, logs['perf_origin']))} "
            f"{'_'.join(map(str, logs['perf_fa_only']))} "
            f"{'_'.join(map(str, logs['perf_final']))} "
            f"{'_'.join(map(str, logs['perf_penalty_trees']))} "
            f"{runtime['tot']}_"
            f"{runtime['performance_origin']}_"
            f"{runtime['fa']}_"
            f"{runtime['repair']}_"
            f"{runtime['fix']}_"
            f"{runtime['performance_fa_only']}_"
            f"{runtime['performance_final']}_"
            f"{runtime['penalty_trees']}_"
            f"{runtime['performance_penalty_trees']}_"
            f"{runtime['ver']} "
            f"{logs['#ver_its']}_{ver_status} "
            f"{logs['orig_fidelity']}_{logs['final_fidelity']} "
        )

        with open(self.get_result_file_name(), "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                if not provisional:
                    # drop the PENDING snapshot(s) of this benchmark, if any
                    f.seek(0)
                    content = f.read()
                    marker = f"_{self.PROVISIONAL_VER_STATUS} "
                    kept = [l for l in content.split("\n")
                            if not (l.startswith(benchmark_string + " ") and marker in l)]
                    if len(kept) != content.count("\n") + 1:
                        f.seek(0)
                        f.truncate()
                        f.write("\n".join(kept))
                f.seek(0, 2)
                f.write(f"\n{line}")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        return
    
    def results_to_latex(self):
        filename = self.get_result_file_name()
        detailed_results = parse_results_file(filename)

        # (1) full main table
        df_summary = create_summary_table(detailed_results)

        # HACK: manual fixes
        # df_summary.loc[df_summary['Base'] == "Beluga (4 Jigs)", "\# It Ver."] = 1
        # df_summary.loc[df_summary['Base'] == "Beluga (5 Jigs)", "\# It Ver."] = 1
        # df_summary.loc[df_summary['Base'] == "Beluga (4 Jigs)", "\# AT"] = 3
        # df_summary.loc[df_summary['Base'] == "Beluga (5 Jigs)", "\# AT"] = 19
        # df_summary.loc[df_summary['Base'] == "Beluga (6 Jigs)", "\# AT"] = 4
        # df_summary.loc[df_summary['Base'] == "Beluga (8 Jigs)", "\# AT"] = 5

        # wherever status if FA_OOM, other runtimes should be "-"
        df_summary.loc[df_summary['Outcome'] == 'FA OOM', \
                    ['Runtime FA', 'Runtime Repair', 'Runtime Ver.']] = "OOM", np.nan, np.nan
        # wherever status is Gurobi TO, verification runtime should be "-"
        df_summary.loc[df_summary['Outcome'] == 'Gurobi TO', ['Runtime Repair', 
                                                            'Runtime Ver.']] = "TO", np.nan

        # manually add some SAFE results for IC3 
        # df_summary.loc[(df_summary['Base'] == "8Blocks") & (df_summary['Prefix'] == "CI"), \
        #                ["Runtime Ver.", "Outcome"]] = 480, "Safe (IC3)" #8m
        # df_summary.loc[df_summary['Base'] == "Beluga (3 Jigs)", "Outcome"] = "Safe (both)"
        # df_summary.loc[df_summary['Base'] == "Beluga (4 Jigs)", ["Runtime Ver.", "Outcome"]] = \
        #                 240, "Safe (IC3)" #4m
        # df_summary.loc[df_summary['Base'] == "Beluga (5 Jigs)", ["Runtime Ver.", "Outcome"]] = \
        #                 18000, "Safe (IC3)" #5h
        # df_summary.loc[(df_summary['Base'] == "1 way line (15, 10)") & \
        #                (df_summary['Prefix'] == "Icy + Park"), \
        #                "Outcome"] = "Safe (both)"
        # df_summary.loc[(df_summary['Base'] == "1 way line (17, 10)") & \
        #                (df_summary['Prefix'] == "Icy + Park"), \
        #                "Outcome"] = "Safe (PPA)"
        # # df_summary.loc[(df_summary['Base'] == "1 way line (15,10)") & \
        # #                (df_summary['Prefix'] == "Icy"), \
        # #                 ["Runtime Ver.", "Outcome"]] = "-", "Loop TO"
        # # df_summary.loc[(df_summary['Base'] == "1 way line (17,10)") & \
        # #                (df_summary['Prefix'] == "Icy"), \
        # #                 ["Runtime Ver.", "Outcome"]] = "-", "Loop TO"
        # df_summary.loc[(df_summary['Base'] == "1 way line (15, 10)") & \
        #                (df_summary['Prefix'] == "Icy"), \
        #                 "Outcome"] = "Endless"
        # df_summary.loc[(df_summary['Base'] == "1 way line (17, 10)") & \
        #                (df_summary['Prefix'] == "Icy"), \
        #                 "Outcome"] = "Endless"
        # df_summary.loc[(df_summary['Base'] == "2 way line (15, 10)") & \
        #                (df_summary['Prefix'] == "Icy + Park"), \
        #                 "Outcome"] = "Safe (PPA)"
        # df_summary.loc[(df_summary['Base'] == "2 way line (15, 10)") & \
        #                (df_summary['Prefix'] == "Icy"), \
        #                 "Outcome"] = "Safe (PPA)"
        # df_summary.loc[(df_summary['Base'] == "2 way line (17, 10)") & \
        #                (df_summary['Prefix'] == "Icy"), \
        #                 "Outcome"] = "Safe (both)"

        # create new column "\# Loop" that sums Gurobi and Verifier iterations,
        # then delete Gurobi Its col.
        # Insert "\# Loop" column at the position of "\# It Gurobi"
        loop_col_idx = df_summary.columns.get_loc('\# It Gurobi')
        # (!) NOTE we're manually adding 1 here because current statistics miss iteration "0"
        df_summary.insert(loop_col_idx, '\# Loop', df_summary['\# It Gurobi'] + df_summary['\# It Ver.'] + 1)
        df_summary.drop(columns=['\# It Gurobi'], inplace=True)

        # Drop any rows containing Flight Diversion (tmp)
        df_summary = df_summary[~df_summary['Base'].str.contains('Flight Diversion', na=False)]

        # 1 way Icy is inevitably unsafe in all its start states
        df_summary.loc[(df_summary['Base'].str.startswith("1 way")) & (df_summary['Prefix'] == "Icy"), \
                    ['\%Unsafe Orig.', '\%Unsafe (No Ver.)', '\%Unsafe MILP', '\%Unsafe Base']] = 100.0
                    #    ['\%Unsafe Orig.', '\%Unsafe MILP']] = 100.0
        
        
        df_formatted = format_final_table(df_summary)
        print_cols = [c for c in df_formatted.columns if c not in ['Base', 'Prefix']]
        print_cols = [c for c in print_cols if not "Ver" in c]
        print_cols = [c for c in print_cols if not "Base" in c]
        # print(print_cols)
        # assert False
        print("\n### Summary Table ###\n")
        print(df_formatted[print_cols].to_string(index=False))
        # or export to LaTeX:
        print(df_formatted[print_cols].to_latex(index=False, escape=False))

        # (2) filter rows keeping only those where (Base, Prefix) are contained in SELECTED_INSTANCES
        #selected_instances = set(SELECTED_INSTANCES)
        df_summary = df_summary[df_summary.apply(lambda row: (row['Base'], row['Prefix']) in SELECTED_INSTANCES, axis=1)]
        df_formatted = df_formatted[df_formatted.apply(lambda row: (row['Base'], row['Prefix']) in SELECTED_INSTANCES, axis=1)]
        # df_formatted = df_summary #format_final_table(df_summary, format_runtimes=False)
        print("\n### Filtered Summary Table ###\n")
        print(df_formatted[print_cols].to_string(index=False))
        # or export to LaTeX:
        print(df_formatted[print_cols].to_latex(index=False, escape=False))

        # print stats
        print_summary_stats(df_summary)

        # or export to LaTeX:
        # print(df_formatted.to_latex(index=False, escape=False))

        # (3) ICAPS26: build a third table for selected instances only
        # (!) TMP off: we don't use ICAPS Table 1 anymore, kept for consistency
        # cut all columns except:
        # 'Benchmark', '\# Faults', '\# AT', '\# Loop', '\# It Ver., 'Runtime FA', 'Runtime Repair', 'Runtime Ver.', 'Outcome'
        # new_print_cols = ['Benchmark', '\# Faults', '\# AT', '\# Loop', '\# It Ver.', 
        #                   'Runtime FA', 'Runtime Repair', 'Runtime Ver.']
        # new_print_cols = ['Benchmark',  '\%Unsafe (No Ver.)', '\%Unsafe MILP', '\%Unsafe Base',
        #                 '\# Faults', '\# AT', 'Runtime FA', 'Runtime Repair', 'Runtime Ver.']
        
        # df_selected = df_formatted.copy()
        # change outcome column by reporting difference in \%Unsafe between Original and Final
        # def outcome_diff(row):
        #     if row['Outcome'] in ['Safe', 'Improved']:
        #         try:
        #             orig_unsafe = float(row['\%Unsafe Orig.'])
        #             final_unsafe = float(row['\%Unsafe MILP'])
        #             diff = orig_unsafe - final_unsafe
        #             return f"{diff:.1f}"
        #         except (ValueError, TypeError):
        #             return "-"
        #     else:
        #         return row['Outcome']
        # df_selected['MILP'] = df_selected.apply(outcome_diff, axis=1)
        # add a 2nd outcome column with difference between Original and Base
        # def outcome_diff_base(row):
        #     if row['Outcome'] not in ['FA OOM', 'Gurobi TO', 'MaxIt', 'No faults']:
        #         try:
        #             orig_unsafe = float(row['\%Unsafe Orig.'])
        #             base_unsafe = float(row['\%Unsafe Base'])
        #             diff = orig_unsafe - base_unsafe
        #             return f"{diff:.1f}"
        #         except (ValueError, TypeError):
        #             return "-"
        #     else:
        #         return row['Outcome']
        # df_selected['Penalty Trees'] = df_selected.apply(outcome_diff_base, axis=1)

        # print(df_selected)
        # df_selected['Benchmark'] = df_selected['Benchmark'].map(SHORT_INSTANCES_NAMES).fillna(df_selected['Benchmark'])

        # print("\n### ICAPS26 Table 1 ###\n")
        # # df_selected = df_selected[new_print_cols + ['Penalty Trees', 'MILP']]
        # df_selected = df_selected[new_print_cols]
        # print(df_selected.to_string(index=False))
        # # or export to LaTeX:
        # print(df_selected.to_latex(index=False, escape=False))
