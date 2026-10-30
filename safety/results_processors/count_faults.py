from utils.repair_methods import resolve_result_path
from results_processors.processor import Processor
from utils.utils import *
from results_processors.helpers.utils import *
from results_processors.helpers.utils import _print_benchmark

TABLE_HEADER = {
        'Base': 'Base',
        'Prefix': 'Prefix',
        'MILP-L1#States': 'MILP-L1#States',
        'Gur#NotRep': 'MILP-L1#NotRep',
        'Gur#SafePi' : 'MILP-L1#SafePi',
        'FALoss#States': 'FALoss#States',
        'FALoss#NotRep': 'FALoss#NotRep',
        'FALoss#SafePi' : 'FALoss#SafePi',
        'MH#States': 'MH#States',
        'MH#NotRep': 'MH#NotRep',
        'MH#SafePi' : 'MH#SafePi',
        'GurFALoss': 'GurFALoss',
        'GurMH': 'GurMH',
        'FALossMH': 'FALossMH'
    }

MULTI_INDEX_COLUMNS = pd.MultiIndex.from_tuples([
    ("Benchmark", ""),
    ("MILP-L1", r"$\lvert S\rvert $"),
    ("MILP-L1", r"\#NotRep"),
    ("MILP-L1", r"$\lvert safe\rvert $"),
    ("FALoss", r"$\lvert S\rvert $"),
    ("FALoss", r"\#NotRep"),
    ("FALoss", r"$\lvert safe\rvert $"),
    ("MH", r"$\lvert S\rvert $"),
    ("MH", r"\#NotRep"),
    ("MH", r"$\lvert safe\rvert $"),
    ("CommonFaults", "GurFALoss"),
    ("CommonFaults","GurMH"),
    ("CommonFaults", "FALossMH")
])

def create_summary_table(detailed_results):
    df = pd.DataFrame(detailed_results)
    df['Base'] = df.apply(lambda row: row['benchmark'] if not row['Prefix'] else row['benchmark'][:-len(row['Prefix'])-1], axis=1)
    df['Base'] = df['Base'].apply(lambda x: '_'.join(x.split('_')[:-2]) if '_' in x else x)
    df['Base'] = df['Base'].map(BASE_RENAMES).fillna(df['Base'])
    df['Prefix'] = df['Prefix'].map(PRINT_PREFIXES).fillna(df['Prefix'])

    # Select + rename columns
    column_order = list(TABLE_HEADER.keys())
    df = df[column_order]
    df = df.rename(columns=TABLE_HEADER)

    # Sort rows
    df['__sort_key'] = df.apply(create_sort_key, axis=1)
    df = df.sort_values(by='__sort_key').drop(columns='__sort_key')

    # escape LaTeX special characters
    df = df.replace('_', r'\_', regex=True)
    df = df.replace('#', r'\#', regex=True)
    df = df.replace('%', r'\%', regex=True)

    df['Benchmark'] = df.apply(_print_benchmark, axis=1)
    # reorder columns to have Benchmark first
    cols = df.columns.tolist()
    cols.insert(0, cols.pop(cols.index('Benchmark')))
    df = df[cols]
    
    df = df.drop(["Prefix", "Base"], axis=1)
    # df.columns = MULTI_INDEX_COLUMNS
    return df

class CountFaultsResults(Processor):
    def get_result_file_name(self):        
        lr_string = f"_{self.tree_config['lr']}" if self.tree_config['model_type'] == "xgb" else ""
        tree_string = f"{self.tree_config['model_type']}_{self.tree_config['num_trees']}_"\
                        f"{self.tree_config['max_depth']}{lr_string}"
        filename = f"{RESULT_DIR}/count_faults-{tree_string}.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        return f"{RESULT_DIR}/count_faults-{tree_string}.txt"
    
    def write_results(self, logs, **kwargs):
        prefix = kwargs.get("prefix")
        benchmark = kwargs.get("benchmark")
        import random
        import time
        time.sleep(1 + int(10 * random.random()))  # to avoid concurrent accesses running in parallel

        with open(self.get_result_file_name(), "a") as f:
            p = f"_{prefix}" if prefix else ""
            benchmark_string = f"{benchmark}{p}"

            f.write(f"\n{benchmark_string} ")
            
            for _, sub_dict in logs.items():
                first = True
                for key in sub_dict.keys():
                    if first:
                        f.write(f"{sub_dict[key]}")
                        first = False
                    else:
                        f.write(f"_{sub_dict[key]}")
                f.write(" ")
        return 
    
    def results_to_latex(self):
        filename = self.get_result_file_name()
        with open(resolve_result_path(filename), 'r') as f:
            data = f.readlines()
        detailed_results = []
        for line in data:
            # print(line, "\n")
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # print(parts)
            benchmark_full, same_faults_comparison, milp_l1_data, fa_loss_data, mh_data  = parts
            prefix, base = extract_prefix_base(benchmark_full)
            # print("Prefix:", prefix, "Base:", base)
            if base is None:
                continue
            
            gur_states, gur_no_rep_faults, gur_safe = map(int, milp_l1_data.split("_"))
            fa_loss_states, fa_loss_no_rep_faults, fa_loss_safe = map(int, fa_loss_data.split("_"))
            mh_loss_states, mh_loss_no_rep_faults, mh_loss_safe = map(int, mh_data.split("_"))
            gur_fa, gur_mh, fa_mh = map(int, same_faults_comparison.split("_"))
            
            detailed_results.append({
                'benchmark': benchmark_full,
                'Prefix': prefix,
                'Base': base,
                'MILP-L1#States': gur_states,
                'Gur#NotRep': gur_no_rep_faults,
                'Gur#SafePi': gur_safe,
                'FALoss#States': fa_loss_states,
                'FALoss#NotRep': fa_loss_no_rep_faults,
                'FALoss#SafePi': fa_loss_safe,
                'MH#States': mh_loss_states,
                'MH#NotRep': mh_loss_no_rep_faults,
                'MH#SafePi': mh_loss_safe,
                'GurFALoss': gur_fa,
                'GurMH': gur_mh,
                'FALossMH': fa_mh
            })
        df = create_summary_table(detailed_results)

        # TMP only keep tot + new faults columns
        cols_to_keep = ['Benchmark', 
                        'MILP-L1#States', 'MILP-L1#NotRep',
                        'FALoss#States', 'FALoss#NotRep', 
                        'MH#States', 'MH#NotRep'
        ]
        df = df[cols_to_keep]

        # print results
        print(df)
        # dfp = df.copy()
        # dfp.columns = MULTI_INDEX_COLUMNS
        print(df.to_latex(
            index=False,
            escape=False,
            multicolumn=True,
            multicolumn_format='c'
        ))        
        # return

        # === Additional: Compute and print #NotRep/#States for each method  ===
        def print_notrep_over_states_from_df(df):
            for method, states_col, notrep_col in [
                ("MILP-L1", "MILP-L1#States", "MILP-L1#NotRep"),
                ("FALoss", "FALoss#States", "FALoss#NotRep"),
                ("MH", "MH#States", "MH#NotRep"),
            ]:
                ratios = df[notrep_col] / df[states_col]
                # print(f"\n#NotRep/#States per benchmark for {method}:")
                # for bench, ratio in zip(df['Benchmark'], ratios):
                #     print(f"{bench}: {ratio:.4f}")
                print(f"Average #NotRep/#States for {method}: {ratios.mean():.4f}")

        # Uncomment to run this after results_to_latex or as needed
        print_notrep_over_states_from_df(df)
