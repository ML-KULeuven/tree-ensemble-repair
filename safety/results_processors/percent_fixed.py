from utils.repair_methods import resolve_result_path
from results_processors.processor import Processor
from utils.utils import *
from results_processors.helpers.utils import *
from results_processors.helpers.utils import _print_benchmark

TABLE_HEADER = {
        'Base': 'Base',
        'Prefix': 'Prefix',
        '#States': '#States',
        'MILP%Fix': 'MILP%Fix',
        'FALoss%Fix': 'FALoss%Fix',
        'MH%Fix': 'MH%Fix',
    }

MULTI_INDEX_COLUMNS = pd.MultiIndex.from_tuples([
    ("Benchmark", ""),
    (r"$\lvert S\rvert $", ""),
    (r"\%Fix", "MILP"),
    (r"\%Fix", "FALoss"),
    (r"\%Fix", "MH"),
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
    df.columns = MULTI_INDEX_COLUMNS
    return df

def parse_values(s):
    per, fix, tot = s.split("_")
    return round(float(per), 2), int(fix), int(tot)

class PercentFixed(Processor):
    def get_result_file_name(self, no_ver=False, suffix=""):        
        lr_string = f"_{self.tree_config['lr']}" if self.tree_config['model_type'] == "xgb" else ""
        tree_string = f"{self.tree_config['model_type']}_{self.tree_config['num_trees']}_"\
                        f"{self.tree_config['max_depth']}{lr_string}"
        
        nover_str = "-NoVer" if no_ver else ""
        suffix_str = f"-{suffix}" if suffix else ""
        # return f"{RESULT_DIR}/percent-fixed-{tree_string}.txt"
        filename = f"{RESULT_DIR}/percent-fixed-{tree_string}{nover_str}{suffix_str}.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        return f"{RESULT_DIR}/percent-fixed-{tree_string}{nover_str}{suffix_str}.txt"
    
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
            print(line, "\n")
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            print(parts)
            benchmark_full, milp_l1, fa_loss, mh = parts
            prefix, base = extract_prefix_base(benchmark_full)
            print("Prefix:", prefix, "Base:", base)
            if base is None:
                continue

            gur_per, gur_fix, gur_tot = parse_values(milp_l1)
            fa_loss_per, fa_loss_fix, fa_loss_tot = parse_values(fa_loss)
            mh_per, mh_fix, mh_tot = parse_values(mh)
           
            
            # assert mh_tot == gur_tot and gur_tot == fa_loss_tot, "The total states used for the three methods is different"
            # if assertion triggered, report nan for the percentages to signal error
            if not (mh_tot == gur_tot == fa_loss_tot):
                print(f"Warning: Total states for MH ({mh_tot}), MILP-L1 ({gur_tot}) and FA Loss ({fa_loss_tot}) are different for benchmark {benchmark_full}. Reporting NaN for percentages and 0 for fixes.")
                gur_per, fa_loss_per, mh_per = float('nan'), float('nan'), float('nan')

            detailed_results.append({
                'benchmark': benchmark_full,
                'Prefix': prefix,
                'Base': base,
                '#States': gur_tot,
                'MILP%Fix': f"{gur_per:.2f}",
                'FALoss%Fix': f"{fa_loss_per:.2f}",
                'MH%Fix': f"{mh_per:.2f}",
            })
        df = create_summary_table(detailed_results)
        print(df.to_latex(
            index=False,
            escape=False,
            multicolumn=True,
            multicolumn_format='c',
            float_format="%.2f"
        ))        
        return
