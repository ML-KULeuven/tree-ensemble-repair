from utils.utils import *

################### FAULT ANALYSIS PARSERS ##############################

def parse_faults(file_path, benchmark, prefix, preserve_non_inputs=False):
    import re
    import pandas as pd
    import veritas  # assuming veritas.FloatT is required

    def parse_number(x):
        x = x.strip()
        try:
            return int(x)
        except ValueError:
            try:
                return float(x)
            except ValueError:
                raise ValueError(f"Cannot parse value: {x}")

    num_inputs = filter_data_columns(benchmark, prefix)

    faults = []
    with open(file_path, 'r') as f:
        content = f.read()
    fault_entries = content.split('###############################')
    
    for entry in fault_entries:
        source_match = re.search(r'Source:\s*\[([^\]]+)\]', entry)
        # target_match = re.search(r'Target:\s*\[([^\]]+)\]', entry)
        selected_action_match = re.search(r'Selected Action:\s*(\d+)', entry)
        applicable_actions_match = re.search(r'Applicable Actions:\s*\[([^\]]+)\]', entry)
        
        if source_match and selected_action_match and applicable_actions_match:
            source = [parse_number(x) for x in source_match.group(1).split(',')]
            # target = [parse_number(x) for x in target_match.group(1).split(',')]
            selected_action = int(selected_action_match.group(1))
            applicable_actions = [int(x.strip()) for x in applicable_actions_match.group(1).split()]
            
            # always drop first column
            # keep following num_inputs columns and drop remaining ones (if any)
            if preserve_non_inputs:
                source = source[1:]
            else:
                source = source[1:1+num_inputs]
            # target = target[1:1+num_inputs]
            faults.append({
                'source': source,
                # 'target': target,
                'selected_action': selected_action,
                'applicable_actions': applicable_actions
            })
   
    # create a dataframe with fault sources
    faults_df = pd.DataFrame(
        [fault['source'] for fault in faults], 
        dtype=veritas.FloatT
    )
    applicable_actions = [fault['applicable_actions'] for fault in faults]
    targets = [fault['selected_action'] for fault in faults]

    # check there's at least one fault - otherwise, we're done (policy is safe)
    # if faults_df.empty:
    #     print(f"\t- No faults found.")
    
    return faults_df, applicable_actions, pd.Series(targets)

def parse_policy_action_not_fault_logs(filename):
    num_sources = 0
    num_not_fault = 0

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            # Count sources
            if line.startswith("Source:"):
                num_sources += 1

            # Count "not a fault"
            if "Policy Action:" in line and "is not a fault." in line:
                num_not_fault += 1

    return num_sources, num_not_fault

def parse_policy_safe_logs(filename):
    num_states = 0
    num_unsafe = 0

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            # Count number of unsafe states
            if line.startswith("Number of unsafe states is:"):
                # get next number in line
                num_unsafe = int(line.split(":")[1].strip())
                
            # Count total nuber of states
            if line.startswith("Total states checked is:"):
                num_states = int(line.split(":")[1].strip())

    return num_states, num_unsafe

# TODO fix when implementing retraining-augment
# def store_faults_in_json(newest_file, benchmark, prefix, tree_config, it):
#     faults_json_path = faults_as_json_file(benchmark, prefix, tree_config["model_type"], tree_config["max_depth"], tree_config["num_trees"], tree_config["lr"], it)
#     def parse_number(x):
#         x = x.strip()
#         try:
#             return int(x)
#         except ValueError:
#             try:
#                 return float(x)
#             except ValueError:
#                 raise ValueError(f"Cannot parse value: {x}")

#     faults_map = {}  # key: tuple(state), value: set(actions)
#     filepaths = [newest_file.removesuffix(f'{it}.txt') + f'{max_it}.txt' for max_it in range(0, it + 1)]
#     for file_path in filepaths:
#         with open(file_path, 'r') as f:
#             content = f.read()
#         fault_entries = content.split('###############################')
        
#         for entry in fault_entries:
#             source_match = re.search(r'Source:\s*\[([^\]]+)\]', entry)
#             selected_action_match = re.search(r'Selected Action:\s*(\d+)', entry)
            
#             if source_match and selected_action_match:
#                 source = [parse_number(x) for x in source_match.group(1).split(',')]
#                 selected_action = int(selected_action_match.group(1))
                
#                 state_key = tuple(source)
#                 if state_key not in faults_map:
#                     faults_map[state_key] = set()
#                 faults_map[state_key].add(selected_action)
   
#     faults_json = [
#         {"state": list(state)[1:], "actions": sorted(list(actions))}
#         for state, actions in faults_map.items()
#     ]

#     # Write JSON file
#     with open(faults_json_path, 'w') as out_f:
#         json.dump(faults_json, out_f, indent=2)

#     print(f"✅ Faults written to {faults_json_path} ({len(faults_json)} unique states)")
    
#     return

def manual_faults_check(at, forbidden_x, forbidden_y, applicable_actions, sanity_check = True):
    """
    Check if the detected faults are valid by:
    - manually computing model output for any forbidden state
    - comparing label with forbidden_y (from PlaJA)
    """
    
    manual_y = pd.Series([
            actions[np.argmax(at.predict(fx)[0][actions])] 
            for fx, actions in zip(forbidden_x.values, applicable_actions)
        ])
    # assert np.all(list(manual_y) == forbidden_y), f"Something's off with faults labels!"
    mismatches = [
        (i, fy, fl) 
        for i, (fy, fl) in enumerate(zip(forbidden_y, manual_y)) 
        if fy != fl
    ]
    if not sanity_check:
        # in this case, we just want to check how many faults are actually faults for the passed policy
        log = {"#faults": len(forbidden_x), "#matches": len(forbidden_x) - len(mismatches)}
        # debug
        # return mismatches, log
        return log

    if mismatches:
        # print("Mismatches found:")
        print("\t\t - WARNING: something's off with faults labels!")
        for idx, fy, fl in mismatches:
            print(f"\t\t --> index {idx}: forbidden_y = {fy}, manual_y = {fl}", \
                  f" - dropping this fault...")
        print()
        # raise AssertionError(f"{len(mismatches)} mismatches found!")

        # drop mismatches from faults lists --> keep all that is not in mismatches idx
        mismatches_idx = [idx for idx, _, _ in mismatches]
        forbidden_x = forbidden_x.drop(mismatches_idx).reset_index(drop=True)
        applicable_actions = [
            actions for i, actions in enumerate(applicable_actions) 
            if i not in mismatches_idx
        ]
        forbidden_y = forbidden_y.drop(mismatches_idx).reset_index(drop=True)

    # return manual_y
    return forbidden_x, applicable_actions, forbidden_y


def attach_faults(forbidden_x, applicable_actions, forbidden_y, fx, aa, fy, \
                  at, no_stacking=False, duplicates_allowed=False, ignore_final_cols=0):
    """
    Attach new faults to the existing ones, removing duplicates.
    """
    
    if no_stacking:
        # throw away faults from previous iterations
        d = fx.duplicated()
        forbidden_x = fx[~d]
        applicable_actions = [a for i, a in enumerate(aa) if not d.iloc[i]]
        forbidden_y = pd.Series(fy[~d])
        return forbidden_x, applicable_actions, forbidden_y, np.sum(d)

    # check that no new faults were already there in previous iterations
    df_i = fx.copy()
    df_i['label'] = pd.Series(fy)
    df = forbidden_x.copy()
    df['label'] = pd.Series(forbidden_y)
    # assert pd.merge(df, df_i, how='inner').empty, \
    #     f"Some faults were already there in previous iterations! " \
    ### debug
    # wdf = pd.merge(df, df_i, how='inner')
    # if len(wdf) > 0:
    #     print(f"{len(wdf)} duplicates found:")
    #     print(wdf)
    #     for row in wdf.iterrows():
    #         print(f"State: {row[1].values[:-1]}, label: {row[1].values[-1]}")
    #         print(f"at_optimized prediction: {at.predict(row[1].values[:-1])}")
    ### end debug
    # di = df_i.duplicated() # duplicates within current iteration
    # df = pd.concat([df, df_i]).reset_index(drop=True) if not df.empty else df_i
    # duplicates = df.duplicated()

    # NEW add suport for ignoring latest column(s), 
    # --> in case we parsed faults with "preserve_non_inputs=True" 
    #     but want to allow duplicates that differ only in non-input features
    if ignore_final_cols > 0:
        state_cols = [c for c in df_i.columns if c != 'label']
        subset_cols = state_cols[:-ignore_final_cols] + ['label']
        di = df_i.duplicated(subset=subset_cols)
        df = pd.concat([df, df_i]).reset_index(drop=True) if not df.empty else df_i
        state_cols = [c for c in df.columns if c != 'label']
        subset_cols = state_cols[:-ignore_final_cols] + ['label']
        duplicates = df.duplicated(subset=subset_cols)
    else:
        di = df_i.duplicated() # duplicates within current iteration
        df = pd.concat([df, df_i]).reset_index(drop=True) if not df.empty else df_i
        duplicates = df.duplicated()


    if not duplicates_allowed:
        # if (di.sum() != duplicates.sum()):
        #     print(f"Some faults were already there in previous iterations! Have a look! TODO: RAISE ASSERTION HERE") 
        #     print("duplicates_allowed: ", duplicates_allowed)
        #  print(len(df_i), len(df), di.sum(), duplicates.sum())
        assert di.sum() == duplicates.sum(), \
            f"Some faults were already there in previous iterations!"
    
    # attach new faults and labels
    forbidden_x = pd.concat([forbidden_x, fx]).reset_index(drop=True)
    forbidden_x = forbidden_x[~duplicates]
    # applicable_actions.extend([a for i, a in enumerate(aa) if not di.iloc[i]])
    forbidden_y = pd.concat([forbidden_y, fy]).reset_index(drop=True) \
                    if not forbidden_y.empty else pd.Series(fy)
    forbidden_y = forbidden_y[~duplicates]

    # correction for when duplicates_ok = True: filter Aa using the same 'duplicates' mask
    Aa_all = applicable_actions + list(aa)
    keep_all = (~duplicates).to_numpy()
    applicable_actions = [a for a, k in zip(Aa_all, keep_all) if k]

    assert len(applicable_actions) == forbidden_x.shape[0] == forbidden_y.shape[0], \
        "Mismatch in faults lengths after attaching new faults!"
 
    # check duplicates were removed
    assert pd.concat([forbidden_x, forbidden_y], axis=1).duplicated().sum()==0,\
            "Duplicates still there!?"
    
    if not duplicates_allowed:
        return forbidden_x, applicable_actions, forbidden_y, np.sum(df_i.duplicated())
    else:
        # retraining: also count duplicates wrt prev. iterations
        return forbidden_x, applicable_actions, forbidden_y, np.sum(df.duplicated())
    
def attach_verification_fault(
    at, fx, aa, fy, Fx, Aa, Fy, no_stacking=False, duplicates_ok=False, ignore_final_cols=0
):
    # attach new fault(s) found with verification
    # to all faults found by inner loop
    if not Fx.empty:
        assert fx.shape[1] == Fx.shape[1], "Mismatch in faults shapes!"

    if fx.shape[0] > 1:
        print("\tWARNING: verification found more than one fault???")

    if at:
        __ = manual_faults_check(at, fx, fy, aa)

    Fx, Aa, Fy, __ = attach_faults(
        Fx, Aa, Fy, fx, aa, fy, at, no_stacking=no_stacking, 
        duplicates_allowed=duplicates_ok, ignore_final_cols=ignore_final_cols
    )
    
    print(f"\t- Added {len(fx)} fault(s) found by verification "\
            f"to the {len(Fx)-len(fx)} found with testing.")
    return Fx, Aa, Fy