from policy.nn_teacher import NNTeacher
from utils.utils import *
from policy.tree_ensemble_helpers import *

def load_training_records(data_file, num_inputs):
    """
    Read the teacher's <state, action> records from a .json file.

    Two layouts are accepted:
      * a JSON document  {"sample": [{"state": [...], "label": a}, ...]}
        (state[0] is a bookkeeping column and is dropped), or a bare JSON list of
        records with "state" and "selected_action"/"label";
      * one JSON record per line (in case PlaJA emitted line-delimited output).
    Returns (states, complete_states, labels).
    """
    with open(data_file, "r") as f:
        text = f.read()
    try:
        doc = json.loads(text)
        if isinstance(doc, dict) and "sample" in doc:
            records, drop_first = doc["sample"], True
        elif isinstance(doc, list):
            records, drop_first = doc, False
        else:
            raise ValueError(f"Unrecognised training-data layout in {data_file}")
    except json.JSONDecodeError:
        records, drop_first = [json.loads(l) for l in text.splitlines() if l.strip()], False

    states, complete_states, labels = [], [], []
    for entry in records:
        full = entry["state"][1:] if drop_first else entry["state"]
        states.append(full[0:num_inputs])
        complete_states.append(full)
        labels.append(entry["selected_action"] if "selected_action" in entry else entry["label"])
    return states, complete_states, labels

class EnsembleLearner():
    def __init__(self, mr_config, tree_config, fconfig, prefix, benchmark, debug):
        self.mr_config = mr_config
        self.tree_config = tree_config
        self.fconfig = fconfig
        self.prefix = prefix
        self.benchmark = benchmark
        self.nn_teacher = NNTeacher(prefix, benchmark)
        self.debug = debug
        
    def get_training_data(self, verbose = True):
        # check how many columns to keep reading interface file
        num_inputs = filter_data_columns(self.benchmark, self.prefix)

        # generate training data / NN stats with PlaJA only if they don't exist yet
        data_file = find_data_file(self.benchmark, self.prefix, self.tree_config["only_goal_data"])
        stat_file = find_stat_file(self.benchmark, self.prefix, self.tree_config["only_goal_data"])
        if not (os.path.exists(data_file) and os.path.exists(stat_file)):
            print(f"Generating training data from neural policy in {data_file}...")
            self.nn_teacher.generate_training_data(data_file, stat_file)

        print("\n\tLoading data from ", data_file)
        states, complete_states, labels = load_training_records(data_file, num_inputs)
        data = {"states": states, "labels": labels, "complete_states": complete_states}

        nn_perf = np.nan
        if verbose:
            nn_perf = self.nn_teacher.get_performance(stat_file)

        df = pd.DataFrame(data['states']).astype(np.float32).astype(veritas.FloatT)
        y = pd.DataFrame(data['labels'])
        df1 = pd.DataFrame(data['complete_states']).astype(np.float32).astype(veritas.FloatT)
        
        if verbose:
            print("\n\tClass distribution in the dataset: ", \
            y.value_counts(normalize=True).to_dict())

        # extract predicted class in case we're dealing with a Regressor
        if y.shape[1] > 1:
            print("\n\tNote: converting raw scores to class labels.\n")
            y = y.values.argmax(axis=1)
        return df, y, df1, nn_perf
    
    def learn(self, cache = True):
        # (!) set cache=True if we want to load previously trained models
        training_states, training_actions, training_complete_states, teacher_performance = \
            self.get_training_data(verbose=True)
        print("Training Data size:", len(training_states))
        if len(training_states) == 0:
            print("This configuration is invalid since none of the policy runs reach the goal")
            return training_states, training_actions, teacher_performance, None
        policy, __ = train_or_load_ensemble(self.benchmark, self.prefix, 
                                            self.mr_config["repair_method"], 
                                            self.tree_config, training_states, 
                                            training_actions, iteration=0, cache=cache)
        return training_states, training_actions, teacher_performance, policy
    
    def load(self, iteration):
        policy, __ = train_or_load_ensemble(self.benchmark, self.prefix, 
                                            self.mr_config["repair_method"], 
                                            self.tree_config, [], [], iteration, cache=True) 
        return policy
    
    def evaluate(self, iteration, uf_approx=False):
        # no evaluation on debug mode
        if self.debug:
            return (0,0,0,0), False
        (gf_train, uf_train, gf_eval, uf_eval), uf_approx = \
            evaluate_ensemble_performance(self.prefix, self.benchmark, 
                                          self.mr_config["repair_method"], self.tree_config, 
                                          iteration, self.fconfig, uf_approx, penalty=False)
        return (gf_train, uf_train, gf_eval, uf_eval), uf_approx
    
    def write_policy_to_file(self, policy, iterations):
        model_path = get_ensemble_file(self.benchmark, self.prefix, 
                                       self.mr_config["repair_method"],
                                       self.tree_config['model_type'], 
                                       self.tree_config["max_depth"], 
                                       self.tree_config["num_trees"], 
                                       self.tree_config["lr"], 
                                       iterations
                    )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "w") as f:
            f.write(policy.to_json())
        return
    
    def retrieve_policy(self, repair_method, iteration):
        """Load an already existing policy from disk (never trains)."""
        model_path = get_ensemble_file(self.benchmark, self.prefix, repair_method,
                                       self.tree_config['model_type'],
                                       self.tree_config["max_depth"],
                                       self.tree_config["num_trees"],
                                       self.tree_config["lr"],
                                       iteration)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"No stored policy at {model_path}")
        print(f"\n\tLoading addtree from file: {model_path}")
        return veritas.AddTree.read(model_path)