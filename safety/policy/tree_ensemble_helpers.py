from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from utils.utils import *


# Domains whose state features are real-valued. Every other benchmark is discrete
# (integer-valued features), where learned split thresholds must be integers.
CONTINUOUS_GAMES = {"bouncing_ball", "cart_pole", "inverted_pendulum", "stopping_car", "follow_car"}


def is_discrete_domain(benchmark):
    """True if the benchmark's state features are all integer-valued."""
    return get_game(benchmark) not in CONTINUOUS_GAMES


def round_splits(t0, i0, t1, i1):
    """Copy tree t0 (from node i0) into t1 (at node i1), rounding split thresholds up.
    For integer inputs `x < v` and `x < ceil(v)` select the same branch, so this
    keeps the policy unchanged on the discrete state space while making the
    thresholds exact integers."""
    if t0.is_internal(i0):
        split = t0.get_split(i0)
        t1.split(i1, split.feat_id, np.ceil(split.split_value))
        round_splits(t0, t0.left(i0),  t1, t1.left(i1))
        round_splits(t0, t0.right(i0), t1, t1.right(i1))
    else:
        for i in range(t0.num_leaf_values()):
            t1.set_leaf_value(i1, i, t0.get_leaf_value(i0, i))


def transform_addtree(at0):
    """Return a copy of AddTree at0 with every split threshold rounded up to an integer."""
    at1 = veritas.AddTree(at0.num_leaf_values(), at0.get_type())
    for t0 in at0:
        t1 = at1.add_tree()
        round_splits(t0, t0.root(), t1, t1.root())
    for i in range(at0.num_leaf_values()):
        at1.set_base_score(i, at0.get_base_score(i))
    return at1


def round_splits_if_discrete(at, benchmark):
    """Round split thresholds of a freshly learned ensemble when the domain is discrete."""
    if is_discrete_domain(benchmark):
        print(f"\tDiscrete domain ({get_game(benchmark)}): rounding split thresholds to integers.")
        return transform_addtree(at)
    return at


def train_or_load_ensemble(benchmark, prefix, repair_method, config, df, y, iteration, cache=False): 
    params = {
                "max_depth": config['max_depth'], 
                "n_estimators": config['num_trees'],
                "learning_rate": config['lr']
            }
    at, model_name = get_model(benchmark, prefix, repair_method, config['model_type'], 
                               params, df, y, iteration, cache=cache) 
    return at, model_name


def get_model_name(benchmark, prefix, model_type, max_depth, num_trees, lr, iteration=0):
    """Model naming function"""

    # p = f"_{prefix}" if prefix is not None else ""
    p = f"_{prefix}" if prefix else ""
    if lr:
        model_name = f"{benchmark}{p}_{max_depth}depth_{num_trees}trees_{lr}LR_it{iteration}"
    else:
        model_name = f"{benchmark}{p}_{max_depth}depth_{num_trees}trees_it{iteration}"
        
    return model_name


def xgb_params(y):
    return {
        "random_state": SEED,
        "n_jobs": 1,
        "objective": "multi:softmax",
        "eval_metric": "merror",
        "multi_strategy": "multi_output_tree",
        "tree_method": "hist",
        "num_class": np.max(np.unique(y)) + 1,  
        # ...
    }

def rf_params():
    return {
        "random_state": SEED,
        "n_jobs": 1,
        # ...
    }

def get_params(model_type, y):
    if model_type == "xgb":
        return xgb_params(y)
    elif model_type == "rf":
        return rf_params()
    else:
        raise NotImplementedError(f"No available params for {model_type} yet!")


def get_model(benchmark, prefix, repair_method, model_type, params, df, y, iteration=0, read_only=False,
              cache=False):
    """ Train or load a pretrained ensemble policy. 
        NOTE: manually set cache=True to load previosuly trained models at first iteration.
    """

    # HACK: for race linetrack instances, we manually copied mod linetrack models 
    #       and start from those
    if benchmark.startswith("race_linetrack") and iteration == 0 and \
        benchmark != "race_linetrack_30_20_16_16":
        cache = True

    model_name = get_model_name(benchmark, prefix, model_type, params['max_depth'], 
                                params['n_estimators'], lr=params['learning_rate'], iteration=iteration)

    model_path = get_ensemble_file(benchmark, prefix, repair_method, model_type,
                                   params['max_depth'], params['n_estimators'],
                                   params['learning_rate'], iteration)
    # print(benchmark, prefix, model_type, params['max_depth'], params['n_estimators'],
    #       params['learning_rate'], iteration)
    # print(f"Model path: {model_path}")
    if os.path.isfile(model_path) and cache:
        # load existing model
        # model = joblib.load(model_path)
        print(f"\n\tLoading {model_type} addtree from file: {model_path}")
        at = veritas.AddTree.read(model_path)
    elif read_only:
        raise FileNotFoundError(f"Model {model_path} not found!")
    else:
        # train new model
        if model_type == "rf":  params.pop("learning_rate", None)
        additional_params = get_params(model_type, y) 
        for k, v in additional_params.items():
            params[k] = v

        if model_type == "xgb":
            model = XGBClassifier(**params)
        elif model_type == "rf":
            model = RandomForestClassifier(**params)
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented yet!")
    
        xtrain, xtest, ytrain, ytest = train_test_split(df, y, test_size=TEST_SIZE, 
                                                        random_state=SEED, shuffle=True)
        
        # if a class is not present in the training set, add a dummy training sample
        expected_labels = set(range(get_num_outputs(benchmark, prefix)))
        missing_labels = expected_labels - set(np.unique(ytrain))
        for missing_class in missing_labels:
            dummy_sample = np.zeros((1, xtrain.shape[1]))
            xtrain = np.vstack((xtrain, dummy_sample))
            ytrain = np.vstack((ytrain, missing_class))  #
            print(f"WARNING: class {missing_class} is missing - adding dummy ex.")

        # Previous version: What if you have training data from classes 0 to 3 but not 4?
        # if len(np.unique(ytrain)) < np.max(np.unique(ytrain)) + 1: 
        #     dummy_sample = np.zeros((1, xtrain.shape[1]))
        #     missing_class = [i for i in range(np.max(np.unique(ytrain)) + 1) \
        #                      if i not in np.unique(ytrain)][0]
        #     xtrain = np.vstack((xtrain, dummy_sample))
        #     ytrain = np.vstack((ytrain, missing_class))
        #     print(f"WARNING: class {missing_class} is missing - adding dummy ex.")
        
        t = time.time()
        model.fit(xtrain, ytrain)
        t = time.time() - t
        print(f'\n\tNew {model_type} ensemble trained in {t:.3f} seconds.')
        # print(f"\tModel name: {model_name}")

        yhat_train = model.predict(xtrain)
        yhat_test = model.predict(xtest)
        print(f"\txgb: train acc. {accuracy_score(ytrain, yhat_train):.3f},"\
              f" test acc. {accuracy_score(ytest,  yhat_test):.3f}")

        # cache model in json format
        at = veritas.get_addtree(model, silent=True)
        is_correct = veritas.test_conversion(at, xtest, model.predict_proba(xtest))
        assert is_correct==True, f"Something wrong with conversion to addtree!"
        # discrete domains: make split thresholds exact integers (after the
        # conversion check, which compares against the float-threshold booster)
        at = round_splits_if_discrete(at, benchmark)
        # save as json
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "w") as f:
            f.write(at.to_json())
    # print("Successfully written to", model_path)
    # print(f"\tWorking with {at.num_leaf_values()}-class AddTree", \
      #    f"({len(at)} trees, {at.max_depth()} depth).")
    return at, model_name
