import os
from utils.utils import BENCHMARK_DIR
############################# HELPER FUNCTIONS TO GET BENCHMARK FILES ##############################
def get_problem(benchmark):
    """
    Get the specific problem instance we are looking at. Gives hints to the number of objects in the domain.
    """
    return '_'.join(benchmark.split("_")[:-2]) or benchmark

def get_game(benchmark):
    """
    Each benchmark instance is part of a Planning Task. Get this task. 
    """
    p_to_g = {
        "blocksworld": "blocksworld",
        "sum_linetrack": "transport_feature",
        "mod_linetrack": "mod_transport",
        "race_linetrack": "race_transport",
        "linetrack": "transport",
        "beluga": "beluga",
        "npuzzle_3_3": "npuzzle",
        "bouncing_ball": "bouncing_ball",
        "stopping_car": "stopping_car",
        "cart_pole": "cart_pole",
        "inverted_pendulum": "inverted_pendulum",
        "flight_diversion": "flight_diversion",
        "one_way_line": "one_way_line",
        "two_way_line": "two_way_line"
    }
    for k in p_to_g.keys():
        if k in benchmark:
            return p_to_g[k]
    assert False, "Cannot find game name for benchmark: " + benchmark


def get_model_file(prefix, benchmark):
    """
    Get the model file for this benchmark. This specifies the state space as an Markov Decision Process or a Labelled Transition System.
    """
    game = get_game(benchmark)
    problem = get_problem(benchmark)
    dir_path = BENCHMARK_DIR
    if game == "blocksworld":
        model_file = f"{dir_path}/{game}/models/cost05/{problem}.jani"
    if game == "transport":
        model_file = f"{dir_path}/{game}/models/{problem}.jani"
    if game == "race_transport" or game == "mod_transport" or game == "one_way_line" or game == "two_way_line":
        model_file = f"{dir_path}/{game}/models/{prefix}/{problem}.jani"
    if game == "beluga":
        model_file = f"{dir_path}/{game}/models/{prefix}/{problem}.jani"
    if game == "bouncing_ball" or game == "stopping_car" or game == "cart_pole" or game == "inverted_pendulum":
       model_file = f"{dir_path}/{game}/models/{problem}.jani"
    if game == "flight_diversion":
       model_file = f"{dir_path}/{game}/models/{prefix}/{problem}.jani"
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file {model_file} does not exist.")
    return model_file

def get_fixed_property_file(prefix, benchmark, num_starts):
    """
    Load a fixed set of start states generated from the generator scripts inside the benchmarks directory. Good for aligning experiments over multiple clusters.
    """
    game = get_game(benchmark)
    problem = get_problem(benchmark)
    dir_path = BENCHMARK_DIR
    if game == "transport":
        property_file = f"{dir_path}/{game}/additional_properties/learning/random_starts_{num_starts}/linetrack/pa_{problem}_random_starts_{num_starts}.jani"
    if game == "blocksworld":
        property_file = f"{dir_path}/blocksworld/additional_properties/cost05/learning/random_starts_{num_starts}/{problem}/pa_{problem}_random_starts_{num_starts}.jani"
    if game == "npuzzle":
        property_file = f"{dir_path}/npuzzle/additional_properties/learning/cost05/random_starts_{num_starts}/npuzzle_3_3/pa_{problem}_random_starts_{num_starts}.jani"
    if game == "race_transport" or game == "mod_transport" or game == "one_way_line" or game == "two_way_line":
        property_file = f"{dir_path}/{game}/additional_properties/learning/{prefix}/random_starts_{num_starts}/{problem}/pa_{problem}_random_starts_{num_starts}.jani"
    if game == "beluga":
        property_file = f"{dir_path}/{game}/additional_properties/learning/{prefix}/random_starts_{num_starts}/{problem}/pa_{problem}_random_starts_{num_starts}.jani"
    if game == "bouncing_ball" or game == "stopping_car" or game == "cart_pole" or game == "inverted_pendulum":
        property_file = f"{dir_path}/{game}/additional_properties/learning/random_starts_{num_starts}/{problem}/pa_{problem}_random_starts_{num_starts}.jani"
    if game == "flight_diversion":
        property_file = f"{dir_path}/{game}/additional_properties/learning/{prefix}/random_starts_{num_starts}/{problem}/pa_{problem}_random_starts_{num_starts}.jani" 
        # assert False, "Flight diversion fixed set of start states not supported still"
        # property_file = f"{dir_path}/{game}/additional_properties/learning/random_starts_{num_starts}/{problem}/pa_{problem}_random_starts_{num_starts}.jani" 
    if not os.path.exists(property_file):
        raise FileNotFoundError(f"Property file {property_file} does not exist.")
    return property_file

def get_compact_property_file(prefix, benchmark):
    """
    Load the property file where the start condition is encoded as a symbolic formula. Start states can be sampled from this start condition using z3.
    """
    game = get_game(benchmark)
    problem = get_problem(benchmark)
    dir_path = BENCHMARK_DIR
    rddps_path = f"{dir_path}/../rddps22"
    if game == "transport":
        property_file = f"{dir_path}/transport/additional_properties/safety_verification/linetrack/compact_starts_no_predicates/no_filtering/pa_compact_starts_no_predicates_{benchmark}_0.jani"
    if game == "blocksworld":
        if prefix == "CA":
            property_file = f"{dir_path}/blocksworld/additional_properties/safety_verification/cost05/{problem}/compact_starts_no_predicates/learned_cost05_non_terminal/pa_compact_starts_no_predicates_{benchmark}_0.jani"
        if prefix == "CI":
            property_file = f"{dir_path}/blocksworld/additional_properties/safety_verification/cost05/{problem}/compact_starts_no_predicates/learned_no_cost/pa_compact_starts_no_predicates_{benchmark}_0.jani"
    if game == "npuzzle":
        if prefix == "CA":
            property_file = f"{dir_path}/npuzzle/additional_properties/safety_verification/cost05/{problem}/compact_starts_3_no_predicates/cost_no_filtering/pa_compact_starts_3_no_predicates_{benchmark}_0.jani"
        if prefix == "CI":
            property_file = f"{dir_path}/npuzzle/additional_properties/safety_verification/cost05/{problem}/compact_starts_3_no_predicates/no_cost_no_filtering/pa_compact_starts_3_no_predicates_{benchmark}_0.jani"
    if game == "race_transport" or game == "mod_transport" or game == "one_way_line" or game == "two_way_line":
        property_file = f"{dir_path}/{game}/additional_properties/safety_verification/{prefix}/compact_starts_no_predicates/{problem}/pa_{problem}_compact_starts_no_predicates.jani"
    if game == "beluga":
        property_file = f"{dir_path}/{game}/additional_properties/safety_verification/{prefix}/compact_starts_no_predicates/{problem}/pa_{problem}_compact_starts_no_predicates.jani"
    if game == "bouncing_ball" or game == "stopping_car" or game == "cart_pole" or game == "inverted_pendulum":
        property_file = f"{dir_path}/{game}/additional_properties/safety_verification/compact_starts_no_predicates/{problem}/pa_{problem}_compact_starts_no_predicates.jani"
    if game == "flight_diversion":
        property_file = f"{dir_path}/{game}/additional_properties/safety_verification/{prefix}/compact_starts_no_predicates/{problem}/pa_{problem}_compact_starts_no_predicates.jani" 
    if not os.path.exists(property_file):
        raise FileNotFoundError(f"Property file {property_file} does not exist.")
    return property_file

def get_property_file(prefix, benchmark, compact, num_starts=0):
    """
    Load the correct property file based on whether it is compact or fixed.
    """
    if compact:
        return get_compact_property_file(prefix, benchmark)
    else:
        return get_fixed_property_file(prefix, benchmark, num_starts)

def get_interface_file(prefix, benchmark):
    """
    An interface file specifies which state variable corresponds to the input features of the tree ensemble. It also specifies the ordering of the action labels.
    """
    game = get_game(benchmark)
    problem = get_problem(benchmark)
    dir_path = BENCHMARK_DIR
    if game == "blocksworld":
        if prefix == "CA":
            interface_file = f"{dir_path}/{game}/networks/{problem}/learned_cost05_non_terminal/{benchmark}.jani2nnet"
        else:
            interface_file = f"{dir_path}/{game}/networks/{problem}/learned_no_cost/{benchmark}.jani2nnet"
    elif game == "transport":
        interface_file = f"{dir_path}/{game}/networks/{problem}/no_filtering/{benchmark}.jani2nnet"
    else: 
        interface_file = f"{dir_path}/{game}/networks/{prefix}/{problem}/{benchmark}.jani2nnet"
    if not os.path.exists(interface_file):
        raise FileNotFoundError(f"Interface file {interface_file} does not exist.")
    return interface_file
