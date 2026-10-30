from utils.utils import *
from utils.plaja_interface import *

class NNTeacher():
    def __init__(self, prefix, benchmark):
        self.prefix = prefix
        self.benchmark = benchmark
    
    def learn(self):
        """
        Run a teacher policy on num_starts randomly sampled start states from the start condition. Save the <state, action> decisions made by the nn policy.
        """
        model_file = get_model_file(self.prefix, self.benchmark)
        property_file = get_property_file(self.prefix, self.benchmark, True)
        interface_file = get_interface_file(self.prefix, self.benchmark)
        sample = "sample"
        max_time = 3600
        command = f"{PLAJA_DIR} --model-file {model_file} --seed 42 --engine QL_AGENT --max-time {max_time} --nn-interface {interface_file} --initial-state-enum {sample} --epsilon-start 1 --save-nnet --applicability-filtering 1 --num-episodes 30000 --max-length-episode 1500"
        command += f" --additional-properties {property_file} --prop 1" if (property_file != None) else " --prop 0"
        t = time.time()
        try:
            # give the subprocess timeout headroom over PlaJA's own --max-time:
            # if both are equal, python kills PlaJA right before it can finish
            # its budget and write the network out
            subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, text=True,
                        # let's use all the memory we have for this
                        # preexec_fn=limit_memory_preexec(DEFAULT_PLAJA_MEM_MIB),
                        timeout=max_time + 600)
        except subprocess.TimeoutExpired:
            print(f"\t- NN training timed out after {time.time() - t:.3f} s.")
        except subprocess.CalledProcessError as e:
            print("Command failed!")
            print("Exit code:", e.returncode)
            print("Output:", e.stdout)
            print("Error output:", e.stderr)

        # print(f"Command executed successfully: {command}")
        print(f"\t- training nn completed in {time.time() - t:.3f} s.")
        
    
    def generate_training_data(self, data_file, stat_file):
        """
        Run a teacher policy on num_starts randomly sampled start states from the start condition. Save the <state, action> decisions made by the nn policy.
        Only the missing artifacts are (re)generated: an existing training-data
        file or stats file is kept as is, so PlaJA is not re-run needlessly.
        """
        need_data = not os.path.exists(data_file)
        need_stats = not os.path.exists(stat_file)
        if not (need_data or need_stats):
            print("\tTraining data and NN stats already exist, skipping PlaJA.")
            return
        # (!) we decided to only always use 1000 start states + compact=True for training data generation
        num_starts = 1000
        compact = True
        plaja_runtime = 7200
        
        model_file = get_model_file(self.prefix, self.benchmark)
        property_file = get_property_file(self.prefix, self.benchmark, compact, num_starts)
        interface_file = get_interface_file(self.prefix, self.benchmark)
        nn_file = get_neural_network_file(self.prefix, self.benchmark)
        if not os.path.exists(nn_file):
            print(f"NN file not found: {nn_file}")
            print("(!) Expected a pretrained teacher in the benchmark archive -- check the "
                  "archive sync (or an old-name .nnet, e.g. race_linetrack_*) before "
                  "relying on retraining from scratch.")
            self.learn()
            if not os.path.exists(nn_file):
                raise FileNotFoundError(
                    f"Teacher network still missing after training attempt: {nn_file}")
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        os.makedirs(os.path.dirname(stat_file), exist_ok=True)
        sample = "sample"
        # if get_game(benchmark) == "blocksworld" or get_game(benchmark) == "beluga":
        #     sample = "inc"
        data_command = f"{PLAJA_DIR} --model-file {model_file} --seed 42 --engine QL_AGENT --max-time {plaja_runtime} --nn-interface {interface_file} --load-nn {nn_file} --initial-state-enum {sample} --epsilon-start 0 --evaluation-mode --applicability-filtering 1 --num-episodes {num_starts} --max-length-episode 1500 --save-agent-actions {data_file} --terminate-cycles --trace-episodes {num_starts}"
        data_command += f" --additional-properties {property_file} --prop 1" if (property_file != None) else "--prop 0"
        # seed = 10
        # performance_flag = "--compute-policy-performance" if not approximate else "--compute-policy-performance-approximately"
        nn_perf_command = f"{PLAJA_DIR} --model-file {model_file} --seed 42 --engine QL_AGENT --save-intermediate-stats --max-time {plaja_runtime} --nn-interface {interface_file} --stats-file {stat_file} --load-nn {nn_file} --initial-state-enum {sample} --epsilon-start 0 --evaluation-mode --applicability-filtering 1 --num-episodes {num_starts} --max-length-episode 1500 --print-stats --terminate-cycles"
        nn_perf_command += f" --additional-properties {property_file} --prop 1" if (property_file != None) else "--prop 0"
        t = time.time()
        if need_data:
            print("RUNNING COMMAND:", data_command)
            try:
                subprocess.run(data_command, shell=True, check=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True,
                            preexec_fn=limit_memory_preexec(DEFAULT_PLAJA_MEM_MIB),
                            timeout=plaja_runtime + 600)
            except subprocess.TimeoutExpired:
                print("\t- training data generation timed out.")
            except subprocess.CalledProcessError as e:
                print("Command failed!", data_command)
                print("Exit code:", e.returncode)
                print("Output:", e.stdout)
                print("Error output:", e.stderr)
        else:
            print(f"\tTraining data already exists, skipping PlaJA: {data_file}")

        if need_stats:
            print("RUNNING COMMAND:", nn_perf_command)
            try:
                subprocess.run(nn_perf_command, shell=True, check=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True,
                            preexec_fn=limit_memory_preexec(DEFAULT_PLAJA_MEM_MIB),
                            timeout=plaja_runtime + 600) 
            except subprocess.TimeoutExpired:
                print("\t- NN stats computation timed out.")
            except subprocess.CalledProcessError as e:
                print("Command failed!")
                print("Exit code:", e.returncode)
                print("Output:", e.stdout)
                print("Error output:", e.stderr)
        else:
            print(f"\tNN stats already exist, skipping PlaJA: {stat_file}")
            

        # print(f"Command executed successfully: {command}")
        print(f"\t- training set generation completed in {time.time() - t:.3f} s.")
        
    def get_performance(self, stat_file):
        print("Stats File:", stat_file)
        stats = pd.read_csv(stat_file)
        last_row = stats.iloc[-1] 
        goal_fract = np.round(last_row["DoneGoal"]/last_row["Episodes"]*100, 2)
        print(f"\t\t--> nn: DoneGoal {goal_fract}%")
        avoid_fract = np.round(last_row["DoneAvoid"]/last_row["Episodes"]*100, 2)
        print(f"\t\t--> nn: DoneAvoid {avoid_fract}%")
        cycle_fract = np.round(last_row["DetectedCycles"]/last_row["Episodes"]*100, 2)
        print(f"\t\t--> nn: DetectedCycles {cycle_fract}%")
        return goal_fract, avoid_fract, cycle_fract