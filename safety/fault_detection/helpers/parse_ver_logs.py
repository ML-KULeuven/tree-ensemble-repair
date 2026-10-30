import re
import json 

def parse_log(file_path):
    total_time = None
    status = None
    path_values = []

    in_path = False

    with open(file_path, 'r') as f:
        for line in f:
            # Check for IC3 total time
            if "IC3 total time:" in line:
                match = re.search(r'IC3 total time:\s*(\d+)\s*ms', line)
                if match:
                    total_time = int(match.group(1))

            # Check for Unsolvable
            if "Unsolvable" in line:
                status = "SAFE"

            # Check for Full Path
            if "Full path:" in line:
                status = "UNSAFE"
                in_path = True
                continue

            # Parse path values if in path mode
            if in_path:
                if line.strip() == "" or "Frames:" in line:
                    in_path = False
                    continue
                if '=' in line:
                    values = re.findall(r'=(\-?\d+)i', line)
                    path_values.append([int(v) for v in values])
                    
    return {
        "total_time_ms": total_time,
        "status": status,
        "path": path_values if status == "UNSAFE" else None
    }
    
def convert_logs_to_json(log_file_path):
    results = parse_log(log_file_path)
    json_path = log_file_path.replace(".txt", ".json")
    with open(json_path, "w") as outfile:
        json.dump(results, outfile, indent=2)
    return json_path, results["status"] == "SAFE"

def parse_verification_log(log_dir):
    # find if log file contains "Safe policy", "Unsafe policy" or "Time limit reached"
    with open(log_dir, 'r') as f:
        content = f.read()
        if "Unsafe policy" in content or "unsafe path" in content:
            return "UNSAFE"
        elif "Safe policy" in content:
            return "SAFE"
        elif "Time limit reached" in content:
            return "TIMEOUT"
        # TODO: deal with OOM if it ever shows up?
        # ...
        else:
            raise ValueError("No valid status in verification log file.")