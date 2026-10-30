# Paths used by the experiments: edit before running!

# Cache for trained models and result files 
CACHE_DIR = "/cw/dtaijupiter/NoCsBack/dtai/lorenzoc/cache_pinacs/fairness-repair-neurips26"

# Cache for downloading and storing datasets through prada
PRADA_DATA_DIR = "/cw/dtaijupiter/NoCsBack/dtai/lorenzoc/data"

# Clone of https://github.com/fatt21/fatt (FATT baseline)
FATT_DIR = "../fatt"

# Clone of https://github.com/UMich-FATML/BuDRO (BuDRO baseline)
BUDRO_DIR = "../BuDRO"

# GNU parallel, used by the run*.sh scripts 
# (falls back to `parallel` on PATH if missing)
PARALLEL_BIN = "/cw/dtaijupiter/NoCsBack/dtai/laurensd/bin/parallel"
