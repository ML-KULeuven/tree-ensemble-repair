#!/bin/bash

# Run main.py for one experiment (-e) on some/all datasets (all folds in parallel)
# Usage: ./run.sh -e orig|lop|milp|milp-b|fatt|budro [--verify] [--dry-run] [--cache] [-m xgb/rf] [-j jobs] datasets
#
# Pipeline order (same datasets and model type, always with --cache):
#   -e orig  ->  -e lop  ->  -e milp / -e milp-b   (->  -e orig --verify, at the very end)
# lop loads the cached original model, milp/milp-b the cached LOP-compressed one.
# fatt and budro are independent of this pipeline.

PARALLEL_BIN=$(python -c "from paths import PARALLEL_BIN; print(PARALLEL_BIN)")
[ -x "$PARALLEL_BIN" ] || PARALLEL_BIN="parallel"   # GNU parallel from PATH
DRY_RUN=false
CACHE_FLAG=""
MODEL_TYPE="xgb"
EXP=""
VERIFY_FLAG=""
JOBS=10

# Parse arguments
DATASETS=()
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true ;;
        --cache) CACHE_FLAG="--cache" ;;
        -m|--model) MODEL_TYPE="$2"; shift ;;
        -e|--exp) EXP="$2"; shift ;;
        --verify) VERIFY_FLAG="--verify" ;;
        -j|--jobs) JOBS="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 -e EXP [OPTIONS] [DATASETS...]"
            echo "Options:"
            echo "  -e, --exp      Experiment to run: orig, lop, milp, milp-b, fatt, budro (required)"
            echo "  --verify       (-e orig only) verify the cached original models with veritas"
            echo "  --dry-run      Print commands without running"
            echo "  --cache        Enable caching (needed to save results)"
            echo "  -m, --model    Model type (default: xgb; ignored by fatt/budro)"
            echo "  -j, --jobs     Parallel jobs (default: $JOBS)"
            echo ""
            echo "Datasets: specify one or more dataset names as arguments"
            echo "If no datasets specified, uses default list"
            exit 0
            ;;
        *) DATASETS+=("$1") ;;
    esac
    shift
done

case $EXP in
    orig|lop|milp|milp-b|fatt|budro) ;;
    "") echo "Error: -e/--exp is required (orig, lop, milp, milp-b, fatt, budro)"; exit 1 ;;
    *) echo "Error: unknown experiment '$EXP' (orig, lop, milp, milp-b, fatt, budro)"; exit 1 ;;
esac
if [ -n "$VERIFY_FLAG" ] && [ "$EXP" != "orig" ]; then
    echo "Error: --verify is only supported with -e orig"; exit 1
fi

# Default datasets if none specified
DEFAULT_DATASETS=(
    "CompasTwoYears"
    "DefaultCreditCardClients"
    "DutchCensus"
    "BankMarketingFull"
    "LawSchool"
    "ACSIncome"
    "ACSEmployment"
    "ACSPublicCoverage"
    # "ACSMobility"
    "ACSTravelTime"
    "MEPSPanel21"
)

# Use specified datasets or defaults
if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=("${DEFAULT_DATASETS[@]}")
fi

# Generate commands for all datasets × 5 folds
commands=()
for dname in "${DATASETS[@]}"; do
    for fold in {0..4}; do
        commands+=("python main.py $dname --fold $fold -m $MODEL_TYPE -e $EXP $VERIFY_FLAG $CACHE_FLAG")
    done
done

echo "Generated ${#commands[@]} commands for -e $EXP (${#DATASETS[@]} datasets × 5 folds)"
if [ -z "$CACHE_FLAG" ]; then
    echo "WARNING: no --cache flag -- results will not be saved to results/"
fi

if $DRY_RUN; then
    echo "DRY RUN MODE - commands will be printed:"
    printf "%s\n" "${commands[@]}"
else
    echo "Running in parallel using GNU parallel..."
    printf "%s\n" "${commands[@]}" | "$PARALLEL_BIN" -j "$JOBS"
fi
