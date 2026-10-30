#!/bin/bash

# Run main_generalization.py for some/all datasets (all folds in parallel)
# Usage: ./run_generalization.sh [--dry-run] [-m xgb/rf] [-j jobs] datasets
# Note: reuses the original and LOP-compressed models,
#       so run "./run.sh -e orig --cache" and "./run.sh -e lop --cache" first.

PARALLEL_BIN=$(python -c "from paths import PARALLEL_BIN; print(PARALLEL_BIN)")
[ -x "$PARALLEL_BIN" ] || PARALLEL_BIN="parallel"   # GNU parallel from PATH
DRY_RUN=false
MODEL_TYPE="xgb"
JOBS=8

DATASETS=()
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true ;;
        -m|--model) MODEL_TYPE="$2"; shift ;;
        -j|--jobs) JOBS="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [DATASETS...]"
            echo "Options:"
            echo "  --dry-run          Print commands without running"
            echo "  -m, --model        Model type (default: $MODEL_TYPE)"
            echo "  -j, --jobs         Parallel jobs (default: $JOBS)"
            echo ""
            echo "Datasets: specify one or more dataset names as arguments"
            echo "If no datasets specified, uses default list"
            exit 0
            ;;
        *) DATASETS+=("$1") ;;
    esac
    shift
done

DEFAULT_DATASETS=(
    "CompasTwoYears"
    "DutchCensus"
    "BankMarketingFull"
    "LawSchool"
    "ACSIncome"
    "ACSEmployment"
    "ACSPublicCoverage"
    "ACSTravelTime"
    "MEPSPanel21"
    "DefaultCreditCardClients"
)

if [ ${#DATASETS[@]} -eq 0 ]; then
    DATASETS=("${DEFAULT_DATASETS[@]}")
fi

commands=()
for dname in "${DATASETS[@]}"; do
    for fold in {0..4}; do
        commands+=("python main_generalization.py $dname --fold $fold -m $MODEL_TYPE")
    done
done

echo "Generated ${#commands[@]} commands (${#DATASETS[@]} datasets × 5 folds)"

if $DRY_RUN; then
    echo "DRY RUN MODE - commands will be printed:"
    printf "%s\n" "${commands[@]}"
else
    echo "Running in parallel using GNU parallel..."
    printf "%s\n" "${commands[@]}" | "$PARALLEL_BIN" -j "$JOBS"
fi
