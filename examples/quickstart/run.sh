#!/bin/bash
set -eu
# Run against this checkout without an external tracking server or environment manager.
TASK_DIR=$(mktemp -d)
trap 'rm -rf "$TASK_DIR"' EXIT
cat > "$TASK_DIR/params.csv" <<'CSV'
1.00,0.00
0.75,0.25
0.50,0.50
0.25,0.75
0.00,1.00
CSV
cat > "$TASK_DIR/values.csv" <<'CSV'
0.00,1.00
3.00,2.00
4.00,5.00
7.00,6.00
8.00,9.00
CSV
python -m torch_bsf fit --params "$TASK_DIR/params.csv" --values "$TASK_DIR/values.csv" --degree 3 --save "$TASK_DIR/model.bsf"
python -m torch_bsf predict --model "$TASK_DIR/model.bsf" --params "$TASK_DIR/params.csv" --save "$TASK_DIR/predictions.csv"
python -m torch_bsf cross-validate --params "$TASK_DIR/params.csv" --values "$TASK_DIR/values.csv" --degree 2 --folds 3
python -m torch_bsf select-degree --params "$TASK_DIR/params.csv" --values "$TASK_DIR/values.csv" --degree 1 --degrees 1 2 3 --folds 3
cat "$TASK_DIR/predictions.csv"
