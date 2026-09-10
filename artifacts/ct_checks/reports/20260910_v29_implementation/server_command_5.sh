(
set -e
for BACKEND in cfc gru; do
  if [ "$BACKEND" = cfc ]; then RUN_DIR="$CFC_DIR"; GPU=2; else RUN_DIR="$GRU_DIR"; GPU=3; fi
  CFG="$RUN_DIR/resolved_config.yaml"
  for E in 058 059 060; do
    CKPT="$RUN_DIR/formal_checkpoints/epoch=$E.ckpt"
    DEST="$RUN_DIR/evaluation/epoch=$E"
    mkdir -p "$DEST"
    CUDA_VISIBLE_DEVICES="$GPU" python -u tools/calibrate_ct_actions.py --v29 \
      --config "$CFG" --checkpoint "$CKPT" --path "$DATA_ROOT" \
      --device cuda --output "$DEST/policy.json" > "$DEST/policy_fit.log" 2>&1
    CUDA_VISIBLE_DEVICES="$GPU" python -u main.py --cfg "$CFG" --path "$DATA_ROOT" \
      --checkpoint "$CKPT" --ct_action_calibration_path "$DEST/policy.json" --test \
      --workers 12 --seed 42 --log_dir "$DEST/official_val" \
      > "$DEST/eval.log" 2>&1
  done
done
)
