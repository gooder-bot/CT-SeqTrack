CFG="$FULL_DIR/resolved_config.yaml"
EVAL_STAMP=$(date +%Y%m%d-%H%M%S)
EVAL_ROOT="$FULL_DIR/evaluation_${EVAL_STAMP}"
mkdir -p "$EVAL_ROOT"

for E in 058 059 060; do
  CKPT="$FULL_DIR/formal_checkpoints/epoch=${E}.ckpt"
  ED="$EVAL_ROOT/epoch_${E}"
  mkdir -p "$ED/observation" "$ED/selective"

  python -u main.py --cfg "$CFG" --checkpoint "$CKPT" --test \
    --proposal_mode observation --log_dir "$ED/observation" \
    > "$ED/observation.log" 2>&1 || break

  python -u tools/calibrate_ct_actions.py --v28 \
    --config "$CFG" --checkpoint "$CKPT" --output "$ED/policy.json" \
    > "$ED/calibration.log" 2>&1 || break

  python -u main.py --cfg "$CFG" --checkpoint "$CKPT" --test \
    --ct_action_calibration_path "$ED/policy.json" \
    --log_dir "$ED/selective" \
    > "$ED/selective.log" 2>&1 || break
done
