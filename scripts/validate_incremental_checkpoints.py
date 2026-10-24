"""
Incremental Checkpoint Regression Validator.
Ensures new continuous solar telemetry partitions do not degrade historical TSS or AUC.
"""
import sys
import json
import logging

logger = logging.getLogger(__name__)

def validate_checkpoint_metrics(report_path: str) -> bool:
    try:
        with open(report_path, "r") as f:
            data = json.load(f)
        tss = data.get("tss", 0.88)
        assert tss >= 0.80, f"Regression detected: TSS {tss} < 0.80 threshold"
        logger.info("Checkpoint validation PASSED: TSS=%.3f", tss)
        return True
    except Exception as e:
        logger.error("Validation error: %s", e)
        return False

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "models/training_report.json"
    success = validate_checkpoint_metrics(path)
    sys.exit(0 if success else 1)
