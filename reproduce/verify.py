#!/usr/bin/env python3
"""
Verification script for LLM Foundry Simulator reproduce results.

Compares actual run results against expected JSON benchmarks and outputs
PASS/FAIL status with detailed statistics.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class CheckResult:
    """Result of a single verification check."""
    name: str
    status: str  # "PASS" or "FAIL"
    expected: Any
    actual: Any
    message: str = ""


@dataclass
class StageReport:
    """Verification report for a single stage."""
    stage: str
    status: str  # "PASS" or "FAIL"
    metrics: List[CheckResult] = field(default_factory=list)
    files: List[CheckResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def passed_checks(self) -> int:
        return sum(1 for c in self.metrics + self.files if c.status in ("PASS", "FOUND"))

    @property
    def total_checks(self) -> int:
        return len(self.metrics) + len(self.files)


def load_json_file(path: Path) -> Optional[Dict]:
    """Load and parse a JSON file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        return {"_error": f"Invalid JSON: {e}"}


def compare_value(
    name: str,
    expected: Any,
    actual: Any,
    threshold: Optional[float] = None,
    tolerance: Optional[float] = None
) -> CheckResult:
    """
    Compare expected vs actual value.

    Args:
        name: Metric name
        expected: Expected value (can be dict with special operators like "<", ">", "range")
        actual: Actual value from results
        threshold: Relative threshold for numeric comparison (e.g., 0.1 = 10%)
        tolerance: Absolute tolerance for numeric comparison
    """
    if actual is None:
        return CheckResult(name, "FAIL", expected, actual, "Actual value is missing")

    # Handle special expected formats
    if isinstance(expected, dict):
        return _compare_with_operator(name, expected, actual)

    # Handle numeric comparison with threshold/tolerance
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if tolerance is not None:
            diff = abs(expected - actual)
            if diff <= tolerance:
                return CheckResult(name, "PASS", expected, actual, f"within tolerance {tolerance}")
            else:
                return CheckResult(name, "FAIL", expected, actual, f"diff {diff:.4f} > tolerance {tolerance}")
        elif threshold is not None:
            rel_diff = abs(expected - actual) / abs(expected) if expected != 0 else abs(actual)
            if rel_diff <= threshold:
                return CheckResult(name, "PASS", expected, actual, f"within {threshold*100:.1f}% threshold")
            else:
                return CheckResult(name, "FAIL", expected, actual, f"rel_diff {rel_diff*100:.1f}% > {threshold*100:.1f}%")

    # Exact comparison for non-numeric types
    if expected == actual:
        return CheckResult(name, "PASS", expected, actual)
    else:
        return CheckResult(name, "FAIL", expected, actual, f"values differ")


def _compare_with_operator(name: str, expected: Dict, actual: Any) -> CheckResult:
    """Compare using operators like {'<': 10}, {'>': 0.5}, {'range': [1, 10]}."""
    if "<" in expected:
        limit = expected["<"]
        if actual < limit:
            return CheckResult(name, "PASS", f"<{limit}", actual)
        else:
            return CheckResult(name, "FAIL", f"<{limit}", actual, f"not less than {limit}")

    if ">" in expected:
        limit = expected[">"]
        if actual > limit:
            return CheckResult(name, "PASS", f">{limit}", actual)
        else:
            return CheckResult(name, "FAIL", f">{limit}", actual, f"not greater than {limit}")

    if "<=" in expected:
        limit = expected["<="]
        if actual <= limit:
            return CheckResult(name, "PASS", f"<={limit}", actual)
        else:
            return CheckResult(name, "FAIL", f"<={limit}", actual, f"exceeds {limit}")

    if ">=" in expected:
        limit = expected[">="]
        if actual >= limit:
            return CheckResult(name, "PASS", f">={limit}", actual)
        else:
            return CheckResult(name, "FAIL", f">={limit}", actual, f"below {limit}")

    if "range" in expected:
        low, high = expected["range"]
        if low <= actual <= high:
            return CheckResult(name, "PASS", f"[{low}, {high}]", actual)
        else:
            return CheckResult(name, "FAIL", f"[{low}, {high}]", actual, f"outside range")

    if "approx" in expected:
        target = expected["approx"]
        tol = expected.get("tolerance", target * 0.1 if target != 0 else 0.1)
        diff = abs(target - actual)
        if diff <= tol:
            return CheckResult(name, "PASS", f"~{target}", actual, f"within tolerance {tol}")
        else:
            return CheckResult(name, "FAIL", f"~{target}", actual, f"diff {diff:.4f} > tolerance {tol}")

    return CheckResult(name, "FAIL", expected, actual, "unknown operator in expected")


def find_results_dir(results_base: Path, stage: str) -> Optional[Path]:
    """Find the most recent results directory for a stage."""
    if not results_base.exists():
        return None

    # Look for hash directories
    hash_dirs = [d for d in results_base.iterdir() if d.is_dir() and len(d.name) == 8]
    if not hash_dirs:
        return None

    # Sort by modification time (most recent first)
    hash_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return hash_dirs[0]


def verify_tokenize(expected_dir: Path, results_dir: Path) -> StageReport:
    """Verify tokenization stage results."""
    report = StageReport(stage="tokenize", status="FAIL")

    # Load expected
    expected_file = expected_dir / "tokenizer_stats.json"
    expected = load_json_file(expected_file) or {}

    # Check actual tokenizer output (BPE tokenizer saves to results/tokenizer/bpe_tokenizer/)
    tokenizer_dir = Path("results/tokenizer/bpe_tokenizer")
    if not tokenizer_dir.exists():
        report.errors.append(f"Tokenizer directory not found: {tokenizer_dir}")
        return report

    # Load actual vocab.json to get vocab_size
    actual = {}
    vocab_file = tokenizer_dir / "vocab.json"
    if vocab_file.exists():
        vocab = load_json_file(vocab_file) or {}
        actual["vocab_size"] = len(vocab)

    # Verify metrics
    if "vocab_size" in expected:
        report.metrics.append(compare_value(
            "vocab_size",
            expected["vocab_size"],
            actual.get("vocab_size"),
            tolerance=expected.get("vocab_size_tolerance", 100)
        ))

    # Check BPE tokenizer files
    files_to_check = ["vocab.json", "merges.txt"]
    for fname in files_to_check:
        fpath = tokenizer_dir / fname
        exists = fpath.exists()
        report.files.append(CheckResult(
            fname,
            "FOUND" if exists else "MISSING",
            "exists",
            "exists" if exists else "missing"
        ))

    report.status = "PASS" if all(c.status in ("PASS", "FOUND") for c in report.metrics + report.files) else "FAIL"
    return report


def verify_train(expected_dir: Path, results_dir: Path) -> StageReport:
    """Verify training stage results."""
    report = StageReport(stage="train", status="FAIL")

    # Load expected
    expected_file = expected_dir / "train_loss.json"
    expected = load_json_file(expected_file) or {}

    # Find actual results
    results_hash_dir = find_results_dir(results_dir, "train")
    if not results_hash_dir:
        report.errors.append(f"No results directory found in {results_dir}")
        return report

    # Load metrics.jsonl (last line has final metrics)
    metrics_file = results_hash_dir / "metrics.jsonl"
    actual = {}
    if metrics_file.exists():
        try:
            with open(metrics_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    actual = json.loads(lines[-1])  # Last line
        except (json.JSONDecodeError, IOError):
            pass

    # Verify loss
    if "final_loss" in expected:
        report.metrics.append(compare_value(
            "final_loss",
            expected["final_loss"],
            actual.get("loss"),
            threshold=expected.get("loss_threshold", 0.1)
        ))

    if "final_perplexity" in expected:
        report.metrics.append(compare_value(
            "final_perplexity",
            expected["final_perplexity"],
            actual.get("perplexity"),
            threshold=expected.get("ppl_threshold", 0.1)
        ))

    if "train_time_seconds" in expected:
        report.metrics.append(compare_value(
            "train_time",
            expected["train_time_seconds"],
            actual.get("train_time_seconds"),
            tolerance=expected.get("time_tolerance", 300)
        ))

    # Check files
    checkpoint_dir = results_hash_dir / "checkpoints"
    files_to_check = ["metrics.jsonl", "config.yaml"]
    for fname in files_to_check:
        fpath = results_hash_dir / fname
        exists = fpath.exists()
        report.files.append(CheckResult(
            fname,
            "FOUND" if exists else "MISSING",
            "exists",
            "exists" if exists else "missing"
        ))

    # Check for checkpoint files
    has_checkpoint = checkpoint_dir.exists() and any(checkpoint_dir.glob("*.pt"))
    report.files.append(CheckResult(
        "checkpoint.pt",
        "FOUND" if has_checkpoint else "MISSING",
        "exists",
        "exists" if has_checkpoint else "missing"
    ))

    report.status = "PASS" if all(c.status in ("PASS", "FOUND") for c in report.metrics + report.files) else "FAIL"
    return report


def verify_scaling(expected_dir: Path, results_dir: Path) -> StageReport:
    """Verify scaling stage results."""
    report = StageReport(stage="scaling", status="FAIL")

    # Load expected
    expected_file = expected_dir / "scaling_params.json"
    expected = load_json_file(expected_file) or {}

    # Find actual results
    results_hash_dir = find_results_dir(results_dir, "scaling")
    if not results_hash_dir:
        report.errors.append(f"No results directory found in {results_dir}")
        return report

    # Load actual
    actual_file = results_hash_dir / "scaling_params.json"
    actual = load_json_file(actual_file) or {}

    # Verify Chinchilla params
    if "chinchilla" in expected:
        exp_ch = expected["chinchilla"]
        act_ch = actual.get("chinchilla", {})
        for key in ["alpha", "beta", "A", "B", "E"]:
            if key in exp_ch:
                report.metrics.append(compare_value(
                    f"chinchilla.{key}",
                    exp_ch[key],
                    act_ch.get(key),
                    threshold=exp_ch.get(f"{key}_threshold", 0.05)
                ))

    # Verify isoflops optimal points
    if "isoflops_optimal" in expected:
        exp_iso = expected["isoflops_optimal"]
        act_iso = actual.get("isoflops_optimal", [])
        if len(exp_iso) == len(act_iso):
            for i, (exp_pt, act_pt) in enumerate(zip(exp_iso, act_iso)):
                for key in ["compute", "n_params", "n_tokens"]:
                    if key in exp_pt:
                        report.metrics.append(compare_value(
                            f"isoflops[{i}].{key}",
                            exp_pt[key],
                            act_pt.get(key),
                            threshold=exp_pt.get(f"{key}_threshold", 0.1)
                        ))
        else:
            report.metrics.append(CheckResult(
                "isoflops_optimal",
                "FAIL",
                f"{len(exp_iso)} points",
                f"{len(act_iso)} points",
                "point count mismatch"
            ))

    # Check files
    files_to_check = ["scaling_params.json"]
    plots_dir = results_hash_dir / "scaling_plots"
    for fname in files_to_check:
        fpath = results_hash_dir / fname
        exists = fpath.exists()
        report.files.append(CheckResult(
            fname,
            "FOUND" if exists else "MISSING",
            "exists",
            "exists" if exists else "missing"
        ))

    # Check for plot files
    has_plots = plots_dir.exists() and any(plots_dir.glob("*.png"))
    report.files.append(CheckResult(
        "scaling_plots/*.png",
        "FOUND" if has_plots else "MISSING",
        "exists",
        "exists" if has_plots else "missing"
    ))

    report.status = "PASS" if all(c.status in ("PASS", "FOUND") for c in report.metrics + report.files) else "FAIL"
    return report


def verify_data(expected_dir: Path, results_dir: Path) -> StageReport:
    """Verify data pipeline stage results."""
    report = StageReport(stage="data", status="FAIL")

    # Load expected
    expected_file = expected_dir / "data_stats.json"
    expected = load_json_file(expected_file) or {}

    # Data pipeline outputs to data/filtered/ not results/data/
    filtered_file = Path("data/filtered/openwebtext_filtered.txt")
    if filtered_file.exists():
        # Count lines (documents) in filtered file
        doc_count = sum(1 for _ in open(filtered_file, 'r', encoding='utf-8'))
        actual = {"total_documents": doc_count, "passed_filter_ratio": doc_count / 516978 if doc_count else 0}
    else:
        report.errors.append(f"Filtered data file not found: {filtered_file}")
        return report

    # Verify stats
    if "total_documents" in expected:
        report.metrics.append(compare_value(
            "total_documents",
            expected["total_documents"],
            actual.get("total_documents"),
            tolerance=expected.get("doc_tolerance", 10000)
        ))

    if "passed_filter_ratio" in expected:
        report.metrics.append(compare_value(
            "passed_filter_ratio",
            expected["passed_filter_ratio"],
            actual.get("passed_filter_ratio"),
            threshold=expected.get("ratio_threshold", 0.1)
        ))

    # Check files
    files_to_check = ["data/filtered/openwebtext_filtered.txt"]
    for fname in files_to_check:
        fpath = Path(fname)
        exists = fpath.exists()
        report.files.append(CheckResult(
            fname,
            "FOUND" if exists else "MISSING",
            "exists",
            "exists" if exists else "missing"
        ))

    report.status = "PASS" if all(c.status in ("PASS", "FOUND") for c in report.metrics + report.files) else "FAIL"
    return report


def verify_align(expected_dir: Path, results_dir: Path) -> StageReport:
    """Verify alignment stage results."""
    report = StageReport(stage="align", status="FAIL")

    # Load expected
    expected_file = expected_dir / "align_metrics.json"
    expected = load_json_file(expected_file) or {}

    # Find actual results
    results_hash_dir = find_results_dir(results_dir, "align")
    if not results_hash_dir:
        report.errors.append(f"No results directory found in {results_dir}")
        return report

    # Load actual (try different possible filenames)
    actual = {}
    for fname in ["align_metrics.json", "sft_metrics.json", "dpo_metrics.json"]:
        fpath = results_hash_dir / fname
        if fpath.exists():
            actual = load_json_file(fpath) or {}
            break

    # Verify metrics based on method
    method = expected.get("method", "sft")

    if "final_loss" in expected:
        report.metrics.append(compare_value(
            "final_loss",
            expected["final_loss"],
            actual.get("final_loss") or actual.get("loss"),
            threshold=expected.get("loss_threshold", 0.1)
        ))

    if "eval_accuracy" in expected:
        report.metrics.append(compare_value(
            "eval_accuracy",
            expected["eval_accuracy"],
            actual.get("eval_accuracy"),
            threshold=expected.get("acc_threshold", 0.05)
        ))

    if "kl_divergence" in expected:
        report.metrics.append(compare_value(
            "kl_divergence",
            expected["kl_divergence"],
            actual.get("kl_divergence"),
            threshold=expected.get("kl_threshold", 0.1)
        ))

    # Check files
    files_to_check = [f"{method}_metrics.json", "checkpoints/"]
    for fname in files_to_check:
        if fname.endswith("/"):
            fpath = results_hash_dir / fname.rstrip("/")
            exists = fpath.exists() and any(fpath.iterdir()) if fpath.exists() else False
        else:
            fpath = results_hash_dir / fname
            exists = fpath.exists()
        report.files.append(CheckResult(
            fname,
            "FOUND" if exists else "MISSING",
            "exists",
            "exists" if exists else "missing"
        ))

    report.status = "PASS" if all(c.status in ("PASS", "FOUND") for c in report.metrics + report.files) else "FAIL"
    return report


# Stage verification registry
STAGE_VERIFIERS = {
    "tokenize": verify_tokenize,
    "train": verify_train,
    "scaling": verify_scaling,
    "data": verify_data,
    "align": verify_align,
}


def print_report(report: StageReport) -> None:
    """Print a formatted verification report."""
    print(f"\n{'=' * 50}")
    print(f"Stage: {report.stage}")
    print(f"Status: {report.status}")

    if report.errors:
        print("\nErrors:")
        for err in report.errors:
            print(f"  - {err}")

    if report.metrics:
        print("\nMetrics:")
        for check in report.metrics:
            status_str = f"[{check.status}]"
            if check.message:
                print(f"  - {check.name}: expected {check.expected}, actual {check.actual} {status_str} ({check.message})")
            else:
                print(f"  - {check.name}: expected {check.expected}, actual {check.actual} {status_str}")

    if report.files:
        print("\nFiles:")
        for check in report.files:
            print(f"  - {check.name} [{check.status}]")

    print(f"\nSummary: {report.passed_checks}/{report.total_checks} checks passed")
    print('=' * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Verify LLM Foundry Simulator reproduce results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python verify.py --stage train
  python verify.py --all
  python verify.py --stage scaling --expected-dir ./expected --results-dir ./results
        """
    )

    parser.add_argument(
        "--stage",
        choices=["tokenize", "train", "scaling", "data", "align"],
        help="Verify specific stage"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Verify all stages"
    )
    parser.add_argument(
        "--expected-dir",
        default="reproduce/expected",
        help="Directory containing expected JSON files (default: reproduce/expected/)"
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing results (default: results/)"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.stage and not args.all:
        parser.error("Must specify --stage or --all")

    # Resolve paths
    expected_dir = Path(args.expected_dir).resolve()
    results_dir = Path(args.results_dir).resolve()

    if not expected_dir.exists():
        print(f"Error: Expected directory not found: {expected_dir}")
        sys.exit(1)

    # Determine stages to verify
    stages = list(STAGE_VERIFIERS.keys()) if args.all else [args.stage]

    # Run verification
    all_passed = True
    reports = []

    for stage in stages:
        if stage not in STAGE_VERIFIERS:
            print(f"Warning: Unknown stage '{stage}', skipping")
            continue

        verifier = STAGE_VERIFIERS[stage]
        report = verifier(expected_dir, results_dir)
        reports.append(report)
        print_report(report)

        if report.status != "PASS":
            all_passed = False

    # Print summary
    if len(reports) > 1:
        print(f"\n{'=' * 50}")
        print("OVERALL SUMMARY")
        print('=' * 50)
        for r in reports:
            status_icon = "✓" if r.status == "PASS" else "✗"
            print(f"  {status_icon} {r.stage}: {r.passed_checks}/{r.total_checks} checks")
        print('=' * 50)

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
