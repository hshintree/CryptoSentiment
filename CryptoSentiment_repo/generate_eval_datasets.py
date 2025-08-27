#!/usr/bin/env python3
"""
Standalone script to generate, validate, and save evaluation datasets.

This script:
1. Uses the enhanced EWMA market labeler
2. Creates Ea (2019-2020), Val (2021), and Eb (2015-2018 & 2022-2023) datasets
3. Validates all characteristics and paper compliance
4. Saves datasets as CSV files in data/ directory
5. Provides comprehensive reporting and validation

Usage:
    python generate_eval_datasets.py
"""
import argparse
import pandas as pd, numpy as np, warnings
import os
from pathlib import Path
from datetime import datetime
from eval_loader_complete import EvalLoaderComplete

# make pandas & sklearn shut up
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def create_output_directory():
    """Create data directory if it doesn't exist."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    return data_dir


def validate_dataset_characteristics(datasets, verbose=True):
    """
    Comprehensive validation of dataset characteristics.
    
    Args:
        datasets: Dictionary from EvalLoaderComplete.create_eval_datasets()
        verbose: Whether to print detailed validation results
        
    Returns:
        Dict with validation results and metrics
    """
    validation_results = {
        "ea_validation": {},
        "eb_validation": {},
        "overall_validation": {},
        "all_checks_passed": True
    }
    
    if verbose:
        print("\n" + "="*80)
        print("🔍 COMPREHENSIVE DATASET VALIDATION")
        print("="*80)
    
    ea_df = datasets['ea_full']
    eb_df = datasets['eb_full']
    
    # ===== EA DATASET VALIDATION =====
    if verbose:
        print("\n📋 EA DATASET (2020 FINE-TUNING) VALIDATION:")
        print("-" * 50)
    
    # Size validation
    ea_size = len(ea_df)
    ea_size_ok = 15000 <= ea_size <= 100000  # Flexible range around ~60k
    validation_results["ea_validation"]["size"] = {"value": ea_size, "ok": ea_size_ok, "target": "~60,000"}
    
    if verbose:
        status = "✅" if ea_size_ok else "❌"
        print(f"{status} Size: {ea_size:,} tweets (target: ~60,000)")
    
    # Year validation
    ea_years = sorted(ea_df['date'].dt.year.unique())
    ea_year_ok = ea_years == [2019, 2020]
    validation_results["ea_validation"]["year"] = {
        "value": ea_years, "ok": ea_year_ok, "target": "[2019, 2020]"
    }
    
    if verbose:
        status = "✅" if ea_year_ok else "❌"
        print(f"{status} Year coverage: {ea_years} (target: [2019, 2020])")
    
    # Label distribution validation
    ea_labels = ea_df['Label'].value_counts()
    ea_balance = ea_labels.min() / ea_labels.max() if ea_labels.max() > 0 else 0
    ea_balance_ok = ea_balance >= 0.25  # At least 25% balance (not too skewed)
    validation_results["ea_validation"]["balance"] = {
        "value": ea_balance, 
        "ok": ea_balance_ok, 
        "target": "≥0.25",
        "distribution": ea_labels.to_dict()
    }
    
    if verbose:
        status = "✅" if ea_balance_ok else "❌"
        print(f"{status} Label balance: {ea_balance:.3f} (target: ≥0.25 for reasonable distribution)")
        print(f"   Distribution: {ea_labels.to_dict()}")
    
    # EWMA features validation
    ea_has_volatility = 'Volatility' in ea_df.columns
    ea_vol_ok = ea_has_volatility and ea_df['Volatility'].notna().all()
    validation_results["ea_validation"]["volatility"] = {"ok": ea_vol_ok, "target": "All tweets have volatility"}
    
    if verbose:
        status = "✅" if ea_vol_ok else "❌"
        print(f"{status} Volatility features: {'Present' if ea_vol_ok else 'Missing/Invalid'}")
        if ea_has_volatility:
            print(f"   Mean volatility: {ea_df['Volatility'].mean():.4f}")
    
    #  ᐅ No Val set in paper-exact experiment → skip whole section
    
    # ===== EB DATASET VALIDATION =====
    if verbose:
        print("\n📋 EB DATASET (EVENT EVALUATION) VALIDATION:")
        print("-" * 50)
    
    # Size validation
    eb_size = len(eb_df)
    eb_size_ok = 15000 <= eb_size <= 80000  # Flexible range around ~40k
    validation_results["eb_validation"]["size"] = {"value": eb_size, "ok": eb_size_ok, "target": "~40,000"}
    
    if verbose:
        status = "✅" if eb_size_ok else "❌"
        print(f"{status} Size: {eb_size:,} tweets (target: ~40,000)")
    
    # Year validation (excluding 2020 only)
    eb_years = sorted(eb_df['date'].dt.year.unique())
    eb_year_ok = (2020 not in eb_years) and len(eb_years) > 1
    validation_results["eb_validation"]["years"] = {
        "value": eb_years, "ok": eb_year_ok, "target": "2015-2019, 2021-2023"
    }
    
    if verbose:
        status = "✅" if eb_year_ok else "❌"
        print(f"{status} Year coverage: {eb_years} (target: excludes 2020 only)")
    
    # Event focus validation
    eb_events = eb_df['Is_Event'].sum()
    eb_event_pct = (eb_events / len(eb_df) * 100) if len(eb_df) > 0 else 0
    eb_event_ok = eb_event_pct >= 10  # At least 10% event tweets
    validation_results["eb_validation"]["events"] = {
        "value": eb_events, 
        "percentage": eb_event_pct,
        "ok": eb_event_ok, 
        "target": "≥10% event tweets"
    }
    
    if verbose:
        status = "✅" if eb_event_ok else "❌"
        print(f"{status} Event focus: {eb_events:,} tweets ({eb_event_pct:.1f}%) are event-related")
    
    # Label actionability
    eb_actionable = (eb_df['Label'] != 'Neutral').sum()
    eb_actionable_pct = (eb_actionable / len(eb_df) * 100) if len(eb_df) > 0 else 0
    eb_actionable_ok = eb_actionable_pct >= 50  # At least 50% actionable signals
    validation_results["eb_validation"]["actionable"] = {
        "value": eb_actionable,
        "percentage": eb_actionable_pct,
        "ok": eb_actionable_ok,
        "target": "≥50% actionable signals"
    }
    
    if verbose:
        status = "✅" if eb_actionable_ok else "❌"
        print(f"{status} Actionable signals: {eb_actionable:,} tweets ({eb_actionable_pct:.1f}%)")
    
    # ===== OVERALL VALIDATION =====
    all_checks = [
        ea_size_ok, ea_year_ok, ea_balance_ok, ea_vol_ok,
        # (val checks removed)
        eb_size_ok, eb_year_ok, eb_event_ok, eb_actionable_ok,
    ]
    
    validation_results["all_checks_passed"] = all(all_checks)
    
    if verbose:
        print("\n" + "="*80)
        if validation_results["all_checks_passed"]:
            print("🎉 ALL VALIDATION CHECKS PASSED! Datasets are paper-compliant.")
        else:
            print("⚠️  Some validation checks failed. Review above for details.")
        print("="*80)
    
    return validation_results


def save_datasets_to_csv(datasets, data_dir, timestamp=None, *, write: bool = True, timestamped: bool = False):
    """
    Save all datasets to CSV files with descriptive names.
    
    Args:
        datasets: Dictionary from EvalLoaderComplete.create_eval_datasets()
        data_dir: Path object for data directory
        timestamp: Optional timestamp string for file naming
        write: Whether to actually write CSV files
        timestamped: Whether to include timestamp in filenames
        
    Returns:
        Dict with saved file paths
    """
    if not write:
        print("⚠️ Skipping CSV writes (--no-save enabled)")
        return {}

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    saved_files = {}
    
    print("\n" + "="*60)
    print("💾 SAVING DATASETS TO CSV FILES")
    print("="*60)
    
    # Main datasets
    files_to_save = [
        ("ea_full", f"#1train{'_' + timestamp if timestamped else ''}.csv", "Ea training dataset (year 2020)"),
        ("eb_full", f"#2val{'_' + timestamp if timestamped else ''}.csv",   "Eb evaluation dataset (events)")
    ]
    
    for key, filename, description in files_to_save:
        if key in datasets:
            filepath = data_dir / filename
            datasets[key].to_csv(filepath, index=False)
            saved_files[key] = str(filepath)
            print(f"✅ Saved {description}: {filepath}")
            print(f"   Shape: {datasets[key].shape}")
    
    # (no CV folds in this set-up)
    
    print(f"\n📊 Total files saved: {len(saved_files)}")
    
    return saved_files


def generate_summary_report(datasets, validation_results, saved_files, timestamp):
    """Generate a comprehensive summary report."""
    report_lines = []
    
    report_lines.append("="*80)
    report_lines.append("📊 EVALUATION DATASETS GENERATION SUMMARY REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Timestamp: {timestamp}")
    report_lines.append("")
    
    # Dataset overview
    report_lines.append("📋 DATASET OVERVIEW:")
    report_lines.append("-" * 40)
    report_lines.append(f"Ea dataset (2020 training): {len(datasets['ea_full']):,} tweets")
    report_lines.append(f"Eb dataset (event evaluation): {len(datasets['eb_full']):,} tweets")
    report_lines.append("")
    
    # Label distributions
    report_lines.append("📊 LABEL DISTRIBUTIONS:")
    report_lines.append("-" * 40)
    ea_labels = datasets['ea_full']['Label'].value_counts()
    eb_labels = datasets['eb_full']['Label'].value_counts()
    
    report_lines.append("Ea dataset (2020 training):")
    for label, count in ea_labels.items():
        pct = (count / len(datasets['ea_full']) * 100)
        report_lines.append(f"  {label}: {count:,} tweets ({pct:.1f}%)")
    
    report_lines.append("Eb dataset (event evaluation):")
    for label, count in eb_labels.items():
        pct = (count / len(datasets['eb_full']) * 100)
        report_lines.append(f"  {label}: {count:,} tweets ({pct:.1f}%)")
    report_lines.append("")
    
    # Validation summary
    report_lines.append("✅ VALIDATION RESULTS:")
    report_lines.append("-" * 40)
    if validation_results["all_checks_passed"]:
        report_lines.append("🎉 ALL VALIDATION CHECKS PASSED!")
    else:
        report_lines.append("⚠️  Some validation checks failed.")
    
    report_lines.append(f"Ea size validation: {'✅' if validation_results['ea_validation']['size']['ok'] else '❌'}")
    report_lines.append(f"Ea balance validation: {'✅' if validation_results['ea_validation']['balance']['ok'] else '❌'}")
    report_lines.append(f"Eb size validation: {'✅' if validation_results['eb_validation']['size']['ok'] else '❌'}")
    report_lines.append(f"Eb event focus validation: {'✅' if validation_results['eb_validation']['events']['ok'] else '❌'}")
    report_lines.append("")
    
    # File locations
    report_lines.append("📁 SAVED FILES:")
    report_lines.append("-" * 40)
    report_lines.append(f"Main datasets: {len(saved_files)} files")
    report_lines.append("")
    
    # Key files
    report_lines.append("🔑 KEY FILES:")
    report_lines.append("-" * 40)
    key_files = ["ea_full", "eb_full"]
    for key in key_files:
        if key in saved_files:
            report_lines.append(f"  {key}: {saved_files[key]}")
    report_lines.append("")
    
    report_lines.append("="*80)
    report_lines.append("🚀 DATASETS READY FOR PAPER IMPLEMENTATION!")
    report_lines.append("="*80)
    
    return "\n".join(report_lines)


def main():
    """Main function to generate, validate, and save evaluation datasets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-save", action="store_true", help="Do not write CSVs to data/ (print only)")
    parser.add_argument("--timestamped", action="store_true", help="Append timestamp to output CSV filenames and report")
    args = parser.parse_args()

    print("🚀 Starting Evaluation Dataset Generation")
    print("="*60)
    
    # Check input file exists
    csv_path = "data/combined_dataset_raw.csv"
    if not Path(csv_path).exists():
        print(f"❌ Error: {csv_path} not found")
        print("Please ensure the combined dataset exists before running this script.")
        return
    
    # Create output directory
    data_dir = create_output_directory()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # 1. Generate datasets
        print("📊 Generating evaluation datasets with enhanced EWMA labeling...")
        loader = EvalLoaderComplete(csv_path, config_path="config.yaml")
        datasets = loader.create_eval_datasets(n_folds=5)
        
        # 2. Validate characteristics
        print("\n🔍 Validating dataset characteristics...")
        validation_results = validate_dataset_characteristics(datasets, verbose=True)
        
        # 3. Save to CSV files (controlled by flags)
        print("\n💾 Saving datasets to CSV files...")
        saved_files = save_datasets_to_csv(
            datasets, data_dir, timestamp, write=(not args.no_save), timestamped=args.timestamped
        )
        
        # 4. Generate summary report
        summary_report = generate_summary_report(datasets, validation_results, saved_files, timestamp)
        
        # Save summary report (timestamped only if requested)
        report_name = f"generation_report_{timestamp}.txt" if args.timestamped else "generation_report.txt"
        report_path = data_dir / report_name
        with open(report_path, 'w') as f:
            f.write(summary_report)
        
        # Print summary
        print(summary_report)
        print(f"\n📄 Full report saved to: {report_path}")
        
        # Quick access guide
        print("\n" + "="*60)
        print("🎯 QUICK ACCESS GUIDE")
        print("="*60)
        print("To use these datasets in your research:")
        print(f"1. Training data (Ea): {saved_files.get('ea_full', 'N/A')}")
        print(f"3. Evaluation data (Eb): {saved_files.get('eb_full', 'N/A')}")
        
        print("\n🎉  Dataset generation finished without Val set – paper-exact configuration.")
        
    except Exception as e:
        print(f"\n❌ Error during dataset generation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✨ Dataset generation complete!")


if __name__ == "__main__":
    main() 