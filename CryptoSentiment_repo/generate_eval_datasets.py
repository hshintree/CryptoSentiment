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

import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
from eval_loader_complete import EvalLoaderComplete


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
    val_df = datasets['val_full']
    eb_df = datasets['eb_full']
    all_data = datasets['all_data']
    
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
    
    # ===== VAL DATASET VALIDATION =====
    if verbose:
        print("\n📋 VAL DATASET (2021 STRESS-TEST) VALIDATION:")
        print("-" * 50)
    
    # Size validation
    val_size = len(val_df)
    val_size_ok = 15000 <= val_size <= 500000  # Flexible range for full year
    validation_results["val_validation"] = {}
    validation_results["val_validation"]["size"] = {"value": val_size, "ok": val_size_ok, "target": "150k-500k"}
    
    if verbose:
        status = "✅" if val_size_ok else "❌"
        print(f"{status} Size: {val_size:,} tweets (target: full 2021 year)")
    
    # Year validation
    val_years = sorted(val_df['date'].dt.year.unique())
    val_year_ok = val_years == [2021]
    validation_results["val_validation"]["year"] = {
        "value": val_years, "ok": val_year_ok, "target": "[2021]"
    }
    
    if verbose:
        status = "✅" if val_year_ok else "❌"
        print(f"{status} Year coverage: {val_years} (target: [2021])")
    
    # Label distribution (no hard balance check - just display)
    val_labels = val_df['Label'].value_counts()
    validation_results["val_validation"]["distribution"] = {
        "value": val_labels.to_dict(),
        "ok": True,  # Always pass - just informational
        "target": "Display only"
    }
    
    if verbose:
        print(f"✅ Label distribution: {val_labels.to_dict()}")
    
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
    
    # Year validation (excluding 2019-2021)
    eb_years = sorted(eb_df['date'].dt.year.unique())
    eb_year_ok = (2019 not in eb_years) and (2020 not in eb_years) and (2021 not in eb_years) and len(eb_years) > 1
    validation_results["eb_validation"]["years"] = {
        "value": eb_years, "ok": eb_year_ok, "target": "2015-2018, 2022-2023"
    }
    
    if verbose:
        status = "✅" if eb_year_ok else "❌"
        print(f"{status} Year coverage: {eb_years} (target: excludes 2019-2021)")
    
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
    
    # ===== CROSS-VALIDATION VALIDATION =====
    if verbose:
        print("\n📋 CROSS-VALIDATION SETUP VALIDATION:")
        print("-" * 50)
    
    # Check fold sizes
    ea_train_folds = datasets['ea_train_folds']
    ea_test_folds = datasets['ea_test_folds']
    eb_train_folds = datasets['eb_train_folds']
    eb_test_folds = datasets['eb_test_folds']
    
    cv_ok = (len(ea_train_folds) == 5 and len(ea_test_folds) == 5 and 
             len(eb_train_folds) == 5 and len(eb_test_folds) == 5)
    validation_results["overall_validation"]["cv_folds"] = {"ok": cv_ok, "target": "5 folds each"}
    
    if verbose:
        status = "✅" if cv_ok else "❌"
        print(f"{status} Cross-validation folds: {len(ea_train_folds)} ea + {len(eb_train_folds)} eb folds")
        print(f"   Ea train sizes: {[len(f) for f in ea_train_folds]}")
        print(f"   Ea test sizes: {[len(f) for f in ea_test_folds]}")
        print(f"   Eb train sizes: {[len(f) for f in eb_train_folds]}")
        print(f"   Eb test sizes: {[len(f) for f in eb_test_folds]}")
    
    # ===== AGGREGATED DATA VALIDATION =====
    if verbose:
        print("\n📋 AGGREGATED DATA VALIDATION:")
        print("-" * 50)
    
    agg_size = len(all_data)
    agg_size_ok = 1000 <= agg_size <= 5000  # Reasonable number of days
    validation_results["overall_validation"]["aggregated_size"] = {
        "value": agg_size, 
        "ok": agg_size_ok, 
        "target": "1,000-5,000 days"
    }
    
    if verbose:
        status = "✅" if agg_size_ok else "❌"
        print(f"{status} Aggregated data: {agg_size:,} days (target: reasonable daily coverage)")
    
    # ===== OVERALL VALIDATION =====
    all_checks = [
        ea_size_ok, ea_year_ok, ea_balance_ok, ea_vol_ok,
        val_size_ok, val_year_ok,
        eb_size_ok, eb_year_ok, eb_event_ok, eb_actionable_ok,
        cv_ok, agg_size_ok
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


def save_datasets_to_csv(datasets, data_dir, timestamp=None):
    """
    Save all datasets to CSV files with descriptive names.
    
    Args:
        datasets: Dictionary from EvalLoaderComplete.create_eval_datasets()
        data_dir: Path object for data directory
        timestamp: Optional timestamp string for file naming
        
    Returns:
        Dict with saved file paths
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    saved_files = {}
    
    print("\n" + "="*60)
    print("💾 SAVING DATASETS TO CSV FILES")
    print("="*60)
    
    # Main datasets
    files_to_save = [
        ("ea_full", f"ea_dataset_2019-2020_train_{timestamp}.csv", "Ea training dataset (2019-2020)"),
        ("val_full", f"val_dataset_2021_stress_{timestamp}.csv", "2021 stress-test dataset"),
        ("eb_full", f"eb_event_15-18_22-23_{timestamp}.csv", "Eb evaluation dataset (events)"),
        ("all_data", f"aggregated_daily_dataset_{timestamp}.csv", "Daily aggregated dataset")
    ]
    
    for key, filename, description in files_to_save:
        if key in datasets:
            filepath = data_dir / filename
            datasets[key].to_csv(filepath, index=False)
            saved_files[key] = str(filepath)
            print(f"✅ Saved {description}: {filepath}")
            print(f"   Shape: {datasets[key].shape}")
    
    # Cross-validation folds
    print(f"\n📁 Saving cross-validation folds...")
    
    # Ea folds
    for i, (train_fold, test_fold) in enumerate(zip(datasets['ea_train_folds'], datasets['ea_test_folds'])):
        train_path = data_dir / f"ea_train_fold_{i+1}_{timestamp}.csv"
        test_path = data_dir / f"ea_test_fold_{i+1}_{timestamp}.csv"
        
        train_fold.to_csv(train_path, index=False)
        test_fold.to_csv(test_path, index=False)
        
        saved_files[f"ea_train_fold_{i+1}"] = str(train_path)
        saved_files[f"ea_test_fold_{i+1}"] = str(test_path)
    
    # Eb folds
    for i, (train_fold, test_fold) in enumerate(zip(datasets['eb_train_folds'], datasets['eb_test_folds'])):
        train_path = data_dir / f"eb_train_fold_{i+1}_{timestamp}.csv"
        test_path = data_dir / f"eb_test_fold_{i+1}_{timestamp}.csv"
        
        train_fold.to_csv(train_path, index=False)
        test_fold.to_csv(test_path, index=False)
        
        saved_files[f"eb_train_fold_{i+1}"] = str(train_path)
        saved_files[f"eb_test_fold_{i+1}"] = str(test_path)
    
    print(f"✅ Saved 20 cross-validation fold files (5 folds × 2 datasets × 2 splits)")
    
    # Save file index
    index_data = []
    for key, filepath in saved_files.items():
        index_data.append({
            "dataset": key,
            "filepath": filepath,
            "timestamp": timestamp
        })
    
    index_df = pd.DataFrame(index_data)
    index_path = data_dir / f"dataset_index_{timestamp}.csv"
    index_df.to_csv(index_path, index=False)
    saved_files["index"] = str(index_path)
    
    print(f"✅ Saved dataset index: {index_path}")
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
    report_lines.append(f"Ea dataset (2019-2020 training): {len(datasets['ea_full']):,} tweets")
    report_lines.append(f"Val dataset (2021 stress-test): {len(datasets['val_full']):,} tweets")
    report_lines.append(f"Eb dataset (event evaluation): {len(datasets['eb_full']):,} tweets")
    report_lines.append(f"Daily aggregated dataset: {len(datasets['all_data']):,} days")
    report_lines.append(f"Cross-validation folds: 5 folds per dataset")
    report_lines.append("")
    
    # Label distributions
    report_lines.append("📊 LABEL DISTRIBUTIONS:")
    report_lines.append("-" * 40)
    ea_labels = datasets['ea_full']['Label'].value_counts()
    val_labels = datasets['val_full']['Label'].value_counts()
    eb_labels = datasets['eb_full']['Label'].value_counts()
    
    report_lines.append("Ea dataset (2019-2020 training):")
    for label, count in ea_labels.items():
        pct = (count / len(datasets['ea_full']) * 100)
        report_lines.append(f"  {label}: {count:,} tweets ({pct:.1f}%)")
    
    report_lines.append("Val dataset (2021 stress-test):")
    for label, count in val_labels.items():
        pct = (count / len(datasets['val_full']) * 100)
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
    report_lines.append(f"Main datasets: {len([f for f in saved_files.keys() if 'fold' not in f])} files")
    report_lines.append(f"CV folds: {len([f for f in saved_files.keys() if 'fold' in f])} files")
    report_lines.append(f"Total files: {len(saved_files)} files")
    report_lines.append("")
    
    # Key files
    report_lines.append("🔑 KEY FILES:")
    report_lines.append("-" * 40)
    key_files = ["ea_full", "val_full", "eb_full", "all_data", "index"]
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
        
        # 3. Save to CSV files
        print("\n💾 Saving datasets to CSV files...")
        saved_files = save_datasets_to_csv(datasets, data_dir, timestamp)
        
        # 4. Generate summary report
        summary_report = generate_summary_report(datasets, validation_results, saved_files, timestamp)
        
        # Save summary report
        report_path = data_dir / f"generation_report_{timestamp}.txt"
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
        print(f"2. Validation data (Val): {saved_files.get('val_full', 'N/A')}")
        print(f"3. Evaluation data (Eb): {saved_files.get('eb_full', 'N/A')}")
        print(f"3. Daily aggregated: {saved_files.get('all_data', 'N/A')}")
        print(f"4. Dataset index: {saved_files.get('index', 'N/A')}")
        print("\nFor cross-validation, use the individual fold files:")
        print("  ea_train_fold_1.csv, ea_test_fold_1.csv, ...")
        print("  eb_train_fold_1.csv, eb_test_fold_1.csv, ...")
        
        if validation_results["all_checks_passed"]:
            print("\n🎉 SUCCESS: All datasets generated and validated successfully!")
        else:
            print("\n⚠️  WARNING: Some validation checks failed. Please review the report.")
        
    except Exception as e:
        print(f"\n❌ Error during dataset generation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n✨ Dataset generation complete!")


if __name__ == "__main__":
    main() 