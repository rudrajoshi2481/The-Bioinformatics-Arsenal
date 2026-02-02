#!/usr/bin/env python3
"""
Download Script for Narratives fMRI Dataset (ds002345)
Uses DataLad to download multi-subject data from OpenNeuro.

Usage:
    python download_data.py --subjects 10 --tasks tunnel pieman
    python download_data.py --all  # Download everything (large!)
    python download_data.py --list  # List available subjects/tasks
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import json


# Dataset configuration
DATASET_ID = "ds002345"
OPENNEURO_URL = f"https://github.com/OpenNeuroDatasets/{DATASET_ID}.git"
DEFAULT_DATA_DIR = Path("/app/tmp/brain_llm/ds002345")

# Available tasks in the Narratives dataset
AVAILABLE_TASKS = [
    "tunnel", "lucy", "prettymouth", "milkyway", "slumlord", "notthefall",
    "reach", "sherlock", "shapessocial", "shapesphysical", "schema",
    "bronx", "black", "forgot", "pieman"
]


def run_command(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return result"""
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  Error: {result.stderr}")
    return result


def check_datalad_installed() -> bool:
    """Check if DataLad is installed"""
    result = run_command(["datalad", "--version"], check=False)
    if result.returncode != 0:
        print("❌ DataLad not installed. Install with: pip install datalad")
        return False
    print(f"✓ DataLad version: {result.stdout.strip()}")
    return True


def check_git_annex_installed() -> bool:
    """Check if git-annex is installed"""
    result = run_command(["git-annex", "version"], check=False)
    if result.returncode != 0:
        print("❌ git-annex not installed. Install with: apt-get install git-annex")
        return False
    version_line = result.stdout.split('\n')[0] if result.stdout else "unknown"
    print(f"✓ git-annex: {version_line}")
    return True


def clone_dataset(data_dir: Path) -> bool:
    """Clone the dataset if not already present"""
    if data_dir.exists() and (data_dir / ".datalad").exists():
        print(f"✓ Dataset already cloned at {data_dir}")
        return True
    
    print(f"\n📥 Cloning dataset to {data_dir}...")
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    
    result = run_command(
        ["datalad", "clone", OPENNEURO_URL, str(data_dir)],
        check=False
    )
    
    if result.returncode != 0:
        print(f"❌ Failed to clone dataset: {result.stderr}")
        return False
    
    print("✓ Dataset cloned successfully")
    return True


def list_available_subjects(data_dir: Path) -> List[str]:
    """List all available subjects in the dataset"""
    subjects = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir() and d.name.startswith("sub-"):
            subjects.append(d.name)
    return subjects


def list_subject_tasks(data_dir: Path, subject: str) -> List[str]:
    """List available tasks for a subject"""
    func_dir = data_dir / subject / "func"
    if not func_dir.exists():
        return []
    
    tasks = set()
    for f in func_dir.iterdir():
        if "_task-" in f.name and f.name.endswith("_bold.nii.gz"):
            # Extract task name from filename
            parts = f.name.split("_")
            for part in parts:
                if part.startswith("task-"):
                    tasks.add(part.replace("task-", ""))
    return sorted(tasks)


def download_subject_data(data_dir: Path, subject: str, tasks: Optional[List[str]] = None) -> bool:
    """Download data for a specific subject"""
    subject_dir = data_dir / subject
    
    if not subject_dir.exists():
        print(f"  ⚠️ Subject {subject} not found in dataset")
        return False
    
    # Download anatomical data
    anat_dir = subject_dir / "anat"
    if anat_dir.exists():
        print(f"  📥 Downloading anatomical data for {subject}...")
        result = run_command(
            ["datalad", "get", str(anat_dir)],
            cwd=data_dir,
            check=False
        )
    
    # Download functional data
    func_dir = subject_dir / "func"
    if func_dir.exists():
        if tasks:
            # Download specific tasks
            for task in tasks:
                pattern = f"{subject}/func/*task-{task}*"
                print(f"  📥 Downloading {subject} task-{task}...")
                result = run_command(
                    ["datalad", "get", pattern],
                    cwd=data_dir,
                    check=False
                )
        else:
            # Download all functional data
            print(f"  📥 Downloading all functional data for {subject}...")
            result = run_command(
                ["datalad", "get", str(func_dir)],
                cwd=data_dir,
                check=False
            )
    
    return True


def verify_download(data_dir: Path, subject: str, task: str) -> dict:
    """Verify that data was downloaded correctly"""
    func_dir = data_dir / subject / "func"
    bold_files = list(func_dir.glob(f"*task-{task}*_bold.nii.gz"))
    
    result = {
        "subject": subject,
        "task": task,
        "bold_found": len(bold_files) > 0,
        "bold_files": [f.name for f in bold_files],
        "is_placeholder": False
    }
    
    if bold_files:
        # Check if file is a git-annex placeholder or actual data
        file_size = bold_files[0].stat().st_size
        result["file_size_mb"] = file_size / (1024 * 1024)
        result["is_placeholder"] = file_size < 1000  # Placeholder files are tiny
    
    return result


def print_dataset_summary(data_dir: Path):
    """Print summary of available data"""
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    
    subjects = list_available_subjects(data_dir)
    print(f"\nTotal subjects: {len(subjects)}")
    
    if subjects:
        print(f"Subject range: {subjects[0]} to {subjects[-1]}")
        
        # Sample a few subjects to show tasks
        sample_subjects = subjects[:3]
        print("\nSample subject tasks:")
        for subj in sample_subjects:
            tasks = list_subject_tasks(data_dir, subj)
            print(f"  {subj}: {', '.join(tasks[:5])}{'...' if len(tasks) > 5 else ''}")
    
    print("\nAvailable tasks in dataset:")
    for task in AVAILABLE_TASKS:
        print(f"  - {task}")


def generate_subject_list(data_dir: Path, output_file: Path):
    """Generate a JSON file with all subjects and their tasks"""
    subjects = list_available_subjects(data_dir)
    
    data = {
        "dataset": DATASET_ID,
        "total_subjects": len(subjects),
        "subjects": {}
    }
    
    for subj in subjects:
        tasks = list_subject_tasks(data_dir, subj)
        data["subjects"][subj] = {
            "tasks": tasks,
            "n_tasks": len(tasks)
        }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Subject list saved to {output_file}")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Download Narratives fMRI dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available subjects and tasks
  python download_data.py --list
  
  # Download first 10 subjects, tunnel task only
  python download_data.py --subjects 10 --tasks tunnel
  
  # Download specific subjects
  python download_data.py --subject-ids sub-001 sub-002 sub-003
  
  # Download all data (WARNING: ~500GB)
  python download_data.py --all
  
  # Verify downloads
  python download_data.py --verify --subjects 5
        """
    )
    
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                       help="Directory to store dataset")
    parser.add_argument("--subjects", type=int, default=None,
                       help="Number of subjects to download (from start)")
    parser.add_argument("--subject-ids", nargs="+", default=None,
                       help="Specific subject IDs to download")
    parser.add_argument("--tasks", nargs="+", default=None,
                       help="Specific tasks to download")
    parser.add_argument("--all", action="store_true",
                       help="Download all subjects and tasks")
    parser.add_argument("--list", action="store_true",
                       help="List available subjects and tasks")
    parser.add_argument("--verify", action="store_true",
                       help="Verify downloaded data")
    parser.add_argument("--generate-list", type=Path, default=None,
                       help="Generate JSON file with subject/task info")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be downloaded without downloading")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NARRATIVES FMRI DATASET DOWNLOADER")
    print("=" * 60)
    
    # Check dependencies
    print("\n📋 Checking dependencies...")
    if not check_datalad_installed():
        sys.exit(1)
    if not check_git_annex_installed():
        sys.exit(1)
    
    # Clone dataset if needed
    if not clone_dataset(args.data_dir):
        sys.exit(1)
    
    # List mode
    if args.list:
        print_dataset_summary(args.data_dir)
        return
    
    # Generate subject list
    if args.generate_list:
        generate_subject_list(args.data_dir, args.generate_list)
        return
    
    # Determine subjects to download
    all_subjects = list_available_subjects(args.data_dir)
    
    if args.subject_ids:
        subjects_to_download = args.subject_ids
    elif args.subjects:
        subjects_to_download = all_subjects[:args.subjects]
    elif args.all:
        subjects_to_download = all_subjects
    else:
        print("\n⚠️ No subjects specified. Use --subjects N, --subject-ids, or --all")
        print("   Run with --list to see available subjects")
        return
    
    # Determine tasks
    tasks = args.tasks if args.tasks else None
    
    print(f"\n📊 Download plan:")
    print(f"  Subjects: {len(subjects_to_download)}")
    print(f"  Tasks: {tasks if tasks else 'all'}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN - Would download:")
        for subj in subjects_to_download[:10]:
            print(f"  - {subj}")
        if len(subjects_to_download) > 10:
            print(f"  ... and {len(subjects_to_download) - 10} more")
        return
    
    # Download data
    print("\n📥 Starting downloads...")
    successful = 0
    failed = 0
    
    for i, subject in enumerate(subjects_to_download):
        print(f"\n[{i+1}/{len(subjects_to_download)}] {subject}")
        if download_subject_data(args.data_dir, subject, tasks):
            successful += 1
        else:
            failed += 1
    
    # Verify if requested
    if args.verify:
        print("\n🔍 Verifying downloads...")
        for subject in subjects_to_download[:5]:  # Verify first 5
            for task in (tasks or ["tunnel"]):
                result = verify_download(args.data_dir, subject, task)
                status = "✓" if result["bold_found"] and not result["is_placeholder"] else "✗"
                size = f"{result.get('file_size_mb', 0):.1f}MB" if result["bold_found"] else "N/A"
                print(f"  {status} {subject} {task}: {size}")
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Data directory: {args.data_dir}")


if __name__ == "__main__":
    main()
