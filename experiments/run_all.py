import os
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
import concurrent.futures
from typing import Dict, List, Tuple

# Configuration paths
BASE_DIR = Path(__file__).parent
OUTPUT_BASE_DIR = BASE_DIR.parent.parent / "results/Self-Reflection/sum/llama"

# Concurrency configuration
MAX_WORKERS = 1  # Maximum concurrency

def ensure_output_dir():
    """Ensure output directory exists"""
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    return OUTPUT_BASE_DIR

def get_output_path(input_file: str) -> str:
    """Get output file path"""
    # Generate output filename from input filename
    # Example: location.py -> sum_llama_loc_50.jsonl
    filename = os.path.basename(input_file)
    name_parts = filename.replace('.py', '').split('_')
    
    # Build output filename
    if len(name_parts) == 1:
        # Simple filename, like location.py -> loc
        key_part = name_parts[0]
    else:
        # Complex filename, like location_fine-grained_category.py -> loc-fine-grained-category
        key_part = '-'.join(name_parts)
    
    output_filename = f"sum_llama_{key_part}_50.jsonl"
    return os.path.join(OUTPUT_BASE_DIR, output_filename)

def run_file(input_file: str) -> Tuple[str, Dict]:
    """Execute a single file and return result"""
    print(f"\nStarting execution: {input_file}")
    
    # Build command
    cmd = ["python", str(input_file)]
    
    try:
        # Execute command
        start_time = time.time()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(BASE_DIR)  # Set working directory
        )
        
        # Wait for execution to complete
        stdout, stderr = process.communicate()
        end_time = time.time()
        
        # Parse output
        result = {
            "success": process.returncode == 0,
            "duration": end_time - start_time,
            "stdout": stdout,
            "stderr": stderr,
            "return_code": process.returncode
        }
        
        # Print execution status
        if result["success"]:
            print(f"✓ Successfully executed {input_file} (Duration: {result['duration']:.2f} seconds)")
        else:
            print(f"✗ Execution failed {input_file}")
            print(f"Error message: {result['stderr']}")
        
        return input_file, result
        
    except Exception as e:
        error_result = {
            "success": False,
            "duration": 0,
            "stdout": "",
            "stderr": str(e),
            "return_code": -1
        }
        print(f"✗ Execution exception {input_file}")
        print(f"Error message: {str(e)}")
        return input_file, error_result

def main():
    # Ensure output directory exists
    ensure_output_dir()
    
    # Create log directory
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Create log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_log_{timestamp}.json"
    
    # Get all Python files
    input_files = [
        os.path.join(BASE_DIR, f) 
        for f in os.listdir(BASE_DIR) 
        if f.endswith('.py') and f != 'run_all.py'
    ]
    
    # Execution result records
    results = {
        "start_time": timestamp,
        "total_files": len(input_files),
        "successful": 0,
        "failed": 0,
        "executions": []
    }
    
    # Use thread pool to execute files
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(run_file, input_file): input_file for input_file in input_files}
        
        # Collect results
        for future in concurrent.futures.as_completed(future_to_file):
            input_file, result = future.result()
            
            # Update statistics
            if result["success"]:
                results["successful"] += 1
            else:
                results["failed"] += 1
            
            # Record execution result
            execution_record = {
                "input_file": input_file,
                "output_file": get_output_path(input_file),
                "start_time": time.strftime("%H:%M:%S", time.localtime(time.time() - result["duration"])),
                "end_time": time.strftime("%H:%M:%S"),
                "duration": result["duration"],
                "success": result["success"],
                "error": result["stderr"] if not result["success"] else None
            }
            results["executions"].append(execution_record)
    
    # Save execution log
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\nExecution Summary:")
    print(f"Total files: {results['total_files']}")
    print(f"Successfully executed: {results['successful']}")
    print(f"Execution failed: {results['failed']}")
    print(f"Detailed log saved to: {log_file}")

if __name__ == "__main__":
    main() 