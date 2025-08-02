import os
import json
import concurrent.futures
from eval import main as eval_main
from typing import List, Dict
import time

def process_single_file(file_path: str) -> Dict:
   
    try:
        print(f"\nStart processing: {os.path.basename(file_path)}")
        print("=" * 50)
        start_time = time.time()
        
        eval_main(file_path)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return {
            "file_name": os.path.basename(file_path),
            "status": "success",
            "processing_time": processing_time
        }
    except Exception as e:
        return {
            "file_name": os.path.basename(file_path),
            "status": "error",
            "error_message": str(e)
        }

def process_all_files(max_workers: int = 3):
    """
    :param max_workers: max number of workers, default is 3
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
    
    data_dir = os.path.join(project_root, 'data', 'nil', 'test', 'results', 'd2t', 'llama')
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"data directory not found: {data_dir}")
    
    jsonl_files = [f for f in os.listdir(data_dir) if f.endswith('.jsonl')]
    
    if not jsonl_files:
        print(f"warning: no JSONL files found in {data_dir}")
        return
    
    file_paths = [os.path.join(data_dir, f) for f in jsonl_files]
    
    print(f"found {len(file_paths)} files to process")
    print(f"using {max_workers} concurrent threads")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_single_file, file_path): file_path 
                         for file_path in file_paths}
        
        for future in concurrent.futures.as_completed(future_to_file):
            result = future.result()
            results.append(result)
            
            if result["status"] == "success":
                print(f"\nfile {result['file_name']} processed successfully")
                print(f"processing time: {result['processing_time']:.2f} seconds")
            else:
                print(f"\nfile {result['file_name']} processing failed")
                print(f"error message: {result['error_message']}")
    
    print("\nProcessing completed! Summary:")
    print("=" * 50)
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")
    print(f"total files: {len(results)}")
    print(f"success: {success_count}")
    print(f"failed: {error_count}")
    
    if error_count > 0:
        print("\nfailed file list:")
        for result in results:
            if result["status"] == "error":
                print(f"- {result['file_name']}: {result['error_message']}")

if __name__ == "__main__":
    process_all_files(max_workers=3) 