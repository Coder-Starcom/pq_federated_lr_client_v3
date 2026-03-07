import pandas as pd
import os

def split_csv(input_csv, rows_per_file=5000, max_files=10):
    """
    Breaks a large CSV into smaller pieces suitable for GitHub.
    """
    if not os.path.exists(input_csv):
        print(f"❌ Error: {input_csv} not found!")
        return

    print(f"Reading {input_csv} in chunks of {rows_per_file} rows...")
    
    # Read in chunks so we don't crash RAM
    reader = pd.read_csv(input_csv, chunksize=rows_per_file)
    
    for i, chunk in enumerate(reader):
        filename = f"client_data_{i+1}.csv"
        chunk.to_csv(filename, index=False)
        print(f"✅ Created {filename} ({len(chunk)} rows)")
        
        # Stop after max_files to keep the repo clean
        if i + 1 >= max_files:
            print(f"✋ Reached limit of {max_files} files. Stopping.")
            break

if __name__ == "__main__":
    # 1. Change this to your actual filename
    TARGET_FILE = "Fraud.csv" 
    
    # 2. Set how many rows per friend (5000 is usually < 2MB)
    ROWS = 5000 
    
    split_csv(TARGET_FILE, rows_per_file=ROWS, max_files=10)
    
    print("\n🚀 Done! You can now push these smaller .csv files to GitHub.")
