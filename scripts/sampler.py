import pandas as pd
import time

# --- Configuration ---
original_filepath = 'complaints.csv'  # 1. UPDATE THIS PATH
lite_filepath = 'complaints_lite.csv'          # 2. This will be our new file
chunk_size = 100000                            # Process in chunks of 100k rows
sample_fraction = 0.01                         # 3. We'll keep 1% of the data. 
                                               # (Adjust if you want more/less)

# --- Script ---
print(f"Starting to sample '{original_filepath}'...")
start_time = time.time()
all_chunks = []

# Use a try-except block to handle potential CSV parsing errors
try:
    with pd.read_csv(original_filepath, chunksize=chunk_size, on_bad_lines='skip') as reader:
        for chunk in reader:
            # Take a random 1% sample of the current chunk
            sampled_chunk = chunk.sample(frac=sample_fraction)
            all_chunks.append(sampled_chunk)

except Exception as e:
    print(f"Error while reading CSV: {e}")
    print("This can happen if the file is corrupt or has an encoding issue.")
    print("Trying to read with 'latin1' encoding...")
    
    # Reset and try again with a different encoding (very common for large, weird CSVs)
    all_chunks = []
    try:
        with pd.read_csv(original_filepath, chunksize=chunk_size, on_bad_lines='skip', encoding='latin1') as reader:
            for chunk in reader:
                sampled_chunk = chunk.sample(frac=sample_fraction)
                all_chunks.append(sampled_chunk)
    except Exception as final_e:
        print(f"Failed again: {final_e}")
        print("Please check the file. The process is stopping.")
        exit()

if all_chunks:
    # Combine all the small, sampled chunks into one DataFrame
    final_df = pd.concat(all_chunks, ignore_index=True)
    
    # Save our new, small, and usable dataset!
    final_df.to_csv(lite_filepath, index=False)
    
    end_time = time.time()
    print("\n--- 🥳 SUCCESS! ---")
    print(f"Created '{lite_filepath}' with {len(final_df)} rows.")
    print(f"Total time: {end_time - start_time:.2f} seconds.")
    print("\nYou can now open your Jupyter Notebook and use this new file.")

else:
    print("No data was processed. Please check your file path and chunk size.")