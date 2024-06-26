#%%
import pandas as pd
import csv
import sys
import uuid
#%%
# Increase the field size limit
csv.field_size_limit(sys.maxsize)

# Path to the DwC archive occurrence file
file_path = '/itf-fi-ml/home/michato/data/occurrence.txt'


# Initialize a list to keep track of problematic lines
problematic_lines = []

# Use a smaller chunk size for initial inspection
chunk_size = 10000
chunks = []
column_headers = None

# Function to check if a line has the correct number of columns
def check_line(line, expected_num_columns):
    return len(line.split('\t')) == expected_num_columns

# Read the file manually line by line
with open(file_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    headers = next(reader)
    expected_num_columns = len(headers)
    for i, line in enumerate(reader, start=2):  # start=2 to account for header line
        if not check_line('\t'.join(line), expected_num_columns):
            problematic_lines.append((i, line))
        else:
            chunks.append(line)

# Create DataFrame from the non-problematic lines
if chunks:
    df = pd.DataFrame(chunks, columns=headers)
else:
    df = pd.DataFrame(columns=headers)

#%% Save the problematic lines to a log file
log_file_path = '/itf-fi-ml/home/michato/data/problematic_lines.log'
with open(log_file_path, 'w', encoding='utf-8') as log_file:
    for line_num, line in problematic_lines:
        log_file.write(f"Line {line_num}: {line}\n")

# Display the DataFrame
df.head()
# %%
# Function to check if a string is a valid UUID
def is_valid_uuid(key):
    try:
        uuid.UUID(key)
        return True
    except (ValueError, TypeError):
        return False

# Filter out valid UUIDs
valid_uuids = [key for key in list(df['datasetKey'].unique()) if is_valid_uuid(key)]
# %%
# Filter the dataframe to only include rows with valid UUIDs
filtered_df = df[df['datasetKey'].isin(valid_uuids)]

# Group by 'datasetKey' and sample 200 rows for each group
subset_df = filtered_df.groupby('datasetKey').apply(lambda x: x.sample(n=200, replace=True)).reset_index(drop=True)
# %%
subset_df.to_csv('data_subset.csv')
# %%
subset_df = pd.read_csv('data_subset.csv')
# %%
