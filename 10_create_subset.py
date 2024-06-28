#%%
import pandas as pd
from dwca.read import DwCAReader
#%%

# Path to the DwC archive occurrence file
file_path = '/itf-fi-ml/home/michato/data/0092709-240506114902167.zip'
def on_bad_lines(ln: list[str]):
   print(ln)
   return None
with DwCAReader(file_path) as dwca:
   print("Core data file is: {}".format(dwca.descriptor.core.file_location)) # => 'occurrence.txt'

   df = dwca.pd_read('occurrence.txt', parse_dates=True, on_bad_lines=on_bad_lines, quotechar='\"', engine="python")
   multimedia_df = dwca.pd_read('multimedia.txt', on_bad_lines='warn')
#%%
df['datasetKey'].unique()
# %%


#%% Group by 'datasetKey' and sample 200 rows for each group
subset_df = filtered_df.groupby('datasetKey').apply(lambda x: x.sample(n=200, replace=True)).reset_index(drop=True)
# %%
subset_df.to_csv('data_subset.csv')
# %%
subset_df = pd.read_csv('data_subset.csv')
# %%
multimedia = pd.read_csv('/itf-fi-ml/home/michato/data/multimedia.txt', sep='\t')
# %%
multimedia[multimedia['gbifID'].isin(subset_df['gbifID'])]
# %%
