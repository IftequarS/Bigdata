'''Bigdata
Cloud Computing Trends and Innovation : A Practical Exploration'''

from azureml.core import Workspace, Dataset
import pandas as pd

ws = Workspace.from_config()

dataset = Dataset.get_by_name(ws, name="ibm_cloud")
df = dataset.to_pandas_dataframe()

df.head()
df.info()
df.isnull().sum()

# Splitting the single column into multiple columns
df = df.iloc[:, 0].str.split(",", expand=True)

# Assign correct column names
df.columns = [
    "interval_start",
    "location",
    "kind",
    "host",
    "method",
    "statuscode",
    "endpoint",
    "aggregated_stats_name",
    "aggregated_stats_value"
]

df.head()
df.info()

df["interval_start"] = pd.to_datetime(
    df["interval_start"],
    unit="s",
    errors="coerce"
)
df["statuscode"] = pd.to_numeric(df["statuscode"], errors="coerce")
df["aggregated_stats_value"] = pd.to_numeric(df["aggregated_stats_value"], errors="coerce")

# Remove duplicates
df = df.drop_duplicates()

# Handle missing values
df.fillna({
    "location": "unknown",
    "kind": "unknown",
    "host": "unknown",
    "method": "unknown",
    "endpoint": "unknown",
}, inplace=True)

df["statuscode"].fillna(df["statuscode"].median(), inplace=True)
df["aggregated_stats_value"].fillna(df["aggregated_stats_value"].median(), inplace=True)

df.isnull().sum()

df.describe()

df.shape

df.to_csv("cleaned_ibm_cloud.csv", index=False)

import os

clean_path = "./outputs/cleaned_ibm_cloud.csv"
os.makedirs("outputs", exist_ok=True)

df.to_csv(clean_path, index=False)

from azureml.core import Workspace

ws = Workspace.from_config()
datastore = ws.get_default_datastore()

datastore.upload(
    src_dir="outputs",
    target_path="cleaned-data",
    overwrite=True
)

from azureml.core import Workspace, Dataset

ws = Workspace.from_config()
datastore = ws.get_default_datastore()

dataset = Dataset.Tabular.from_delimited_files(
    path=(datastore, "outputs/cleaned_ibm_cloud.csv"),
    validate=False,
    infer_column_types=False,
    separator=",",
    header=True
)

dataset = dataset.register(
    workspace=ws,
    name="cleaned ibm cloud",
    description="Cleaned metrics dataset with schema inference disabled",
    create_new_version=True
)

print("Dataset registered successfully")

*******************************************************************************************************************************

Output:-
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1048575 entries, 0 to 1048574
Data columns (total 1 columns):
 #   Column                                                                                                     Non-Null Count    Dtype 
---  ------                                                                                                     --------------    ----- 
 0   interval_start,location,kind,host,method,statusCode,endpoint,aggregated_stats_name,aggregated_stats_value  1048575 non-null  object
dtypes: object(1)
memory usage: 8.0+ MB
interval_start,location,kind,host,method,statusCode,endpoint,aggregated_stats_name,aggregated_stats_value    0
dtype: int64
interval_start	location	kind	host	method	statuscode	endpoint	aggregated_stats_name	aggregated_stats_value
0	1705951500	datacenter1	CLIENT	component41	GET	200	endpoint892	avg	108022.47
1	1705951500	datacenter1	CLIENT	component41	GET	200	endpoint892	median	77053.5
2	1705951500	datacenter1	CLIENT	component41	GET	200	endpoint892	min	37770.0
3	1705951500	datacenter1	CLIENT	component41	GET	200	endpoint892	max	3062620.0
4	1705951500	datacenter1	CLIENT	component41	GET	200	endpoint892	count	300.0
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1048575 entries, 0 to 1048574
Data columns (total 9 columns):
 #   Column                  Non-Null Count    Dtype 
---  ------                  --------------    ----- 
 0   interval_start          1048575 non-null  object
 1   location                1048575 non-null  object
 2   kind                    1048575 non-null  object
 3   host                    1048575 non-null  object
 4   method                  1048575 non-null  object
 5   statuscode              1048575 non-null  object
 6   endpoint                1048575 non-null  object
 7   aggregated_stats_name   1048575 non-null  object
 8   aggregated_stats_value  1048575 non-null  object
dtypes: object(9)
memory usage: 72.0+ MB
interval_start            0
location                  0
kind                      0
host                      0
method                    0
statuscode                0
endpoint                  0
aggregated_stats_name     0
aggregated_stats_value    0
dtype: int64
statuscode	aggregated_stats_value
count	1.048575e+06	1.048575e+06
mean	2.337673e+02	4.280070e+05
std	8.153035e+01	2.079348e+06
min	-1.000000e+00	-1.669836e+02
25%	2.000000e+02	2.896320e+00
50%	2.000000e+02	1.974808e+04
75%	2.000000e+02	2.251042e+05
max	5.040000e+02	2.373539e+08
(1048575, 9)
"Datastore.upload" is deprecated after version 1.0.69. Please use "Dataset.File.upload_directory" to upload your files             from a local directory and create FileDataset in single method call. See Dataset API change notice at https://aka.ms/dataset-deprecation.
Uploading an estimated of 1 files
Uploading outputs/metrics_dataset_cleaned.csv
Uploaded outputs/metrics_dataset_cleaned.csv, 1 files out of an estimated total of 1
Uploaded 1 files
$AZUREML_DATAREFERENCE_2ccecc2786f7426581f897f7d3b1998f
Dataset registered successfully
{'infer_column_types': 'False', 'activity': 'to_pandas_dataframe'}
{'infer_column_types': 'False', 'activity': 'to_pandas_dataframe', 'activityApp': 'TabularDataset'}
interval_start,location,kind,host,method,statusCode,endpoint,aggregated_stats_name,aggregated_stats_value
0	1705951500,datacenter1,CLIENT,component41,GET,...
1	1705951500,datacenter1,CLIENT,component41,GET,...
2	1705951500,datacenter1,CLIENT,component41,GET,...
3	1705951500,datacenter1,CLIENT,component41,GET,...
4	1705951500,datacenter1,CLIENT,component41,GET,...
