import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#------------ 1. BASIC PYTHON FILE LOADING ------------

# Read a text file
# with open("data.txt", "r") as vin_file:
#     vin_data = vin_file.read()
# print(vin_data)

# Read a text file line by line
# with open("data.txt", "r") as vin_file:
#     vin_lines = vin_file.readlines()
# print(vin_lines)

# File modes:
# "r"  = read
# "w"  = write (overwrites existing content)
# "a"  = append
# "rb" = read binary
# "wb" = write binary


def vin_read_text(vin_path):
    with open(vin_path, "r") as vin_file:
        vin_data = vin_file.read()
    return vin_data


def vin_read_lines(vin_path):
    with open(vin_path, "r") as vin_file:
        vin_lines = vin_file.readlines()
    return vin_lines


#------------ 2. NUMPY FILE LOADING ------------

# Import NumPy using: import numpy as np

# A. Load a single NumPy array from .npy
# vin_data = np.load("data.npy")
# print(vin_data)
# print(type(vin_data))
# print(vin_data.shape)

# Save a single NumPy array
# vin_array = np.array([10, 20, 30, 40])
# np.save("data.npy", vin_array)


def vin_load_npy(vin_path):
    return np.load(vin_path)


# B. Load multiple arrays from .npz
# vin_data = np.load("dataset.npz")
# vin_X = vin_data["X"]
# vin_y = vin_data["y"]
# print(vin_X)
# print(vin_y)

# Save multiple arrays in .npz
# vin_X = np.array([1, 2, 3, 4])
# vin_y = np.array([5, 6, 7, 8])
# np.savez("dataset.npz", X=vin_X, y=vin_y)

# Save compressed .npz
# np.savez_compressed("dataset.npz", X=vin_X, y=vin_y)


def vin_load_npz(vin_path):
    vin_loaded = np.load(vin_path)
    vin_data = {}
    for vin_key in vin_loaded.files:
        vin_data[vin_key] = vin_loaded[vin_key]
    vin_loaded.close()
    return vin_data


# C. Load numeric CSV data using NumPy
# vin_data = np.loadtxt("data.csv", delimiter=",")
# print(vin_data)

# If the CSV has a header or missing values
# vin_data = np.genfromtxt("data.csv", delimiter=",", skip_header=1)
# print(vin_data)

# D. Load numeric TXT data
# vin_data = np.loadtxt("data.txt")
# print(vin_data)

# For tab-separated numeric TXT data
# vin_data = np.loadtxt("data.txt", delimiter="\t")


def vin_load_numeric_csv(vin_path):
    return np.loadtxt(vin_path, delimiter=",")


def vin_load_csv_with_header(vin_path):
    return np.genfromtxt(vin_path, delimiter=",", skip_header=1)


def vin_load_numeric_txt(vin_path):
    return np.loadtxt(vin_path)


def vin_load_tab_separated_txt(vin_path):
    return np.loadtxt(vin_path, delimiter="\t")


#------------ 3. PANDAS FILE LOADING ------------

# Import Pandas using: import pandas as pd

# A. Load CSV
# vin_df = pd.read_csv("data.csv")

# Skip the first two rows
# vin_df = pd.read_csv("data.csv", skiprows=2)

# Specify a separator
# vin_df = pd.read_csv("data.csv", sep="\t")

# Use the first column as the row index
# vin_df = pd.read_csv("data.csv", index_col=0)

# Read only selected columns
# vin_df = pd.read_csv("data.csv", usecols=["Age", "Label"])

# Treat selected strings as missing values
# vin_df = pd.read_csv("data.csv", na_values=["NA", "?"])


def vin_load_csv(vin_path):
    return pd.read_csv(vin_path)


# B. Load Excel file (.xlsx)
# vin_df = pd.read_excel("data.xlsx")
# vin_df = pd.read_excel("data.xlsx", sheet_name="Sheet1")


def vin_load_excel(vin_path, vin_sheet=0):
    return pd.read_excel(vin_path, sheet_name=vin_sheet)


# C. Load JSON file
# vin_df = pd.read_json("data.json")


def vin_load_json(vin_path):
    return pd.read_json(vin_path)


# D. Load TSV file (tab-separated)
# vin_df = pd.read_csv("data.tsv", sep="\t")


def vin_load_tsv(vin_path):
    return pd.read_csv(vin_path, sep="\t")


# E. Load tabular TXT file
# vin_df = pd.read_csv("data.txt", sep="\t")


def vin_load_tabular_txt(vin_path):
    return pd.read_csv(vin_path, sep="\t")


# F. Load Pickle file (.pkl)
# vin_df = pd.read_pickle("data.pkl")


def vin_load_pickle(vin_path):
    return pd.read_pickle(vin_path)


#------------ 4. INSPECT A PANDAS DATASET ------------


def vin_inspect_dataframe(vin_df):
    print(vin_df.head())
    print(vin_df.tail())
    print(vin_df.shape)
    print(vin_df.columns)
    print(vin_df.dtypes)
    vin_df.info()
    print(vin_df.describe())
    print(vin_df.isnull().sum())


# Individual commands:
# print(vin_df.head())          # First 5 rows
# print(vin_df.tail())          # Last 5 rows
# print(vin_df.shape)           # Number of rows and columns
# print(vin_df.columns)         # Column names
# print(vin_df.dtypes)           # Data types
# vin_df.info()                  # Dataset information
# print(vin_df.describe())      # Statistical summary
# print(vin_df.isnull().sum())  # Missing values per column


#------------ 5. BASIC PANDAS OPERATIONS ------------


def vin_pandas_operations(vin_df):
    # Select one column
    vin_age = vin_df["Age"]

    # Select multiple columns
    vin_subset = vin_df[["Age", "Label"]]

    # Select a row by position
    vin_row = vin_df.iloc[0]

    # Select a specific cell
    vin_value = vin_df.iloc[0, 1]

    # Select rows based on a condition
    vin_filtered = vin_df[vin_df["Age"] > 30]

    # Remove rows containing missing values
    vin_clean = vin_df.dropna()

    # Fill missing values in Age with zero
    vin_filled = vin_df.copy()
    vin_filled["Age"] = vin_filled["Age"].fillna(0)

    # Remove the ID column
    vin_without_id = vin_df.drop("ID", axis=1)

    # Convert Age to numeric; invalid values become NaN
    vin_numeric = vin_df.copy()
    vin_numeric["Age"] = pd.to_numeric(
        vin_numeric["Age"], errors="coerce"
    )

    return vin_age, vin_subset, vin_row, vin_value, vin_filtered, vin_clean, vin_filled, vin_without_id, vin_numeric


#------------ 6. SEPARATE FEATURES (X) AND TARGET (y) ------------

# A. Separate X and y from a Pandas DataFrame using the target column name
# Example: the column "Label" is the output/target.
# vin_X = vin_df.drop(columns=["Label"])  # All columns except Label
# vin_y = vin_df["Label"]                 # Only the target column

# The same operation using axis=1:
# vin_X = vin_df.drop("Label", axis=1)
# vin_y = vin_df["Label"]

# B. Separate X and y by column position using Pandas
# Use this when the last column contains the target.
# vin_X = vin_df.iloc[:, :-1]  # All rows, all columns except last
# vin_y = vin_df.iloc[:, -1]   # All rows, only the last column

# C. Separate X and y from a NumPy array
# Assume the last column contains the target.
# vin_array = np.loadtxt("data.csv", delimiter=",", skiprows=1)
# vin_X = vin_array[:, :-1]  # All rows, all columns except last
# vin_y = vin_array[:, -1]   # All rows, only the last column

# Alternative NumPy method: remove the target column by its index.
# vin_X = np.delete(vin_array, -1, axis=1)
# vin_y = vin_array[:, -1]


def vin_split_pandas_xy(vin_df, vin_target_column):
    """Return features X and target y from a Pandas DataFrame."""
    vin_X = vin_df.drop(columns=[vin_target_column])
    vin_y = vin_df[vin_target_column]
    return vin_X, vin_y


def vin_split_numpy_xy(vin_array, vin_target_index=-1):
    """Return features X and target y from a 2D NumPy array."""
    vin_X = np.delete(vin_array, vin_target_index, axis=1)
    vin_y = vin_array[:, vin_target_index]
    return vin_X, vin_y


# D. Split features and labels into training and testing sets
# This works with NumPy arrays, Pandas DataFrames, and Pandas Series.
# vin_X_train, vin_X_test, vin_y_train, vin_y_test = train_test_split(
#     vin_X, vin_y, test_size=0.2, random_state=42
# )
# test_size=0.2 means 80% training data and 20% testing data.
# random_state=42 makes the split reproducible.

# For classification, preserve the class proportions using stratify:
# vin_X_train, vin_X_test, vin_y_train, vin_y_test = train_test_split(
#     vin_X, vin_y, test_size=0.2, random_state=42, stratify=vin_y
# )


def vin_split_train_test(vin_X, vin_y, vin_test_size=0.2):
    """Return X_train, X_test, y_train, y_test."""
    return train_test_split(
        vin_X,
        vin_y,
        test_size=vin_test_size,
        random_state=42
    )


#------------ 7. CONVERT BETWEEN PANDAS AND NUMPY ------------

# Pandas DataFrame to NumPy array
# vin_array = vin_df.to_numpy()

# NumPy array to Pandas DataFrame
# vin_df = pd.DataFrame(vin_array)

# NumPy array to DataFrame with column names
# vin_df = pd.DataFrame(
#     vin_array,
#     columns=["Feature1", "Feature2", "Label"]
# )


def vin_pandas_to_numpy(vin_df):
    return vin_df.to_numpy()


def vin_numpy_to_pandas(vin_array, vin_columns=None):
    return pd.DataFrame(vin_array, columns=vin_columns)


#------------ 8. SAVE DATASETS ------------

# Save a Pandas DataFrame as CSV
# vin_df.to_csv("data.csv", index=False)

# Save a Pandas DataFrame as Excel
# vin_df.to_excel("data.xlsx", index=False)

# Save a Pandas DataFrame as JSON
# vin_df.to_json("data.json")

# Save a Pandas DataFrame as Pickle
# vin_df.to_pickle("data.pkl")

# Save a NumPy array as NPY
# np.save("data.npy", vin_array)

# Save multiple NumPy arrays as NPZ
# np.savez("dataset.npz", X=vin_X, y=vin_y)


def vin_save_datasets(vin_df, vin_array, vin_X, vin_y):
    vin_df.to_csv("data.csv", index=False)
    vin_df.to_excel("data.xlsx", index=False)
    vin_df.to_json("data.json")
    vin_df.to_pickle("data.pkl")
    np.save("data.npy", vin_array)
    np.savez("dataset.npz", X=vin_X, y=vin_y)


#------------ 9. QUICK LOADING CHEAT SHEET ------------

# Plain text file       : open("data.txt", "r")
# Numeric TXT           : np.loadtxt("data.txt")
# Numeric CSV           : np.loadtxt("data.csv", delimiter=",")
# CSV with header       : np.genfromtxt("data.csv", delimiter=",", skip_header=1)
# CSV using Pandas      : pd.read_csv("data.csv")
# TSV                   : pd.read_csv("data.tsv", sep="\t")
# Excel                 : pd.read_excel("data.xlsx")
# JSON                  : pd.read_json("data.json")
# NPY                   : np.load("data.npy")
# NPZ                   : np.load("data.npz")
# Pickle                : pd.read_pickle("data.pkl")


#------------ 10. EXAMPLE: TRY THE FUNCTIONS ------------

if __name__ == "__main__":
    print("Data loading reference file")
    print("Import NumPy and Pandas are ready.")
    print("To load a file, call the matching function with its file path.")
    print('Example: vin_df = vin_load_csv("data.csv")')
    print('Example: vin_array = vin_load_npy("data.npy")')
    print('Example: vin_data = vin_load_npz("dataset.npz")')

