
vin_X = vin_df.drop("Label", axis=1)
vin_y = vin_df["Label"]

# Positional alternative: last column is target
vin_X = vin_df.iloc[:, :-1]
vin_y = vin_df.iloc[:, -1]

#------------ 4. Encode categorical labels

vin_encoder = LabelEncoder()
vin_y_encoded = vin_encoder.fit_transform(vin_y)

# Example: encode a categorical feature column
vin_gender_encoder = LabelEncoder()
vin_df["Gender_encoded"] = vin_gender_encoder.fit_transform(vin_df["Gender"])

#------------ 5. One-hot encode categorical features

vin_encoded_df = pd.get_dummies(vin_df, columns=["Gender"])

# Or using OneHotEncoder
vin_onehot = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
vin_gender_array = vin_onehot.fit_transform(vin_df[["Gender"]])

#------------ 6. Train-test split

vin_X_train, vin_X_test, vin_y_train, vin_y_test = train_test_split(
    vin_X, vin_y, test_size=0.2, random_state=42
)

# For classification, preserve class proportions
vin_X_train, vin_X_test, vin_y_train, vin_y_test = train_test_split(
    vin_X, vin_y, test_size=0.2, random_state=42, stratify=vin_y
)

#------------ 7. Standardization (mean=0, standard deviation=1)

vin_scaler = StandardScaler()
vin_X_train_scaled = vin_scaler.fit_transform(vin_X_train)
vin_X_test_scaled = vin_scaler.transform(vin_X_test)

#------------ 8. Min-max normalization (0 to 1)

vin_minmax = MinMaxScaler()
vin_X_train_normalized = vin_minmax.fit_transform(vin_X_train)
vin_X_test_normalized = vin_minmax.transform(vin_X_test)

#------------ 9. Convert to NumPy arrays

vin_X_array = vin_X.to_numpy()
vin_y_array = vin_y.to_numpy()

#------------ 10. Convert to PyTorch tensors

import torch

vin_X_tensor = torch.tensor(vin_X_array, dtype=torch.float32)
vin_y_tensor = torch.tensor(vin_y_array, dtype=torch.float32)

# For class labels used by CrossEntropyLoss, use integer class IDs:
vin_y_class = torch.tensor(vin_y_array, dtype=torch.long)

#------------ 11. Shuffle data (NumPy)

vin_indices = np.random.permutation(len(vin_X_array))
vin_X_shuffled = vin_X_array[vin_indices]
vin_y_shuffled = vin_y_array[vin_indices]

#------------ 12. Check class balance

print(pd.Series(vin_y).value_counts())
print(pd.Series(vin_y).value_counts(normalize=True))

#------------ 13. Simple preprocessing pipeline example

# Replace this with the dataset's actual feature and target columns
vin_X = vin_df.drop("Label", axis=1)
vin_y = vin_df["Label"]

vin_X_train, vin_X_test, vin_y_train, vin_y_test = train_test_split(
    vin_X, vin_y, test_size=0.2, random_state=42, stratify=vin_y
)

vin_scaler = StandardScaler()
vin_X_train = vin_scaler.fit_transform(vin_X_train)
vin_X_test = vin_scaler.transform(vin_X_test)

vin_X_train = torch.tensor(vin_X_train, dtype=torch.float32)
vin_X_test = torch.tensor(vin_X_test, dtype=torch.float32)
vin_y_train = torch.tensor(vin_y_train.to_numpy(), dtype=torch.long)
vin_y_test = torch.tensor(vin_y_test.to_numpy(), dtype=torch.long)

print(vin_X_train.shape, vin_X_test.shape)
print(vin_y_train.shape, vin_y_test.shape)

#------------ 14. Important: avoid data leakage
# Split first.
# Fit scaler/encoder on training data only.
# Transform test data using the already-fitted scaler/encoder.

