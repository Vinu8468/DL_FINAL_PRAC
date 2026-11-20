            print(vin_key, vin_data[vin_key].shape)

    return vin_clean_data


# Example:
# vin_mat_data = vin_read_mat("dataset.mat")
# vin_X = vin_mat_data["X"]       # Use the actual variable name
# print(vin_X)


# Save arrays to a MAT file:
# from scipy.io import savemat
# import numpy as np
# vin_X = np.array([[1, 2], [3, 4]])
# savemat("output.mat", {"X": vin_X})

# Note: MATLAB v7.3 MAT files are HDF5-based. Try reading them with h5py.


#------------
# 5. DICOM FILES (.dcm)
# DICOM is commonly used for medical images such as CT and MRI.
# Install if needed: pip install pydicom matplotlib


def vin_read_dicom(vin_path):
    try:
        import pydicom
    except ImportError as vin_error:
        raise ImportError("Install pydicom using: pip install pydicom") from vin_error

    vin_dicom = pydicom.dcmread(vin_path)

    print("Patient ID:", getattr(vin_dicom, "PatientID", "Not available"))
    print("Modality:", getattr(vin_dicom, "Modality", "Not available"))
    print("Study Date:", getattr(vin_dicom, "StudyDate", "Not available"))
    print("Image dimensions:", getattr(vin_dicom, "Rows", "?"),
          "x", getattr(vin_dicom, "Columns", "?"))

    vin_image = vin_dicom.pixel_array

    # Apply rescale parameters when present (commonly needed for CT values).
    vin_slope = float(getattr(vin_dicom, "RescaleSlope", 1))
    vin_intercept = float(getattr(vin_dicom, "RescaleIntercept", 0))
    vin_image = vin_image.astype("float32") * vin_slope + vin_intercept

    return vin_dicom, vin_image


# Example:
# vin_dicom, vin_image = vin_read_dicom("scan.dcm")
# print("Pixel array shape:", vin_image.shape)
# print("Minimum:", vin_image.min())
# print("Maximum:", vin_image.max())


# Display a DICOM image:
# import matplotlib.pyplot as plt
# plt.imshow(vin_image, cmap="gray")
# plt.title("DICOM Image")
# plt.axis("off")
# plt.show()


#------------
# 6. CONVERT BIOLOGICAL SEQUENCES TO NUMERICAL DATA
# Example encoding for DNA. This is a simple character encoding, not a learned embedding.


def vin_encode_dna(vin_sequence):
    vin_mapping = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    vin_encoded = []

    for vin_base in vin_sequence.upper():
        if vin_base not in vin_mapping:
            raise ValueError("Unknown DNA character: " + vin_base)
        vin_encoded.append(vin_mapping[vin_base])

    return vin_encoded


# Example:
# vin_sequence = "ACGTN"
# vin_encoded = vin_encode_dna(vin_sequence)
# print(vin_encoded)


#------------
# 7. BASIC FILE AND PACKAGE CHECKS

# Check whether a file exists:
# vin_path = "dataset.fasta"
# print(os.path.exists(vin_path))

# Get file extension:
# print(os.path.splitext(vin_path)[1])

# Install packages when required:
# pip install numpy pandas h5py scipy pydicom matplotlib


#------------
# 8. QUICK FORMAT REFERENCE

# FASTA  -> vin_read_fasta("file.fasta")
# FASTQ  -> vin_read_fastq("file.fastq")
# HDF5   -> vin_read_hdf5("file.h5")
# MAT    -> vin_read_mat("file.mat")
# DICOM  -> vin_read_dicom("file.dcm")

# The examples are commented so this reference file can be opened or run
# without requiring all optional packages or example data files.

