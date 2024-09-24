import gdown
import pandas as pd

# File ID extracted from the link
file_id = "1UbQLrXi-TSLJw44307AZkWn2kdAuMo83"

# The download URL
url = f"https://drive.google.com/uc?id={file_id}"

# Download the file using gdown
output = 'Diabetes.csv'  # This is the filename it will save as
gdown.download(url, output, quiet=False)

# Read the CSV file into a DataFrame
df = pd.read_csv(output)
print(df.head())


