import gdown
import pandas as pd

# File ID extracted from the link
file_id = "1UbQLrXi-TSLJw44307AZkWn2kdAuMo83"

# The download URL
url = f"https://drive.google.com/uc?id={file_id}"

# Download the file using gdown
output = 'Diabetes1.csv'  
gdown.download(url, output, quiet=False)

# Read the CSV file into a DataFrame
data = pd.read_csv(output)
print(data.head())


