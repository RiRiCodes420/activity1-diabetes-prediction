from setuptools import find_packages, setup
from typing import List

# Constant for the '-e .' entry
HYPHEN_E_DOT = "-e ."

# Function to get and clean the list of requirements from a file
def get_requirements(file_path: str) -> List[str]:
    # Open the file and read the lines
    with open(file_path, 'r') as file_obj:
        # Read all lines and strip newline characters
        requirements = [req.strip() for req in file_obj.readlines()]
    
    # Remove the '-e .' entry if it exists
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    
    return requirements  # Return the cleaned list of requirements

# Setup configuration for the package
setup(
    name="activity1-diabetes-prediction",
    version="0.0.1",
    author="Rianna",
    author_email="rianna20furtado02@gmail.com",
    packages=find_packages(),  # Automatically find all packages
    install_requires=get_requirements('requirements.txt'),  # Call the function to get requirements
)