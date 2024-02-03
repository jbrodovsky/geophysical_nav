# Check to see if a virtual environment is active
if (-not $env:VIRTUAL_ENV) {
    Write-Host "A virtual environment is not currently active"
    # Prompt the user to create a virtual environment
    $response = Read-Host "Would you like to create a local virtual environment? (y/n)"
    if ($response -eq "y") {
        # Create a virtual environment
        python -m venv .\venv
        # Activate the virtual environment
        .\venv\Scripts\Activate.ps1
        pip install --upgrade pip
    } else {
        Write-Host "Please activate a virtual environment before running this script"
        exit
    }
}

# Clone the pyins repository
Write-Host "Cloning the pyins repository"
git clone https://github.com/nmayorov/pyins.git

# Navigate into the pyins directory
Set-Location -Path .\pyins

# Install pyins with pip
Write-Host "Installing pyins"
pip install .

# Run the tests to ensure everything is working correctly
Write-Host "Running tests"
pytest

# Navigate back to the original directory
Write-Host "Removing the pyins directory"
Set-Location -Path ..
Remove-Item -Recurse -Force .\pyins

Write-Host "pyins has been successfully installed"