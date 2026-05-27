#!/bin/bash

sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=wavelet_seafloor
#SBATCH -N 1
#sbatch -n 2
#SBATCH --mem=16gb
#SBATCH -t 04:00:00
#SBATCH -A open
#SBATCH -p standard
#SBATCH -o logs/wavelet_%j.log
#SBATCH -e logs/wavelet_%j.err

set -e

# --- Configuration ---
REPO_DIR="/storage/home/vzt5134/work/wavelet_analysis/ocean_floor_wavelet_analysis"
JSON_DIR="/storage/home/vzt5134/work/wavelet_analysis/seafloor_json_mapping"
RESULTS_DIR="/storage/home/vzt5134/work/wavelet_analysis/wavelet_analysis_results"
PYTHON_SCRIPT="automatic_global_data_gaussian_mixture.py"

# Force Python to write outputs instantly to the log file (no buffering)
export PYTHONUNBUFFERED=1

cd "\$REPO_DIR"

# Ensure directories exist
mkdir -p logs
mkdir -p "\$RESULTS_DIR"

# ==============================================================================
# PHASE 1: ENVIRONMENT SETUP (Only runs if venv is missing)
# ==============================================================================
echo "=== [Background] Step 1: Loading Python & Setting up Venv ==="
module load python

if [ ! -d "venv" ]; then
    echo "Creating virtual environment and installing packages..."
    python3 -m venv venv
    source venv/bin/activate
    
    echo "Installing/Updating packages from requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "Virtual environment already exists! Activating without reinstalling packages."
    source venv/bin/activate
fi

# ==============================================================================
# PHASE 2: GENERATE MAPPING (Runs every time to catch new coordinates)
# ==============================================================================
echo "=== [Background] Step 2: Generating Seafloor JSON Mapping ==="
python3 -u create_seafloor_json_mapping.py 

# ==============================================================================
# PHASE 3: ANALYSIS LOOP WITH RESTART LOGIC
# ==============================================================================
if [ ! -d "\$JSON_DIR" ]; then
    echo "Error: Directory \$JSON_DIR does not exist." >&2
    exit 1
fi

# FIXED: Added backslash to escape the variable context
cd "\$JSON_DIR"

echo "=== [Background] Step 3: Starting GMM analysis loop at \$(date) ==="
for file in *.json; do
    [ -e "\$file" ] || continue
    clean_name="\${file%.json}"    
    
    # Check if the results directory for this specific tile already exists
    if [ -d "\$RESULTS_DIR/\$clean_name" ]; then
        echo "Tile \$clean_name already processed. Skipping..."
        continue
    fi
    
    echo "Processing tile: \$clean_name"
    python3 -u "\$REPO_DIR/\$PYTHON_SCRIPT" "\$clean_name"
done

echo "All tasks processed successfully at \$(date)."
EOT

echo "Entire workflow has been submitted to the queue!"


# #!/bin/bash

# sbatch <<EOT
# #!/bin/bash
# #SBATCH --job-name=wavelet_seafloor
# #SBATCH -N 1
# #SBATCH --mem=16gb
# #SBATCH -t 02:00:00
# #SBATCH -A open
# #SBATCH -p standard
# #SBATCH -o logs/wavelet_%j.log
# #SBATCH -e logs/wavelet_%j.err

# set -e

# # --- Configuration ---
# REPO_DIR="/storage/home/vzt5134/work/wavelet_analysis/ocean_floor_wavelet_analysis"
# JSON_DIR="/storage/home/vzt5134/work/wavelet_analysis/seafloor_json_mapping"
# PYTHON_SCRIPT="automatic_global_data_gaussian_mixture.py"

# # Force Python to write outputs instantly to the log file (no buffering)
# export PYTHONUNBUFFERED=1

# cd "\$REPO_DIR"

# # Ensure the logs directory exists so Slurm doesn't fail writing outputs
# mkdir -p logs

# # ==============================================================================
# # PHASE 1: ENVIRONMENT SETUP (Only runs if venv is missing)
# # ==============================================================================
# echo "=== [Background] Step 1: Loading Python & Setting up Venv ==="
# module load python

# if [ ! -d "venv" ]; then
#     echo "Creating virtual environment and installing packages..."
#     python3 -m venv venv
#     source venv/bin/activate
    
#     echo "Installing/Updating packages from requirements.txt..."
#     pip install --upgrade pip
#     pip install -r requirements.txt
# else
#     echo "Virtual environment already exists! Activating without reinstalling packages."
#     source venv/bin/activate
# fi

# # ==============================================================================
# # PHASE 2: GENERATE MAPPING (Runs every time to catch new coordinates)
# # ==============================================================================
# echo "=== [Background] Step 2: Generating Seafloor JSON Mapping ==="
# python3 -u create_seafloor_json_mapping.py 

# # ==============================================================================
# # PHASE 3: ANALYSIS LOOP
# # ==============================================================================
# if [ ! -d "\$JSON_DIR" ]; then
#     echo "Error: Directory \$JSON_DIR does not exist." >&2
#     exit 1
# fi

# cd "\$JSON_DIR"

# echo "=== [Background] Step 3: Starting GMM analysis loop at \$(date) ==="
# for file in *.json; do
#     [ -e "\$file" ] || continue
#     clean_name="\${file%.json}"    
#     echo "Processing tile: \$clean_name"
    
#     python3 -u "\$REPO_DIR/\$PYTHON_SCRIPT" "\$clean_name"
# done

# echo "All tasks processed successfully at \$(date)."
# EOT

# echo "Entire workflow has been submitted to the queue!"




