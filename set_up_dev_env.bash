

conda remove --name data_algebra_dev_env --all --yes
conda env create -f data_algebra_dev_env.yaml
conda activate data_algebra_dev_env
# sym link to source files
pip install --no-deps -e "$(pwd)"  


