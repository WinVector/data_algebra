
pip uninstall -y data_algebra

rm -f coverage.txt
rm -rf dist build data_algebra.egg-info docs

find . -name .DS_Store -exec rm {} \;
find . -name '*~' -exec rm {} \;
find . -name __pycache__ -exec rm -rf {} \; 1>/dev/null 2>/dev/null
find . -name .ipynb_checkpoints -exec rm -rf {} \; 1>/dev/null 2>/dev/null

