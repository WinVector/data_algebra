
rm -rf dist build data_algebra.egg-info data_algebra/__pycache__
pip uninstall -y data_algebra
python3 setup.py sdist bdist_wheel
pip install dist/data_algebra-*.tar.gz

