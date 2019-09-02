
rm -rf dist build data_algebra.egg-info data_algebra/__pycache__ tests/__pycache__
pip uninstall -y data_algebra
pytest
pytest --cov data_algebra tests > coverage.txt
cat coverage.txt
python3 setup.py sdist bdist_wheel
pip install dist/data_algebra-*.tar.gz
twine check dist/*
