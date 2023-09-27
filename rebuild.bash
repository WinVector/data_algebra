
bash ./clean.bash
pip install --no-deps -e "$(pwd)"
conda list --export > data_algebra_dev_env_package_list.txt
pytest --cov-report term-missing --cov data_algebra tests > coverage.txt
pdoc3 --force -o ./docs ./data_algebra
cat coverage.txt
python3 setup.py sdist bdist_wheel
twine check dist/*

