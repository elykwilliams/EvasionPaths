SHELL:=/bin/bash
.ONESHELL:

# make install will recreate the environment and install source files
install:
	conda env create --file environment.yml	--force
	source ~/.bashrc
	conda activate EvasionPaths-env
	python3 setup.py install 

# will pull from github and install source files
update:
	git pull
	source ~/.bashrc
	conda activate EvasionPaths-env
	python3 setup.py install

