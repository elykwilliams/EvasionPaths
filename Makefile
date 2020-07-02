SHELL:=/bin/bash
.ONESHELL:

install:
	conda env create --file environment.yml	--force
	source ~/.bashrc
	conda activate EvasionPaths-env
	python3 setup.py install 

update:
	git pull
	source ~/.bashrc
	conda activate EvasionPaths-env
	python3 setup.py install

