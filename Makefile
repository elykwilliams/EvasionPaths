SHELL := /bin/bash
.ONESHELL:

VENV := .env
PYTHON := ${VENV}/bin/python3
PIP := ${VENV}/bin/pip3


.PHONY: pre-req
pre-req:        ## Install/update python, pip, and virtualenv
	sudo apt install python3 python3-pip python3-venv


env:    pre-req ## Create virtual environment
	python3 -m venv ${VENV}


install: env    ## Install dependencies and package
	${PIP} install -r requirements.txt
	${PIP} install -e .


.PHONY: update
update:         ## Get recent updates and reinstall
	git pull --rebase origin master:master
	make install


conda-env:	environment.yml      ## Create conda environment with dependencies
	conda env create --file environment.yml --force


conda-install:  conda-env       ## Install package
	conda activate EvasionPaths-env
	pip install -e .


.PHONY: conda-update
conda-update:   ## Get recent updates and reinstall
	git pull --rebase origin master:master
	conda activate EvasionPaths-env
	make conda-install

.PHONY: help
help: ## Show this help
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'