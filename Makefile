SHELL := /bin/bash
.ONESHELL:

VENV := venv
PYTHON := ${VENV}/bin/python3
PIP := ${VENV}/bin/pip3


.PHONY: pre-req
pre-req:        ## Install/update python, pip, and virtualenv
	sudo apt install python3 python3-pip python3-venv


venv:    pre-req ## Create virtual environment
	python3 -m venv ${VENV}


install: env    ## Install dependencies and package
	${PIP} install -e .


.PHONY: update
update:         ## Get recent updates and reinstall
	git stash
	git pull --rebase
	git stash pop
	make install

.PHONY: help
help: ## Show this help
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'