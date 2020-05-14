
install:
	conda env create --file environment.yml	--force
	conda activate EvasionPaths-env
	python3 setup.py install

	
