from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="videoslides",
    version='0.0.2',
    description='Package made to obtain a text transcription from a video, with a flow from video to frames to structured frames to transcription in a json file ',
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=["videoslides", "functions"],
    package_dir={'':'src'},
    install_requires=[
        "easyocr ~= 1.4.1",
        "opencv-python-headless ~= 4.1.2.30",
        "stanza ~= 1.4.0",
        "numpy ~= 1.19.5",
        "matplotlib ~= 3.3.4",
        "pytube ~= 12.0.0",
        "scikit-image ~= 0.17.2",
        "validators ~= 0.20.0",
        "nltk ~= 3.6.7"  
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent", 
        "Intended Audience :: Developers"
    ],
    url="https://github.com/Zes7one/VideoSlides",
    author="Franco Palma",
    author_email="franco.pm10@gmail.com "

) 
        # "easyocr ~= 1.4.1",
        # "opencv-python-headless ~= 4.1.2.30",
        # "stanza ~= 1.4.0",
        # "numpy ~= 1.19.5",
        # "matplotlib ~= 3.3.4",
        # "pytube ~= 12.0.0",
        # "scikit-image ~= 0.17.2",
        # "validators ~= 0.20.0",
        # "nltk ~= 3.6.7"  

        # "pathlib ~= ",
        # "json ~= ",
        # "re ~= ",

# Comando para correr el setup -> crear paquete
# py setup.py bdist_wheel

# Comando para instalar localmente el paquete ( se corre cada vez que se cambia el setup.py)
# pip install -e .
# -e : evita que se copie el codigo y se linkea a la carpeta src solamente

# pip install check-manifest
# check-manifest --create
# git add MANIFEST.in
# py setup.py sdist

# py setup.py bdist_wheel sdist
# cd dist
# dir 

# Subir a Pypi
# pip install twine
# twine upload dist/*