from setuptools import setup, find_packages

setup(
    name="calicropyield",
    version="0.1.0",
    description="A multi-modal data downloader and processing library for California crop yield benchmarking",
    author="Hamid Kamangir",
    author_email="hamid.kamangir@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "rasterio",
        "geopandas",
        "shapely",
        "gdown",
        "numpy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)