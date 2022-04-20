from setuptools import setup


def readme():
    with open("README.md") as f:
        return f.read()


MAJOR, MINOR, MICRO = 0, 1, 0
__VERSION__ = "{}.{}.{}".format(MAJOR, MINOR, MICRO)

setup(
    name="corrupted-text",
    version=__VERSION__,
    description=(
        "Corruption of text datasets; model-independent and inspired by"
        "real-world corruption causes."
    ),
    long_description_content_type="text/markdown",
    long_description=readme(),
    # keywords="",
    url="https://github.com/testingautomated-usi/corrupted-text",
    author="Michael Weiss",
    author_email="michael.weiss@usi.ch",
    license="MIT",
    packages=["corrupted_text"],
    install_requires=[
        "polyleven==0.7",  # Fast levenshtein distance implementation
        "numpy>=1.16.4",
    ],
    extras_require={
        "lint": [
            "flake8==3.8.2",
            "black==22.3.0",
            "isort==5.6.4",
            "docstr-coverage==2.2.0",
        ],
        "test": ["pytest>=6.2.5", "datasets>=1.0.0"],
        "acc_measurement": ["datasets>=1.0.0", "tensorflow>=2.6.0"],
    },
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
