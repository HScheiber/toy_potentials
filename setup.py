from setuptools import setup, find_packages

install_requires = ["numpy", "scipy", "deeptime", "tqdm", "matplotlib", "joblib"]


CLASSIFIERS = """\
Development Status :: 3 - Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
Operating System :: OS Independent
Programming Language :: Python :: 3
Topic :: Scientific/Engineering
"""

setup(
    name="toy_potentials",
    version="0.0.1",
    python_requires=">=3.5.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=install_requires,
    description="Toy Potentials: A package for generating toy potentials for testing MD simulation and ensemble modeling strategies.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    author="Hayden Scheiber",
    author_email="hscheibe@amgen.com",
    classifiers=CLASSIFIERS.splitlines(),
)
