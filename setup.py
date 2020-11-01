from setuptools import find_packages, setup
setup(
    name="S3EN",
    packages=find_packages(include=["S3EN.estimator"]),
    version="0.1.0",
    description="My first Python library",
    author="Charles Ayotte-Trépanier",
    license="MIT",
    install_requires=["tensorflow==2.3.1",
                      "scikit-learn==0.23.2",
                      "pandas==1.1.3"],
    setup_requires=["pytest-runner"],
    tests_require=["pytest==6.1.2"],
    test_suite="tests",
    url="https://github.com/Selraghc/S3EN"
)