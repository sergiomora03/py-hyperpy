import setuptools

with open("README.md", "r") as fh:
    README = fh.read()

# This call to setup() does all the work
setuptools.setup(
    name="py-hyperpy",
    version="0.0.3",
    description="HyperPy: An automatic hyperparameter optimization framework",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/sergiomora03/py-hyperpy",
    author="Sergio A. Mora Pardo",
    author_email="sergiomora823@gmail.com",
    license="MIT",
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    include_package_data=True,
    python_requires='>=3.7'
)