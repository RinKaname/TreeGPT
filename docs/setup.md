# setup.py Documentation

The `setup.py` script is the standard setup file for the `TreeGPT` project. It uses `setuptools` to manage the packaging and distribution of the project, including its dependencies and metadata.

## Key Components

-   **Metadata**: The script defines essential metadata for the project, such as its name (`treegpt`), version, author, and a brief description. It also reads the `README.md` file to provide a long description for package managers like PyPI.
-   **Dependencies**: The `install_requires` argument reads the `requirements.txt` file to specify the project's core dependencies. This ensures that all necessary packages are installed when the project is set up.
-   **Extra Dependencies**: The `extras_require` argument defines optional sets of dependencies for different purposes:
    -   `dev`: Includes tools for development, such as `pytest` for testing and `black` for code formatting.
    -   `viz`: Includes packages for visualization, such as `torch-geometric` and `tensorboard`.
-   **Entry Points**: The `entry_points` argument defines console scripts that are created when the package is installed. In this case, it creates a `treegpt-train` command, which is intended to be a shortcut for running the training script.
-   **Classifiers**: The `classifiers` provide metadata to PyPI to help users find the project. They specify the development status, intended audience, license, and supported Python versions.

## Installation

To install the project and its dependencies, you can run the following command from the root of the repository:

```bash
pip install .
```

To install the development dependencies as well, you can use:

```bash
pip install .[dev]
```