# Contributing to retroml

We welcome contributions to the `retroml` project! Your help is invaluable in making this project better. Here are some guidelines to help you get started.

## Table of Contents

* [How to Contribute](#how-to-contribute)
* [Reporting Bugs](#reporting-bugs)
* [Suggesting Enhancements](#suggesting-enhancements)
* [Your First Code Contribution](#your-first-code-contribution)
* [Setting Up Your Development Environment](#setting-up-your-development-environment)
* [Code Style and Guidelines](#code-style-and-guidelines)
* [Submitting Pull Requests](#submitting-pull-requests)
* [Code of Conduct](#code-of-conduct)

## How to Contribute

There are many ways to contribute to `retroml`, not just by writing code!

* **Report bugs:** If you find a bug, please let us know.
* **Suggest enhancements:** Have an idea for a new feature or an improvement? Share it!
* **Write code:** Fix bugs, implement new features, improve documentation.
* **Improve documentation:** Clearer explanations, more examples, tutorials are always welcome.
* **Review pull requests:** Help us maintain code quality by reviewing contributions from others.

## Reporting Bugs

If you encounter a bug, please open an issue on our [GitHub Issues page](https://github.com/Agile-Creative-Labs/retroml/issues).

When reporting a bug, please include:
* A clear and concise description of the bug.
* Steps to reproduce the behavior.
* Expected behavior.
* Actual behavior.
* Screenshots (if applicable).
* Your operating system and Python/`retroml` version.

## Suggesting Enhancements

Have a great idea for `retroml`? We'd love to hear it! Please open an issue on our [GitHub Issues page](https://github.com/Agile-Creative-Labs/retroml/issues) and label it as an "enhancement" or "feature request".

When suggesting an enhancement, please describe:
* The problem you're trying to solve.
* Your proposed solution.
* Any alternative solutions you've considered.

## Your First Code Contribution

If you're looking to make your first contribution, look for issues labeled `good first issue` or `help wanted` on our [issues page](https://github.com/Agile-Creative-Labs/retroml/issues).

## Setting Up Your Development Environment

1.  **Fork the Repository:** Click the "Fork" button at the top right of the `retroml` [GitHub repository](https://github.com/Agile-Creative-Labs/retroml).
2.  **Clone Your Fork:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/retroml.git](https://github.com/YOUR_USERNAME/retroml.git)
    cd retroml
    ```
    (Replace `YOUR_USERNAME` with your GitHub username)
3.  **Create a Virtual Environment:**
    It's recommended to work in a virtual environment to avoid conflicts with your system's Python packages.
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```
4.  **Install Dependencies:**
    ```bash
    pip install -e . # Installs retroml in editable mode
    pip install -r requirements-dev.txt # If you have a dev requirements file
    ```
    *(Note: If you don't have a `requirements-dev.txt`, you might need to list common development tools here like `pytest`, `flake8`, `black`, etc.)*

## Code Style and Guidelines

* **Follow PEP 8:** We generally adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code style. Tools like `flake8` or `black` can help.
* **Docstrings:** Please add clear docstrings to new functions, classes, and modules following common Python conventions (e.g., NumPy style or Google style).
* **Tests:** New features should come with corresponding unit tests. Bug fixes should include a test that reproduces the bug before the fix and passes after the fix.
* **Comments:** Use comments to explain complex logic, but prefer self-documenting code.

## Submitting Pull Requests

1.  **Create a New Branch:**
    ```bash
    git checkout -b feature/your-awesome-feature-name
    ```
    (Use `bugfix/` for bug fixes, `feat/` for new features, `docs/` for documentation improvements.)
2.  **Make Your Changes:** Write your code, add tests, update documentation.
3.  **Run Tests:** Ensure all existing tests pass and your new tests pass.
    ```bash
    pytest # or your chosen testing command
    ```
4.  **Lint Your Code:** (If applicable)
    ```bash
    flake8 . # or black .
    ```
5.  **Commit Your Changes:** Write clear and concise commit messages.
    ```bash
    git commit -m "feat: Add awesome new feature"
    ```
6.  **Push Your Branch:**
    ```bash
    git push origin feature/your-awesome-feature-name
    ```
7.  **Open a Pull Request (PR):**
    Go to your fork on GitHub and you'll see a prompt to open a new pull request.
    * Clearly describe the changes in your PR.
    * Reference any related issues (e.g., `Closes #123`).
    * Provide screenshots or GIFs if the changes are visual.
    * Be prepared for feedback and discussion during the review process.

## Code of Conduct

Please note that `retroml` has a [Code of Conduct](CODE_OF_CONDUCT.md) that all contributors are expected to follow.
