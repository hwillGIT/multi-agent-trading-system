<general_rules>
When contributing to this repository, please adhere to the following rules:

1.  **Code Quality and Formatting**: Use the provided `Makefile` commands to maintain code quality. Run `make format` to apply `black` and `isort` for consistent code formatting. Run `make lint` to check for linting errors and type inconsistencies using `flake8` and `mypy`.

2.  **Code Reusability**: Before implementing new functionality, always search within the `agents/` and `core/` directories to determine if similar functionality already exists. The `agents/` directory contains individual, specialized agent implementations, while the `core/` directory provides the base framework, shared utilities, and exception classes. Reusing and extending existing components is highly encouraged.
</general_rules>

