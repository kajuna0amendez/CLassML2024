from setuptools import setup, find_packages

setup(
    name="JAX ML Class",  # Name of the project
    version="1.0.0",  # Version number
    description="ML Class",  # Short description
    long_description=open("README.md").read(),  # Long description from README
    long_description_content_type="text/markdown",  # Format of the README file
    author="Andres Mendezx",
    author_email="andres.mendez@cinvestav.mx",
    url="",  # Project homepage
    packages=find_packages(),  # Automatically find packages in the directory
    install_requires=[
        "jax==0.4.26.dev20240404+cd7251190",
        "numpy==1.26.4",
        "pydantic==2.7.1",
        "requests==2.32.3",
        ""
        
    ],
    extras_require={
        # Specific dependencies for frontend
        "frontend": [
            "PySide6",  # For GUI development
        ],
        # Specific dependencies for backend
        "backend": [
            "Flask",  # Web framework for backend
            "SQLAlchemy",  # Database ORM
        ],
        # Development and testing tools
        "dev": [
            "pytest",
            "flake8",
        ],
    },
    classifiers=[
        # Metadata about the project
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10.12",  # Specify compatible Python versions
    entry_points={
        # Entry points for CLI or scripts
        "console_scripts": [
        ],
    },
)