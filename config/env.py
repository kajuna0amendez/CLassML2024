"""
This module is used to set the environment variables
"""
import os

def set_username(username: str="Andres Mendez")-> None:
    """
    Set the username
    """
    os.environ["USERNAME"] = username


def eliminate_preallocate_memory()-> None:
    """
    Eliminate the preallocate memory
    """
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
