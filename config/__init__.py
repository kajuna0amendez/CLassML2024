"""
My Package
A package that provides environment configuration.
"""

from .env import set_username, eliminate_preallocate_memory

# Expose the variables in the public API
__all__ = ["set_username", "eliminate_preallocate_memory"]
