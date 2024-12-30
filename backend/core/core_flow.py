"""
Basic Pipeline for MLFlow
"""
from metaflow import FlowSpec, step
from config import set_username

set_username()

class MLFlow(FlowSpec):
    """
    Basic Pipeline
    """
    @step
    def start(self):
        """
        Start the flow
        """
        self.next(self.end)

    @step
    def end(self):
        """
        End the flow
        """
        print("Flow is done!")

if __name__ == "__main__":
    MLFlow()
