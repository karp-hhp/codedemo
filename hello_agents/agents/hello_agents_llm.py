import os

class HelloAgentsLLM:
    def __init__(self, model:str |None = None, apiKey:str = None, baseUrl:str = None, timeout:int = None):
        self.model = model or os.getenv("")