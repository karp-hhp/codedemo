import os
from openai import OpenAI
import dotenv
from dotenv import load_dotenv

load_dotenv()

class HelloAgentsLLM:
    def __init__(self, model:str |None = None, apiKey:str = None, baseUrl:str = None, timeout:int = None):
        self.model = model or os.getenv("DEEPSEEK_MODEL")
        apikey = apiKey or os.getenv("DEEPSEEK_API_KEY")
        base_url = baseUrl or os.getenv("DEEPSEEK_BASE_URL")

        print(f"self.model:{self.model}, apikey:{apikey}, base_url:{base_url}")
        self.client = OpenAI(api_key=apikey, base_url=base_url)

    def think(self, temperature: float = 0, verbose: bool = False) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that write python code."},
                {"role": "user", "content": "写一个快速排序算法"},
            ],
            temperature=temperature,
            stream=True
        )
        collected_content = []
        for chunk in response:
            if not chunk.choices:
                continue
            
            content = chunk.choices[0].delta.content or ""
            print(content, end="", flush=True)
            
            collected_content.append(content)
            
        
        print()
        return "".join(collected_content)


if __name__ == "__main__":
    helloAgentsLLM = HelloAgentsLLM()
    answer = helloAgentsLLM.think(0.3, verbose=True)
    print("\n\n" + "=" * 20 + " 只取 content " + "=" * 20)
    print(answer)