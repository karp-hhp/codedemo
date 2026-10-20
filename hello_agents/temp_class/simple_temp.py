AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用工具一步步地解决问题。

# 可用工具：
- `get_weather(city: str)`:查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

#输出格式要求：
你每次的回复必须严格遵循以下格式，包含一对Thought和Action：

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须是以下之一：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务: Finish[最终答案]

# 重要提示：
- 每次只输出一对Thought-Action
- Action必须在同一行，不要换行
- 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束

请开始吧！
"""

import requests

def get_weather(city: str) -> str:
    """通过调用wttr.in API查询真实的天气信息

    Args:
        city (str): _description_

    Returns:
        str: _description_
    """
    url = f"https://wttr.in/{city}?format=j1"
    
    try:
        response = requests.get(url=url)
    except requests.exceptions.RequestException as e:
        return f"error: check weather error - {e}"
    except (KeyError, IndexError) as e:
        return f"error: parse weather data error, prosobile city is disable - {e}"
    
import os
from tavily import TavilyClient
from dotenv import load_dotenv

# 读取当前目录的.env,写入os.environ
load_dotenv()

def get_attraction(city: str, weather: str) -> str:
    """根据城市和天气，使用Tavily Search Api搜索并返回优化后的景点推荐

    Args:
        city (str): _description_
        weather (str): _description_

    Returns:
        str: _description_
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    
    if not api_key:
        return "error: not config TAVILY_API_KEY in environment"
    
    tavily = TavilyClient(api_key=api_key)
    
    query = f"'{city}' 在 '{weather}' 天气下最值得去的旅游景点推荐及理由"
    
    try:
        response = tavily.search(query=query, search_depth="basic", include_answer=True)
        
        if response.get("answer"):
            return response["answer"]
        
        # 如果没有综合性的回答，则格式化原始结果
        formatted_result = []
        for result in response.get("result", []):
            formatted_result.append(f"- {result['title']}: {result['content']}")
            
        if not formatted_result:
            return "sorry, not find "
        
        return "根据搜索，为您找到以下信息:\n" + "\n".join(formatted_result)
        
    except Exception as e:
        return f"error: invoke tavily error - {e}"
    


