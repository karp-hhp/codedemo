from serpapi import SerpApiClient
from dotenv import load_dotenv
import os

from ..tools.tool_executor import ToolExecutor

load_dotenv()
def search(query: str) -> str:
    """
    一个基于SerpApi的实战网页搜索引擎工具。它会智能地解析搜索结果，
    优先返回直接答案或知识图谱信息。
    """
    try:
        api_key = os.getenv("SERP_API_KEY")
        if not api_key:
            return "error: serp api key not config in env"
        
        params = {
            "engine":"google",
            "q":query,
            "api_key":api_key,
            "gl":"cn",
            "hl":"zh-cn",
        }
        
        client = SerpApiClient(params)
        results = client.get_dict()
        
        print(f"SerpApiClient results: {results}")
        
        # 智能解析：优先寻找最直接的答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        
        return  f"sorry, not find info about '{query}'"
    except Exception as e:
        print(f"error serpapi e:{e}")
        return f"error serpapi e:{e}"
    
    
REACT_AGENT_PROMPT = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下：
{tools}

请严格按照以下格式进行回应：
Thought：你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action： 你决定采取的行动，必须是以下格式之一：
- '{{tool_name}}[{{tool_input}}]':调用一个可用工具。
- 'Finish[最终答案]': 当你认为已经获取到最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。

现在，请开始解决以下问题：
Question: {question}
History: {history}
"""

    

if __name__ == "__main__":
    # infos = search("华为的最新款手机介绍")
    # print(f"infos: ${infos}")
    print("begin run tool executor")
    tool_executor = ToolExecutor()
    
    search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.register_tool("search", search_desc, search)
    
    print("\n--- 可用的工具 ---")
    print(tool_executor.get_available_tools())
    
    tool_name = "search"
    tool_input = "英伟达最新的GPU型号是什么"
    
    tool_func = tool_executor.get_tool(tool_name=tool_name)
    if tool_func:
        observation = tool_func(tool_input)
        print(f"observation result: {observation}")
    else:
        print(f"错误:未找到名为 '{tool_name}' 的工具。")
