import os

class ToolExecutor:
    def __init__(self):
        self.tools: dict[str, dict[str, any]] = {}

    def register_tool(self, tool_name:str, description:str, func:callable):
        if tool_name in self.tools:
            print(f"tool has exist, current tool will be override")
        self.tools[tool_name] = {"description":description, "func":func}
        print(f"tool {tool_name} has registered")
        

    def get_tool(self, tool_name:str) -> callable:
        return self.tools.get(tool_name, {}).get("func")

    def get_available_tools(self) -> str:
        # join 用 "\n" 做分隔符，把一个字符串列表粘成一个字符串
        return "\n".join([f"- {info['description']}" for name, info in self.tools.items()])