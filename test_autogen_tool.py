import asyncio
import os
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


def load_env(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


async def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"{city} 今天 23 摄氏度，晴。"


async def main() -> None:
    load_env()

    model_client = OpenAIChatCompletionClient(
        model=os.environ.get("OPENAI_MODEL", "gpt-5.4"),
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
            "structured_output": True,
        },
        temperature=0,
    )

    agent = AssistantAgent(
        name="weather_agent",
        model_client=model_client,
        tools=[get_weather],
        system_message="你是一个简洁的中文助手。需要天气信息时必须调用工具。",
        reflect_on_tool_use=True,
        model_client_stream=True,
    )

    await Console(agent.run_stream(task="北京今天天气怎么样？请先查天气，再用一句话回答。"))
    await model_client.close()


asyncio.run(main())
