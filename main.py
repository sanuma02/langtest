import os
import requests
from langchain.tools import tool
from markdownify import markdownify
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.chat_models import init_chat_model

country_preference = {
    "Costa Rica" : "Toucans",
    "Mexico" : "Hummingbirds",
    "Colombia" : "Trogons"
}

ALLOWED_DOMAINS = ["https://langchain-ai.github.io/"]
LLMS_TXT = 'https://langchain-ai.github.io/langgraph/llms.txt'


model = init_chat_model("anthropic:claude-sonnet-4-5")
llms_txt_content = requests.get(LLMS_TXT).text

system_prompt = f"""
You are an expert Python developer and technical assistant.
Your primary role is to help users with questions about LangGraph and related tools.

Instructions:

1. If a user asks a question you're unsure about — or one that likely involves API usage,
   behavior, or configuration — you MUST use the `fetch_documentation` tool to consult the relevant docs.
2. When citing documentation, summarize clearly and include relevant context from the content.
3. Do not use any URLs outside of the allowed domain.
4. If a documentation fetch fails, tell the user and proceed with your best expert understanding.

You can access official documentation from the following approved sources:

{llms_txt_content}

You MUST consult the documentation to get up to date documentation
before answering a user's question about LangGraph.

Your answers should be clear, concise, and technically accurate.
"""

#define tools
@tool
def get_bird_preference(country: str) -> str:
    """Get bird preference by country"""
    preference = 'Undefined'
    if country in country_preference:
        preference = country_preference[country]
    return preference

@tool
def fetch_documentation(url: str) -> str:  
    """Fetch and convert documentation from a URL"""
    if not any(url.startswith(domain) for domain in ALLOWED_DOMAINS):
        return (
            "Error: URL not allowed. "
            f"Must start with one of: {', '.join(ALLOWED_DOMAINS)}"
        )
    response = requests.get(url, timeout=10.0)
    response.raise_for_status()
    return markdownify(response.text)



agent = create_agent(
    model=model,
    tools=[fetch_documentation],
    system_prompt=system_prompt,
    name="Agentic RAG",
)

# Run the agent
response = agent.invoke({
    'messages': [
        HumanMessage(content=(
            "Write a short example of a langgraph agent using the "
            "prebuilt create react agent. the agent should be able "
            "to look up stock pricing information."
        ))
    ]
})

print(response)
print('_______________')
print(response.keys())