import os
import prompts.bird_game
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

country_preference = {
    "Costa Rica" : "Toucans",
    "Mexico" : "Hummingbirds",
    "Colombia" : "Trogons"
}

model = init_chat_model("anthropic:claude-sonnet-4-5")

#define tool
@tool
def get_bird_preference(country: str) -> str:
    """Get bird preference by country"""
    preference = 'Undefined'
    if country in country_preference:
        preference = country_preference[country]
    return preference

# get prompt
system_prompt = prompts.bird_game.data
agent = create_agent(
    model=model,
    tools=[get_bird_preference],
    system_prompt=system_prompt,
)

# Run the agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Create 5 trivia questions regarding birds of Costa Rica"}]}
)

print(response)
print('_______________')
print(response.keys())