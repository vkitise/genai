import os
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage
os.environ["COHERE_API_KEY"] = "N8FmJLuZKK6lkB4pkCG6SingpSBGEDWSfbI2P362"
text = open("GenAI8.txt", encoding="utf-8").read()
prompt = f"""
Summary:
Key Points:
Conclusion:
{text}
"""
llm = ChatCohere(
    model="command-r-08-2024",
    cohere_api_key=os.environ["COHERE_API_KEY"]
)
response = llm.invoke([HumanMessage(content=prompt)])
print(response.content)
