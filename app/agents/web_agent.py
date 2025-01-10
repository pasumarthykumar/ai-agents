import os
from langchain.agents import initialize_agent, AgentType
from duckduckgo_search import DDGS
from langchain.tools import tool
from langchain.chat_models import ChatOpenAI
from twilio.rest import Client
from langchain.prompts import PromptTemplate
from decouple import config

OPENAI_API_KEY = config("OPENAI_API_KEY") 

TWILIO_ACCOUNT_SID = config("TWILIO_ACCOUNT_SID") 
TWILIO_AUTH_TOKEN = config("TWILIO_AUTH_TOKEN") 
TWILIO_WHATSAPP_NUMBER = config("TWILIO_WHATSAPP_NUMBER") 


@tool
def duckduckgo_search(query: str) -> str:
    """
    Use DuckDuckGo to perform a web search and return summarized information.
    """
    ddgs = DDGS()
    results = ddgs.text(query, max_results=5)  # Perform DuckDuckGo search
    if not results:
        return "No relevant information found."

    # Extract meaningful information from search results
    summary = []
    for res in results:
        summary.append(f"{res['title']}: {res['body']} (Source: {res['href']})")

    # Return a summarized result
    return "\n".join(summary)


    
# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)



web_agent = initialize_agent(
    tools=[duckduckgo_search],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

