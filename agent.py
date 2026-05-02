# ============================================================
# PROJECT: AI Agent with Web Search + Memory
# Day 14 — LangChain Agents, ReAct, DuckDuckGo, Memory
# Author: Varshini Marathi
# ============================================================

# pip install langchain langchain-openai duckduckgo-search langchain-community

import os
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY_HERE"

from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory
from langchain import hub

# ── TOOLS ────────────────────────────────────────────────────
search = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    """Search the internet for real-time information about any topic."""
    return search.run(query)

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Input: math expression as string."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def word_counter(text: str) -> str:
    """Count the number of words in a given text."""
    count = len(text.split())
    return f"Word count: {count}"

tools = [web_search, calculator, word_counter]

# ── AGENT SETUP ──────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
prompt = hub.pull("hwchase17/react")

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)

# ── INTERACTIVE LOOP ─────────────────────────────────────────
print("=" * 55)
print("🤖 AI Agent with Web Search + Memory")
print("   Ask anything — I can search the web!")
print("   Type 'quit' to exit.")
print("=" * 55)

chat_history = []

while True:
    user_input = input("\nYou: ").strip()
    if not user_input:
        continue
    if user_input.lower() in ["quit", "exit"]:
        print("Goodbye! 👋")
        break

    result = agent_executor.invoke({
        "input": user_input,
        "chat_history": chat_history
    })

    answer = result["output"]
    print(f"\nAgent: {answer}")

    # Store in memory
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": answer})
