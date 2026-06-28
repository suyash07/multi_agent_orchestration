import warnings
warnings.filterwarnings("ignore")

from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain_core.messages import HumanMessage, AIMessage
from tools import tools

# ─────────────────────────────────────────
# 1. THE REASONING AGENT
# This is the LLM that decides which tools
# to call based on the user's question
# ─────────────────────────────────────────

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""
You are a senior bank analyst assistant with access to a 
customer churn prediction system.

You help bank analysts, customer service reps, and managers
understand and act on customer churn risk.

You have access to four tools:
- get_portfolio_summary: overall churn picture
- get_churn_drivers: why customers are churning
- get_high_risk_customers: who is most at risk
- predict_churn: churn probability for one customer

RULES:
1. Always use a tool to answer. Never make up numbers.
2. Be concise and actionable. This is a business context.
3. When listing customers, prioritize actionability.
4. If asked something outside your tools, say so honestly.
5. Always end with a clear recommended action.
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

agent = create_openai_tools_agent(llm, tools, prompt)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)

# ─────────────────────────────────────────
# 2. CONVERSATION LOOP
# Keeps chat history so the agent
# remembers earlier questions
# ─────────────────────────────────────────

def run_chat():
    print("\n" + "="*60)
    print("BANK CHURN ANALYST - AI Assistant")
    print("="*60)
    print("Ask me anything about customer churn.")
    print("Type 'quit' to exit.\n")
    print("Example questions:")
    print("  - Give me an overview of churn")
    print("  - Which customers should I call this week?")
    print("  - Why are German customers churning?")
    print("  - Predict churn for a 45 year old German customer")
    print("    with $120k balance, 2 products, inactive")
    print("="*60 + "\n")

    chat_history = []

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        try:
            response = executor.invoke({
                "input": user_input,
                "chat_history": chat_history
            })

            answer = response["output"]

            # update chat history for memory
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=answer))

            print(f"\nAssistant: {answer}\n")
            print("-" * 60 + "\n")

        except Exception as e:
            print(f"\nError: {e}\n")
            print("Try rephrasing your question.\n")

if __name__ == "__main__":
    run_chat()