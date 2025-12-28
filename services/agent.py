from langchain_ollama.chat_models import ChatOllama
from langsmith import traceable
from langchain.agents import create_agent
from dotenv import load_dotenv
from tools.tools import retrieve_context
from models.inputschema import UserInput


load_dotenv(dotenv_path=".env", override=True)


class CustomerSupportAgent:
    def __init__(self):
        self.model = ChatOllama(
            model="gpt-oss:120b-cloud",
            temperature=0.5,
            streaming=True
        )
        self.prompt = (
            "You are a customer support agent for a laptop company and you specialize in providing troubleshooting steps."
            "Use the tool to help answer user queries."
        )

    @traceable(run_type='chain')
    def run_agent(self, query: UserInput):
        tools = [retrieve_context]
        agent = create_agent(self.model, tools, system_prompt=self.prompt)

        response = None

        for event in agent.stream(
            {"messages": [{"role": "user", "content": query.message}]},
            stream_mode="values",
        ):
            response = event["messages"][-1]
            response.pretty_print()

        return response


if __name__ == "__main__":
    agent = CustomerSupportAgent()
    query = (
        "How to fix no power issue?"
        "Once you get the answer, look up common extensions of that method."
    )
    agent.run_agent(UserInput(message=query))
