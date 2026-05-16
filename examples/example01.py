"""
Example01: A simple example of using the Agent class.

set:
    export OCI_PROFILE=DEFAULT
    export OCI_REGION=us-chicago-1

    in a .env file and source
"""
from locus.agent import Agent

MODEL = "oci:openai.gpt-5.5"

SYSTEM_PROMPT = """
You are a helpful assistant.
Provide always a clear an complete answer.
"""

agent = Agent(model=MODEL,
              system_prompt=SYSTEM_PROMPT)

print(agent.run_sync("What is the capital of France?").text)
