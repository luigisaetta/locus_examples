"""
Author: L. Saetta
Last update: 2026-05-19
License: MIT
Description: Minimal example showing a synchronous Locus Agent call.

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

agent = Agent(model=MODEL, system_prompt=SYSTEM_PROMPT)

print(agent.run_sync("What is the capital of France?").text)
