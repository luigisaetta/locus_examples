"""
Example02: With streaming.

set:
    export OCI_PROFILE=DEFAULT
    export OCI_REGION=us-chicago-1

    in a .env file and source
"""

import asyncio
from locus.agent import Agent
from locus.core.events import (
    TerminateEvent,
    ThinkEvent,
    ToolCompleteEvent,
    ToolStartEvent,
)

MODEL = "oci:openai.gpt-5.5"

SYSTEM_PROMPT = """
You are a helpful assistant.
Provide always a clear an complete answer.
"""

agent = Agent(model=MODEL, system_prompt=SYSTEM_PROMPT)

QUESTION = "25*4 is greater or lower than 101? Explain your answer."


async def example_events():
    """Run an agent with streaming and print the emitted Locus events."""
    print("")

    print("Question: " + QUESTION)
    print("")
    print("Events received:")

    async for event in agent.run(QUESTION):
        print(f"\n  Event Type: {event.event_type}")
        print(f"  Timestamp:  {event.timestamp}")

        if isinstance(event, ThinkEvent):
            print(f"  Iteration:  {event.iteration}")
            print(f"  Tool Calls: {len(event.tool_calls)}")
            if event.reasoning:
                preview = (
                    event.reasoning[:80] + "..."
                    if len(event.reasoning) > 80
                    else event.reasoning
                )
                print(f"  Reasoning:  {preview}")

        elif isinstance(event, ToolStartEvent):
            print(f"  Tool Name:  {event.tool_name}")
            print(f"  Arguments:  {event.arguments}")

        elif isinstance(event, ToolCompleteEvent):
            print(f"  Tool Name:  {event.tool_name}")
            print(f"  Result:     {event.result}")
            print(f"  Duration:   {event.duration_ms:.1f}ms")

        elif isinstance(event, TerminateEvent):
            print(f"  Reason:     {event.reason}")
            print(f"  Iterations: {event.iterations_used}")
            if event.final_message:
                print(f"  Answer:\n\n{event.final_message}")

    print()


def main():
    """Run the streaming example from the command line."""
    asyncio.run(example_events())


if __name__ == "__main__":
    main()
