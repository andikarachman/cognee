import asyncio
import cognee
import os
import time

from pprint import pprint

# By default cognee uses OpenAI's gpt-5-mini LLM model
# Provide your OpenAI LLM API KEY
# os.environ["LLM_API_KEY"] = ""


async def cognee_demo():
    # Get file path to document to process
    from pathlib import Path

    current_directory = Path(__file__).resolve().parent.parent
    file_path = os.path.join(current_directory, "data", "alice_in_wonderland.txt")

    print("\n⏱️  Starting cognee pipeline...\n")
    
    # Prune data
    start = time.time()
    await cognee.prune.prune_data()
    print(f"✓ prune_data() completed in {time.time() - start:.2f}s")
    
    # Prune system
    start = time.time()
    await cognee.prune.prune_system(metadata=True)
    print(f"✓ prune_system() completed in {time.time() - start:.2f}s\n")

    # Add document
    start = time.time()
    await cognee.add(file_path)
    print(f"✓ add() completed in {time.time() - start:.2f}s")
    
    # Cognify (process)
    start = time.time()
    await cognee.cognify()
    print(f"✓ cognify() completed in {time.time() - start:.2f}s\n")

    # Query 1
    print("Query 1: List characters...")
    start = time.time()
    answer = await cognee.search("List me all the important characters in Alice in Wonderland.")
    print(f"✓ search() completed in {time.time() - start:.2f}s")
    pprint(answer)
    print()

    # Query 2
    print("Query 2: How did Alice end up in Wonderland?")
    start = time.time()
    answer = await cognee.search("How did Alice end up in Wonderland?")
    print(f"✓ search() completed in {time.time() - start:.2f}s")
    pprint(answer)
    print()

    # Query 3
    print("Query 3: Alice's personality...")
    start = time.time()
    answer = await cognee.search("Tell me about Alice's personality.")
    print(f"✓ search() completed in {time.time() - start:.2f}s")
    pprint(answer)


# Cognee is an async library, it has to be called in an async context
asyncio.run(cognee_demo())
