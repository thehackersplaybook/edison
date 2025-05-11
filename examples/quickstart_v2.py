from edison.edison_deep_research import EdisonDeepResearch
import asyncio
from dotenv import load_dotenv


load_dotenv(".env", override=True)


async def main():
    """
    Quickstart example for using EdisonDeepResearch.
    This function initializes the EdisonDeepResearch class and performs a deep research operation based on user input.
    """
    researcher = EdisonDeepResearch()
    query = input("Enter your research query: ")
    result = await researcher.deep_stream_async(query=query, verbose=True, version="v2")

    async for chunk in result:
        pass


if __name__ == "__main__":
    asyncio.run(main())
