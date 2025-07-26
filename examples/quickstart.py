from edison import Edison
from dotenv import load_dotenv

import os

load_dotenv(".env", override=True)


def main():
    edison = Edison(api_key=os.getenv("OPENAI_API_KEY"))
    response = edison.generate_text_response("What is the capital of France?")
    print(response)


if __name__ == "__main__":
    main()
