import os
import openai
from langsmith.wrappers import wrap_openai
from langsmith import traceable
from dotenv import load_dotenv
from config import get_settings

# Load environment variables from .env file
load_dotenv()

settings = get_settings()
open_ai_key = settings.OPEN_AI_KEY
open_ai_org = settings.OPEN_AI_ORG
lang_smith_key = settings.LANG_SMITH_KEY

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = lang_smith_key
os.environ['OPENAI_API_KEY'] = open_ai_key


# Auto-trace LLM calls in-context
client = wrap_openai(openai.Client())

@traceable # Auto-trace this function
def pipeline(user_input: str):
    result = client.chat.completions.create(
        messages=[{"role": "user", "content": user_input}],
        model="gpt-3.5-turbo"
    )
    return result.choices[0].message.content

pipeline("Hi, OpenAI!")