from totr.config import Config
from totr.llm import LocalLLM, Message

llm = LocalLLM(Config.from_json())

print(llm.complete("What is the meaning of life?", None))
print(llm.chat([Message(role="user", content="What is the meaning of life?")], None))
