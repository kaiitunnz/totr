from totr.config import Config
from totr.retriever import ElasticsearchRetriever

retriever = ElasticsearchRetriever(Config.from_json())

results = retriever.retrieve_titles("injuries")
print(results)
