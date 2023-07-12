from indexing import index

# function to provide the gpt with context of the given docs
def get_similiar_docs(query,k=1,score=False):
  if score:
    similar_docs = index.similarity_search_with_score(query,k=k)
  else:
    similar_docs = index.similarity_search(query,k=k)
  return similar_docs

def get_conversation_string():
  conversation_string = ""