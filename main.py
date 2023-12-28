import argparse

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

def parse_cli_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--doc", default="")
  args = parser.parse_args()
  
  if not args.doc:
    print("error: pass path to txt file as --doc argument. example: python main.py --doc=path/to/myfile")
    return None
  
  return args

embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0
)

args = parse_cli_args()

if args is not None:
  loader = TextLoader(str(args.doc))
  docs = loader.load_and_split(
      text_splitter=text_splitter
  )

  db = Chroma.from_documents(
      docs,
      embedding=embeddings,
      persist_directory="emb"
  )

  # question example
  results = db.similarity_search(
      "What is an interesting fact about the English language?"
  )

  for result in results:
      print("\n")
      print(result.page_content)
