from langchain.embeddings.openai import OpenAIEmbeddings
import os

def main():
    os.environ["OPENAI_API_TYPE"] = "azure"
    os.environ["OPENAI_API_VERSION"] = "2023-05-15"
    os.environ["OPENAI_API_BASE"] = "https://ai-ramsol-traning.openai.azure.com/"
    os.environ["OPENAI_API_KEY"] = "5b60d2473952443cafceeee0b2797cf4"
    embedding_function = OpenAIEmbeddings(
                            api_key="5b60d2473952443cafceeee0b2797cf4",
                            openai_api_base="https://ai-ramsol-traning.openai.azure.com/",
                            openai_api_type="azure",
                            api_version="2023-05-15",
                            deployment="embedding-dev",
                            model="text-embedding-ada-002")
    return embedding_function
if __name__ == '__main__':
    main()