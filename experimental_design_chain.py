from operator import itemgetter

from langchain_community.retrievers.pubmed import PubMedRetriever
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.runnables import RunnableLambda, RunnableParallel

from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

from langchain.globals import set_debug
set_debug(False)

# 在代码中使用环境变量
import os
ZHIPUAI_API_KEY = os.getenv("ZHIPU_API_KEY")

# 初始化zhipu客户端
model = ChatZhipuAI(
    temperature=0.1,
    api_key=ZHIPUAI_API_KEY,
    model_name="glm-4",
)

template1 = """
你是一个生物医学和健康科学领域的专家，并且你对PubMed很了解，需要根据以下中文文本：'{text}'，
提取出一组英文关键字，保证提取出的关键字适合在PubMed搜索，
最终的结果只返回英文关键字就可以了，其余不用返回
"""
prompt1 = PromptTemplate.from_template(template1)
chain1 = (
    {
        'text': itemgetter('query')
    }
    | prompt1
    | model
)

def get_context(response):
    retriever = PubMedRetriever(top_k_results=5)  
    search = ' AND '.join(response.content.split(','))
    documents = retriever.get_relevant_documents(search)
    if documents:
        context = '\n\n'.join([f'第{i+1}篇文献： {document.page_content}' for i,document in enumerate(documents)])
        informations = []
        for document in documents:
            if isinstance(document.metadata['Title'], dict):
                informations.append(f"Title: {document.metadata['Title']['#text']}\nDOI: {document.metadata['DOI']}\nPublished: {document.metadata['Published']}")
            else:
                informations.append(f"Title: {document.metadata['Title']}\nDOI: {document.metadata['DOI']}\nPublished: {document.metadata['Published']}")
        
        return {'context': context, 'informations': informations}
    else:
        search = ' OR '.join(response.content.split(','))
        documents = retriever.get_relevant_documents(search)
        context = '\n\n'.join([f'第{i+1}篇文献： {document.page_content}' for i,document in enumerate(documents)])
        informations = []
        for document in documents:
           if isinstance(document.metadata['Title'], dict):
                informations.append(f"Title: {document.metadata['Title']['#text']}\nDOI: {document.metadata['DOI']}\nPublished: {document.metadata['Published']}")
           else:
                informations.append(f"Title: {document.metadata['Title']}\nDOI: {document.metadata['DOI']}\nPublished: {document.metadata['Published']}")
        return {'context': context, 'informations': informations}

template2 = """
你是一个生物医学和健康科学领域的专家，而且精通中英双语。
现在需要全面参考多篇来自PubMed网站的文献回答用户问题，务必要完整看过全部文献后再回答问题，
确保你的回复完全依据给出的文献信息，不要编造答案。
如果所给出的相关文献不足以回答用户的问题，请直接回复"我无法回答您的问题"。

已知的PubMed文献内容:
{context}
用户问：
{question}

请用中文回答用户问题。
"""
rev_chain = chain1 | RunnableLambda(get_context)
prompt2 = PromptTemplate.from_template(template2)
chain = (
    {
        'context': rev_chain | itemgetter('context'),
        'question': itemgetter('query')
    }
    | prompt2
    | model
)

sequence = RunnableParallel(
    {
        "informations": rev_chain | itemgetter('informations'),
        "output": chain
    }
)

if __name__ == "__main__":
    query = "我想提取脐带间质干细胞外泌体，应该怎么做"
    response = sequence.invoke({'query': query})
    print(response['output'].content, end='\n\n')
    print("以上回答参考的PubMed文献信息如下:")
    for one in response['informations']:
        print(one)