from langchain_community.retrievers.pubmed import PubMedRetriever
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatZhipuAI
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# 在代码中使用环境变量
import os
ZHIPUAI_API_KEY = os.getenv("ZHIPU_API_KEY")

# 初始化zhipu客户端
llm = ChatZhipuAI(
    temperature=0.1,
    api_key=ZHIPUAI_API_KEY,
    model_name="glm-4",
)


'''
基于prompt生成文本
'''
def get_context (query):
    prompt_template = PromptTemplate.from_template(
        "你是一个生物医学和健康科学领域的专家，并且你对PubMed很了解，需要根据以下中文文本：'{text}'，提取出一组英文关键字，保证提取出的关键字适合在PubMed搜索，\
            最终的结果只返回英文关键字就可以了，其余不用返回"
    )
    prompt = prompt_template.format(text=query)
    response = llm.invoke(prompt)

    search = ' AND '.join(response.content.split(','))
    retriever = PubMedRetriever(top_k_results=2)  
    documents = retriever.get_relevant_documents(search)
    if documents:
        context = '\n\n'.join([document.page_content for document in documents])
        informations = []
        for document in documents:
            if isinstance(document.metadata['Title'], dict):
                informations.append(f"Title: {document.metadata['Title']['#text']}\nDOI: {document.metadata['DOI']}\nPublished: {document.metadata['Published']}")
            else:
                informations.append(f"Title: {document.metadata['Title']}\nDOI: {document.metadata['DOI']}\nPublished: {document.metadata['Published']}")
        
        return context,informations
    else:
        search = ' OR '.join(response.content.split(','))
        documents = retriever.get_relevant_documents(search)
        context = '\n\n'.join([document.page_content for document in documents])
        informations = []
        for document in documents:
           if isinstance(document.metadata['Title'], dict):
                informations.append(f"Title: {document.metadata['Title']['#text']}\nDOI: {document.metadata['DOI']}\nPublished: {document.metadata['Published']}")
           else:
                informations.append(f"Title: {document.metadata['Title']}\nDOI: {document.metadata['DOI']}\nPublished: {document.metadata['Published']}")
        return context,informations



"""
利用LLM设计相关实验
"""
def get_experiment(query):
    context, informations = get_context(query)
    prompt_template = """
    你是一个生物医学和健康科学领域的专家，同时精通中英双语。
    你的任务是根据下述给定的已知的上下文信息回答用户问题。
    确保你的回复完全依据下述已知信息。不要编造答案。
    如果下述已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。

    已知上下文信息:
    {context}
    用户问：
    {question}

    请用中文回答用户问题。
    """
    prompt_template = PromptTemplate.from_template(template=prompt_template)
    prompt = prompt_template.format(context=context, question=query)
    response = llm.invoke(prompt)
    return response.content, informations

if __name__ == "__main__":
    query = "设计一个实验验证hsp60蛋白的功能"
    result, informations = get_experiment(query)
    print(result, '\n\n')
    print('以上结果参考于相关PubMed文献，具体文献信息如下：')
    for information in informations:
        print(information, end='\n\n')
    