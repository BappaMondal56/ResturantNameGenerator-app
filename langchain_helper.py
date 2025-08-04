import os
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import streamlit as st

os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
os.environ["OPENAI_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]

llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",
    temperature=0.7
)

def generate_restaurant_name_and_items(cuisine):
    # Chain 1 : Restaurant Name
    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template="I want to open a restaurant for {cuisine} food. Suggest a funny, modern Gen-Z name. Only one name please and no more text."
    )
    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")

    # Chain 2 : Menu Items
    prompt_template_items = PromptTemplate(
        input_variables=['restaurant_name'],
        template="Suggest Some menu items for {restaurant_name} food. Return it just as a comma seperated list without any heading ."
    )
    food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")

    chain = SequentialChain(
        chains=[name_chain, food_items_chain],
        input_variables=['cuisine'],
        output_variables=['restaurant_name', 'menu_items']
    )
    response = chain.invoke({'cuisine': cuisine})
    return response

if __name__ == '__main__':
    print(generate_restaurant_name_and_items("Indian"))