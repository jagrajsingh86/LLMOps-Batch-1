import sys
import os
from operator import itemgetter
from typing import List, Optional, Dict, Any

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from utils.model_loader import Modelloader
from exceptions.custom_exception import DocumentPortalException
from logger import GLOBAL_LOGGER as log
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import PromptType


class ConversationalRAG:

    def __init__(self):
        pass

    def _load_llm(self):
        pass

    @staticmethod
    def _format_docs(docs) -> str:
        pass

    def _build_lcel_chain(self):
        pass


    #----------Public Methods/APIs----------#

    def load_retriever_from_faiss(self):
        pass

    def invoke():
        pass

    
