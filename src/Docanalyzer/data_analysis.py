import os
import sys

# Add the project root to sys.path so that sibling packages (utils, logger, etc.) can be imported
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.model_loader import Modelloader
from utils.document_ops import concat_for_analysis
from exceptions.custom_exception import DocumentPortalException
from logger import GLOBAL_LOGGER as log
from model.models import *
# JsonOutputParser: parses LLM text output into a Python dict validated against a Pydantic model
from langchain_core.output_parsers import JsonOutputParser
# OutputFixingParser: wraps another parser; if parsing fails, it asks the LLM to fix the output
from langchain_classic.output_parsers import OutputFixingParser
import uuid
from prompt.prompt_library import document_analysis_prompt
from prompt.prompt_library import PROMPT_REGISTRY


class DocumentAnalyzer:
    def __init__(self):
        # Use the global logger directly (GLOBAL_LOGGER is already a configured logger instance)
        self.log = log
        self.log.info("Initializing DocumentAnalyzer")
        try:
            # Modelloader reads config/config_loader.yaml and API keys from env/.env
            self.loader = Modelloader()
            # Load the configured LLM (e.g. Google Gemini) that will be used in the chain
            self.llm = self.loader.load_llm()

            # --- RAG Chain component 1: Output Parser ---
            # JsonOutputParser validates the LLM's raw text against the Metadata Pydantic schema
            self.parser = JsonOutputParser(pydantic_object=Metadata)
            # OutputFixingParser adds a self-healing layer: if the LLM's JSON is malformed,
            # it sends the error back to the LLM and asks it to correct the output automatically
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
            # --- RAG Chain component 2: Prompt Template ---
            # Fetch the pre-defined ChatPromptTemplate from the central registry
            self.prompt = PROMPT_REGISTRY["document_analysis"]

        except Exception as e:
            log.error("Failed to initialize DocumentAnalyzer", error=str(e))
            raise DocumentPortalException("Initialization error in DocumentAnalyzer", e) from e
    
    def analyze_document(self, document_text:str)->dict:
        """Analyze the document and extract metadata & Summary."""
        try:
            # --- Build the LangChain LCEL chain using the pipe (|) operator ---
            # The chain flows left-to-right:
            #   1. self.prompt        – fills the ChatPromptTemplate with the input variables
            #   2. self.llm           – sends the formatted prompt to the LLM and gets raw text back
            #   3. self.fixing_parser – parses the raw text into a validated Python dict (with auto-fix)
            chain = self.prompt | self.llm | self.fixing_parser
            self.log.info("Running document analysis chain")

            # Invoke the chain with the two variables the prompt template expects:
            #   - format_instructions: auto-generated JSON schema description from the parser
            #   - document_text: the actual document content to analyze
            response = chain.invoke({
                "format_instructions": self.parser.get_format_instructions(),
                "document_text": document_text
            })

            self.log.info("Document analysis completed successfully")
            # response is a Python dict matching the Metadata model fields
            return response
        
        except Exception as e:
            log.error("Failed to analyze document", error=str(e))
            raise DocumentPortalException("Error during document analysis", e) from e
        

