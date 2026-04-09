import os
import sys
from dotenv import load_dotenv
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_classic.output_parsers import OutputFixingParser
from utils.model_loader import Modelloader
from logger import GLOBAL_LOGGER as log
from exceptions.custom_exception import DocumentPortalException
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import SummaryResponse, PromptType


class DocumentComparerLLM:
    def __init__(self):
        load_dotenv(override=True)
        self.loader = Modelloader()
        self.llm = self.loader.load_llm()
        self.parser = JsonOutputParser(pydantic_object=SummaryResponse)
        self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
        self.prompt = PROMPT_REGISTRY[PromptType.DOCUMENT_COMPARISON.value]
        self.chain = self.prompt | self.llm | self.parser
        log.info("DocumentComparerLLM initialized successfully")

    def compare_documents(self, combined_docs: str) -> pd.DataFrame:
        try:
            inputs = {
                "combined_docs": combined_docs,
                "format_instructions": self.parser.get_format_instructions()
            }

            log.info("Invoking document comparison chain")
            response = self.chain.invoke(inputs)
            log.info("Document comparison completed successfully", response_preview=str(response)[:200])
            return self._format_reponse(response)

        except Exception as e:
            log.error("Error comparing documents", error=str(e))
            raise DocumentPortalException("Error in document comparison", e) from e

    def _format_reponse(self, response_parsed: list[dict]) -> pd.DataFrame: #type: ignore
        try:
            # Convert list of dicts to DataFrame
            df = pd.DataFrame(response_parsed)
            return df
        except Exception as e:
            log.error("Failed to format comparison response", error=str(e))

