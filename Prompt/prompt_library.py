from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#Prompt for document analysis
document_analysis_prompt = ChatPromptTemplate.from_template("""
You are a highly capable assistant trained to analyze and summarize documents.
Return ONLY valid JSON matching the exact schema below.

{format_instructions}

Analyze this document:
{document_text}
""")

# Central dictionary for all the prompt templates in the project, keyed by a unique name. This allows for easy retrieval and reuse of prompts across different modules.
PROMPT_REGISTRY = {
    "document_analysis": document_analysis_prompt,
}