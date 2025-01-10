from langchain.chat_models import ChatOpenAI

class TextAgent:
    """Handles general text-based queries using the LLM."""
    
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)

    def process_text_query(self, query: str) -> str:
        """
        Send the query to the LLM and return the response.
        """
        try:
            # Generate response from LLM
            response = self.llm.invoke(query)
            return response
        except Exception as e:
            return f"Error processing text query: {str(e)}"
