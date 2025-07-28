# backend/llm_module.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Model Parameters
model_name = "gemini-1.5-flash"
DEFAULT_TEMPERATURE = 0.2 # A good starting point for factual responses

def generate_answer(question: str, chunks_content: list[str], temperature: float = DEFAULT_TEMPERATURE) -> str:
    """
    Generates an answer to the question based on provided code chunks.

    Args:
        question (str): The user's question.
        chunks_content (list[str]): A list of code/comment strings (chunk content) to use as context.
        temperature (float): Controls the randomness of the output. Lower is more deterministic.

    Returns:
        str: The generated answer.
    """
    context = "\n\n".join(chunks_content)
    prompt = f"""
    You are a senior software engineer AI assistant.
    Your goal is to provide accurate, concise, and helpful answers to questions about code, acting as a technical expert.

    Instructions:
    1.  Explain function behavior line-by-line if a function is asked (e.g., `reverse()`).
    2.  Use ONLY the provided code context to answer the question. Do not use outside knowledge.
    3.  If the answer is not available in the provided context, state "I cannot answer this question based on the provided code context."
    4.  Read the code context carefully and analyze function names, logic, comments, and code patterns.
    5.  Mention function names and file names (if given) when applicable.
    6.  Generate clear, meaningful, developer-friendly, and context-aware answers.
    7.  Maintain a professional and helpful tone.

   
Mermaid Diagram Generation (Always Generate if Possible):

- Always generate a Mermaid diagram (flowchart using `graph TD`) if the function or logic can be visualized.
- Use plain, human-readable **labels** inside nodes.
-  DO NOT include raw C/C++ syntax inside Mermaid labels (e.g., `for(i = 0; i < n; i++)`, `printf("hello")`).
-  DO NOT use quotation marks (`"`) or semicolons (`;`) inside node labels.
-  Instead, simplify such code to labels like:
    - `Loop from 0 to n`
    - `Check if i is prime`
    - `Print value`
    - `Increment counter`
- Use:
    - `[]` for steps
    - `{{}}` for decision nodes (conditions)
    - `-->` for arrows

- Enclose the Mermaid code block in triple backticks:

```mermaid
graph TD
    A[Start] --> B[Initialize i]
    B --> C{{i < n?}}
    C -- Yes --> D[Check prime]
    C -- No --> E[End]


    If you generate a Mermaid diagram, briefly explain its purpose before the diagram block.And explain the complete function line by line perfectly
    generate a diagram to add significant value to the answer.

    Code Context:
    ```
    {context}
    ```

    Question: {question}

    Answer:
    """
    # Configure generation with the temperature parameter
    generation_config = {
        "temperature": temperature
    }

    model = genai.GenerativeModel(model_name)
    res = model.generate_content(prompt, stream=False, generation_config=generation_config)
    
    if res.text:
        return res.text
    else:
        return "No answer could be generated."