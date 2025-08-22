# backend/llm_module.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Model Parameters
model_name = "gemini-1.5-flash"
DEFAULT_TEMPERATURE = 0.2

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
    
    prompt = f"""You are a senior software engineer AI assistant.
Your goal is to provide accurate, concise, and helpful answers to questions about code, acting as a technical expert.

Instructions:
1. Explain function behavior line-by-line if a function is asked.
2. Use ONLY the provided code context to answer the question. Do not use outside knowledge.
3. If the answer is not available in the provided context, state "I cannot answer this question based on the provided code context."
4. Read the code context carefully and analyze function names, logic, comments, and code patterns.
5. Mention function names and file names when applicable.
6. Generate clear, meaningful, developer-friendly, and context-aware answers.
7. Maintain a professional and helpful tone.

MERMAID DIAGRAM GENERATION (MANDATORY):
- You MUST generate a Mermaid diagram for ANY function or logic flow.
- Use flowchart format: graph TD
- Use plain, human-readable labels inside nodes.
- DO NOT include raw code syntax in labels.
- DO NOT use quotation marks or semicolons in node labels.
- Simplify code to readable labels like:
  * Loop from 0 to n
  * Check if condition is true
  * Print result
  * Return value
- Node types:
  * A[Process Step] for actions
  * B{{{{Decision Point}}}} for conditions
  * C((Start/End)) for entry/exit points
- Use --> for arrows
- Use -- Yes --> and -- No --> for decision branches

RESPONSE FORMAT:
1. Brief explanation of what the code does
2. Line-by-line breakdown
3. Mermaid diagram (REQUIRED)
4. Summary

Example Mermaid format:
```mermaid
graph TD
    A((Start)) --> B[Initialize variables]
    B --> C{{{{Check condition}}}}
    C -- Yes --> D[Execute logic]
    C -- No --> E[Skip to end]
    D --> F[Return result]
    E --> F
    F --> G((End))
```

Code Context:
```
{context}
```

Question: {question}

Provide your complete response with the mandatory Mermaid diagram:"""

    # Configure generation with the temperature parameter
    generation_config = {
        "temperature": temperature
    }
    
    try:
        model = genai.GenerativeModel(model_name)
        res = model.generate_content(prompt, stream=False, generation_config=generation_config)
        
        if res.text:
            response_text = res.text
            
            # Force diagram generation if missing
            if "```mermaid" not in response_text:
                response_text += "\n\n" + generate_fallback_diagram(context, question)
            
            return response_text
        else:
            return "No answer could be generated."
            
    except Exception as e:
        return f"Error generating response: {str(e)}"

def generate_fallback_diagram(context: str, question: str) -> str:
    """
    Generates a fallback Mermaid diagram when LLM fails to create one.
    """
    # Analyze the code to create appropriate diagram
    context_lower = context.lower()
    
    diagram = "## Mermaid Diagram:\n```mermaid\ngraph TD\n    A((Start))"
    
    # Detect common patterns
    has_function = any(keyword in context_lower for keyword in ['def ', 'function', 'void ', 'int ', 'return'])
    has_loop = any(keyword in context_lower for keyword in ['for', 'while', 'loop'])
    has_condition = any(keyword in context_lower for keyword in ['if', 'else', 'elif', '?', 'switch'])
    has_input = any(keyword in context_lower for keyword in ['input', 'scanf', 'cin', 'read'])
    has_output = any(keyword in context_lower for keyword in ['print', 'printf', 'cout', 'output', 'return'])
    
    current_node = 'A'
    
    if has_function:
        next_node = chr(ord(current_node) + 1)
        diagram += f" --> {next_node}[Initialize function parameters]"
        current_node = next_node
    
    if has_input:
        next_node = chr(ord(current_node) + 1)
        diagram += f"\n    {current_node} --> {next_node}[Read input values]"
        current_node = next_node
    
    if has_condition:
        next_node = chr(ord(current_node) + 1)
        diagram += f"\n    {current_node} --> {next_node}{{{{Check condition}}}}"
        condition_node = next_node
        
        # True branch
        true_node = chr(ord(next_node) + 1)
        diagram += f"\n    {condition_node} -- Yes --> {true_node}[Execute true branch]"
        
        # False branch  
        false_node = chr(ord(next_node) + 2)
        diagram += f"\n    {condition_node} -- No --> {false_node}[Execute false branch]"
        
        current_node = chr(ord(next_node) + 3)
        diagram += f"\n    {true_node} --> {current_node}[Continue execution]"
        diagram += f"\n    {false_node} --> {current_node}"
    
    if has_loop:
        next_node = chr(ord(current_node) + 1)
        diagram += f"\n    {current_node} --> {next_node}[Initialize loop]"
        current_node = next_node
        
        loop_check = chr(ord(current_node) + 1)
        diagram += f"\n    {current_node} --> {loop_check}{{{{Loop condition}}}}"
        
        loop_body = chr(ord(current_node) + 2)
        diagram += f"\n    {loop_check} -- Yes --> {loop_body}[Execute loop body]"
        diagram += f"\n    {loop_body} --> {loop_check}"
        
        current_node = chr(ord(current_node) + 3)
        diagram += f"\n    {loop_check} -- No --> {current_node}[Exit loop]"
    
    if has_output:
        next_node = chr(ord(current_node) + 1)
        diagram += f"\n    {current_node} --> {next_node}[Generate output]"
        current_node = next_node
    
    # End node
    end_node = chr(ord(current_node) + 1)
    diagram += f"\n    {current_node} --> {end_node}((End))"
    diagram += "\n```"
    
    return diagram