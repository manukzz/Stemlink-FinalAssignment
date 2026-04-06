import gradio as gr
import requests
import os
import time

# Configuration
API_URL = "http://127.0.0.1:8000"  

def process_pdf(file):
    """
    Uploads the PDF to the FastAPI /index-pdf endpoint.
    """
    if file is None:
        return gr.update(interactive=False), "❌ No file uploaded."

    try:
        # Prepare file for upload
        files = {'file': (os.path.basename(file.name), open(file.name, 'rb'), 'application/pdf')}
        
        # Call Backend
        response = requests.post(f"http://127.0.0.1:8000/index-pdf", files=files)
        
        if response.status_code == 200:
            data = response.json()
            msg = f"✅ {data.get('message', 'Indexed successfully')} (Chunks: {data.get('chunks_indexed', 'N/A')})"
            return gr.update(interactive=True, placeholder="Ask a question about your document..."), msg
        else:
            return gr.update(interactive=False), f"⚠️ Error {response.status_code}: {response.text}"
            
    except requests.exceptions.ConnectionError:
        return gr.update(interactive=False), "❌ Connection Error: Ensure FastAPI backend is running."
    except Exception as e:
        return gr.update(interactive=False), f"❌ Unexpected Error: {str(e)}"


def format_response_with_citations(answer, citations):
    """
    Helper to format the final markdown response with citations if present.
    """
    text = answer
    
    if citations:
        text += "\n\n---\n**Sources & Citations:**\n"
        for source, details in citations.items():
            
            text += f"* **[{source}]**: {details}\n"
            
    return text


def rag_logic_stream(history):
    """
    Calls the FastAPI /qa endpoint and streams the result.
    """
    user_message = history[-1]["content"]
    
    try:
        # Call Backend
        payload = {"question": user_message}
        response = requests.post(f"http://127.0.0.1:8000/qa", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            answer_text = data.get("answer", "No answer provided.")
            citations = data.get("citations", {})
            
            # Combine answer + citations for display
            full_response = format_response_with_citations(answer_text, citations)
            
        else:
            full_response = f"⚠️ Backend Error {response.status_code}: {response.text}"

    except requests.exceptions.ConnectionError:
        full_response = "❌ Connection Error: Could not reach the API. Is it running?"

    
    history.append({"role": "assistant", "content": ""})
    
    
    
    chunk_size = 5 
    for i in range(0, len(full_response), chunk_size):
        time.sleep(0.01)
        history[-1]["content"] = full_response[: i + chunk_size]
        yield history
        
    
    history[-1]["content"] = full_response
    yield history


# --- GRADIO INTERFACE ---

custom_css = """
.gradio-container {max-width: 900px !important; margin: auto;}
#chatbot-container {border-radius: 15px;}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.HTML("<center><h1>📄 RAG Agent Demo</h1><p>Upload a PDF to the Multi-Agent API</p></center>")
    
    history_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="1. Upload PDF", file_types=[".pdf"])
            status_msg = gr.Markdown("*Status: Waiting for upload...*")
            
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Agent Response",
                height=500,
                elem_id="chatbot-container"
                
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="🔒 Upload a PDF to start chatting",
                    scale=9,
                    interactive=False, 
                    container=False
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)

    # --- EVENT HANDLING ---

    
    pdf_input.upload(
        fn=process_pdf, 
        inputs=[pdf_input], 
        outputs=[msg, status_msg]
    )

   
    def add_user_message(user_message, history):
        if not user_message or user_message.strip() == "":
            return "", history
        history.append({"role": "user", "content": user_message})
        return "", history

    msg.submit(add_user_message, [msg, history_state], [msg, history_state]).then(
        rag_logic_stream, [history_state], [chatbot]
    )
    submit_btn.click(add_user_message, [msg, history_state], [msg, history_state]).then(
        rag_logic_stream, [history_state], [chatbot]
    )

if __name__ == "__main__":
    demo.launch()