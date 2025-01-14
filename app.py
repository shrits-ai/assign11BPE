import gradio as gr
import json

class BPETokenizer:
    def __init__(self, vocab_path):
        # Load pre-trained vocabulary
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

    def encode(self, text):
        """Encode a piece of text into BPE tokens."""
        for token in sorted(self.vocab, key=len, reverse=True):  # Sort tokens by length in descending order
            text = text.replace(token, f' {token} ')  # Replace tokens with space-separated versions
        return text.split()  # Split text into tokens

# Load the pre-trained tokenizer
vocab_path = "bpe_vocab_5000.json"
bpe_tokenizer = BPETokenizer(vocab_path)

# Gradio Functions
def encode_text(text):
    """Encode user-provided text with the pre-trained tokenizer."""
    if not text.strip():
        return "Please enter some text to encode."  # Handle empty input
    tokens = bpe_tokenizer.encode(text)
    return " | ".join(tokens)  # Use a separator to display tokens clearly

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Bengali BPE Tokenizer")
    gr.Markdown(
        """
        This app encodes Bengali text into Byte Pair Encoding (BPE) tokens using a pre-trained tokenizer.
        Enter Bengali text below and press "Encode" to view the tokenized output.
        """
    )
    
    with gr.Row():
        input_text = gr.TextArea(label="Enter Bengali Text to Encode", lines=5, placeholder="Type Bengali text here...")
        output_tokens = gr.Textbox(label="Encoded Tokens", lines=5, interactive=False)
    
    encode_button = gr.Button("Encode")
    encode_button.click(encode_text, inputs=input_text, outputs=output_tokens)

# Launch the app
demo.launch(share=True)
