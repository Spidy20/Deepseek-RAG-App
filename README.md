# Deploy DeepSeek Model on AWS EC2 🚀 | Build a RAG App 🤖 with LangChain & ChromaDB! 🔥

### [Watch this tutorial►](https://youtu.be/YWmnD_QcZQU)
<img src="https://github.com/Spidy20/Deepseek-RAG-App/blob/master/yt_thumb.jpg">

- This video demonstrates how to deploy the DeepSeek model on AWS EC2 and build a RAG (Retrieval-Augmented Generation) application using LangChain & ChromaDB. You'll learn to set up an EC2 instance, configure dependencies, run the DeepSeek Ollama API, and integrate it with a Streamlit-based chat app to process and analyze PDF documents with AI-powered responses. 🚀
### Implementation Architecture
<img src="https://github.com/Spidy20/Deepseek-RAG-App/blob/master/deepseek-RAG-V-1.0.png">

### Used Services
- **AWS EC2**: Responsible for managing the backend of the Document Extractor using the Boto3 SDK.
- **AWS EC2**: Deploy and run the DeepSeek model efficiently on a scalable cloud instance.
- **Streamlit**: Build an interactive chat interface to test DeepSeek’s AI responses.
- **ChromaDB**: Store and retrieve vector embeddings for RAG-based document processing.
- **Ollama**: Serve and run the DeepSeek model locally on EC2 with optimized inference.

### Implementation Setup

1. **Set Up EC2 Instance** – Launch an AWS EC2 instance and configure security settings.
2. **Connect to EC2** – Access the instance via SSH and install necessary dependencies.
3. **Install Ollama** – Set up Ollama to run and manage the DeepSeek model.
4. **Test DeepSeek Model** – Run a quick shell test to verify model functionality.
5. **Install App Dependencies** – Install Streamlit, ChromaDB, and LangChain for the RAG app.
6. **Develop the Chat App** – Integrate DeepSeek with Streamlit for real-time chat interaction.
7. **Configure Security & API Access** – Set up EC2 security groups and expose the API.
8. **Test & Deploy** – Upload a PDF, query the model, analyze responses, and finalize deployment.  

# Commands

# Ubuntu Commands  

```sh
# Update System Packages
apt update

# To download Ollama
curl -fsSL https://ollama.com/install.sh | sh

# To download Model visit
# https://ollama.com/library/deepseek-r1:7b

# Download model 
ollama run deepseek-r1:7b

# Check API Serving
ollama serve

# Model API
curl http://localhost:11434/api/chat -d '{
  "model": "deepseek-r1:8b",
  "messages": [{ "role": "user", "content": "Write python script for hello world" }],
  "stream": false
}'

# To install requirements
python3 -m pip install -r requirements.txt

# To run App
python3 -m streamlit run app.py
```

### Give Star⭐ to this repository, and fork it to support me. 

### [Buy me a Coffee☕](https://www.buymeacoffee.com/spidy20)
### [Donate me on PayPal(It will inspire me to do more projects)](https://www.paypal.me/spidy1820)
