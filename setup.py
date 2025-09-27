from setuptools import setup, find_packages

setup(
    name="case-study-chatbot",
    version="0.1.0",
    description="AI-powered case study analysis chatbot",
    packages=find_packages(),
    python_requires="^>=3.10",
    install_requires=[
        "python-dotenv^>=1.0.0",
        "python-docx^>=0.8.11",
        "sentence-transformers^>=2.2.2",
        "faiss-cpu^>=1.7.4",
        "networkx^>=3.1.0",
        "google-generativeai^>=0.3.0",
        "openai^>=1.3.0",
        "gradio^>=3.50.0",
        "numpy^>=1.24.0",
        "pandas^>=2.0.0",
        "requests^>=2.31.0",
        "tqdm^>=4.65.0",
        "rich^>=13.5.0",
    ],
)
