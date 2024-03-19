# RAGTune

RAGTune is an automated tuning and optimization tool for the RAG (Retrieval-Augmented Generation) pipeline. This tool allows you to evaluate different LLMs (Large Language Models), embedding models, query transformations, and rerankers.

## Getting Started

### Prerequisites

Before you begin, ensure you have met the following requirements:
- You have installed Python 3.10 or higher.
- Some tests require a lot of API call, make sure you are willing to spend on the API calls.

### Cloning the Repository

To clone the RAGTune repository, run the following command:

```bash
git clone https://github.com/misbahsy/RAGTune.git
cd RAGTune
```

### Installing Dependencies

Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

![Unstructured](https://www.unstructured.io/) is used as a document loader, make sure to install the dependencies mentioned by Unstructured.

libmagic-dev (filetype detection)
poppler-utils (images and PDFs)
tesseract-ocr (images and PDFs, install tesseract-lang for additional language support)
libreoffice (MS Office docs)
pandoc (EPUBs, RTFs and Open Office docs). Please note that to handle RTF files, you need version 2.14.2 or newer. Running either make install-pandoc or ./scripts/install-pandoc.sh will install the correct version for you.

### Setting Up Environment Variables

Create a `.env` file in the root directory of the project and add your API keys:

```plaintext
OPENAI_API_KEY=sk-xyz
COHERE_API_KEY=cohere-xyz
ANTHROPIC_API_KEY=anthropic-xyz
```

Replace `xyz` with your actual API keys.

### Running the Streamlit App

To run the Streamlit app, execute the following command:

```bash
streamlit run Home.py
```

This will start the Streamlit server and open the app in your default web browser.

## Customization

### Adding LLMs

To add a new LLM for evaluation, you can modify the `pages/1_LLM.py` file. Import the necessary modules and add the LLM to the evaluation chain.

### Adding Embedding Models

To add a new embedding model, update the `pages/2_embeddings.py` file. Import the embedding model and integrate it with the existing evaluation pipeline.

## Tips

- Feel free to experiment by adding and varying different RAG parameters.
- Use virtual environments to manage dependencies and avoid conflicts.
- Keep your API keys confidential and do not commit them to the repository.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is open-sourced under the MIT License. See the `LICENSE` file for more information.
