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

[Unstructured](https://www.unstructured.io/) is used as a document loader, make sure to install the dependencies mentioned by Unstructured.
```plaintext
libmagic-dev (filetype detection)
poppler-utils (images and PDFs)
tesseract-ocr (images and PDFs, install tesseract-lang for additional language support)
libreoffice (MS Office docs)
pandoc (EPUBs, RTFs and Open Office docs). Please note that to handle RTF files, you need version 2.14.2 or newer. Running either make install-pandoc or ./scripts/install-pandoc.sh will install the correct version for you.
```

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

### Dataset

Make sure to first add your files and the corresponding Questions and Groud truth answers for the evaluations. The app will throw errors if this step is not completed. 

### Step-wise Approach

Each section is divided into steps to help you run the app in a step-wise manner. Skipping a step will throw errors. 

Additionally, if you notice an error box on a new tab, most likely it needs for you to run the prior steps before seeing the visual results.

### Evaluation Metrics

This repo uses metrics defined by [Ragas](https://docs.ragas.io/en/stable/concepts/metrics/index.html#). Please refer to Ragas Docs for a deeper understanding of the evaluation metrics.

| Metric Name | Metric Definition |
|----------------------------------|-----------------------------------------------------------------------|
| `ragas.metrics.answer_relevancy` | Scores the relevancy of the answer according to the given question. |
| `ragas.metrics.answer_similarity`| Scores the semantic similarity of ground truth with generated answer. |
| `ragas.metrics.answer_correctness`| Measures answer correctness compared to ground truth as a combination of factuality and semantic similarity. |
| `ragas.metrics.context_precision` | Average Precision is a metric that evaluates whether all of the relevant items selected by the model are ranked higher or not. |
| `ragas.metrics.context_recall`    | Estimates context recall by estimating TP and FN using annotated answer and retrieved context. |
| `ragas.metrics.context_entity_recall`| Calculates recall based on entities present in ground truth and context. |

### Rerankers

Rerankers are imported from [Rerankers](https://github.com/AnswerDotAI/rerankers) library. Please make sure to add your API keys for the providers you intend to use. For some HF rerankers, you might have to accept the terms of use on the HF website.

## Customization

### Adding LLMs

A few LLMs are added a starting point, feel free to modify the list or add your own LLM provider of choice in the list below:

```python
llm_options = {
    "Cohere - command-light": lambda: ChatCohere(model_name="command-light", temperature=temperature, max_tokens=max_tokens),
    "Cohere - command": lambda: ChatCohere(model_name="command", temperature=temperature, max_tokens=max_tokens),
    "Cohere - command-r": lambda: ChatCohere(model_name="command-r", temperature=temperature, max_tokens=max_tokens),
    "OpenAI - gpt-3.5-turbo": lambda: ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature, max_tokens=max_tokens),
    "OpenAI - gpt-4-turbo-preview": lambda: ChatOpenAI(model_name="gpt-4-turbo-preview", temperature=temperature, max_tokens=max_tokens),
    "OpenAI - gpt-4": lambda: ChatOpenAI(model_name="gpt-4", temperature=temperature, max_tokens=max_tokens),
    "Anthropic - claude-3-opus-20240229": lambda: ChatAnthropic(model_name="claude-3-opus-20240229", temperature=temperature, max_tokens=max_tokens),
    "Anthropic - claude-3-sonnet-20240229": lambda: ChatAnthropic(model_name="claude-3-sonnet-20240229", temperature=temperature, max_tokens=max_tokens),
    "Anthropic - claude-3-haiku-20240307": lambda: ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=temperature, max_tokens=max_tokens),
    # "Ollama - Gemma": lambda: ChatOllama(model_name="gemma", temperature=temperature, max_tokens=max_tokens),
}
```
Make sure to install any langchain dependencies for the provider and load any necessary api keys in the Home.py file. 

You can also add any embedding models, rerankers, query transformation techniques, etc. 

## Tips

- Feel free to experiment by adding and varying different RAG parameters.
- Use virtual environments to manage dependencies and avoid conflicts.
- Keep your API keys confidential and do not commit them to the repository.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## License

This project is open-sourced under the MIT License. See the `LICENSE` file for more information.
