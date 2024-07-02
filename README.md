<h3>Text Summarization</h3>
Abstract text summarization involves generating a concise and coherent summary that captures the core ideas and essential points of a longer text. Unlike extractive summarization, which selects and concatenates sentences or phrases directly from the source document, abstract summarization creates new sentences that convey the main information in a more fluid and readable manner. This process can be particularly challenging as it requires a deeper understanding of the content, context, and nuances of the original text.
<h3>Table of Contents</h3>
<ul>
    <li>Installation</li>
    <li>Dataset</li>
    <li>Preprocessing</li>
    <li>Model Training</li>
    <li>Evaluation</li>
    <li>Results</li>
    <li>Usage</li>
    <li>Contributing</li>
    <li>License</li>
</ul>

<h3>Installation</h3>
Before running the notebook, ensure that you have the following libraries installed: <br>

```bash
pip install accelerate -U
pip install transformers[sentencepiece] datasets sacrebleu rouge_score py7zr
```
<h3>Additional dependencies:</h3>
<ul>
    <li>nltk</li>
    <li>torch</li>
    <li>numpy</li>
    <li>pandas</li>
    <li>matplotlib</li>
    <li>tqdm</li>
</ul>

<h3>Dataset</h3>
The dataset used in this notebook is stored in a CSV file located on Google Drive. It is loaded into the notebook with the following command: <br>

```bash
dataset = pd.read_csv("/content/drive/MyDrive/summary/Datasets.csv")
dataset.rename(columns={'Task Description': 'dialogue', 'Summary': 'summary'}, inplace=True)
```
<h3>Preprocessing</h3>
Preprocessing steps include: <br>

<ul>
  <li>Tokenization</li>
  <li>Lemmatization</li>
  <li>Stopwords removal</li>
</ul>
The nltk library is used for these preprocessing tasks. Ensure the necessary NLTK packages are downloaded: <br>

```bash
nltk.download('punkt')
nltk.download('wordnet')
# nltk.download('stopwords')
```
<h3>Model Training</h3>
The notebook uses a pre-trained transformer model for sequence-to-sequence learning from the Hugging Face transformers library. The key components include:<br>
<ul>
  <li>AutoModelForSeq2SeqLM</li>
  <li>AutoTokenizer</li>
  <li>DataCollatorForSeq2Seq</li>
  <li>TrainingArguments</li>
  <li>Trainer</li>
</ul>
<h3>Training Loss</h3>

![Training and Validation Loss](https://github.com/Bikas0/Text-Summarization/assets/66817101/ae1daab4-43c7-4cb9-abc1-8fd4371baabb)

The graph depicts the training and validation loss over the course of training a machine learning model. Understanding this graph is crucial for diagnosing the performance and learning behavior of the model.

<h5>Key Components of the Graph</h5>
<ul>
  <li>X-axis (Steps): Represents the number of training iterations or steps.</li>
  <li>Y-axis (Loss): Represents the loss value, which is a measure of how well the model's predictions match the target values.</li>
  <li>Training Loss (Blue Line): Training Loss (Blue Line): Indicates the error on the training dataset. It is the loss calculated on the same data that the model was trained on.</li>
  <li>Validation Loss (Red Line): Indicates the error on the validation dataset. It is the loss calculated on data that was not used for training but for validating the model's performance.</li>
</ul>

<h3>Evaluation</h3>
Evaluation metrics used in this notebook include:<br>
<ul>
  <li>ROUGE</li>
  <li>BLEU</li>
  <li>SacreBLEU</li>
</ul>
These metrics are used to assess the quality of the generated summaries.
<h3>ROUGE</h3>
ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a set of metrics used for evaluating automatic summarization and machine translation models. It compares the overlap between the words, phrases, and sequences in the generated summary and a set of reference summaries (usually human-written). <br>

![rouge_scores](https://github.com/Bikas0/Text-Summarization/assets/66817101/abae47bc-9163-4fe5-90cd-1a016f3f97be)

<h3>Results</h3>
The results of the model training and evaluation are visualized using matplotlib. Performance metrics and loss curves are plotted to provide insights into the model's learning process.

<h3>Usage</h3>
To use this notebook:
<ul>
  <li>Clone the repository.</li>
  <li>Ensure all dependencies are installed.</li>
  <li>Load the dataset from your Google Drive.</li>
  <li>Run the notebook cells sequentially.</li>
  <li>Contributing</li>
  <li>Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.</li>
</ul>

<h3>License</h3>
This project is licensed under the MIT License.



