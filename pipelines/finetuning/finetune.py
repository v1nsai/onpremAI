from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import random
import PyPDF2
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'text_pdf_preprocessing_and_finetuning',
    default_args=default_args,
    description='A DAG to preprocess text and pdf documents and fine-tune a pre-trained model',
    schedule_interval=timedelta(days=1),
)

def extract_text_from_documents(file_paths):
    texts = []
    for file_path in file_paths:
        if file_path.endswith('.txt'):
            with open(file_path, 'r') as file:
                texts.append(file.read())
        elif file_path.endswith('.pdf'):
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfFileReader(file)
                num_pages = reader.numPages
                text = ''
                for page in range(num_pages):
                    text += reader.getPage(page).extract_text()
                texts.append(text)
    return texts

def preprocess_text(texts):
    # Example preprocessing: Lowercasing and removing special characters
    processed_texts = [text.lower().replace('\n', ' ') for text in texts]
    return processed_texts

def create_batches(texts, batch_size=16):
    random.shuffle(texts)
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    return batches

def fine_tune_model(batches, model_path, epochs=3):
    tokenizer = AutoTokenizer.from_pretrained('TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF')
    model = AutoModelForCausalLM.from_pretrained('TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF')

    optimizer = AdamW(model.parameters(), lr=2e-5)

    model.train()
    for epoch in range(epochs):
        for batch in batches:
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
            labels = inputs.input_ids
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    model.save_pretrained(model_path)

def run_pipeline():
    # Define the file paths (in practice, get these from a data source or Airflow variables)
    file_paths = ['path/to/doc1.txt', 'path/to/doc2.pdf']

    # Step 1: Extract text from documents
    texts = extract_text_from_documents(file_paths)

    # Step 2: Preprocess text
    processed_texts = preprocess_text(texts)

    # Step 3: Create batches
    batches = create_batches(processed_texts)

    # Step 4: Fine-tune the model
    fine_tune_model(batches, 'path/to/save/model')

# Define tasks
extract_text_task = PythonOperator(
    task_id='extract_text',
    python_callable=extract_text_from_documents,
    op_args=[['path/to/doc1.txt', 'path/to/doc2.pdf']],
    dag=dag,
)

preprocess_text_task = PythonOperator(
    task_id='preprocess_text',
    python_callable=preprocess_text,
    op_args=[[extract_text_task.output]],
    dag=dag,
)

create_batches_task = PythonOperator(
    task_id='create_batches',
    python_callable=create_batches,
    op_args=[[preprocess_text_task.output]],
    dag=dag,
)

fine_tune_model_task = PythonOperator(
    task_id='fine_tune_model',
    python_callable=fine_tune_model,
    op_args=[[create_batches_task.output, 'path/to/save/model']],
    dag=dag,
)

# Define task dependencies
extract_text_task >> preprocess_text_task >> create_batches_task >> fine_tune_model_task
