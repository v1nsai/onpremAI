from airflow import DAG
from airflow.utils.dates import days_ago
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from airflow.operators.python import PythonOperator

# Define the default arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

# Define the DAG
dag = DAG(
    'pytorch_model_training',
    default_args=default_args,
    description='A simple DAG to train a PyTorch model',
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
)

# Define the functions for the tasks
def preprocess_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    torch.save(train_dataset, '/tmp/train_dataset.pth')
    torch.save(test_dataset, '/tmp/test_dataset.pth')

def train_model():
    train_dataset = torch.load('/tmp/train_dataset.pth')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(28*28, 128)
            self.fc2 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = x.view(-1, 28*28)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(5):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), '/tmp/model.pth')

def evaluate_model():
    test_dataset = torch.load('/tmp/test_dataset.pth')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    class SimpleNN(nn.Module):
        def __init__(self):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(28*28, 128)
            self.fc2 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = x.view(-1, 28*28)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = SimpleNN()
    model.load_state_dict(torch.load('/tmp/model.pth'))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')

# Define the tasks
preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

# Set task dependencies
preprocess_data_task >> train_model_task >> evaluate_model_task