import random
import time
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from ADGSyn.utils.util import DemoDataset, demo_collate
from model.DGAT import DemoDualStreamModel


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


DEMO_HYPERPARAM_SPACE = {
    'learning_rate': [0.001, 0.0005],
    'batch_size': [32, 64],
    'num_epochs': [10, 20],
    'dropout_rate': [0.1, 0.2]
}


def generate_demo_hyperparams():
    return {
        'lr': random.choice(DEMO_HYPERPARAM_SPACE['learning_rate']),
        'batch_size': random.choice(DEMO_HYPERPARAM_SPACE['batch_size']),
        'num_epochs': random.choice(DEMO_HYPERPARAM_SPACE['num_epochs']),
        'dropout': random.choice(DEMO_HYPERPARAM_SPACE['dropout_rate'])
    }


def demo_train_epoch(model, device, data_loader, optimizer):
    model.train()
    total_loss = 0
    for batch_idx, (data1, data2, y) in enumerate(data_loader):
        data1, data2, y = data1.to(device), data2.to(device), y.to(device)
        optimizer.zero_grad()
        
        output = model(data1, data2)
        loss = nn.CrossEntropyLoss()(output, y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx >= 5:
            break
    
    return total_loss / (batch_idx + 1)


def demo_evaluate_model(model, device, data_loader):
    model.eval()
    total_preds = []
    total_labels = []

    with torch.no_grad():
        for batch_idx, (data1, data2, y) in enumerate(data_loader):
            data1, data2 = data1.to(device), data2.to(device)
            output = model(data1, data2)
            preds = torch.softmax(output, dim=1).cpu().numpy()
            total_preds.extend(preds[:, 1])
            total_labels.extend(y.cpu().numpy())
            
            if batch_idx >= 3:
                break

    return np.array(total_labels), np.array(total_preds)


def demo_train_and_validate(params, device, dataset, num_folds=3):
    fold_accuracies = []
    dataset_size = len(dataset)
    fold_size = dataset_size // num_folds

    for fold in range(num_folds):
        print(f"Training fold {fold + 1}...")
        
        indices = list(range(dataset_size))
        test_indices = indices[fold * fold_size: (fold + 1) * fold_size]
        train_indices = [i for i in indices if i not in test_indices]

        train_data = dataset.get_subset(train_indices)
        test_data = dataset.get_subset(test_indices)

        train_loader = DataLoader(train_data, batch_size=params['batch_size'],
                                  shuffle=True, collate_fn=demo_collate)
        test_loader = DataLoader(test_data, batch_size=params['batch_size'],
                                 shuffle=False, collate_fn=demo_collate)

        model = DemoDualStreamModel(dropout=params['dropout']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])

        best_acc = 0.0

        for epoch in range(params['num_epochs']):
            train_loss = demo_train_epoch(model, device, train_loader, optimizer)
            
            if epoch % 5 == 0:
                true_labels, pred_probs = demo_evaluate_model(model, device, test_loader)
                predicted_labels = (pred_probs > 0.5).astype(int)
                current_acc = accuracy_score(true_labels, predicted_labels)

                if current_acc > best_acc:
                    best_acc = current_acc

                print(f"Epoch {epoch}: Loss={train_loss:.4f}, Acc={current_acc:.4f}")

        fold_accuracies.append(best_acc)

    return np.mean(fold_accuracies)


def save_demo_records(records, filename='demo_search.log'):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Demo Hyperparameter Search Records\n")
        f.write("=" * 50 + "\n")
        for i, record in enumerate(records):
            line = f"Iteration {i+1}: Accuracy={record['score']:.4f} | Params={record['params']} | Time={record['time']:.1f}s\n"
            f.write(line)


if __name__ == "__main__":
    print("Starting Demo Dual-Stream Attention Model Training...")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        dataset = DemoDataset()
        print(f"Dataset size: {len(dataset)}")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Creating mock dataset...")
        dataset = None

    if dataset is None:
        print("Cannot load dataset, exiting...")
        exit()

    best_params = None
    best_score = 0.0
    search_records = []

    for iteration in range(5):
        print(f"\n=== Demo Hyperparameter Search Iteration {iteration + 1}/5 ===")

        current_params = generate_demo_hyperparams()
        print("Current parameters:", current_params)

        start_time = time.time()
        try:
            avg_accuracy = demo_train_and_validate(current_params, device, dataset)
            duration = time.time() - start_time

            search_records.append({
                'params': current_params,
                'score': avg_accuracy,
                'time': duration
            })

            if avg_accuracy > best_score:
                best_score = avg_accuracy
                best_params = current_params
                print(f"New best parameters found! Accuracy: {avg_accuracy:.4f}")

            print(f"Current best accuracy: {best_score:.4f}")
            print(f"Current best parameters: {best_params}")
            
        except Exception as e:
            print(f"Error during training: {e}")
            continue

    print("\n=== Demo Hyperparameter Search Completed ===")
    print(f"Best validation accuracy: {best_score:.4f}")
    print("Best parameter combination:")
    if best_params:
        for key, value in best_params.items():
            print(f"  {key}: {value}")

    save_demo_records(search_records)
    print("Demo training completed!")
