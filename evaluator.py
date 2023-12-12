from datasets import load_metric
from sklearn.metrics import confusion_matrix,f1_score,recall_score,precision_score,accuracy_score
import json
import torch
from transformers import BertTokenizer


class ML_Evaluator:

    def __init__(self, model, tokenizer):
        self.model = model
        self.metric = load_metric('accuracy')
        self.tokenizer = tokenizer

    def evaluate(self, dataloader):
        all_predictions = []
        all_true_labels = []

        for x,y in dataloader:
            inputs = x
            labels = y
            predictions = self.model.predict(inputs)
            all_predictions.extend(predictions)
            all_true_labels.extend(labels)

        # 计算评价指标
        self.metric.compute(predictions=all_predictions, references=all_true_labels)

        # 生成混淆矩阵
        cm = confusion_matrix(all_true_labels, all_predictions)
        f1 = f1_score(all_true_labels, all_predictions, average='macro')
        recall = recall_score(all_true_labels, all_predictions, average='macro')
        precision = precision_score(all_true_labels, all_predictions, average='macro')
        accuracy = accuracy_score(all_true_labels, all_predictions)
        # Combine metrics in a dictionary
        results = {
            "confusion_matrix": cm.tolist(),  # Convert numpy array to list for serialization
            "f1_score": f1,
            "recall": recall,
            "precision": precision,
            "accuracy": accuracy
        }

        # Convert the dictionary to a JSON string
        results_json = json.dumps(results)

        return results_json


class BertEvaluator:
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def evaluate(self, dataloader):
        self.model.eval()
        all_predictions = []
        all_true_labels = []

        with torch.no_grad():
            for x,y in dataloader:
                inputs = x.to(self.device)
                labels = y.to(self.device)

                outputs = self.model(inputs)
                if not isinstance(outputs, torch.Tensor):
                    outputs = outputs.logits
                _, predicted = torch.max(outputs.data, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_true_labels.extend(labels.cpu().numpy())

        f1 = f1_score(all_true_labels, all_predictions, average='macro')
        recall = recall_score(all_true_labels, all_predictions, average='macro')
        precision = precision_score(all_true_labels, all_predictions, average='macro')
        accuracy = accuracy_score(all_true_labels, all_predictions)
        # Combine metrics in a dictionary
        results = {
            "f1_score": f1,
            "recall": recall,
            "precision": precision,
            "accuracy": accuracy
        }

        return results

