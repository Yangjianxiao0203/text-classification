from datasets import load_metric
from sklearn.metrics import confusion_matrix,f1_score,recall_score,precision_score,accuracy_score
import json
import torch

class Evaluator:
    def __init__(self,model,config):
        self.config = config
        self.model = model
        self.metric = load_metric('accuracy')
        self.debug = config["debug_mode"]

    @torch.no_grad()
    def eval(self,dataloader):
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        all_predictions = []
        all_true_labels = []
        for x,y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_pred = self.model(x)
            y_pred = torch.argmax(y_pred,dim=-1)
            all_predictions.extend(y_pred.cpu().numpy())
            all_true_labels.extend(y.cpu().numpy())
            if self.debug:
                break

        self.metric.compute(predictions=all_predictions, references=all_true_labels)
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

        return results
