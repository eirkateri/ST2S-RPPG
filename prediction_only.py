import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Load configuration from YAML file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

class CnnClassifier(nn.Module):
    def __init__(self, pretrained_model):
        super(CnnClassifier, self).__init__()
        self.pretrained_model = pretrained_model
        self.pretrained_model.output_layer = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pretrained_model(x)
        x = nn.functional.softmax(x, dim=1)
        return x

if __name__ == "__main__":
    # Load the pretrained model
    pretrained_model = torch.load(config['data_paths']['model_data'] + "/model.h")
    model = CnnClassifier(pretrained_model)
    model.eval()

    # Load the test dataset
    df = pd.read_csv(config['data_paths']['target_csv'])
    test_loader = DataLoader(df, batch_size=config['training']['batch_size'], shuffle=False)

    results = []
    for inputs, targets in tqdm(test_loader):
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        results.extend(preds.cpu().numpy())

    # Save the results to a CSV file
    result_df = pd.DataFrame({"Predicted Class": results})
    result_df.to_csv(config['output']['results_csv_path'], index=False)