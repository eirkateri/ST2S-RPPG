import os
import pandas as pd
import yaml


class BinaryClassGenerator:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        self.source_folder = config['data_paths']['source_folder']
        self.save_folder = config['data_paths']['generated_data_folder']
        self.threshold = config.get('classification', {}).get('threshold', 0.5)

    def generate_binary_classes(self):
        # Ensure the output directory exists
        os.makedirs(self.save_folder, exist_ok=True)

        # List all CSV files in the source directory
        csv_files = [file for file in os.listdir(self.source_folder) if file.endswith('.csv')]

        # Process each CSV file
        for file_name in csv_files:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(os.path.join(self.source_folder, file_name))
            df = df.sort_values(by='Absolute Difference', ascending=True)
            df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: 1 if x > self.threshold else 0)
            df = df.iloc[:, [0, -1]]

            df = df.rename(columns={'Absolute Difference': 'Class'})
            new_file_name = file_name.replace('.csv', '_class.csv')
            df.to_csv(os.path.join(self.save_folder, new_file_name), index=False)

        print(f"Binary class generation completed. Files saved to {self.save_folder}.")
