import os
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import cv2
import numpy as np
from tqdm import tqdm
import yaml
from collections import Counter


def run_mlp_experiments():
    def extract_person_id(filename):
        return filename.split('_')[0]

    def extract_features(image_filenames, image_folder, df, label_column):
        features = []
        for filename in tqdm(image_filenames):
            image_path = os.path.join(image_folder, filename)
            img = cv2.imread(image_path)
            if img is None:
                continue
            img_resized = cv2.resize(img, (128, 128))
            hog = cv2.HOGDescriptor()
            features_hog = hog.compute(img_resized).flatten()
            label = df.loc[df['Image Name'] == filename, label_column].iloc[0]
            features.append((features_hog, label))
        return features

    # Load configuration from YAML file
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    experiment_type = config['experiment']['type']
    csv_file = config['data_paths']['csv_file']
    image_folder = config['data_paths']['image_folder']
    results_folder = config['data_paths']['results_folder']

    df = pd.read_csv(csv_file)

    # Extract person IDs
    df['Person ID'] = df['Image Name'].apply(extract_person_id)
    unique_person_ids = df['Person ID'].unique()

    for person_id in unique_person_ids:
        print(f"Processing for Person ID: {person_id}")

        # Load the test person's data
        test_df = pd.read_csv(f'/path/to/your/test/data/MMSE_{person_id}_class.csv')
        train_df = df[df['Person ID'] != person_id]

        if experiment_type == "LOPO":
            train_features = extract_features(train_df['Image Name'].tolist(), image_folder, train_df, 'Class')
            test_features = extract_features(test_df['Image Name'].tolist(), image_folder, test_df, 'Class')

            X_train = np.array([x[0] for x in train_features])
            y_train = np.array([x[1] for x in train_features])
            X_test = np.array([x[0] for x in test_features])
            y_test = np.array([x[1] for x in test_features])

            # Perform 10-fold cross-validation on the LOPO test data
            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            all_predictions = np.zeros((len(X_test), 10), dtype=int)  # Store predictions from each fold

            fold_idx = 0
            for train_index, test_index in kf.split(X_test):
                X_test_fold = X_test[test_index]
                y_test_fold = y_test[test_index]

                # Train the MLP Classifier on the full training set
                classifier = MLPClassifier(solver='sgd', momentum=0.9,
                                           hidden_layer_sizes=(200, 200, 200, 200, 200, 200))
                classifier.fit(X_train, y_train)

                # Make predictions for this fold
                y_pred = classifier.predict(X_test_fold)
                all_predictions[test_index, fold_idx] = y_pred
                fold_idx += 1

            # Majority voting
            final_predictions = []
            for i in range(len(X_test)):
                vote_count = Counter(all_predictions[i, :])
                final_class = vote_count.most_common(1)[0][0]
                final_predictions.append(final_class)

            accuracy = accuracy_score(y_test, final_predictions)
            print(f"LOPO_{person_id} - 10-Fold CV Accuracy with Majority Voting: {accuracy}")

            # Save results
            results_df = pd.DataFrame({'Test Labels': y_test, 'Predictions': final_predictions})
            results_df.to_csv(os.path.join(results_folder, f'{person_id}_LOPO_10fold_majority_predictions.csv'),
                              index=False)

        elif experiment_type == "LOPO_with_Samples":
            train_df_with_samples = train_df.copy()
            for class_label in [0, 1]:
                samples_from_test = test_df[test_df['Class'] == class_label].sample(3)
                train_df_with_samples = train_df_with_samples.append(samples_from_test)

            train_features = extract_features(train_df_with_samples['Image Name'].tolist(), image_folder,
                                              train_df_with_samples, 'Class')
            test_features = extract_features(test_df['Image Name'].tolist(), image_folder, test_df, 'Class')

            X_train = np.array([x[0] for x in train_features])
            y_train = np.array([x[1] for x in train_features])
            X_test = np.array([x[0] for x in test_features])
            y_test = np.array([x[1] for x in test_features])

            # Perform 10-fold cross-validation on the LOPO test data
            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            all_predictions = np.zeros((len(X_test), 10), dtype=int)  # Store predictions from each fold

            fold_idx = 0
            for train_index, test_index in kf.split(X_test):
                X_test_fold = X_test[test_index]
                y_test_fold = y_test[test_index]

                # Train the MLP Classifier on the full training set with samples
                classifier = MLPClassifier(solver='sgd', momentum=0.9,
                                           hidden_layer_sizes=(200, 200, 200, 200, 200, 200))
                classifier.fit(X_train, y_train)

                # Make predictions for this fold
                y_pred = classifier.predict(X_test_fold)
                all_predictions[test_index, fold_idx] = y_pred
                fold_idx += 1

            # Majority voting
            final_predictions = []
            for i in range(len(X_test)):
                vote_count = Counter(all_predictions[i, :])
                final_class = vote_count.most_common(1)[0][0]
                final_predictions.append(final_class)

            accuracy = accuracy_score(y_test, final_predictions)
            print(f"LOPO_with_Samples_{person_id} - 10-Fold CV Accuracy with Majority Voting: {accuracy}")

            # Save results
            results_df = pd.DataFrame({'Test Labels': y_test, 'Predictions': final_predictions})
            results_df.to_csv(
                os.path.join(results_folder, f'{person_id}_LOPO_with_Samples_10fold_majority_predictions.csv'),
                index=False)

        elif experiment_type == "Same_Person":
            # Extract features for the same person
            features = extract_features(test_df['Image Name'].tolist(), image_folder, test_df, 'Class')

            # Prepare data for cross-validation
            X = np.array([x[0] for x in features])
            y = np.array([x[1] for x in features])

            # Perform 10-fold cross-validation
            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            all_predictions = np.zeros((len(X), 10), dtype=int)  # Store predictions from each fold

            fold_idx = 0
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Train the MLP Classifier
                classifier = MLPClassifier(solver='sgd', momentum=0.9,
                                           hidden_layer_sizes=(200, 200, 200, 200, 200, 200))
                classifier.fit(X_train, y_train)

                # Make predictions for this fold
                y_pred = classifier.predict(X_test)
                all_predictions[test_index, fold_idx] = y_pred
                fold_idx += 1

            # Majority voting
            final_predictions = []
            for i in range(len(X)):
                vote_count = Counter(all_predictions[i, :])
                final_class = vote_count.most_common(1)[0][0]
                final_predictions.append(final_class)

            accuracy = accuracy_score(y, final_predictions)
            print(f"Same_Person_{person_id} - 10-Fold CV Accuracy with Majority Voting: {accuracy}")

            # Save results
            results_df = pd.DataFrame({'Test Labels': y, 'Predictions': final_predictions})
            results_df.to_csv(os.path.join(results_folder, f'{person_id}_Same_Person_10fold_majority_predictions.csv'),
                              index=False)
