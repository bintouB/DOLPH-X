import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings('ignore')


class MLEvaluator:
    """
    Trains and evaluates 11 classification models on the DOLPH-X dataset.
    Handles data preparation, model initialisation, training, evaluation,
    and serialisation for downstream XAI notebooks.
    """

    def __init__(self, train_df, test_df):
        """
        Initialise the evaluator with training and test datasets.

        Args:
            train_df : training DataFrame (after SMOTE balancing)
            test_df  : test DataFrame (original, unmodified)
        """
        self.train_df        = train_df
        self.test_df         = test_df
        self.models          = {}   # untrained model instances
        self.trained_models  = {}   # trained model instances
        self.results         = None # evaluation results DataFrame
        self.feature_names   = []   # feature column names
        self.X_train         = None
        self.X_test          = None
        self.y_train         = None
        self.y_test          = None
        self.all_predictions = {}   # per-model predictions on the test set

    def prepare_data(self):
        """
        Separates features (X) from target (y) for train and test sets.
        Retains only numeric columns and fills missing values with 0.
        """
        print('Preparing data...')

        train_df = self.train_df.copy()
        test_df  = self.test_df.copy()

        target_col = 'DIFFICULTY_encoded'
        if target_col not in train_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")

        self.X_train = train_df.drop(columns=[target_col])
        self.y_train = train_df[target_col]
        self.X_test  = test_df.drop(columns=[target_col])
        self.y_test  = test_df[target_col]

        # Retain only numeric columns
        self.X_train = self.X_train.select_dtypes(include=[np.number])
        self.X_test  = self.X_test.select_dtypes(include=[np.number])

        # Fill missing values with 0
        self.X_train = self.X_train.fillna(0)
        self.X_test  = self.X_test.fillna(0)

        self.feature_names = list(self.X_train.columns)

        print(f'Data ready')
        print(f'  Features : {len(self.feature_names)} columns')
        print(f'  Train    : {self.X_train.shape[0]} rows')
        print(f'  Test     : {self.X_test.shape[0]} rows')
        print(f'  Classes  : {sorted(self.y_train.unique())}')

    def initialize_models(self):
        """
        Initialises 11 classification models with their configurations.
        Models are not yet trained at this stage.
        """
        print('\nInitialising models...')

        self.models = {
            'SVM': SVC(
                random_state=42,
                probability=True,   # required for AUC-ROC computation
                kernel='rbf',       # non-linear decision boundary
                C=1.0               # regularisation strength
            ),
            'Logistic_Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,      # maximum iterations for convergence
                solver='liblinear'  # efficient solver for small datasets
            ),
            'Decision_Tree': DecisionTreeClassifier(
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5,
                weights='uniform'   # all neighbours weighted equally
            ),
            'Naive_Bayes': GaussianNB(),
            'Random_Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            'Extra_Trees': ExtraTreesClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            'Bagging': BaggingClassifier(
                base_estimator=DecisionTreeClassifier(max_depth=8),
                n_estimators=50,
                random_state=42
            ),
            'Gradient_Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=42,
                early_stopping=True,        # stop if no improvement
                validation_fraction=0.1     # 10% of train for validation
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42,
                eval_metric='mlogloss',     # multi-class log loss
                use_label_encoder=False
            ),
        }

        print(f'{len(self.models)} models initialised')

    def evaluate_model(self, model, model_name):
        """
        Evaluates a trained model and returns its performance metrics.

        Args:
            model      : trained model instance
            model_name : model name (string)
        Returns:
            dict : accuracy, precision, recall, F1-score, AUC-ROC, predictions
        """
        y_pred = model.predict(self.X_test)

        # Prediction probabilities for AUC-ROC
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(self.X_test)
        elif hasattr(model, 'decision_function'):
            y_pred_proba = model.decision_function(self.X_test)

        accuracy  = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
        recall    = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
        f1        = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)

        # AUC-ROC (multi-class One-vs-Rest)
        auc_roc = None
        if y_pred_proba is not None:
            try:
                if len(np.unique(self.y_test)) == 2:
                    auc_roc = roc_auc_score(self.y_test, y_pred_proba[:, 1])
                else:
                    auc_roc = roc_auc_score(
                        self.y_test,
                        y_pred_proba,
                        multi_class='ovr',
                        average='weighted'
                    )
            except Exception as e:
                print(f'AUC-ROC could not be computed for {model_name}: {e}')
                auc_roc = None

        return {
            'Model'    : model_name,
            'Accuracy' : accuracy,
            'Precision': precision,
            'Recall'   : recall,
            'F1_Score' : f1,
            'AUC_ROC'  : auc_roc,
            'y_pred'   : y_pred,
        }

    def train_and_evaluate_all(self):
        """
        Main pipeline method:
        1. Prepare data
        2. Initialise models
        3. Train and evaluate each model
        4. Return a DataFrame of results sorted by F1-Score
        The trained evaluator is automatically saved to evaluator.pkl.
        """
        print('Starting model evaluation pipeline...')
        print('=' * 60)

        self.prepare_data()
        self.initialize_models()

        all_results = []

        for model_name, model in self.models.items():
            print(f'\nTraining {model_name}...')
            try:
                model.fit(self.X_train, self.y_train)
                self.trained_models[model_name] = model
                self.all_predictions[model_name] = model.predict(self.X_test)

                result = self.evaluate_model(model, model_name)
                all_results.append(result)

                print(f'  Accuracy  : {result["Accuracy"]:.4f}')
                print(f'  Precision : {result["Precision"]:.4f}')
                print(f'  Recall    : {result["Recall"]:.4f}')
                print(f'  F1-Score  : {result["F1_Score"]:.4f}')
                if result['AUC_ROC'] is not None:
                    print(f'  AUC-ROC   : {result["AUC_ROC"]:.4f}')

            except Exception as e:
                print(f'Error with {model_name}: {e}')
                continue

        results_df = pd.DataFrame(all_results)
        results_df = results_df.drop(columns=['y_pred'])
        results_df = results_df.sort_values('F1_Score', ascending=False)
        results_df = results_df.reset_index(drop=True)

        self.results = results_df

        # Auto-save the trained evaluator
        self.save()

        print('\n' + '=' * 60)
        print('FINAL RESULTS')
        print('=' * 60)
        print(results_df.to_string(index=False))
        return results_df

    def save(self, filepath='evaluator.pkl'):
        """
        Serialises the entire evaluator (trained models + datasets)
        for reuse in XAI notebooks.

        Args:
            filepath : output file path (default: evaluator.pkl)
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Evaluator saved to '{filepath}'")

    @staticmethod
    def load(filepath='evaluator.pkl'):
        """
        Loads a previously saved evaluator.

        Args:
            filepath : path to the pickle file
        Returns:
            MLEvaluator instance with all trained models loaded
        """
        with open(filepath, 'rb') as f:
            evaluator = pickle.load(f)
        print(f"Evaluator loaded from '{filepath}'")
        print(f'  Available models: {list(evaluator.trained_models.keys())}')
        return evaluator
