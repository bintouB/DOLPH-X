import numpy as np

class XAIEvaluator:
    """
    Base class shared by LIME, MC-LIME, SHAP and DiCE evaluators.
    Centralises shared logic: instance selection, common attributes,
    and utility methods.
    """

    # Models retained for XAI analysis
    SELECTED_MODELS  = ['Random_Forest', 'XGBoost', 'Decision_Tree']

    # Consensus filter uses only the two best-performing models
    CONSENSUS_MODELS = ['Random_Forest', 'XGBoost']

    # Label encoding order
    CLASS_NAMES = ['E+P+', 'E+P-', 'E-P+', 'E-P-']

    def __init__(self, evaluator):
        """
        Initialise the XAIEvaluator from a trained MLEvaluator instance.

        Args:
            evaluator : trained MLEvaluator containing models and datasets
        """
        self.evaluator       = evaluator
        self.X_train         = evaluator.X_train
        self.X_test          = evaluator.X_test
        self.y_test          = evaluator.y_test
        self.y_train         = evaluator.y_train
        self.feature_names   = evaluator.feature_names
        self.all_predictions = evaluator.all_predictions
        self.trained_models  = {
            k: v for k, v in evaluator.trained_models.items()
            if k in self.SELECTED_MODELS
        }
        self.selected_samples = {}

        # Build class mapping from encoded labels to profile names
        unique_labels = sorted(self.y_test.unique())
        self.class_mapping = {
            code: self.CLASS_NAMES[i]
            for i, code in enumerate(unique_labels)
        }
        self.reverse_mapping = {v: k for k, v in self.class_mapping.items()}
        self.classes = self.CLASS_NAMES


    def select_samples(self):
        """
        Selects representative instances for XAI analysis.
        Consensus filter applied on CONSENSUS_MODELS only (RF + XGBoost).
        Up to 4 instances are selected per profile using random_state=42.
        This method is shared by LIME, MC-LIME, SHAP and DiCE.
        """
        print('\nSELECTING REPRESENTATIVE INSTANCES')
        print('=' * 50)
        print(f'Class mapping: {self.class_mapping}')

        for encoded_class, class_name in self.class_mapping.items():
            class_indices = self.y_test[
                self.y_test == encoded_class
            ].index.tolist()

            # Keep only instances correctly predicted by all consensus models
            correctly_predicted = []
            for idx in class_indices:
                pos = list(self.y_test.index).index(idx)
                if all(
                    preds[pos] == encoded_class
                    for model_name, preds in self.all_predictions.items()
                    if model_name in self.CONSENSUS_MODELS
                ):
                    correctly_predicted.append(pos)

            print(f'\n{class_name}: {len(correctly_predicted)} consensus '
                  f'/ {len(class_indices)} total')

            if len(correctly_predicted) >= 4:
                np.random.seed(42)
                selected = np.random.choice(
                    correctly_predicted, 4, replace=False
                ).tolist()
                print('  -> 4 instances selected')
            elif correctly_predicted:
                selected = correctly_predicted
                print(f'  -> {len(selected)} instance(s) available (minority profile)')
            else:
                selected = []
                print('  -> No valid instances')

            self.selected_samples[class_name] = selected

        total = sum(len(v) for v in self.selected_samples.values())
        print(f'\nTotal selected: {total} instances')
