import numpy as np
import pandas as pd

class ResultGenerator():
    def __init__(self, model, data_loader, settings, paths):
        self.path_results = paths.path_result[0]
        self.data_loader = data_loader
        self.settings = settings
        self.model = model

    def get_output_csv(self):
        """------------- Load test data --------------"""
        test_df = self.data_loader.load_test_data()
        labels = self.data_loader.labels
        """------------- Predict test data --------------"""
        predictions = self.model.predict(test_df)
        """------------- Save Results --------------"""
        df_submission = pd.DataFrame(columns=['Protein Id', 'GO Term Id', 'Prediction'])
        l = []
        for k in list(self.data_loader.test_protein_ids):
            l += [k] * predictions.shape[1]

        df_submission['Protein Id'] = l
        df_submission['GO Term Id'] = labels * predictions.shape[0]
        df_submission['Prediction'] = predictions.ravel()
        df_submission.to_csv(self.path_results + "submission.tsv", header=False, index=False, sep="\t")