import numpy as np
import pandas as pd
import progressbar


class DataLoader:
    def __init__(self, paths, settings):
        self.labels = None
        self.train_df = None
        self.labels_df = None
        self.train_terms_limited = None
        self.test_embeddings = None
        self.test_protein_ids = None
        self.train_embeddings = None
        self.train_protein_ids = None
        self.train_terms = None
        self.dataset_path = paths.raw_dataset
        self.num_of_labels = settings.num_of_labels

    def import_data(self):
        self._load_train_data()
        self._convert_embeddings_to_df()
        self._limit_goterm_labels()
        self._label_onehot_encoding()

        return self.train_df, self.labels_df

    def _load_train_data(self):
        self.train_terms = pd.read_csv(self.dataset_path + "/Train/train_terms.tsv", sep="\t")
        self.train_protein_ids = np.load(self.dataset_path + '/t5embeds/train_ids.npy')
        self.train_embeddings = np.load(self.dataset_path + '/t5embeds/train_embeds.npy')

    def load_test_data(self):
        self.test_protein_ids = np.load(self.dataset_path + '/t5embeds/test_ids.npy')
        self.test_embeddings = np.load(self.dataset_path + '/t5embeds/test_embeds.npy')

        column_num = self.test_embeddings.shape[1]
        self.test_df = pd.DataFrame(self.test_embeddings,
                                    columns=["Column_" + str(i) for i in range(1, column_num + 1)])
        return self.test_df

    def _convert_embeddings_to_df(self):
        print("----------- Convert embeddings to Data Frame ----------- ")
        column_num = self.train_embeddings.shape[1]
        self.train_df = pd.DataFrame(self.train_embeddings,
                                     columns=["Column_" + str(i) for i in range(1, column_num + 1)])
        return self.train_df

    def _limit_goterm_labels(self):
        print("----------- limitting Go Terms ----------- ")
        # Take value counts in descending order and fetch first 1500 `GO term ID` as labels
        self.labels = self.train_terms['term'].value_counts().index[:self.num_of_labels].tolist()
        # Fetch the train_terms data for the relevant labels only
        self.train_terms_limited = self.train_terms.loc[self.train_terms['term'].isin(self.labels)]

    def _label_onehot_encoding(self):
        print("----------- OneHot Encoding ----------- ")
        # Setup progressbar settings.
        # This is strictly for aesthetic.
        bar = progressbar.ProgressBar(maxval=self.num_of_labels, \
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

        # Create an empty dataframe of required size for storing the labels,
        # i.e, train_size x num_of_labels (142246 x 1500)
        train_size = self.train_protein_ids.shape[0]  # len(X)
        self.train_labels = np.zeros((train_size, self.num_of_labels))

        # Convert from numpy to pandas series for better handling
        series_train_protein_ids = pd.Series(self.train_protein_ids)

        # Loop through each label
        bar.start()
        for i in range(self.num_of_labels):
            # For each label, fetch the corresponding train_terms data
            n_train_terms = self.train_terms_limited[self.train_terms_limited['term'] == self.labels[i]]

            # Fetch all the unique EntryId aka proteins related to the current label(GO term ID)
            label_related_proteins = n_train_terms['EntryID'].unique()

            # In the series_train_protein_ids pandas series, if a protein is related
            # to the current label, then mark it as 1, else 0.
            # Replace the ith column of train_Y with with that pandas series.
            self.train_labels[:, i] = series_train_protein_ids.isin(label_related_proteins).astype(float)

            # Progress bar percentage increase
            bar.update(i + 1)

        # Notify the end of progress bar
        bar.finish()

        self.labels_df = pd.DataFrame(data=self.train_labels, columns=self.labels)

        return self.labels_df
