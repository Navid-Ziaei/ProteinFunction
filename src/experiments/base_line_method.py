import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.visualization import plot_go_term_hist, plot_aspects_pie_plot
# Required for progressbar widget
import progressbar

train_terms = pd.read_csv("D:/Navid/Projects2/ProteinFunction/data/Train/train_terms.tsv", sep="\t")
print(train_terms.shape)

train_protein_ids = np.load('D:/Navid/Projects2/ProteinFunction/data/t5embeds/train_ids.npy')
train_embeddings = np.load('D:/Navid/Projects2/ProteinFunction/data/t5embeds/train_embeds.npy')

print(train_protein_ids.shape)
# Now lets convert embeddings numpy array(train_embeddings) into pandas dataframe.
column_num = train_embeddings.shape[1]
train_df = pd.DataFrame(train_embeddings, columns=["Column_" + str(i) for i in range(1, column_num + 1)])
print(train_df.shape)

plot_go_term_hist(train_terms, show_plot=False, save_fig=True, save_path='')

# Set the limit for label
num_of_labels = 1500

# Take value counts in descending order and fetch first 1500 `GO term ID` as labels
labels = train_terms['term'].value_counts().index[:num_of_labels].tolist()

# Fetch the train_terms data for the relevant labels only
train_terms_updated = train_terms.loc[train_terms['term'].isin(labels)]

plot_aspects_pie_plot(train_terms_updated)

# Setup progressbar settings.
# This is strictly for aesthetic.
bar = progressbar.ProgressBar(maxval=num_of_labels, \
                              widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

# Create an empty dataframe of required size for storing the labels,
# i.e, train_size x num_of_labels (142246 x 1500)
train_size = train_protein_ids.shape[0]  # len(X)
train_labels = np.zeros((train_size, num_of_labels))

# Convert from numpy to pandas series for better handling
series_train_protein_ids = pd.Series(train_protein_ids)

# Loop through each label
bar.start()
for i in range(num_of_labels):
    # For each label, fetch the corresponding train_terms data
    n_train_terms = train_terms_updated[train_terms_updated['term'] == labels[i]]

    # Fetch all the unique EntryId aka proteins related to the current label(GO term ID)
    label_related_proteins = n_train_terms['EntryID'].unique()

    # In the series_train_protein_ids pandas series, if a protein is related
    # to the current label, then mark it as 1, else 0.
    # Replace the ith column of train_Y with with that pandas series.
    train_labels[:, i] = series_train_protein_ids.isin(label_related_proteins).astype(float)

    # Progress bar percentage increase
    bar.update(i + 1)

# Notify the end of progress bar
bar.finish()

# Convert train_Y numpy into pandas dataframe
labels_df = pd.DataFrame(data=train_labels, columns=labels)
print(labels_df.shape)

INPUT_SHAPE = [train_df.shape[1]]
BATCH_SIZE = 5120

model = tf.keras.Sequential([
    tf.keras.layers.BatchNormalization(input_shape=INPUT_SHAPE),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=512, activation='relu'),
    tf.keras.layers.Dense(units=num_of_labels, activation='sigmoid')
])

# Compile models
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['binary_accuracy', tf.keras.metrics.AUC()],
)

history = model.fit(
    train_df, labels_df,
    batch_size=BATCH_SIZE,
    epochs=5
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss']].plot(title="Cross-entropy")
history_df.loc[:, ['binary_accuracy']].plot(title="Accuracy")
plt.show()

test_embeddings = np.load('D:/Navid/Projects2/ProteinFunction/data/t5embeds/test_embeds.npy')

# Convert test_embeddings to dataframe
column_num = test_embeddings.shape[1]
test_df = pd.DataFrame(test_embeddings, columns=["Column_" + str(i) for i in range(1, column_num + 1)])
print(test_df.shape)

predictions = model.predict(test_df)
df_submission = pd.DataFrame(columns=['Protein Id', 'GO Term Id', 'Prediction'])
test_protein_ids = np.load('D:/Navid/Projects2/ProteinFunction/data/t5embeds/test_ids.npy')
l = []
for k in list(test_protein_ids):
    l += [k] * predictions.shape[1]

df_submission['Protein Id'] = l
df_submission['GO Term Id'] = labels * predictions.shape[0]
df_submission['Prediction'] = predictions.ravel()
df_submission.to_csv("submission.tsv", header=False, index=False, sep="\t")
