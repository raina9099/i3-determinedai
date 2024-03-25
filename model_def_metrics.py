# This script reports training and testing losses as metrics
# to the Determined master via a Determined core.Context.
# This allows you to view metrics in the WebUI.


# NEW: Import Determined.
import determined as det

import numpy as np
import pandas as pd
import implicit
import scipy.sparse as sparse

def buildData(path="kafka_ratings.csv"):
    data = pd.read_csv(path)
    data.columns = ['timestamp', 'user', 'movie', 'rating']

    movie_id_map = {name: ind for ind, name in enumerate(data['movie'].unique())}
    id_movie_map = {ind: name for ind, name in enumerate(data['movie'].unique())}

    user_id_map = {name: ind for ind, name in enumerate(data['user'].unique())}
    id_user_map = {ind: name for ind, name in enumerate(data['user'].unique())}

    data['movie_id'] = data['movie'].replace(movie_id_map).astype(int)
    data['user_id'] = data['user'].replace(user_id_map).astype(int)

    matrix_user_item = sparse.csr_matrix((data['rating'], (data['user_id'], data['movie_id'])))
    return matrix_user_item, data, user_id_map, id_user_map, id_movie_map, movie_id_map

def evaluate_mse(model, true_matrix):
    prediction_matrix = model.user_factors @ model.item_factors.T * 5
    mse = np.nanmean((true_matrix.toarray() - prediction_matrix) ** 2)
    return mse

def split_train_validation_test(matrix, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    coo_matrix = matrix.tocoo()
    num_ratings = len(coo_matrix.data)

    # Generate a random permutation of indices
    permutation = np.random.permutation(num_ratings)
    num_train_ratings = int(train_ratio * num_ratings)
    num_val_ratings = int(val_ratio * num_ratings)
    num_test_ratings = int(test_ratio * num_ratings)

    # Split the permutation into training and validation indices
    train_indices = permutation[:num_train_ratings]
    val_indices = permutation[num_train_ratings: num_train_ratings+num_val_ratings]
    test_indices = permutation[num_train_ratings+num_val_ratings:]

    # Extract the corresponding user, item, and rating values for training
    train_data = coo_matrix.data[train_indices]
    train_row = coo_matrix.row[train_indices]
    train_col = coo_matrix.col[train_indices]

    train_matrix = sparse.coo_matrix((train_data, (train_row, train_col)), shape=matrix.shape)

    # Extract the corresponding user, item, and rating values for validation
    val_data = coo_matrix.data[val_indices]
    val_row = coo_matrix.row[val_indices]
    val_col = coo_matrix.col[val_indices]

    val_matrix = sparse.coo_matrix((val_data, (val_row, val_col)), shape=matrix.shape)

    # Extract the corresponding user, item, and rating values for testing
    test_data = coo_matrix.data[test_indices]
    test_row = coo_matrix.row[test_indices]
    test_col = coo_matrix.col[test_indices]

    test_matrix = sparse.coo_matrix((test_data, (test_row, test_col)), shape=matrix.shape)

    return train_matrix, val_matrix, test_matrix

# def train(model, train_matrix_csr, core_context):
#     model.fit(train_matrix_csr)
#     train_loss = evaluate_mse(model, train_matrix_csr)
#     core_context.train.report_training_metrics(
#         steps_completed=1,
#         metrics={"train_loss": train_loss},
#     )

def train_one_epoch(model, train_matrix_csr, core_context, epoch_idx):
    model.fit(train_matrix_csr)
    train_loss = evaluate_mse(model, train_matrix_csr)
    core_context.train.report_training_metrics(
        steps_completed=epoch_idx+1,
        metrics={"train_loss": train_loss},
    )

# def test(model, val_matrix_csr, core_context):
#     val_loss = evaluate_mse(model, val_matrix_csr)
#     core_context.train.report_validation_metrics(
#         steps_completed=1,
#         metrics={"val_loss": val_loss},
#     )
    
def test_one_epoch(model, val_matrix_csr, core_context, epoch_idx):
    val_loss = evaluate_mse(model, val_matrix_csr)
    core_context.train.report_validation_metrics(
        steps_completed=epoch_idx+1,
        metrics={"val_loss": val_loss},
    )

def main(core_context):
    matrix_user_item, data, user_id_map, id_user_map, id_movie_map, movie_id_map = buildData()

    # Generating train, test, val splits
    train_matrix, validation_matrix, test_matrix = split_train_validation_test(matrix_user_item, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    train_matrix_csr = train_matrix.tocsr()
    validation_matrix_csr = validation_matrix.tocsr()
    test_matrix_csr = test_matrix.tocsr()

    factors=100
    regularizations=[0.2, 0.1, 0.01, 0.05, 0.001, 0.0001]
    iterations=10
    for idx, regularization in enumerate(regularizations):
        model = implicit.als.AlternatingLeastSquares(factors=factors, regularization=regularization, iterations=iterations, calculate_training_loss=True)
    # matrix_conf = (matrix * alpha_val).astype('float')
        train_one_epoch(model, train_matrix_csr, core_context, idx)
        test_one_epoch(model, validation_matrix_csr, core_context, idx)


# Docs snippet start: modify main loop core context
if __name__ == "__main__":
    # NEW: Establish new determined.core.Context and pass to main
    # function.
    with det.core.init() as core_context:
        main(core_context=core_context)
# Docs snippet end: modify main loop core content