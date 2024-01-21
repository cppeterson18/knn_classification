"""
Christian Peterson
Bank Certification Analysis
(Utilizes K-Nearest Neighbors Algorithm)
[November 2023]
"""

import statistics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn import neighbors, metrics
from sklearn.metrics import make_scorer, precision_score, recall_score, \
    f1_score, confusion_matrix

INSTITUTIONS_FILE = "/Users/Peterson/Desktop/ds2500/institutions.csv"
BANKLIST_FILE = "/Users/Peterson/Desktop/ds2500/banklist.csv"
CLASS_COL = "failure"
FOLD_AMT = 4
FAILED_BOOL = 1

# [ATTN!] - sometimes Accuracy, Precision, Recall is abbreviated as A,P,R etc.

def dataframe(filename):
    """
    Parameters: a .csv file name (filename).

    Returns: a created DataFrame with cleaned 
    column titles (to ensure a consistent structure).

    Does: creates a Pandas DataFrame using read_csv().
    Each column title in the DataFrame is passed into
    convert_col_strs() to ensure consistent, clean naming
    amongst all of the columns (string format is the same
    for all columns). The columns are all renamed with 
    their cleaned titles using rename() and a dictionary
    of the former title and it's corresponding cleaned title.
    """

    df = pd.read_csv(filename, sep = ",", encoding = "cp1252")

    col_dct = {}

    col_names_lst = list(df.columns.values)
    new_columns = convert_col_strs(col_names_lst)

    for i,name in enumerate(col_names_lst):
        col_dct[name] = new_columns[i]

    df.rename(columns = col_dct, inplace = True)

    return df



def convert_col_strs(column_names):
    """
    Parameters: a list of column names (column_names).

    Returns: a list of column titles that have all been
    modified into lowercase characters, with removed
    whitespaces.

    Does: iterates through each column title, through
    each character in that specific title string, removing
    any whitespaces and ensuring that each letter is
    lowercase; adds each modified column title to a list.
    """

    clean_names = []

    omit_chars = [" "]

    for string in column_names:
        for char in omit_chars:

            string = string.lower()
            string = string.replace(char, "")

            clean_names.append(string)
            
    return clean_names



def add_col(main_df, parameter_df,
            shared_col_name, col_title):
    """
    Parameters: a parent DataFrame (main_df), a subset/child
    DataFrame (parameter_df), a name of a shared column
    (shared_col_name), and the name of the column that's
    being added (col_title).

    Returns: a merged DataFrame with a new column filled,
    with a "0" or "1" based on Binary Classification.

    Does: copies the shared column from a sub DataFrame,
    adding the value "1" for every present row, for the purposes
    of Binary Classification and K Nearest Neighbors' predictions.
    The parent DF and sub DF are merged using this shared column, and 
    each row from the sub DF will have a "1" value corresponding to it,
    while the rest of the rows' are paired with a "0" where there is a
    NaN value present (means that those rows were only a part
    of the parent DF). 
    """

    parameter_sub_df = parameter_df[[shared_col_name]].copy()

    # 1 == failed
    parameter_sub_df[col_title] = 1  

    merged_df = pd.merge(main_df, parameter_sub_df, how = "left", 
                         on = shared_col_name)

    merged_df[col_title].fillna(0, inplace = True)
    
    return merged_df



def convert_vals_type(df, col_name, conversion_type):
    """
    Parameters: a DataFrame (df), a specific column
    name (col_name), and a string that represents what type 
    the values need to be converted to (conversion_type).

    Returns: a new Pandas' Series in which all values have
    been converted to floats.

    Does: a specific column of the DataFrame is pulled out 
    as a Series, and all values in that Series are converted 
    using the specified conversion_type.
    """
    
    if conversion_type == "float":

        series = df[col_name]

        pd.to_numeric(series)
        
    return series



def normalize_df(features_cols, df, col_name):
    """
    Parameters: a list of columns in which their values need
    to be normalized (features_cols), a DataFrame in which those
    columns are located in (df), and the column name as each column is
    normalized one-at-a-time (col_name).

    Returns: a DataFrame in which missing values (NaN values) have
    been dropped and values in specific columns have been normalized.

    Does: drops all NaN values in the columns that need to be
    normalized; the DataFrame is modified. For each column passed
    (a part of `features_cols`) it's values are normalized using
    Min/Max Normalization after all values have been converted to
    floats and each column has been converted to a Pandas' Series.
    """

    df.dropna(subset = features_cols, inplace = True)
    
    float_series = convert_vals_type(df, col_name, "float")
    
    col_min = float_series.min()
    col_max = float_series.max()

    df[f"{col_name}_normalized"] = (float_series - col_min) / (col_max - col_min)
    
    return df



def find_x_y(df, feature_col_names, class_col_name):
    """
    Parameters: a DataFrame (df), columns that represent the
    features (feature_col_names), a column that represents 
    the labels (class_col_name).

    Returns: a data split of 2 arrays that represent the
    features and labels accordingly.

    Does: uses Pandas and NumPy to split the data from
    the DataFrame, converting both into arrays.
    """

    features = np.array(df.loc[:, feature_col_names])

    labels = np.array(df[class_col_name])

    return (features, labels)



def training_testing(df, columns, class_col_name):
    """
    Parameters: a DataFrame (df), columns that represent the
    features (columns), a column that represents the labels (class_col_name).

    Returns: (1.) the features and labels split, and (2.) the 
    split of the training and testing data obtained using train_test_split.
    All are arrays.

    Does: finds the data features and data labels split using find_x_y(),
    and uses that split with Scikit-learn's train_test_split() to find the 
    training and testing split for that data. 
    """

    x_and_y = find_x_y(df,
                       columns,
                       class_col_name)

    X_data = x_and_y[0]
    y_data = x_and_y[1]
    
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, 
                                                        random_state = 0)

    return X_data, y_data, X_train, X_test, y_train, y_test



def nearest_neighbors(num_neighbors, training_features, training_labels):
    """
    Parameters: the K-Value/number of neighbors to be considered
    (num_neighbors), the training features split (training_features),
    and the training labels split (training_labels).

    Returns: a Tuple of the classifier object, and the classifier object
    fitted with the training features set and training labels set.

    Does: uses Scikit-learn's neighbors library to create a KNN
    object (classifier) and the .fit() method to fit it with training data.
    """

    knn_obj = neighbors.KNeighborsClassifier(n_neighbors = num_neighbors)

    fitted_model = knn_obj.fit(training_features, training_labels)

    return (knn_obj, fitted_model)



def fold_data(splits):
    """
    Parameters: an int value specifying the number of needed splits
    for the data folding operation (splits).

    Returns: a folded classifier object, folded `splits` times, in which
    the data splits can be extracted (similar to train_test_split()).

    Does: uses Scikit-learn's KFold function/method to return a folded
    classifier object, based on a specified number of splits.
    """

    folded_data = KFold(n_splits = splits, random_state = 0, shuffle = True)

    return folded_data



def get_scores(y_test, y_predictions, 
               class_col = None, spec_class = None, f1 = False):
    """
    Parameters: the actual occurrences for the class column (labels)
    for each of the banks (y_test), the model's prediction for 
    the outcomes of each bank (y_predictions), a column of all of the 
    labels i.e. class column (class_col), a specific label needed for the
    F1 Score to denote which label we're focusing on in accordance with
    Binary Classification (spec_class), a boolean to denote whether the
    3 scores need to be calculated or the F1 Score only (f1).

    Returns: if the F1 Score isn't the sole score calculation, the
    function will return a dictionary with the corresponding Accuracy,
    Precision, and Recall scores (typically per fold or whatever is
    specified). Or, if f1 is True, then the F1 Score is the only thing 
    being returned as a float.

    Does: uses Skikit-learn's metrics library to calculate either the
    Accuracy and Precision and Recall scores or the F1 Score using the
    model's predicted outcomes and the actual outcomes. 
    """

    if f1 == False:

        accuracy_score = metrics.accuracy_score(y_true = y_test, 
                                                y_pred = y_predictions)
        
        precision_score = metrics.precision_score(y_true = y_test, 
                                                y_pred = y_predictions,
                                                average = "macro")
        
        recall_score = metrics.recall_score(y_true = y_test, 
                                                y_pred = y_predictions,
                                                average = "macro")

        fold_scores = {"accuracy": accuracy_score, 
                "precision": precision_score,
                "recall": recall_score}
        
        return fold_scores
    
    # Calculate the F1 Score
    if f1 == True:

        f1_score = metrics.f1_score(y_true = y_test,
                                    y_pred = y_predictions,
                                    labels = class_col,
                                    pos_label = spec_class)

        return f1_score



def calc_mean(lst_of_dcts, score_type):
    """
    Parameters: a list of dictionaries 
    ([{accuracy: score, precision: score, recall: score}, {}, ...])
    for each fold (lst_of_dcts), a string: "accuracy", 
    "precision", or "recall" (score_type).

    Returns: the mean of all scores of a specific type;
    Used to calculate the mean per K-Value. 

    Does: finds the score per score type in each
    dictionary, appends it to a list, and the
    mean is found using NumPy.
    """

    score_type_lst = []

    for dct in lst_of_dcts:
        score_per_fold = dct[score_type]
        score_type_lst.append(score_per_fold)
    
    avg_score = np.mean(score_type_lst)

    return avg_score



def find_best(results_dct, operation = "max"):
    """
    Parameters: a dictionary with K-Value as the key
    and the mean specific score as the key (results_dct),
    and a string that specifies if the optimal (max) 
    value/mean is to be found or the minimum mean.

    Returns: either the minimum or maximum score and
    the K-Value (key) that it belongs to.

    Does: Gets a list of all mean scores in the dictionary.
    Then, either the maximum or minimum value is found, and
    the corresponding key it "belongs to" or matches with is
    found (as a Tuple). This is all done based on if the minimum or 
    maximum mean score is requested in the function call (operation).
    """

    if operation == "max":

        key_means_lst = list(results_dct.values())
        the_mean = max(key_means_lst)

        for k,v in results_dct.items():
            if v == the_mean:
                key = k   
    

    if operation == "min":

        key_means_lst = list(results_dct.values())
        the_mean = min(key_means_lst)

        for k,v in results_dct.items():
            if v == the_mean:
                key = k 
    
    return (key, the_mean)



def optimal_k(k_values, X_data, y_data, X_test, X_train, y_test, 
              y_train, score_type, to_return, operation = None):
    """
    Parameters:
    k_values: a list of all acceptable k values to be computed 
    X_test: an array for the testing features
    X_train: an array for the training features 
    y_test: an array for the testing labels (y_test)
    y_train: an array for the training labels 
    score_type: a string to denote what mean score is being 
        calculated (can be "Accuracy", "Precision", "Recall")
    to_return: a string used to specify if the optimal value
        is to be returned or the full dictionary (all k's means per a score type)
    operation: string used to specify if optimal (max) is to be found or (min)

    Returns: either a float representing the optimal (or minimum) mean value for 
    a specific score and a specific K-Value or a dictionary of mean values per
    one specific score for each K-Value.

    Does: iterates through each K-Value, creating a classifier, and then 
    folding the classifier to further train the model. For each K-Value, 
    the function iterates through each fold, the training data is fit to
    the original object to create a model, and the actual occurences and a 
    prediction of occurrences are passed to get_scores() to get the Accuracy, 
    Precision, and Recall scores for each fold (in a dictionary).
    To find a mean score for one K-Value, each fold's score dictionary is added 
    to a list, passed to calc_mean(), returned, and added to a dictionary 
    ({K-Value}: mean value). If only an optimal or minimal value is to be 
    returned, the dictionary is passed to find_best() to find the "min" or "max" 
    mean. Otherwise, the dictionary is returned.  

    Note: used `key` at times to represent K-Value 
    """

    all_keys_means = {}

    for i,k_value in enumerate(k_values):
        knn_tup = nearest_neighbors(k_value, X_train, y_train)
        knn_obj = knn_tup[0]

        folded_data = fold_data(FOLD_AMT)

        all_folds_scores = []

        for train_i, test_i in folded_data.split(X_data):
            X_train, X_test, y_train, y_test = X_data[train_i], \
                X_data[test_i], y_data[train_i], y_data[test_i]
            
            knn_obj.fit(X_train, y_train)
            y_predictions = knn_obj.predict(X_test)
            
            # A,P,R scores for EACH FOLD
            performance_dct = get_scores(y_test, y_predictions)
    
            # add to list of dictionaries for each fold 
            all_folds_scores.append(performance_dct)

        # find the mean A or P or R of ALL folds for each K-Value (key) 
        mean_per_key = calc_mean(all_folds_scores, score_type)
        
        all_keys_means[k_value] = mean_per_key
    
    # if optimal needed, return the optimal k per request
    if to_return == "value":
        optimal_k = find_best(all_keys_means, operation)

        return optimal_k
    
    # or return all means for each key
    if to_return == "dct_only":
        return all_keys_means



def get_filtered_columns(df, parameter_col, parameter_col2, 
                         parameter, parameter2):
    """
    Parameters: a DataFrame (df), columns that provide specific parameters or
    identifiers for a specific row (parameter_col & parameter_col2), the actual
    parameters/identifiers (parameter & parameter2).

    Returns: the row that has the specific parameters associated with it in the
    DataFrame, and the position of that row in the DataFrame as a Tuple.

    Does: retrieves a specific row by first applying the first parameter,
    and returning a DataFrame of all rows that have that parameter present.
    This new DataFrame is filtered through again with the second parameter
    to find one specific row and its corresponding index/position.
    """
    
    rows_df = df.loc[df[parameter_col] == parameter]

    row = rows_df.loc[rows_df[parameter_col2] == parameter2]

    position = df.index.get_loc(row.index[0])

    return (row, position)



def test(k_value, X_train, X_test, y_train, y_test, 
         class_col = None, spec_class = None, operation = "Default"):
    """
    Parameters: a specific K-Value (k_value), an array for the
    training features (X_train), an array for the testing features 
    (X_test), an array for the training labels (y_train), an array
    for the testing labels (y_test), a column of all of the labels/class
    column (class_col), a specific label needed for the
    F1 Score (spec_class), and the operation that the user wants to perform
    (can be default - obtain Accuracy & Precision & Recall scores, 
    calculate F1 Score, or get the confusion matrix). 

    Returns: it returns either a dictionary of mean Accuracy, Precision,
    and Recall scores (if selected), a float value representing the F1
    Score (if selected), or a 2d Array confusion matrix (if selected).

    Does: is based on whatever operation needs to be completed: calculate
    mean Accuracy, Precision, Recall scores for one K-Value, or the F1 Score
    for one K-Value, or the confusion matrix for one K-Value. For all operations,
    the nearest_neighbors() function is called, returning a fitted model 
    (with training) data, and the model's predicted outcomes are computed. 
    For whatever operation is being completed, this function will reference 
    get_scores() to obtain the mean scores or the F1, or get_confusion_matrix().
    """

    knn_tup = nearest_neighbors(k_value, X_train, y_train)
    knn_model = knn_tup[1] 
        
    y_predictions = knn_model.predict(X_test)
  
    if operation == "Default":  
        k_scores_dct = get_scores(y_test, y_predictions)

        return k_scores_dct
    
    if operation == "f1":
        f1_score = get_scores(y_test, y_predictions, 
                              class_col,
                              spec_class,
                              f1 = True)
        return f1_score
    
    if operation == "confusion_matrix":
        confusion_matrix = get_confusion_matrix(y_test, y_predictions)
        
        return confusion_matrix
    

    
def get_confusion_matrix(y_test, y_predictions):
    """
    Parameters: the actual occurrences for the class column (labels)
    for each of the banks (y_test) and the model's prediction for 
    the outcomes of each bank (y_predictions).

    Returns: a confusion matrix. 
    
    Does: uses Sklearn's metrics library to create a confusion matrix
    by using the model's predictions of the occurences and the actual
    occurrences.
    """

    confusion_matrix = metrics.confusion_matrix(y_true = y_test,
                                                y_pred = y_predictions)

    return confusion_matrix



def get_heatmap(confusion_matrix, title, xlabels, ylabels,
                entire_x_label, entire_y_label):
    """
    Parameters: a confusion matrix (confusion_matrix) and a
    title for the plot (title) and X-Axis labels and Y-Axis 
    labels (per cell - xlabels, ylabels and per axis -
    entire_x_label, entire_y_label).

    Returns: shows the heatmap using the optimal K-Value for 
    the mean recall score.

    Does: Uses Seaborn to make a heatmap and provides annotations
    and a title.
    """

    heatmap = sns.heatmap(confusion_matrix,
                          annot = True, fmt = 'd',
                          xticklabels = xlabels, 
                          yticklabels = ylabels)

    # Add additional features
    plt.title(title)
    plt.xlabel(entire_x_label)
    plt.ylabel(entire_y_label)
    plt.show()

    return heatmap



def normalize_lst(lst):
    """
    Parameters: a list of floats (lst).

    Returns: a list of floats that have been normalized to a new 
    (common) scale.

    Does: finds the minimum and maximum values in the list by using 
    descriptive_stat() for both. Next, the function applies the Min/Max
    Normalization formula to each value in the list, creating a new list 
    of normalized values.
    """
    
    min_num = min(lst)
    max_num = max(lst)

    normalized_vals = []

    for val in lst:
        numerator = val - min_num
        denominator = max_num - min_num

        scaled_val = numerator / denominator
        normalized_vals.append(scaled_val)
    
    return normalized_vals



def plot_dicts(dcts_lst, indep_var_lst, colors, labels,
               x_label, y_label, title):
    """
    Parameters: a list of dictionaries of dependent variable 
    values (dcts_lst), a list of
    independent variable values (indep_var_lst), a list of colors
    (colors), a list of labels (labels), and a label for the x-axis,
    y-axis, and a title (same names).

    Returns: shows a line plot.

    Does: iterates through each dictionary object's values in the dependent
    variable list and normalizes them, to assign the dependent variables, 
    assigns the independent variables (K-Values), and also labels and colors 
    each plotted line; there is one line for each dictionary 
    (one dictionary for each mean score -- Accuracy,
    Precision, and Recall). Next, the maximum mean value in each
    dictionary is plotted to convey why each specific K-Value was chosen. 
    """

    for i,dct in enumerate(dcts_lst):
        depend_var_lst = list(dct.values())

        normalized_depend_vars = normalize_lst(depend_var_lst)

        line_plot = plt.plot(indep_var_lst, normalized_depend_vars, 
                               color = colors[i], label = labels[i])

        max_val = max(normalized_depend_vars)
        k_value = indep_var_lst[normalized_depend_vars.index(max_val)]

        point = plt.plot(k_value, max_val, 
                 color = colors[i], label = f"{labels[i]} (Max)",
                 marker = "X")

    # add more features    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    
    plt.show()

    return line_plot



def main():

    # Read in the .csv files into a dataframe
    institutions_df = dataframe(INSTITUTIONS_FILE)
    failed_banklist_df = dataframe(BANKLIST_FILE)
    

    # Add a column to `institutions` specifying the failed banks, make sure 
    # all values have been converted 
    modif_institutions_df = add_col(institutions_df, failed_banklist_df, 
                                    "cert", CLASS_COL)


    # Normalize specific columns' values using Max/Min Method
    # Remove all empty rows in the consolidated dataframe and convert all 
    # values to floats 
    normalized_needed = ["asset", "dep", "depdom", "netinc", "offdom", "roa", 
                         "roaptx", "roe"]
    for col_name in normalized_needed:
        normalized_institutions_df = normalize_df(normalized_needed, 
                                                  modif_institutions_df, 
                                                  col_name)
        

    # Assign X (features) and y (labels) where `normalized_needed` is the features 
    # and find the train/test split
    normalized_cols = [f"{col_name}_normalized" for col_name \
                       in normalized_needed]

    X_data, y_data, X_train, \
        X_test, y_train, y_test = training_testing(normalized_institutions_df,
                                                   normalized_cols,
                                                   CLASS_COL)

   
    # [Q1] - What is the optimal value of k if we care most about accuracy?
    # Create the Classifier for many different values of `k`, fit the model 
    # with the training data, find the precision, accuracy, recall for each
    k_values = [i for i in range(4, 19)]

    optimal_k_accuracy = optimal_k(k_values, X_data, y_data, X_test, 
                                   X_train, y_test, y_train, 
                                   "accuracy", "value", "max")
    print(f"OPTIMAL K FOR ACCURACY: {optimal_k_accuracy}\n")


    # [Q2] - What is the lowest mean accuracy for any value of k?
    # Use the optimal_k function, data splits, etc to find
    minimal_k_accuracy = optimal_k(k_values, X_data, y_data, X_test, 
                                   X_train, y_test, y_train, 
                                   "accuracy", "value", "min")
    print(f"K FOR MINIMAL ACCURACY: {round(minimal_k_accuracy[1], 4)}\n")


    # [Q3] - What is the optimal value of k if we care most about mean 
    # precision across all 4 folds?
    # Use the optimal_k function, data splits, etc to find
    optimal_k_precision = optimal_k(k_values, X_data, y_data, X_test, 
                                    X_train, y_test, y_train, 
                                   "precision", "value", "max")
    print(f"OPTIMAL K FOR PRECISION: {optimal_k_precision}\n")


    # [Q4] - What is the optimal value of k if we care most about mean recall 
    # across all 4 folds?
    # Use the optimal_k function, data splits, etc to find
    optimal_k_recall = optimal_k(k_values, X_data, y_data, X_test, 
                                 X_train, y_test, y_train, 
                                 "recall", "value", "max")
    print(f"OPTIMAL K FOR RECALL: {optimal_k_recall}\n")


    # [Q5] - (We care the most about accuracy for the 3 parts of Q5)
    # What is the f1 score for banks that failed?  Give your answer 
    # as a float with 4 digits after the decimal.
    # Using the optimal K-Value for mean accuracy, find the F1 Score using 
    # Scikit-learn, the column for if the banks
    # failed or not, and the "boolean" variable to determine if the bank failed (1)
    k_neigh_optimal = optimal_k_accuracy[0]
    
    failed_f1_score = test(k_neigh_optimal, X_train, X_test,
                              y_train, y_test, 
                              normalized_institutions_df[CLASS_COL],
                              FAILED_BOOL, operation = "f1")
    
    print(f"F1 SCORE FOR FAILED BANKS: {round(failed_f1_score, 4)}\n")

   
    # [Q5b] - How many banks did your model predict to NOT fail, and in fact 
    # did not?
    # Find the confusion matrix, extract the "True Negative" value to answer 
    # this question
    accuracy_confusion_matrix = test(k_neigh_optimal, X_train, X_test,
                              y_train, y_test, operation = "confusion_matrix")
    
    true_negative = accuracy_confusion_matrix[0][0]

    print(f"BANKS PREDICTED ACTUALLY NOT TO FAIL: {true_negative}\n")

    
    # [Q5c] - Does your model correctly predict what happened to Southern 
    # Community Bank of Fayetteville, GA?
    # Filter through the DataFrame to find the specific row, and find the 
    # occurrence/label/class
    needed_row, row_index = get_filtered_columns(normalized_institutions_df, 
                                                 "name", "city", 
                                                 "Southern Community Bank", 
                                                 "Fayetteville")
    
    actual_occurrence = needed_row[CLASS_COL]

    # Obtain the model, get the splitted values of the row, and make a 
    # prediction (to compare with what actually happened)
    knn_model = nearest_neighbors(k_neigh_optimal, X_train, y_train)[1]

    southern_X_data, southern_y_data = find_x_y(needed_row, normalized_cols, 
                                                CLASS_COL)

    southern_bank_pred = knn_model.predict(southern_X_data)

    print(f"""MODEL'S PREDICTION: {southern_bank_pred}
    ACTUAL PREDICTION: {actual_occurrence}\n""")

    
    # [P1] - A heatmap showing the confusion matrix when the value of k is 
    # optimal if we care most about recall
    # Get and plot the confusion matrix as a heatmap (using the optimal 
    # K-Value for RECALL)
    k_neigh_recall = optimal_k_recall[0]

    recall_confusion_matrix = test(k_neigh_recall, X_train, X_test,
                              y_train, y_test, operation = "confusion_matrix")
    title = "Heatmap - Confusion Matrix with Optimal K-Value (Recall)"

    xlabels = ["Positive", "Negative"]
    ylabels = ["Positive", "Negative"]
    entire_x_label, entire_y_label = "Predicted Condition", "Actual Condition"
    
    get_heatmap(recall_confusion_matrix, title, xlabels, ylabels,
                entire_x_label, entire_y_label)

 
    # [P2] - A plot showing why you picked those optimal values of k. 
    # (how different values of k correlate with accuracy, precision, and recall)
    # Obtain all scores (accuracy, precision, recall) for each of the k values, 
    # plot them in a scatterplot x-axis = k values, y-axis = (normalized?) scores
    k_accuracies_dct = optimal_k(k_values, X_data, y_data, X_test, 
                                 X_train, y_test, y_train, 
                                 "accuracy", "dct_only")
    
    k_precisions_dct = optimal_k(k_values, X_data, y_data, X_test, 
                                 X_train, y_test, y_train, 
                                 "precision", "dct_only")
    
    k_recalls_dct = optimal_k(k_values, X_data, y_data, X_test, 
                              X_train, y_test, y_train, 
                              "recall", "dct_only")

    scores_dct_lst = [k_accuracies_dct, k_precisions_dct, k_recalls_dct]
    colors = ["red", "green", "blue"]
    labels = ["Mean Accuracy", "Mean Precision", "Mean Recall"]
    x_label = "K-Values"
    y_label = "Scores for Mean Accuracy, Precision, Recall"
    title = "K-Values vs their Normalized Mean Accuracy, Precision, and Recall Scores"

    plot_dicts(scores_dct_lst, k_values, colors, labels, 
               x_label, y_label, title)
    
main()