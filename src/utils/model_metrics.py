"""
Created on Wed Nov 7 2018

@author: Renato Durrer renato.durrer@axa-winterthur.ch
         Fabien Tarrade fabien.tarrade@axa.ch
"""
import itertools
from sklearn.metrics import (log_loss, f1_score, accuracy_score, average_precision_score, precision_score,
                             recall_score, roc_auc_score, mean_squared_error, r2_score)
from sklearn.model_selection import learning_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplt

def plot_acc_loss(steps_loss_train, loss_train,
                  steps_acc_train=None, accuracy_train=None,
                  steps_loss_eval=None, loss_eval=None,
                  steps_acc_eval=None, accuracy_eval=None):

    # plot the training loss and accuracy
    fig = plt.figure(figsize=(9, 3), dpi=100)
    plt.subplots_adjust(wspace=0.6)
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    # accuracy
    if accuracy_train is not None:
        ax1.plot(steps_acc_train, accuracy_train, 'b', label='training accuracy')
    if accuracy_eval is not None:
        ax1.plot(steps_acc_eval, accuracy_eval, 'r', label='validation accuracy');
    #if accuracy_train is not None and accuracy_eval is not None:
    #    delta=0.05
    #    y_min = min([min(x_list) for x_list in [accuracy_train, accuracy_eval]]) - delta
    #    y_max = max([max(x_list) for x_list in [accuracy_train, accuracy_eval]]) + delta
    #    ax1.set_ylim(y_min, y_max)
    ax1.set_title('Accuracy')
    ax1.set_xlabel("Number of epoch ")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="best")
    # loss
    if loss_train is not None:
        ax2.plot(steps_loss_train, loss_train, label="training loss")
    if loss_eval is not None:
        ax2.plot(steps_loss_eval, loss_eval, label="validation loss")
    ax2.set_title("Loss")
    ax2.set_xlabel("Number of epoch ")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="best");

    print('Loss:')
    if loss_train is not None:
        print('  - loss [training dataset]: {0:.3f}'.format(loss_train[-1]))
    if loss_eval is not None:
        print('  - loss [validation dataset: {0:.3f}'.format(loss_eval[-1]))
    print('')
    print('Accuracy:')
    if accuracy_train is not None:
        print('  - accuracy [training dataset]: {:.2f}%'.format(100 * accuracy_train[-1]))
    if accuracy_eval is not None:
        print('  - accuracy [validation dataset: {:.2f}%'.format(100 * accuracy_eval[-1]))

def print_metrics(y_t, y_pred_t, mode=''):
    """
    Print metrics of various kind

    Parameters
    ----------
    y_t :

    y_pred_t :

    mode : string

    """
    print('Model performance on the {} dataset:'.format(mode))

    #mse = mean_squared_error(y_t, y_pred_t)
    #logloss = log_loss(y_t, y_pred_t)
    accuracy = accuracy_score(y_t, y_pred_t)
    f1 = f1_score(y_t, y_pred_t)
    precision_micro = precision_score(y_t, y_pred_t, average='micro')
    precision_macro = precision_score(y_t, y_pred_t, average='macro')
    avg_precision = average_precision_score(y_t, y_pred_t)
    precision = precision_score(y_t, y_pred_t)
    recall = recall_score(y_t, y_pred_t, average='binary')
    auc = roc_auc_score(y_t, y_pred_t)
    r2 = r2_score(y_t, y_pred_t)

    print('   Metric             {}'.format(mode.title()))
    print('accuracy........... {0:8.4f}'.format(accuracy))
    print('recall............. {0:8.4f}'.format(recall))
    print('auc................ {0:8.4f}'.format(auc))
    print('precision (p=0.5).. {0:8.4f}'.format(precision))
    print('precision (avg).... {0:8.4f}'.format(avg_precision))
    print('precision (micro).. {0:8.4f}'.format(precision_micro))
    print('precision (macro).. {0:8.4f}'.format(precision_macro))
    print('f1.................  {0:8.4f}'.format(f1))
    print('r2.................  {0:8.4f}'.format(r2))
    #print('logloss............  {0:8.4f}'.format(logloss))
    #print('mse................  {0:8.4f}'.format(mse))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Parameters
    ----------
    cm : matrix
        confusion matrix

    classes : list
        list of names of the classes

    normalize : bool, optional
        normalizes confusion matrix if True

    title : string, optional
        title of confusion matrix

    cmap : optional
        some plot object
        default: plt.cm.Blues
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized Confusion Matrix'
    else:
        title = 'Confusion Matrix'

    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def print_confusion_matrix(cm, cr, label, mode=''):
    """
    Print confusino matrix for binary classification

    Parameters
    ----------
    cm : matrix
        confusion matrix

    cr : string

    label :

    mode : optional
    """
    print('Confusion matrics of the {} data set:\n'.format(mode))

    print('confusion matrix: \n {} \n'.format(cm))
    true_negative = cm[0, 0]
    true_positive = cm[1, 1]
    false_negative = cm[1, 0]
    false_positive = cm[0, 1]

    print('True labels:')
    for i, j in zip(np.sum(cm, axis=1), label):
        print('{}  {:,}'.format(j, i))
    print('')
    print('Predicted labels:')
    for i, j in zip(np.sum(cm, axis=0), label):
        print('{}  {:,}'.format(j, i))

    total = true_negative + true_positive + false_negative + false_positive
    accuracy = (true_positive + true_negative) / total
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    misclassification_rate = (false_positive + false_negative) / total
    f1 = (2 * true_positive) / (2 * true_positive + false_positive + false_negative)
    print('\n accuracy................. {0:.4f}'.format(accuracy))
    print(' precision................ {0:.4f}'.format(precision))
    print(' recall................... {0:.4f}'.format(recall))
    print(' misclassification_rate... {0:.4f}'.format(misclassification_rate))
    print(' f1....................... {0:.4f}\n'.format(f1))

    print('classification report: \n {} \n '.format(cr))


def plot_learning_curve(estimator, title, x, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curves.
    Does not work with Keras/Tensorflow

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    x : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).

    train_sizes: array-like
        e.g. np.linspace(.1, 1.0, 5)
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, x, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid(True)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()


def auc_plot(model, title, x_test, y_test, x_train=None, y_train=None, prefit=True):
    """
    Creates a simple AUC ROC plot.

    Parameters
    ----------
    model : scikit-learn model
        model that is evaluated. May be prefitted. Needs to have attribute .predict_proba

    title : string
        title of the plot

    x_test : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y_test : array-like, shape (n_samples) or (n_samples, n_features)
        Target relative to x_test for classification

    x_train : array-like, shape (n_samples, n_features), optional
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y_train : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to x_train for classification

    prefit : bool, optional
        default False
        wheter or not to use a prefitted model. If true x_train and y_train do not need to be passed.
    """
    if not prefit:
        model.fit(x_train, y_train)
    y_probas = model.predict_proba(x_test)
    skplt.metrics.plot_roc(y_test, y_probas)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 2, 1])
    plt.show()


def wrongly_classified(model, x_test, y_test, x_train=None, y_train=None, prefit=True, return_correct=False):
    """
    Returns the features of the wrongly classified points.

    Parameters
    ----------
    model : clf
        classifier implementing methods: fit & predict

    x_test : array-like, shape (n_samples, n_features)
        Feature vector, where n_samples is the number of samples and
        n_features is the number of features.

    y_test : array-like, shape (n_samples) or (n_samples, n_features)
        Target relative to x_test for classification

    x_train : array-like, shape (n_samples, n_features), optional
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y_train : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to x_train for classification

    prefit : bool, optional
        if False model is fitted first. x_train and y_train need to be passed in this case.
        default: True

    return_correct : bool, optional
        if True,

    Returns
    -------
    DataFrame
        DataFrame containing wrongly (or correct) classified data points
    """
    if not prefit:
        model.fit(x_train, y_train)
    df = pd.DataFrame(model.predict(x_test), columns=['predicted_label'], index=x_test.index.values)
    df['true_label'] = y_test
    df2 = pd.DataFrame(x_test)
    df = pd.concat([df, df2], axis='columns')
    if return_correct:
        return df[df['true_label'] == df['predicted_label']]
    else:
        return df[df['true_label'] != df['predicted_label']]


def check_confidence(model, min_conf, x_test, y_test):
    """
    Prints accuracy for minimal confidence for classification.

    Parameters
    ----------
    model : pretrained scikit-learn model (needs, fit, predict & predict_proba)

    min_conf : float,
        minimal confidence for classificaton

    x_test : array-like, shape (n_samples, n_features)
        Feature vector, where n_samples is the number of samples and
        n_features is the number of features.

    y_test : array-like, shape (n_samples) or (n_samples, n_features)
        Target relative to x_test for classification

    Returns
    -------
    None
    """
    predictions = model.predict_proba(x_test)
    wrong_class = pd.DataFrame(predictions[model.predict(x_test) != y_test])
    right_class = pd.DataFrame(predictions[model.predict(x_test) == y_test])
    wrong_class_counts = (wrong_class.apply(lambda row: max(row), axis=1) > min_conf).value_counts()
    right_class_counts = (right_class.apply(lambda row: max(row), axis=1) > min_conf).value_counts()
    baseline = accuracy_score(y_test, model.predict(x_test))
    Accuracy = right_class_counts[True] / (right_class_counts[True] + wrong_class_counts[True])
    Classifications = (right_class_counts[True] + wrong_class_counts[True]) / len(y_test)
    print('With a minimal confidence of {} we\'d have {} accuracy and {} of the datapoints would'
          ' \nbe classified. '
          'The baseline is given as {}'.format(min_conf,
                                               round(Accuracy, 4),
                                               round(Classifications, 4),
                                               round(baseline, 4)))
