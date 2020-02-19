"""
Created on Wed Nov 7 2018

@author: Renato Durrer renato.durrer@axa.ch
         Janine Kaufmann janine.kaufmann@axa.ch
         Fabien Tarrade fabien.tarrade@axa.ch
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import (log_loss, f1_score, accuracy_score, average_precision_score, precision_score,
                             recall_score, roc_auc_score, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from gensim.models import FastText
from scipy.optimize import minimize
import nltk
import time


class W2VTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn wrapper for gensim.models.FastText.

    Parameters
    ----------
    sentences : iterable of iterables
        The `sentences` iterable can be simply a list of lists of tokens, but for larger corpora,
        consider an iterable that streams the sentences directly from disk/network.
        See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
        or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.
        If you don't supply `sentences`, the model is left uninitialized -- use if you plan to initialize it
        in some other way.
    sg : int {1, 0}
        Defines the training algorithm. If 1, skip-gram is used, otherwise, CBOW is employed.
    size : int
        Dimensionality of the feature vectors.
    window : int
        The maximum distance between the current and predicted word within a sentence.
    alpha : float
        The initial learning rate.
    min_alpha : float
        Learning rate will linearly drop to `min_alpha` as training progresses.
    seed : int
        Seed for the random number generator. Initial vectors for each word are seeded with a hash of
        the concatenation of word + `str(seed)`. Note that for a fully deterministically-reproducible run,
        you must also limit the model to a single worker thread (`workers=1`), to eliminate ordering jitter
        from OS thread scheduling. (In Python 3, reproducibility between interpreter launches also requires
        use of the `PYTHONHASHSEED` environment variable to control hash randomization).
    min_count : int
        Ignores all words with total frequency lower than this.
    max_vocab_size : int
        Limits the RAM during vocabulary building; if there are more unique
        words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM.
        Set to `None` for no limit.
    sample : float
        The threshold for configuring which higher-frequency words are randomly downsampled,
        useful range is (0, 1e-5).
    workers : int
        Use these many worker threads to train the model (=faster training with multicore machines).
    hs : int {1,0}
        If 1, hierarchical softmax will be used for model training.
        If set to 0, and `negative` is non-zero, negative sampling will be used.
    negative : int
        If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
        should be drawn (usually between 5-20).
        If set to 0, no negative sampling is used.
    cbow_mean : int {1,0}
        If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
    iter : int
        Number of iterations (epochs) over the corpus.
    trim_rule : function
        Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
        be trimmed away, or handled using the default (discard if word count < min_count).
        Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
        or a callable that accepts parameters (word, count, min_count) and returns either
        :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
        Note: The rule, if given, is only used to prune vocabulary during build_vocab() and is not stored as part
        of the model.
    sorted_vocab : int {1,0}
        If 1, sort the vocabulary by descending frequency before assigning word indexes.
    batch_words : int
        Target size (in words) for batches of examples passed to worker threads (and
        thus cython routines).(Larger batches will be passed if individual
        texts are longer than 10000 words, but the standard cython code truncates to that maximum.)
    min_n : int
        Min length of char ngrams to be used for training word representations.
    max_n : int
        Max length of char ngrams to be used for training word representations. Set `max_n` to be
        lesser than `min_n` to avoid char ngrams being used.
    word_ngrams : int {1,0}
        If 1, uses enriches word vectors with subword(ngrams) information.
        If 0, this is equivalent to word2vec.
    bucket : int
        Character ngrams are hashed into a fixed number of buckets, in order to limit the
        memory usage of the model. This option specifies the number of buckets used by the model.
    callbacks : :obj: `list` of :obj: `~gensim.models.callbacks.CallbackAny2Vec`
        List of callbacks that need to be executed/run at specific stages during training.
    """
    def __init__(self, sentences=None, sg=0, hs=0, size=100, alpha=0.025,
                 window=5, min_count=5, max_vocab_size=None, word_ngrams=1,
                 sample=0.001, seed=1, workers=3, min_alpha=0.0001, negative=5,
                 cbow_mean=1, iter=5, null_word=0, min_n=3, max_n=6,
                 sorted_vocab=1, bucket=2000000, trim_rule=None,
                 batch_words=10000, callbacks=()):
        self.sentences = sentences
        self.sg = sg
        self.hs = hs
        self.size = size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.word_ngrams = word_ngrams
        self.sample = sample
        self.seed = seed
        self.workers = workers
        self.min_alpha = min_alpha
        self.negative = negative
        self.cbow_mean = cbow_mean
        self.iter = iter
        self.null_word = null_word
        self.min_n = min_n
        self.max_n = max_n
        self.sorted_vocab = sorted_vocab
        self.bucket = bucket
        self.trim_rule = trim_rule
        self.batch_words = batch_words
        self.callbacks = callbacks
        self.n_features_ = None
        self.model_ = None

    def fit(self, X, y=None):
        """
        A reference implementation of a fitting function for a transformer.
        Parameters
        ----------
        X : list
            list of input texts

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        self.n_features_ = 1

        tokenized_corpus = self.tokenize(X)

        self.model_ = FastText(sentences=tokenized_corpus,
                               sg=self.sg,
                               hs=self.hs,
                               size=self.size,
                               alpha=self.alpha,
                               window=self.window,
                               min_count=self.min_count,
                               max_vocab_size=self.max_vocab_size,
                               word_ngrams=self.word_ngrams,
                               sample=self.sample,
                               seed=self.seed,
                               workers=self.workers,
                               min_alpha=self.min_alpha,
                               negative=self.negative,
                               cbow_mean=self.cbow_mean,
                               iter=self.iter,
                               null_word=self.null_word,
                               min_n=self.min_n,
                               max_n=self.max_n,
                               sorted_vocab=self.sorted_vocab,
                               bucket=self.bucket,
                               trim_rule=self.trim_rule,
                               batch_words=self.batch_words,
                               callbacks=self.callbacks
                               )

        # Return the transformer
        return self

    def transform(self, X):
        """
        A reference implementation of a transform function.
        Parameters
        ----------
        X : list
            list of input texts

        Returns
        -------
        X_transformed : array, shape (n_samples, size)

        """
        # Check if fit had been called
        check_is_fitted(self, 'n_features_')

        tokenized_corpus = self.tokenize(X)

        w2v_feature_array = self.averaged_word_vectorizer(tokenized_corpus)

        return w2v_feature_array

    def average_word_vectors(self, words, vocabulary):
        """
        Returns the mean feature vector of a text based on its tokens.
        Parameters
        ----------
        words : list
            list of tokens of which a feature vector is calculated

        vocabulary : list
            list of known words

        Returns
        -------
        Feature vector representation
        """
        feature_vector = np.zeros((self.size,), dtype="float64")
        nwords = 0.

        for word in words:
            if word in vocabulary:
                nwords = nwords + 1.
                feature_vector = np.add(feature_vector, self.model_.wv[word])

        if nwords:
            feature_vector = np.divide(feature_vector, nwords)

        return feature_vector

    def averaged_word_vectorizer(self, corpus):
        """
        Calculates the feature representation of a sorpus (list of list of words)
        Parameters
        ----------
        corpus : list
            list of list of words. Input features of which a vector representation is calculated

        Returns
        -------
        np.array
            array of feature representations
        """
        vocabulary = set(self.model_.wv.index2word)
        features = [self.average_word_vectors(tokenized_sentence, vocabulary) for tokenized_sentence in corpus]
        return np.array(features)

    def tokenize(self, X):
        """
        Tokenizes X using nltk.word_tokenize
        Parameters
        ----------
        X : list
            list of strings which are tokenized

        Returns
        -------
        list
            list of tokens
        """
        docs = [doc for doc in X]
        tokenized_corpus = []
        for doc in docs:
            tokens = nltk.word_tokenize(doc)
            tokenized_corpus.append(tokens)
        return tokenized_corpus


scorer = {'accuracy': accuracy_score, 'log_loss': log_loss, 'f1': f1_score,
          'average_precision': average_precision_score, 'precision': precision_score,
          'roc_auc': roc_auc_score, 'mean_squared_error': mean_squared_error,
          'r2': r2_score, 'recall': recall_score}


class VotingClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that taks a bunch of classifiers (pipelines) and creates a combined one.
    Applies a soft voting rule based on a weighted average of the class probabilities. Thereby,
    the weights found such that the scoring function is optimized.

    Parameters
    ----------
    models : list,
        list of scikit-learn pipelines or list of tuples (pipeline, fitting_params)

    weights: optional, array-like, shape (n_models,)
        weights used for the final classifier. If specified
        fit does not optimize the weights further.

    split_data: bool, optional
        default True, if True, then different data sets are used
        for training and optimizing the weights. If False, all passed data
        is used for both.

    test_size: float, optional
        default 0.33, only relevant if split_data=True, relative size of the
        data-set used for optimizing the weights.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    models : list of pipelines
    fit_params : list,
        list of fitting parameter for the models pipelines
    weights : array-like, shape (n_models,)
        weights calculated by fit or specified at initialization
    scorer_ : sklearn.metrics function,
        scorer function used to optimize the weights

    Methods
    -------
    fit :
    predict :
    predict_proba :
    get_weights :
    """
    def __init__(self, models=None, weights=None, split_data=True, test_size=0.33):
        self.weights = weights
        self.split_data = split_data
        self.test_size = test_size
        self.models = []
        self.fit_params = []
        for model in models:
            if type(model) is tuple:
                self.models.append(model[0])
                self.fit_params.append(model[1])
            else:
                self.models.append(model)
                self.fit_params.append({})

    def fit(self, X, y, prefit=False, scoring='accuracy'):
        """
        Fits the passed pipelines and finds the optimal weights.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        prefit : Bool, wheter or not pipelines are already fitted
        scoring : string, score that is used to optimize the weights.
                  can be chosen from:
                  'accuracy', 'log_loss', 'f1', 'average_precision', 'precision',
                  'roc_auc', 'mean_squared_error', 'r2', 'recall'
        Returns
        -------
        self : object
            Returns self.
        """
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.scorer_ = scorer[scoring]
        self.probs_ = []
        if self.split_data:
            self.X_train_, self.X_, self.y_train_, self.y_ = train_test_split(X, y,
                                                                              test_size=self.test_size,
                                                                              random_state=15)
        else:
            self.X_ = X
            self.y_ = y
            self.X_train_ = X
            self.y_train_ = y
        if self.models is None:
            print("Error! No model passed.")
        if not prefit:
            for j in range(len(self.models)):
                self.models[j].fit(self.X_train_, self.y_train_, **self.fit_params[j])
                print('Fitted model nr. {}'.format(j+1))
        else:
            self.X_ = X
            self.y_ = y

        for model in self.models:
            self.probs_.append(model.predict_proba(self.X_))

        # check whether weights need to be found
        if self.weights is None:
            # initialize weights
            init_weights = np.ones(len(self.models)) / len(self.models)

            # optimize score as a function of the weights using scipy.optimize.minimize
            # bounds = [(0, 1)] * len(self.models)
            # TODO find a way to minimize this function / optimize the weights
            then = time.time()
            opt_res = minimize(self.to_optimize_, init_weights[:-1], method='Nelder-Mead')
            print('The optimization took {} seconds.'.format(time.time() - then))
            print(opt_res)
            w_n = 1 - sum(abs(opt_res['x']))
            weights = np.append(abs(opt_res['x']), abs(w_n))
            self.weights = weights / sum(weights)

        # Return the classifier
        return self

    def to_optimize_(self, w_i):
        """
        Function to be optimized for finding the optimal weights

        Parameters
        ----------
        w_i : array-like, shape (n_models-1,)

        Returns
        -------
        float, the negative score calculated using the scoring function.
        """
        # calculate the weights
        w_n = 1 - sum(abs(w_i))
        weights = np.append(abs(w_i), abs(w_n))
        self.weights = weights / sum(weights)

        # calculate the new weighted class probabilities
        probs = np.array(self.probs_)
        probs = np.moveaxis(probs, 0, 1)
        probabilities = np.dot(self.weights, probs)

        # calculate the predictions given the class probabilities
        predictions = [self.classes_[x] for x in np.argmax(probabilities, axis=1)]

        # return the negative score
        score = self.scorer_(self.y_, predictions)
        return -1 * score

    def predict(self, X):
        """
        Predicts the most likely label for an input vector X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is returned
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        predictions = [self.classes_[x] for x in np.argmax(self.predict_proba(X), axis=1)]
        return predictions

    def predict_proba(self, X):
        """
        Calculates the probability for each clas for an input vector X.
        Parameters
        -------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples, n_classes)
            The propability for each label for each sample is returned.
        """
        if self.models is None:
            print("Error! No model passed.")

        probs = []

        for model in self.models:
            probs.append(model.predict_proba(X))

        probs = np.array(probs)
        probs = np.moveaxis(probs, 0, 1)

        probabilities = np.dot(self.weights, probs)
        # probabilities = np.dot(self.weights, probs) / np.sum(self.weights)

        return probabilities

    def get_weights(self):
        """

        Returns
        -------
        list
            List of the weights for the different classifiers.
        """
        return self.weights
