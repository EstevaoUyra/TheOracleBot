import numpy as np


def glove_generator(df, batch_size, method='average', embedder=None, only_labels=False):

    from deeppavlov.models.embedders.glove_embedder import GloVeEmbedder
    assert type(df.sentence[0]) is not str
    data = df.copy()

    if embedder is None:
        embedder = GloVeEmbedder(load_path="data/models/glove.txt",
                                 pad_zero=False)
    i = 0
    while True:

        batch_labels = data.label[i: i + batch_size]

        if only_labels:
            yield batch_labels

        else:
            batch_questions = embedder(data.question[i: i + batch_size])
            batch_sents = embedder(data.sentence[i: i + batch_size])

            if method == 'concat':
                batch_questions = np.vstack([Q.ravel() for Q in batch_questions])
                batch_sents = np.vstack([S.ravel() for S in batch_sents])
            elif method =='average':
                batch_questions = np.vstack([np.array(Q).mean(axis=0) for Q in batch_questions])
                batch_sents = np.vstack([np.array(S).mean(axis=0) for S in batch_sents])
            else:
                raise NotImplementedError

            yield np.hstack((batch_questions, batch_sents)), batch_labels

        i += batch_size
        if i >= data.shape[0]:
            data = data.sample(frac=1.)
            i = 0
