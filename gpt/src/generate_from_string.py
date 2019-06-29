#!/usr/bin/env python3

import json
import os
import numpy as np
import tensorflow as tf

from . import model, sample, encoder


def continue_string(raw_text, length=None,
                    model_name='345M',  # 117M
                    seed=None,
                    nsamples=1,
                    batch_size=1,
                    temperature=1.,
                    top_k=30,
                    models_dir='data/models',
                    ):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """
    print('\n\n\n\n', '---' * 30, 'using model', model_name)
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        context_tokens = enc.encode(raw_text)
        generated = 0
        gen_text = []
        for _ in range(nsamples // batch_size):
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]
            for i in range(batch_size):
                generated += 1
                text = enc.decode(out[i])
                gen_text.append(text)
    return gen_text[0]


def get_future_prediction(name):
    tag_seed = "children life death money career family"
    input_text = tag_seed + "In the year of {} {} will begin".format(np.random.randint(2020, 2070), name)
    time_markers = ["Next month {} will".format(name),
                    "In autumn it will be",
                    "By the end of winter {} will have".format(name),
                    "In the end, {} will die because of".format(name)]
    time_index = 0
    default_time_marker = "And then "

    new_text = input_text
    while len(new_text.split()) - len(input_text.split()) < 400:
        new_text += continue_string(new_text, length=80)
        period_position = new_text.rfind('.')
        new_text = new_text[:period_position+1]
        new_text += "\n"
        if time_index < len(time_markers):
            new_text += time_markers[time_index]
            time_index += 1
        else:
            new_text += default_time_marker

        if '<|endoftext|>' in new_text:
            new_text = new_text[:new_text.rfind('<|endoftext|>')]

    return new_text[len(tag_seed):new_text.rfind('\n')]


if __name__ == '__main__':
    # enc, output, context = start_model()
    generated = continue_string('You have tried for a long time to make this model work, but')[0]
    print('\n\n', generated)
