# %%
import tensorflow as tf

# set memory growth to avoid OOM while running GA
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# %% [markdown]
# # Load data and create tokenizers

# %%
import tensorflow as tf
import sentencepiece as spm
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from pathlib import Path

ds_name = "rus_300000.csv"
ds_size: int

max_len = 25

if ds_name.count('_') == 2:
    und_1 = ds_name.find('_')
    und_2 = ds_name.find('_', und_1 + 1)
    ds_size = int(ds_name[und_1+1:und_2])

else:
    ds_size = int(ds_name[4:-4])

lines = open(Path(ds_name), encoding="utf-8").read().strip().split("\n")
eng, rus = [], []
for line in lines:
    parts = line.split('\t')
    if len(parts) == 3:
        _, en, ru = parts
    elif len(parts) == 2:
        en, ru = parts
    eng.append(en)
    rus.append(ru)

with open('bpe_input.txt', 'w', encoding='utf-8') as f:
    for e, r in zip(eng, rus):
        f.write(e + '\n')
        f.write(r + '\n')

spm.SentencePieceTrainer.train(
    input='bpe_input.txt',
    model_prefix='bpe',
    vocab_size=12000,
    character_coverage=1.0,
    model_type='bpe',
    user_defined_symbols=['<start>', 'pad', '<end>']
)

sp = spm.SentencePieceProcessor()
sp.load('bpe.model')
VOCAB_SIZE = sp.get_piece_size()
start_id = sp.piece_to_id('<start>')
end_id   = sp.piece_to_id('<end>')
pad_id   = sp.piece_to_id('<pad>')

# English inputs
seq_en = [sp.encode(e, out_type=int) for e in eng]
# Russian decoder input/output
rus_in_ids  = [[start_id] + sp.encode(r, out_type=int) for r in rus]
rus_out_ids = [sp.encode(r, out_type=int) + [end_id] for r in rus]

input_tensor     = pad_sequences(seq_en,       maxlen=max_len, padding='post', value=pad_id)
target_tensor_in = pad_sequences(rus_in_ids,   maxlen=max_len, padding='post', value=pad_id)
target_tensor_out= pad_sequences(rus_out_ids,  maxlen=max_len, padding='post', value=pad_id)

# %% [markdown]
# # Load saved model

# %%
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Dense, Dropout,
    Layer, LayerNormalization, MultiHeadAttention, Add
)
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable()
class PositionalEncoding(Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        # precompute the (1, seq_len, d_model) constant
        pos = np.arange(seq_len)[:, np.newaxis]
        i   = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        angles = pos * angle_rates
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        self.pos_encoding = tf.constant(angles[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        # x shape = (batch, seq_len, d_model)
        return x + self.pos_encoding

    def get_config(self):
        # so that load_model can reinstantiate this layer
        config = super().get_config()
        config.update({
            "seq_len": int(self.pos_encoding.shape[1]),
            "d_model": int(self.pos_encoding.shape[2]),
        })
        return config

@register_keras_serializable()
class PaddingMask(Layer):
    def __init__(self, pad_id, **kwargs):
        super().__init__(**kwargs)
        self.pad_id = pad_id

    def call(self, x):
        # x shape = (batch, seq_len)
        mask = tf.cast(tf.not_equal(x, self.pad_id), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def get_config(self):
        config = super().get_config()
        config.update({"pad_id": self.pad_id})
        return config


def feed_forward(x, d_model, d_ff, dropout_rate=0.1):
    ff = Dense(d_ff, activation="relu")(x)
    ff = Dropout(dropout_rate)(ff)
    ff = Dense(d_model)(ff)
    return ff


def build_transformer(
    vocab_in,
    vocab_out,
    seq_len,
    num_layers,
    d_model,
    num_heads,
    d_ff,
    dropout_rate=0.1,
    pad_id=0
):
    # Inputs
    enc_inputs = Input(shape=(seq_len,), name='encoder_inputs')
    dec_inputs = Input(shape=(seq_len,), name='decoder_inputs')

    # Embeddings + PositionalEncoding
    enc_embed = Embedding(vocab_in, d_model, name='encoder_embedding')(enc_inputs)
    dec_embed = Embedding(vocab_out, d_model, name='decoder_embedding')(dec_inputs)

    enc_embed = PositionalEncoding(seq_len, d_model, name='pos_enc')(enc_embed)
    dec_embed = PositionalEncoding(seq_len, d_model, name='pos_dec')(dec_embed)

    # Masks
    enc_padding_mask = PaddingMask(pad_id, name='enc_pad_mask')(enc_inputs)
    dec_padding_mask = PaddingMask(pad_id, name='dec_pad_mask')(dec_inputs)

    # Encoder
    x = enc_embed
    for i in range(num_layers):
        attn = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model//num_heads,
            name=f'enc_mha_{i}'
        )(x, x, attention_mask=enc_padding_mask)
        attn = Dropout(dropout_rate)(attn)
        x = LayerNormalization(epsilon=1e-6)(Add()([x, attn]))

        ff = feed_forward(x, d_model, d_ff, dropout_rate)
        x = LayerNormalization(epsilon=1e-6)(Add()([x, ff]))
    enc_out = x

    # Decoder
    x = dec_embed
    for i in range(num_layers):
        # self‑attention
        attn1 = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model//num_heads,
            name=f'dec_mha1_{i}'
        )(x, x, attention_mask=dec_padding_mask, use_causal_mask=True)
        attn1 = Dropout(dropout_rate)(attn1)
        x = LayerNormalization(epsilon=1e-6)(Add()([x, attn1]))

        # cross‑attention
        attn2 = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model//num_heads,
            name=f'dec_mha2_{i}'
        )(x, enc_out, attention_mask=enc_padding_mask)
        attn2 = Dropout(dropout_rate)(attn2)
        x = LayerNormalization(epsilon=1e-6)(Add()([x, attn2]))

        # feed‑forward
        ff = feed_forward(x, d_model, d_ff, dropout_rate)
        x = LayerNormalization(epsilon=1e-6)(Add()([x, ff]))
    dec_out = x

    outputs = Dense(vocab_out, activation='softmax', name='outputs')(dec_out)
    return Model([enc_inputs, dec_inputs], outputs, name='transformer')


# Hypers:
seq_len 	    = max_len
d_model 	    = 384 # 128
num_heads     = 8 # 4
d_ff 	        = 512
num_layers    = 1 # 3

dropout_rate  = 0.153 # 0.12

batch_size    = 64
epochs 	      = 6
learning_rate = 0.0001 # 0.001

decay_steps   = 600
decay_rate    = 0.97


tf.random.set_seed(42)
model = build_transformer(
    num_layers=num_layers,
    vocab_in=VOCAB_SIZE,
    vocab_out=VOCAB_SIZE,
    seq_len=seq_len,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    dropout_rate=dropout_rate
)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    staircase=False
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

base_scc = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=False,
    reduction='none'            # per‑token losses
)

start_id = sp.piece_to_id('<start>')
end_id   = sp.piece_to_id('<end>')
pad_id   = sp.piece_to_id('<pad>')

def masked_scc(y_true, y_pred):
    # y_true: shape = (batch, seq_len), ints
    # y_pred: shape = (batch, seq_len, V)
    per_token = base_scc(y_true, y_pred)   # shape = (batch, seq_len)
    # 1 is where we want loss, 0 otherwise
    valid = tf.logical_and(
        tf.not_equal(y_true, start_id),
        tf.not_equal(y_true, pad_id)
    )
    mask = tf.cast(valid, tf.float32)
    return tf.reduce_sum(per_token * mask) / tf.reduce_sum(mask)

def masked_accuracy(y_true, y_pred):
    y_true_int = tf.cast(y_true, tf.int64)            # shape=(batch,seq_len), int64
    pred_ids   = tf.argmax(y_pred, axis=-1)           # shape=(batch,seq_len), int64

    matches = tf.equal(y_true_int, pred_ids)          # shape=(batch,seq_len), bool

    valid_positions = tf.logical_and(
        tf.not_equal(y_true_int, start_id),
        tf.not_equal(y_true_int, pad_id)
    )                                                  # shape=(batch,seq_len)
    mask = tf.cast(valid_positions, tf.float32)        # 1.0 for real tokens, 0.0 for pad/<start>

    correct = tf.cast(matches, tf.float32) * mask      # float32
    return tf.reduce_sum(correct) / tf.reduce_sum(mask)


model.compile(
    optimizer=optimizer,
    loss=masked_scc,
    metrics=[masked_accuracy]
)

model.summary()

# %% [markdown]
# ---
# ---
# ---

# %% [markdown]
# # Train model

# %% [markdown]
# ## Define callback to make predictions after each epoch

# %%
import sentencepiece as spm
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import Callback
from pathlib import Path

# --- 1. Load or train your SentencePiece model ---
sp = spm.SentencePieceProcessor()
sp.load('bpe.model')  # ensure this model has <pad>,<start>,<end> as user_defined_symbols

# Special token IDs
PAD_ID   = sp.piece_to_id('<pad>')
START_ID = sp.piece_to_id('<start>')
END_ID   = sp.piece_to_id('<end>')

# --- 2. Utility for encoding inputs ---
def encode_en(sentence):
    # encode English text to subword IDs
    return sp.encode(sentence, out_type=int)

# --- 3. Prediction callback using SentencePiece ---
class PredictionCallback(Callback):
    def __init__(self, sample_inputs, sp_processor, max_len):
        super().__init__()
        self.sample_inputs = sample_inputs
        self.sp = sp_processor
        self.max_len = max_len

    def on_epoch_end(self, epoch, logs=None):
        print(f'\n〰 Epoch {epoch+1} translation samples 〰')
        for sent in self.sample_inputs:
            print(self._translate(sent))

    def _translate(self, sentence):
        # Encode source
        enc_ids = encode_en(sentence)
        enc_seq = pad_sequences([enc_ids], maxlen=self.max_len, padding='post', value=PAD_ID)

        # Initialize decoder input with <start>
        dec_input = [START_ID]
        for _ in range(self.max_len - 1):
            dec_seq = pad_sequences([dec_input], maxlen=self.max_len, padding='post', value=PAD_ID)
            # model expects [encoder_inputs, decoder_inputs]
            preds = self.model.predict([enc_seq, dec_seq], verbose=0)
            # pick next token
            next_id = int(np.argmax(preds[0][len(dec_input)-1]))
            if next_id == END_ID:
                break
            dec_input.append(next_id)

        # Convert IDs back to pieces
        pieces = [self.sp.id_to_piece(i) for i in dec_input[1:]]
        # Join, but merge subword ballots (SentencePiece uses '▁' prefix to mark word boundaries)
        translation = self.sp.decode(dec_input[1:])
        return f'{sentence} -> {translation}'

# sample_sentences = [
#     "What is your name?",
#     "Do you like music?",
#     "Could you help me?",
#     "They will not be happy.",
#     "I have to do that right now.",
#     "She would like to watch a movie this evening.",
#     "Where are you headed, Tom?",
#     "He arrived as soon as he could.",
#     "It have been raining for at least three hours.",
#     "Why didn’t you tell me that you were planning to leave early?"
# ]

sample_sentences = [
    "What are the benefits of regular exercise?",
    "What do you think about climate change?",
    "She is passionate about environmental issues.",
    "What are the main causes of global warming?",
    "What are the implications of artificial intelligence?",
    "He has a unique perspective on the topic.",
    "He enjoys hiking in the mountains.",
    "What are the challenges of remote work?",
    "How can we improve mental health awareness?",
    "The dog barks at strangers.",
    "She wrote an article about the importance of education."
]

pred_cb = PredictionCallback(sample_sentences, sp, max_len)

print_hypers = lambda: print(f'\n{ds_name=} seq_len={max_len} {d_model=} {num_layers=} {num_heads=} {batch_size=} {dropout_rate=} {learning_rate=} {decay_steps=} {decay_rate=} {VOCAB_SIZE=}\n')

# %% [markdown]
# ## Run training

# %%
print_hypers()
model.fit(
    [input_tensor, target_tensor_in], target_tensor_out,
    batch_size=batch_size, epochs=8,
    validation_split=0.1, callbacks=[pred_cb]
)
print_hypers()

# %%
print_hypers()
model.fit(
    [input_tensor, target_tensor_in], target_tensor_out,
    batch_size=batch_size, epochs=2,
    validation_split=0.1, callbacks=[pred_cb]
)
print_hypers()

# %% [markdown]
# ## Save the model

# %% [markdown]
# Saving the model and tokenizer files

# %%
model.save('my_transformer_model.keras')

from google.colab import files

files.download('my_transformer_model.keras')
files.download('bpe.model')
files.download('bpe.vocab')

# %% [markdown]
# ---------

# %% [markdown]
# # Genetic algorithm

# %%
!pip install pygad -q

import pygad
import tensorflow as tf

# hyperparameter search space
gene_space = [
    # num_layers: choose from [1, 2, 3, 4]
    [1, 2, 3, 4],
    # d_model: choose from [128, 256, 384, 512]
    {"low": 128, "high": 512, "step": 128},
    # num_heads: choose from [4, 6, 8]
    [4, 6, 8],
    # d_ff: choose from [256, 512, 768, 1024]
    {"low": 256, "high": 1024, "step": 256},
    # dropout_rate: float in [0.0, 0.4]
    {"low": 0.0, "high": 0.4},
    # learning_rate: float in [1e-4, 1e-2]
    {"low": 1e-5, "high": 5e-2},
    # batch_size: choose from [64, 128, 256]
    [64, 128, 256],
]


def fitness_func(ga, solution, sol_idx):
    num_layers, d_model, num_heads, d_ff, dropout, lr, batch_size = solution
    batch_size = int(batch_size)

    # build and split the dataset per solution
    full_ds = tf.data.Dataset.from_tensor_slices(
        ((input_tensor, target_tensor_in), target_tensor_out
    ).shuffle(10000, reshuffle_each_iteration=True)
    val_size = int(0.1 * len(input_tensor))
    val_ds   = full_ds.take(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    train_ds = full_ds.skip(val_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # build and compile model
    model = build_transformer(
        vocab_in=VOCAB_SIZE,
        vocab_out=VOCAB_SIZE,
        seq_len=max_len,
        num_layers=int(num_layers),
        d_model=int(d_model),
        num_heads=int(num_heads),
        d_ff=int(d_ff),
        dropout_rate=float(dropout)
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(float(lr)),
                  loss=masked_scc,
                  metrics=[masked_accuracy])

    history = model.fit(
        train_ds,
        epochs=1,
        validation_data=val_ds,
        verbose=1
    )

    val_acc = history.history["val_masked_accuracy"][-1]

    # clean up
    del history, model
    tf.keras.backend.clear_session()
    import gc; gc.collect()

    return val_acc


# progress callback
def on_gen(ga_instance):
    gen = ga_instance.generations_completed
    best_fit = ga_instance.best_solutions_fitness[-1]
    print(f"Generation {gen:2d} — Best fitness: {best_fit:.4f}")

ga = pygad.GA(
    gene_space=gene_space,
    num_generations=3,
    num_parents_mating=4,
    sol_per_pop=4,
    num_genes=len(gene_space),
    fitness_func=fitness_func,
    mutation_percent_genes=20,
    parent_selection_type="tournament",
    crossover_type="single_point",
    on_generation=on_gen
)

ga.run()

# print best solution
best_solution, best_fitness, _ = ga.best_solution()
print("\n=== Best GA solution ===")
print(f"num_layers     = {int(best_solution[0])}")
print(f"d_model        = {int(best_solution[1])}")
print(f"num_heads      = {int(best_solution[2])}")
print(f"d_ff           = {int(best_solution[3])}")
print(f"dropout_rate   = {best_solution[4]:.4f}")
print(f"learning_rate  = {best_solution[5]:.6f}")
print(f"batch_size     = {int(best_solution[6])}")
print(f"Validation accuracy = {best_fitness:.4f}")


