from flask import Flask, request, render_template_string
import sentencepiece as spm
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time
from tensorflow.keras.layers import Layer


keras_model_filename = 'my_transformer_model_subword_bugfixed.keras'
bpe_model_filename = 'bpe_subword_bugfixed.model'

sp = spm.SentencePieceProcessor()
sp.load(bpe_model_filename)
PAD_ID   = sp.piece_to_id('<pad>')
START_ID = sp.piece_to_id('<start>')
END_ID   = sp.piece_to_id('<end>')

def masked_scc(y_true, y_pred):
    base = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
    per_token = base(y_true, y_pred)
    valid = tf.logical_and(
        tf.not_equal(y_true, START_ID),
        tf.not_equal(y_true, PAD_ID)
    )
    mask = tf.cast(valid, tf.float32)
    return tf.reduce_sum(per_token * mask) / tf.reduce_sum(mask)

def masked_accuracy(y_true, y_pred):
    y_true_int = tf.cast(y_true, tf.int64)
    pred_ids = tf.argmax(y_pred, axis=-1)
    matches = tf.equal(y_true_int, pred_ids)
    valid = tf.logical_and(
        tf.not_equal(y_true, START_ID),
        tf.not_equal(y_true, PAD_ID)
    )
    mask = tf.cast(valid, tf.float32)
    return tf.reduce_sum(tf.cast(matches, tf.float32) * mask) / tf.reduce_sum(mask)

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

model = load_model(
    keras_model_filename,
    custom_objects={
        'masked_scc':        masked_scc,
        'masked_accuracy':   masked_accuracy,
        'Custom>PositionalEncoding': PositionalEncoding,
        'Custom>PaddingMask':        PaddingMask
    },
    safe_mode=False
)

max_len = 25

def decode_sequence(input_sentence):
    enc_ids = sp.encode(input_sentence, out_type=int)
    enc_seq = pad_sequences([enc_ids], maxlen=max_len, padding='post', value=PAD_ID)
    dec_input = [START_ID]
    for _ in range(max_len - 1):
        dec_seq = pad_sequences([dec_input], maxlen=max_len, padding='post', value=PAD_ID)
        preds = model.predict([enc_seq, dec_seq], verbose=0)
        next_id = int(np.argmax(preds[0][len(dec_input)-1]))
        if next_id == END_ID:
            break
        dec_input.append(next_id)
    return sp.decode(dec_input[1:])


# Flask app setup
app = Flask(__name__)

template = '''
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Simple Translator</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">

    <style>
      body {
        font-family: 'Inter', sans-serif;
        background: var(--bg);
        color: var(--text);
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        margin: 0;
        transition: background 0.3s, color 0.3s;
      }
      :root {
        --bg: #f4f6f8;
        --container-bg: #ffffff;
        --text: #333333;
        --card-shadow: rgba(0,0,0,0.1);
        --button-bg: #4a90e2;
        --button-hover: #357ab8;
        --border: #ccc;
        --textarea-bg: #ffffff;
        --toggle-opacity: 0.4;
        --faded-text: #999999;
      }
      .dark {
        --bg: #2b2b2b;
        --container-bg: #3c3c3c;
        --text: #e0e0e0;
        --card-shadow: rgba(0,0,0,0.5);
        --button-bg: #6a8caf;
        --button-hover: #587b9a;
        --border: #555;
        --textarea-bg: #4a4a4a;
        --toggle-opacity: 0.4;
        --faded-text: #777;
      }
      .container {
        background: var(--container-bg);
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 16px var(--card-shadow);
        width: 100%;
        max-width: 600px;
        position: relative;
      }
      h1, h2 { margin-top: 0; }
      textarea {
        width: 100%;
        height: 120px;
        font-size: 1rem;
        padding: 10px;
        /* Use a consistent border width to prevent shifting */
        border: 3px solid var(--border);
        border-radius: 8px;
        resize: none;
        background: var(--textarea-bg);
        color: var(--text);
        box-sizing: border-box;
        /* Remove width transition; only color changes on focus */
        transition: border-color 0.2s;
      }
      /* Disable red underlines */
      textarea {
        spellcheck: false;
      }
      /* Highlight border on hover/focus of input field */
      #input-textarea:hover,
      #input-textarea:focus {
        outline: none;
        border-color: gray;
      }
      .actions {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        margin-top: 10px;
      }
      .translate-btn {
        background: var(--button-bg);
        color: #fff;
        padding: 10px 20px;
        font-size: 1rem;
        border: 2px solid transparent;
        border-radius: 8px;
        cursor: pointer;
        transition: background 0.3s, border-color 0.3s;
      }
      .translate-btn:hover { background: var(--button-hover); border-color: #fff; }
      .translate-btn:active { background: var(--button-hover); border-color: #000; }
      .theme-toggle {
        position: fixed;
        top: 16px;
        right: 16px;
        background: transparent;
        border: none;
        font-size: 1.4rem;
        cursor: pointer;
        opacity: var(--toggle-opacity);
        transition: opacity 0.3s;
        color: var(--text);
      }
      .theme-toggle:hover { opacity: 1; }
      .result { margin-top: 20px; transition: color 0.3s; }
      .time {
        font-size: 0.85rem;
        color: #888;
        margin-top: 4px;
      }
    </style>
    <script>
      document.addEventListener('DOMContentLoaded', () => {
        const form = document.getElementById('translate-form');
        const input = document.getElementById('input-textarea');
        const themeToggle = document.getElementById('theme-toggle');
        
        const outputDiv  = document.getElementById('output-area');
        const outputArea = outputDiv
          ? outputDiv.getElementsByTagName('textarea')[0]
          : null;

        const darkMode = localStorage.getItem('darkMode') === 'true';
        if (darkMode) {
          document.body.classList.add('dark');
          themeToggle.textContent = '☀️';
        } else {
          themeToggle.textContent = '🌙';
        }

        // on new input, fade previous translation
        input.addEventListener('input', () => {
          if (outputArea) {
            outputArea.style.color = 'var(--faded-text)';
          }
        });

        input.addEventListener('keypress', event => {
          if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            form.submit();
          }
        });

        themeToggle.addEventListener('click', () => {
          document.body.classList.toggle('dark');
          const isDark = document.body.classList.contains('dark');
          localStorage.setItem('darkMode', isDark);
          themeToggle.textContent = isDark ? '☀️' : '🌙';
        });
      });
    </script>
  </head>
  <body>
    <button id="theme-toggle" class="theme-toggle">🌙</button>
    <div class="container">
      <h1>Translate Text</h1>
      <form id="translate-form" method="post">
        <textarea id="input-textarea" name="input_text" spellcheck="false" placeholder="Enter text to translate...">{{ input_text or '' }}</textarea>
        <div class="actions">
          <button type="submit" class="translate-btn">Translate</button>
        </div>
      </form>
      {% if output_text is not none %}
      <div class="result" id="output-area">
        <h2>Translation:</h2>
        <textarea readonly>{{ output_text }}</textarea>
        <div class="time">Model inference time: {{ elapsed_time }} seconds</div>
      </div>
      {% endif %}
    </div>
  </body>
  <script>
    const themeToggle = document.getElementById('theme-toggle');

    // 1) Initial state: read localStorage and set html.dark if needed
    const isDark = localStorage.getItem('darkMode') === 'true';
    document.documentElement.classList.toggle('dark', isDark);
    themeToggle.textContent = isDark ? '☀️' : '🌙';

    // 2) Click handler: flip the class on <html>, update localStorage + button icon
    themeToggle.addEventListener('click', () => {
      const nowDark = document.documentElement.classList.toggle('dark');
      localStorage.setItem('darkMode', nowDark);
      themeToggle.textContent = nowDark ? '☀️' : '🌙';
    });
  </script>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    input_text = ''
    output_text = None
    elapsed_time = None
    if request.method == 'POST':
        input_text = request.form.get('input_text', '')
        if input_text.strip():
            start = time.time()
            output_text = decode_sequence(input_text)
            elapsed_time = f"{(time.time() - start):.3f}"
    return render_template_string(
        template,
        input_text=input_text,
        output_text=output_text,
        elapsed_time=elapsed_time
    )

if __name__ == '__main__':
    app.run(debug=True)
