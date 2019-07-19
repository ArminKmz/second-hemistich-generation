import tensorflow as tf
import random

class Embedding:
  def __init__(self, input_size, embedding_size):
    with tf.variable_scope("Embedding", reuse=tf.AUTO_REUSE):
      self.word_embeddings = tf.get_variable(name='word_embedding', shape=[input_size, embedding_size], initializer=tf.random_uniform_initializer(-1, 1))

  def __call__(self, input):
    with tf.variable_scope("Embedding", reuse=tf.AUTO_REUSE):
      embedded = tf.nn.embedding_lookup(self.word_embeddings, input)
      return embedded

class EncoderRNN:
  def __init__(self, input_size, hidden_size, n_layers):
    with tf.variable_scope("EncoderRNN", reuse=tf.AUTO_REUSE):
      self.hidden_size = hidden_size
      self.input_size = input_size
      self.n_layers = n_layers

      cells = [tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, use_peepholes=True, state_is_tuple=True) for n in range(n_layers)]
      self.lstm = tf.contrib.rnn.MultiRNNCell(cells)

  def __call__(self, input, hidden):
    with tf.variable_scope("EncoderRNN", reuse=tf.AUTO_REUSE):
      output, hidden = tf.nn.dynamic_rnn(self.lstm, input, initial_state=hidden)
      return output, hidden

  def init_hidden(self, n_batches):
    return self.lstm.zero_state(n_batches, dtype=tf.float32)

class DecoderRNN:
  def __init__(self, input_size, hidden_size, output_size, n_layers):
    with tf.variable_scope("DecoderRNN", reuse=tf.AUTO_REUSE):
      self.input_size = input_size
      self.hidden_size = hidden_size
      self.output_size = output_size
      self.n_layers = n_layers
      cells = [tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, use_peepholes=True, state_is_tuple=True) for n in range(n_layers)]
      self.lstm = tf.contrib.rnn.MultiRNNCell(cells)
      self.W = tf.get_variable('W', shape=[hidden_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
      self.b = tf.get_variable('b', shape=[output_size], initializer=tf.zeros_initializer())

  def __call__(self, input, hidden):
    with tf.variable_scope("DecoderRNN", reuse=tf.AUTO_REUSE):
      output, hidden = self.lstm(input, hidden)
      output = tf.tensordot(output, self.W, axes=1) + self.b
      return output, hidden

class AttnDecoderRNN:
  def __init__(self, input_size, hidden_size, output_size, n_layers):
    with tf.variable_scope("AttnDecoderRNN", reuse=tf.AUTO_REUSE):
      self.input_size = input_size
      self.hidden_size = hidden_size
      self.output_size = output_size
      self.n_layers = n_layers

      cells = [tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, use_peepholes=True, state_is_tuple=True) for n in range(n_layers)]
      self.lstm = tf.contrib.rnn.MultiRNNCell(cells)
      self.W = tf.get_variable('W', shape=[hidden_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
      self.b = tf.get_variable('b', shape=[output_size], initializer=tf.zeros_initializer())

  def __call__(self, input, hidden):
    with tf.variable_scope("AttnDecoderRNN", reuse=tf.AUTO_REUSE):
      output, hidden = self.attn_cell(input, hidden)
      output = tf.tensordot(output, self.W, axes=1) + self.b
      return output, hidden

  def init_hidden(self, batch_size, hidden):
    return self.attn_cell.get_initial_state(hidden.h, batch_size, dtype=tf.float32)

  def set_memory(self, memory):
    with tf.variable_scope("AttnDecoderRNN", reuse=tf.AUTO_REUSE):
      self.attn = tf.contrib.seq2seq.BahdanauAttention(num_units=self.hidden_size, memory=memory)
      self.attn_cell = tf.contrib.seq2seq.AttentionWrapper(self.lstm, self.attn, alignment_history=True)

class Seq2SeqModel:
  def __init__(self, encoder, decoder, embedding, max_len, begin_char):
    with tf.variable_scope("Seq2Seq", reuse=tf.AUTO_REUSE):
      self.encoder = encoder
      self.decoder = decoder
      self.embedding = embedding

      attention = (type(decoder) == AttnDecoderRNN)

      self.input_data = tf.placeholder(shape=[None, max_len], dtype=tf.int64)
      self.target_data = tf.placeholder(shape=[None, max_len], dtype=tf.int64)
      self.teacher_forcing = tf.placeholder(tf.float32, [])
      self.learning_rate = tf.placeholder(tf.float32, [])

      output_size = self.decoder.output_size
      batch_size = tf.shape(self.input_data)[0]

      enc_embbeded_input = self.embedding(self.input_data)
      enc_output, enc_hidden = self.encoder(enc_embbeded_input, self.encoder.init_hidden(batch_size))

      dec_hidden = enc_hidden

      dec_input = tf.tile([begin_char], tf.shape(self.input_data)[0:1])
      dec_input = tf.dtypes.cast(dec_input, tf.int64)
      prediction = [dec_input]
      self.loss = 0

      if attention:
        self.decoder.set_memory(enc_output)
        dec_hidden = self.decoder.init_hidden(batch_size, dec_hidden[-1])

      for t in range(1, max_len):
        dec_embedded_input = self.embedding(dec_input)
        dec_output, dec_hidden = self.decoder(dec_embedded_input, dec_hidden)
        self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                      labels=tf.one_hot(self.target_data[:, t], depth=output_size, dtype=tf.float32),
                      logits=dec_output))
        next_char = tf.argmax(dec_output, axis=1)
        prediction.append(next_char)
        use_teacher_forcing = tf.cond(random.random() < self.teacher_forcing, lambda: True, lambda: False)
        dec_input = tf.cond(use_teacher_forcing, lambda: self.target_data[:, t], lambda: next_char)
      if attention:
        self.attn_weights = dec_hidden.alignment_history.stack()
      self.prediction = tf.transpose(tf.convert_to_tensor(prediction))
      self.loss /= max_len
      optimizer = tf.train.AdamOptimizer(self.learning_rate)
      self.opt = optimizer.minimize(self.loss)
