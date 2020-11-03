import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    https://tomekkorbak.com/2020/06/26/implementing-attention-in-pytorch/
    https://towardsdatascience.com/attention-seq2seq-with-pytorch-learning-to-invert-a-sequence-34faf4133e53

    code taken from:
    https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/attention.html
    Applies attention mechanism on the `context` using the `query`.

    **Thank you** to IBM for their initial implementation of :class:`Attention`. Here is
    their `License
    <https://github.com/IBM/pytorch-seq2seq/blob/master/LICENSE>`__.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`

    Example:

         >>> attention = Attention(256)
         >>> query = torch.randn(5, 1, 256)
         >>> context = torch.randn(5, 5, 256)
         >>> output, weights = attention(query, context)
         >>> output.size()
         torch.Size([5, 1, 256])
         >>> weights.size()
         torch.Size([5, 1, 5])
    """

    def __init__(self, dimensions, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()

    def forward(self, query, context, context_lengths=None):
        """
        Args:
            query (:class:`torch.FloatTensor` [batch size, output length, dimensions]): Sequence of
                queries to query the context.
            context (:class:`torch.FloatTensor` [batch size, query length, dimensions]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, output_len, dimensions = query.size()
        query_len = context.size(1)

        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        # TODO: Include mask on PADDING_INDEX?

        # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        if context_lengths is not None:
            mask = torch.arange(query_len, dtype=torch.long, device=context_lengths.get_device()
                                ).expand(batch_size, query_len) < context_lengths
            attention_scores = attention_scores.masked_fill(mask, float("-inf"))

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
        # (batch_size, output_len, dimensions)
        mix = torch.bmm(attention_weights, context)

        # concat -> (batch_size * output_len, 2*dimensions)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)

        return output, attention_weights


class AttnDecoderRnn(nn.Module):
    def __init__(self, output_size, hidden_size, rnn_layers=(0, 2), embedding_dropout_p=0.1, sos=0, eos=1, pad=-1):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.sos = sos
        self.eos = eos
        self.pad = pad

        embedding = nn.Embedding(self.output_size, self.hidden_size)
        if embedding_dropout_p == 0:
            self.embedding = embedding
        else:
            self.embedding = nn.Sequential(*[embedding, nn.Dropout(embedding_dropout_p)])

        self.attn = Attention(hidden_size)

        assert len(rnn_layers) == 2
        self.num_rnn_layers = rnn_layers
        self.rnn1 = nn.GRU(self.hidden_size, self.hidden_size, num_layers=rnn_layers[0], batch_first=True) if \
            rnn_layers[0] > 0 else None
        self.rnn2 = nn.GRU(self.hidden_size, self.hidden_size, num_layers=rnn_layers[1], batch_first=True) if \
            rnn_layers[1] > 0 else None
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, encoder_outputs, encoder_output_lengths, input_labels=None, input_labels_lenghts=None,
                max_length=100):
        teacher_forcing = input_labels is not None
        batch_size = encoder_outputs.shape[0]
        input_labels_start = torch.full([batch_size, 1], fill_value=self.sos, dtype=torch.long)
        # input_labels_end = torch.full([batch_size, 1], fill_value=self.eos, dtype=torch.long)
        hidden1 = None
        hidden2 = None
        if teacher_forcing:
            input_labels = torch.cat((input_labels_start, input_labels), dim=1)
            embedded = self.embedding(input_labels)
            output = self.dropout(embedded)
            if self.rnn1 is not None:
                # TODO: pack
                output, hidden1 = self.rnn1(output, hidden1)
            output, _ = self.attn(output, encoder_outputs)
            if self.rnn2 is not None:
                output, hidden2 = self.rnn2(output, hidden2)
            output = F.log_softmax(self.fc(output), dim=-1)
        else:
            input_labels = input_labels_start
            results = []
            for _ in range(max_length):
                embedded = self.embedding(input_labels)
                output = self.dropout(embedded)
                if self.rnn1 is not None:
                    # TODO: pack
                    output, hidden1 = self.rnn1(output, hidden1)
                output, _ = self.attn(output, encoder_outputs)
                if self.rnn2 is not None:
                    output, hidden2 = self.rnn2(output, hidden2)
                output = F.log_softmax(self.fc(output), dim=-1)
                results.append(output)
                input_labels = torch.topk(output, 1, dim=-1)[1].unsqueeze(1)
            output = torch.stack(results, dim=1)
        return output
