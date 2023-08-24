import torch
import torch.nn as nn
import numpy as np
from Gen_atom import atom
from dataset import TrainDataset
import math


class Multiplication(nn.Module):
    def __init__(self):
        super(Multiplication, self).__init__()

    @staticmethod
    def forward(atom_embedding, position_embedding):
        return atom_embedding * position_embedding


class Concatenate(nn.Module):
    def __init__(self):
        super(Concatenate, self).__init__()

    @staticmethod
    def forward(atom_embedding, position_embedding):
        return np.concatenate((atom_embedding, position_embedding))


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x, y):
        token_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        position_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        # Concatenate along the last dimension
        input_tensor = torch.cat((token_tensor, position_tensor), dim=-1)
        return self.layers(input_tensor).squeeze(0)


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], \
                                        query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # Get the dot product between queries and keys, and apply mask
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # Apply softmax
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        # Get the weighted average of the values
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


class PadAtom(object):
    def __init__(self):
        self.pad_name = 'atom_pad'


class PadPos(object):
    def __init__(self):
        self.pad_name = 'pos_pad'


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """

    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian distance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter) ** 2 /
                      self.var ** 2)


def word_pos_embedding(data):
    # Decide which operation you are going to use
    operation = SelfAttention(25, 5)
    processed_data = []
    gdf = GaussianDistance(0, 24, 1)
    for sentence in data:
        pad_atom = []
        pad_pos = []
        tokens = []
        positions = []
        for element in sentence:
            if isinstance(element, str):

                # Apply get_property() to the string token
                processed_token = atom(element).get_property()
                tokens.append(processed_token)

            elif isinstance(element, list):
                # Apply gdf.expand() to the list (position)
                element = np.array(element)
                x_position, y_position, z_position = gdf.expand(element)
                processed_position = x_position + y_position + z_position
                positions.append(processed_position)

            elif isinstance(element, PadPos):
                # Normalize pad position
                pad_pos.append(np.zeros(25))
            elif isinstance(element, PadAtom):
                # Normalize pad atom
                pad_atom.append(np.zeros(25))

        tokens = torch.tensor(tokens, dtype=torch.float32)
        positions = torch.tensor(positions, dtype=torch.float32)

        tokens = tokens.unsqueeze(0) # Add batch dimension
        positions = positions.unsqueeze(0) # Add batch dimension

        sentence_embed = operation(values=tokens, keys=tokens, query=positions, mask=None)
        processed_data.append(sentence_embed.squeeze(0))  # Remove batch dimension

    return torch.stack(processed_data)


class MyTransform(nn.Module):
    def __init__(self, input_size=1, output_size=50):
        super(MyTransform, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return x


class regressoionHead(nn.Module):

    def __init__(self, d_embedding: int):
        super().__init__()
        self.layer1 = nn.Linear(d_embedding, d_embedding // 2)
        self.layer2 = nn.Linear(d_embedding // 2, d_embedding // 4)
        self.layer3 = nn.Linear(d_embedding // 4, d_embedding // 8)
        self.layer4 = nn.Linear(d_embedding // 8, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = self.dropout(self.relu(self.layer1(x)))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))

        return self.layer4(x)


class Transformer(nn.Module):
    def __init__(self, d_model, nhead, nhid, nlayers, dropout=0.1,
                 device='cuda'):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.device = device
        # Define target encoding here
        self.trg_embedding = MyTransform(1, 25).to(self.device)

        # Define transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, nhid,
                                                   dropout).to(self.device)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         nlayers).to(
            self.device)

        # Define transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, nhid,
                                                   dropout).to(self.device)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer,
                                                         nlayers).to(
            self.device)
        self.output_layer = regressoionHead(25)
        # self.output_layer = nn.Linear(1,25)

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        nn.init.xavier_normal_(self.token_encoder.weight)

    def generate_padding_mask(self, src):
        mask = []
        for sentence in src:
            sent_mask = []
            # step over by 2, as we are treating (Atom, Position) as a single unit
            for i in range(0, len(sentence) - 1, 2):
                if isinstance(sentence[i], PadAtom) or isinstance(
                        sentence[i + 1], PadPos):
                    sent_mask.append(True)  # padding position, to be ignored
                else:
                    sent_mask.append(False)  # non-padding position
            mask.append(sent_mask)
        return torch.tensor(mask, dtype=torch.bool, device=self.device)

    def forward(self, src, tgt):
        # Generate sentence padding mask
        src_mask = self.generate_padding_mask(src)

        # Apply the word embedding and position embedding
        src = (word_pos_embedding(src) * math.sqrt(self.d_model)).to(
            self.device)
        print(src.shape)

        # Apply 'fake' target embedding
        # mask_tgt = self.trg_embedding(tgt).to(self.device)

        # Apply transpose to tensors
        src_tensor = (src).transpose(0, 1)
        # mask_tgt_tensor = (mask_tgt).transpose(0, 1)

        # The mask should have the same size as the source tensor, but with 0s in padding positions and 1s elsewhere
        src_mask = (src_mask == 1)

        # Pass through the encoder, decoder layer
        memory = self.transformer_encoder(src_tensor,
                                          src_key_padding_mask=src_mask).to(
            self.device)
        # output = self.transformer_decoder(mask_tgt_tensor, memory).to(self.device)
        # Pass through the output layer
        memory = memory.mean(dim=0, keepdim=True)
        # memory, _ = memory.max(dim=1)
        # memory = memory[:, 0:1, :]
        output = self.output_layer(memory).to(self.device)

        return output


if __name__ == '__main__':
    path1 = '/Users/huangshixun/Desktop/Transformer/superconductors/data/pre_train_new.json'
    dataset1 = TrainDataset(path1)
    data = [dataset1[-1][0]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src = [['He', [0.0, 0.0, 0.0], 'H', [0.0, 1.2, 1.3], PadAtom(), PadPos()]]
    # src is batch[sentence[words, posi, words, posi]]
    tgt_mask = dataset1[0][1]
    tgt = dataset1[-1][2]
    model_param = {'embedding_size': 25, 'num_head': 5, 'num_hid': 5,
                   'num_layer': 8, 'dropout': 0.1}
    # Load model
    model = Transformer(d_model=model_param['embedding_size'],
                        nhead=model_param['num_head'],
                        nhid=model_param['num_hid'],
                        nlayers=model_param['num_layer'],
                        dropout=model_param['dropout'], device=device)
    out = model(data, tgt_mask)
    print(out.shape, out, tgt)
