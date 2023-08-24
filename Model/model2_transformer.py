import torch
import torch.nn as nn


class MyTransform(nn.Module):
    def __init__(self, input_size=1, output_size=128):
        super(MyTransform, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return x


class regressoionHead(nn.Module):

    def __init__(self, d_embedding: int):
        super(regressoionHead, self).__init__()
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
        self.trg_embedding = MyTransform(1, 128).to(self.device)

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
        self.output_layer = regressoionHead(128*2)
        self.apply(self.init_weights)
        # self.output_layer = nn.Linear(1,25)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

    def forward(self, src):

        # Pass through the encoder, decoder layer
        memory = self.transformer_encoder(src).to(self.device)
        # output = self.transformer_decoder(mask_tgt_tensor, memory).to(self.device)
        # Pass through the output layer
        memory_mean = memory.mean(dim=0, keepdim=True)
        memory_max, _ = memory.max(dim=0, keepdim=True)
        memory = torch.cat([memory_mean, memory_max], dim=-1)
        output = self.output_layer(memory).to(self.device)

        return output


if __name__ == '__main__':
    data = torch.tensor([[0.1600, 0.5397, 1.7337, 0.2970, 0.2363, 0.5491, 0.3194, 1.1294, 2.0190,
                          0.4896, 0.8091, 0.1595, 1.1421, 0.3137, 0.7828, 1.8759, 0.3503, 0.5531,
                          1.1190, 1.6995, 0.3066, 2.3907, 0.6251, 0.4950, 0.2699, 0.9430, 0.3143,
                          0.5226, 1.1251, 1.3510, 0.6072, 0.1575, 0.5506, 0.5583, 1.1268, 1.0759,
                          0.7992, 0.7499, 0.6093, 1.1374, 0.2797, 0.4704, 0.3368, 0.7476, 0.9159,
                          1.0864, 0.8567, 0.7403, 0.3516, 0.5408, 0.7488, 0.6287, 0.3454, 0.3152,
                          0.5073, 1.3430, 0.6573, 0.8365, 0.2307, 0.3203, 0.2964, 0.4253, 0.6734,
                          0.4317, 0.9930, 0.5945, 0.7758, 0.3722, 1.6896, 0.4777, 0.5983, 1.1925,
                          0.5099, 0.1180, 1.9143, 1.0512, 0.3168, 1.1002, 0.7133, 1.0903, 0.2945,
                          0.2562, 0.4917, 0.7594, 0.4127, 0.2504, 0.1751, 0.9419, 0.6902, 0.3308,
                          1.7344, 2.2334, 0.4162, 1.4434, 0.5577, 0.2287, 0.2745, 2.1668, 0.5274,
                          1.7610, 1.1297, 2.0200, 1.5471, 1.4957, 1.0069, 0.7236, 1.6936, 2.2418,
                          1.5155, 1.0574, 0.3751, 0.6284, 1.5711, 1.5218, 0.6946, 0.5498, 1.7158,
                          0.1393, 0.3705, 0.4113, 0.9541, 0.4886, 1.1572, 0.5038, 1.2056, 0.9116,
                          0.7462, 0.9380], [0.1600, 0.5397, 1.7337, 0.2970, 0.2363, 0.5491, 0.3194, 1.1294, 2.0190,
                                            0.4896, 0.8091, 0.1595, 1.1421, 0.3137, 0.7828, 1.8759, 0.3503, 0.5531,
                                            1.1190, 1.6995, 0.3066, 2.3907, 0.6251, 0.4950, 0.2699, 0.9430, 0.3143,
                                            0.5226, 1.1251, 1.3510, 0.6072, 0.1575, 0.5506, 0.5583, 1.1268, 1.0759,
                                            0.7992, 0.7499, 0.6093, 1.1374, 0.2797, 0.4704, 0.3368, 0.7476, 0.9159,
                                            1.0864, 0.8567, 0.7403, 0.3516, 0.5408, 0.7488, 0.6287, 0.3454, 0.3152,
                                            0.5073, 1.3430, 0.6573, 0.8365, 0.2307, 0.3203, 0.2964, 0.4253, 0.6734,
                                            0.4317, 0.9930, 0.5945, 0.7758, 0.3722, 1.6896, 0.4777, 0.5983, 1.1925,
                                            0.5099, 0.1180, 1.9143, 1.0512, 0.3168, 1.1002, 0.7133, 1.0903, 0.2945,
                                            0.2562, 0.4917, 0.7594, 0.4127, 0.2504, 0.1751, 0.9419, 0.6902, 0.3308,
                                            1.7344, 2.2334, 0.4162, 1.4434, 0.5577, 0.2287, 0.2745, 2.1668, 0.5274,
                                            1.7610, 1.1297, 2.0200, 1.5471, 1.4957, 1.0069, 0.7236, 1.6936, 2.2418,
                                            1.5155, 1.0574, 0.3751, 0.6284, 1.5711, 1.5218, 0.6946, 0.5498, 1.7158,
                                            0.1393, 0.3705, 0.4113, 0.9541, 0.4886, 1.1572, 0.5038, 1.2056, 0.9116,
                                            0.7462, 0.9380]]).unsqueeze(0)
    model = Transformer(128,4,4,4, device='cpu')
    print(data.size())
    output = model(data).size()
    print(output)
