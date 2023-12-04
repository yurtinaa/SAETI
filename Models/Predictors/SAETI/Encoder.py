from torch import nn
import torch


class TimeEncoder(nn.Module):
    def __init__(self, n_features, latent_dim, hidden_size, size_seq,

                 kernel=5):
        super().__init__()

        self.n_features = n_features
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.padd = (kernel - 1) // 2
        self.cnns1 = nn.ModuleList([nn.Conv1d(
            in_channels=2,
            out_channels=64,
            padding=self.padd,
            kernel_size=(kernel,)) for i in torch.arange(n_features)])

        self.cnns2 = nn.ModuleList([nn.Conv1d(
            in_channels=64,
            out_channels=32,
            padding=self.padd,
            kernel_size=(kernel,)) for i in torch.arange(n_features)])

        self.cnns3 = nn.ModuleList([nn.Conv1d(
            in_channels=32,
            out_channels=1,
            padding=self.padd,
            kernel_size=(kernel,)) for i in torch.arange(n_features)])

        self.gru_enc = nn.GRU(n_features, hidden_size,
                              batch_first=True, dropout=0,
                              bidirectional=True)

        self.lat_layer = nn.GRU(hidden_size * 2, latent_dim,
                                batch_first=True, dropout=0,
                                bidirectional=False)
        self.lat = nn.Linear(latent_dim * size_seq, latent_dim)
        self.flatten = nn.Flatten()
        self.leaky = nn.LeakyReLU()
        self.tang = nn.Tanh()

    def __cnns(self, x, snip):
        result = torch.empty(x.shape, device=x.device)
        for i, cnn in enumerate(self.cnns3):
            input_cnn = torch.cat((x[:, i:i + 1, :],
                                   snip[:, i:i + 1, :]),
                                  dim=1)
            input_cnn = self.leaky(self.cnns1[i](input_cnn))
            input_cnn = self.leaky(self.cnns2[i](input_cnn))
            result[:, i:i + 1, :] = self.leaky(cnn(input_cnn))
        return result

    def forward(self, x, snip):
        x = x.transpose(1, 2)
        x = self.__cnns(x, snip)
        #
        x = x.transpose(1, 2)
        x, _ = self.gru_enc(x)
        x, _ = self.lat_layer(x)
        x = self.flatten(x)
        x = self.lat(x)
        return self.leaky(x)
