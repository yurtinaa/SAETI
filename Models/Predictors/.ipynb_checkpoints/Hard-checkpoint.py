from torch import nn
import torch

from Models.Predictors.OnlyCnns import OnlyCnns
from Models.Classifier import Classifier


class Hard(OnlyCnns):
    def __init__(self, size_subsequent: int,
                 count_snippet: int,
                 dim: int,
                 classifier: Classifier,
                 snippet_list,
                 device,
                 hidden_dim=128,
                 num_layers=1,
                 cell='gru',
                 bidirectional=False,
                 inside_count=0,
                 batch_norm=False,
                 kernel=5,
                 config=None
                 ):
        super().__init__(
            size_subsequent=size_subsequent,
            count_snippet=count_snippet,
            dim=dim,
            inside_count=inside_count,
            classifier=classifier,
            bidirectional=bidirectional,
            snippet_list=snippet_list,
            device=device,
            kernel=kernel,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            cell=cell,
            batch_norm=batch_norm,
            config=config
        )
        self.size_subsequent = self.size_subsequent - 1
        self.gru_dim = nn.ModuleList([self.rnn_type[cell](input_size=self.last_cnn,
                                                          hidden_size=hidden_dim,
                                                          bidirectional=bidirectional,
                                                          batch_first=True) for i in range(dim)])


        self.leakyRelu = nn.LeakyReLU()

    def augmentation(self, x):
        with torch.no_grad():
            snippet = self.classifier(x).argmax(dim=1).cpu()
        snip = self.snippet_tensor(snippet)
        last = snip[:, :, -1]
        snip = snip[:, :, :-1]
        result_x = torch.zeros(size=(x.shape[0],
                                     self.dim,
                                     self.hidden_dim),
                               device=self.device)
        for i, cnn in enumerate(self.cnns3):
            input_cnn = torch.cat((x[:, i:i + 1, :], snip[:, i:i + 1, :]), dim=1)
            input_cnn = self.leakyRelu(self.cnns1[i](input_cnn))
            input_cnn = self.leakyRelu(self.cnns2[i](input_cnn))
            if len(self.inside) > 0:
                for j, inside in self.inside[i]:
                    input_cnn = inside(input_cnn)
            result = self.leakyRelu(cnn(input_cnn))
            result, _ = self.gru_dim[i](result.transpose(1, 2))
            result_x[:, i:i + 1, :] = self.leakyRelu(result[:, None, -1, :])
        return result_x, last

    # def augmentation_print(self, x):
    #     with torch.no_grad():
    #         snippet = self.classifier(x).argmax(dim=1).cpu()
    #     snippet = self.snippet_tensor(snippet)
    #     last = snippet[:, :, -1]
    #     snip = snippet[:, :, :-1]
    #     result_x = []
    #     for i, cnn in enumerate(self.cnns3):
    #         input_cnn = torch.cat((x[:, i:i + 1, :], snip[:, i:i + 1, :]), dim=1)
    #         input_cnn = self.leakyRelu(self.cnns1[i](input_cnn))
    #         input_cnn = self.leakyRelu(self.cnns2[i](input_cnn))
    #         for inside_cnn in self.inside:
    #             input_cnn = self.leakyRelu(inside_cnn[i](input_cnn))
    #         result = self.leakyRelu(cnn(input_cnn))
    #         result, _ = self.gru_dim[i](result.transpose(1, 2))
    #         result_x.append(self.leakyRelu(result[:, None, -1, :]))
    #     return torch.cat(result_x, 1), last, snippet

    def fl(self):
        print('flatten')

        for rnn in self.rnns:
            rnn.flatten_parameters()
        for rnn in self.gru_dim:
            rnn.flatten_parameters()
