import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        drop_prob=0.5

        # embedding layer
        self.word_emb = nn.Embedding(vocab_size, embed_size)

        # define the LSTM
        # https://pytorch.org/docs/stable/nn.html#lstm
        self.lstm = nn.LSTM(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True  # If True, then the input and output tensors are provided as (batch, seq, feature).
        )

        # define a dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # define the final, fully-connected output layer
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, features, captions):

        # features dimension: (batch_size, embed_dim)
        # captions: (batch_size, max_caption_length)

        embeddings = self.word_emb(captions)  # (batch_size, max_caption_length, embed_dim)

        # turn features into (batch_size, 1, embed_dim) tensor then append embedings
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)

        out, hidden = self.lstm(embeddings) # (batch_size, max_caption_length + 1, hidden_size)

        # remove the last output which takes '<end>' as input
        out_captions = self.fc(out[:, :-1, :])  # (batch_size, max_caption_length, vocab_size)

        return out_captions


    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass

