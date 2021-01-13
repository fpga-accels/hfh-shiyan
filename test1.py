import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from gensim.models import Word2Vec
def load_training_data(path='training_label.txt'):
    # 读取 training 需要的数据
    # 如果是 'training_label.txt'，需要读取 label，如果是 'training_nolabel.txt'，不需要读取 label
    if 'training_label' in path:
        with open(path, 'r',encoding='utf-8') as f:
            lines = f.readlines()
            # lines是二维数组，第一维是行line(按回车分割)，第二维是每行的单词(按空格分割)
            lines = [line.strip('\n').split(' ') for line in lines]
        # 每行按空格分割后，第2个符号之后都是句子的单词
        x = [line[2:] for line in lines]
        # 每行按空格分割后，第0个符号是label
        y = [line[0] for line in lines]
        return x, y
    else:
        with open(path, 'r',encoding='utf-8') as f:
            lines = f.readlines()
            # lines是二维数组，第一维是行line(按回车分割)，第二维是每行的单词(按空格分割)
            x = [line.strip('\n').split(' ') for line in lines]
        return x

def load_testing_data(path='testing_data'):
    # 读取 testing 需要的数据
    with open(path, 'r',encoding='utf-8') as f:
        lines = f.readlines()
        # 第0行是表头，从第1行开始是数据
        # 第0列是id，第1列是文本，按逗号分割，需要逗号之后的文本
        X = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
        X = [sen.split(' ') for sen in X]
    return X

def evaluation(outputs, labels):
    # outputs => 预测值，概率（float）
    # labels => 真实值，标签（0或1）
    outputs[outputs>=0.5] = 1 # 大于等于 0.5 为正面
    outputs[outputs<0.5] = 0 # 小于 0.5 为负面
    accuracy = torch.sum(torch.eq(outputs, labels)).item()
    return accuracy



def train_word2vec(x):
    # 训练 word to vector 的 word embedding
    # window：滑动窗口的大小，min_count：过滤掉语料中出现频率小于min_count的词
    model = Word2Vec(x, size=250, window=5, min_count=5, workers=12, iter=10, sg=1)
    return model

# # 读取 training 数据
# print("loading training data ...")
# train_x, y = load_training_data('training_label.txt')
# train_x_no_label = load_training_data('training_nolabel.txt')
#
# # 读取 testing 数据
# print("loading testing data ...")
# test_x = load_testing_data('testing_data.txt')
#
# # 把 training 中的 word 变成 vector
# # model = train_word2vec(train_x + train_x_no_label + test_x) # w2v_all
# model = train_word2vec(train_x + test_x) # w2v
#
# # 保存 vector
# print("saving model ...")
# # model.save('w2v_all.model')
# model.save('w2v.model')

# 数据预处理
class Preprocess():
    def __init__(self, sentences, sen_len, w2v_path):
        self.w2v_path = w2v_path   # word2vec的存储路径
        self.sentences = sentences  # 句子
        self.sen_len = sen_len    # 句子的固定长度
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []

    def get_w2v_model(self):
        # 读取之前训练好的 word2vec
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size

    def add_embedding(self, word):
        # 这里的 word 只会是 "<PAD>" 或 "<UNK>"
        # 把一个随机生成的表征向量 vector 作为 "<PAD>" 或 "<UNK>" 的嵌入
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        # 它的 index 是 word2idx 这个词典的长度，即最后一个
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)

    def make_embedding(self, load=True):
        print("Get embedding ...")
        # 获取训练好的 Word2vec word embedding
        if load:
            print("loading word to vec model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError
        # 遍历嵌入后的单词
        for i, word in enumerate(self.embedding.wv.vocab):
            print('get words #{}'.format(i+1), end='\r')
            # 新加入的 word 的 index 是 word2idx 这个词典的长度，即最后一个
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding[word])
        print('')
        # 把 embedding_matrix 变成 tensor
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        # 将 <PAD> 和 <UNK> 加入 embedding
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix

    def pad_sequence(self, sentence):
        # 将每个句子变成一样的长度，即 sen_len 的长度
        if len(sentence) > self.sen_len:
        # 如果句子长度大于 sen_len 的长度，就截断
            sentence = sentence[:self.sen_len]
        else:
        # 如果句子长度小于 sen_len 的长度，就补上 <PAD> 符号，缺多少个单词就补多少个 <PAD>
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.sen_len
        return sentence

    def sentence_word2idx(self):
        # 把句子里面的字变成相对应的 index
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            print('sentence count #{}'.format(i+1), end='\r')
            sentence_idx = []
            for word in sen:
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                else:
                # 没有出现过的单词就用 <UNK> 表示
                    sentence_idx.append(self.word2idx["<UNK>"])
            # 将每个句子变成一样的长度
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)

    def labels_to_tensor(self, y):
        # 把 labels 转成 tensor
        y = [int(label) for label in y]
        return torch.LongTensor(y)


from torch.utils.data import DataLoader, Dataset


class TwitterDataset(Dataset):
    """
    Expected data shape like:(data_num, data_len)
    Data can be a list of numpy array or a list of lists
    input data shape : (data_num, seq_len, feature_dim)

    __len__ will return the number of data
    """

    def __init__(self, X, y):
        self.data = X
        self.label = y

    def __getitem__(self, idx):
        if self.label is None: return self.data[idx]
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_embedding=True):
        super(LSTM_Net, self).__init__()
        # embedding layer
        self.embedding = torch.nn.Embedding(embedding.size(0), embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # 是否将 embedding 固定住，如果 fix_embedding 为 False，在训练过程中，embedding 也会跟着被训练
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None)
        # x 的 dimension (batch, seq_len, hidden_size)
        # 取用 LSTM 最后一层的 hidden state 丢到分类器中
        x = x[:, -1, :]
        x = self.classifier(x)
        return x
import numpy as np

def training(batch_size, n_epoch, lr, train, valid, model, device):
    # 输出模型总的参数数量、可训练的参数数量
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))

    loss = nn.BCELoss()  # 定义损失函数为二元交叉熵损失 binary cross entropy loss
    t_batch = len(train)  # training 数据的batch size大小
    v_batch = len(valid)  # validation 数据的batch size大小
    optimizer = optim.Adam(model.parameters(), lr=lr)  # optimizer用Adam，设置适当的学习率lr
    total_loss, total_acc, best_acc = 0, 0, 0
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0

        # training
        model.train()  # 将 model 的模式设为 train，这样 optimizer 就可以更新 model 的参数
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.long)  # 因为 device 为 "cuda"，将 inputs 转成 torch.cuda.LongTensor
            labels = labels.to(device,
                               dtype=torch.float)  # 因为 device 为 "cuda"，将 labels 转成 torch.cuda.FloatTensor，loss()需要float

            optimizer.zero_grad()  # 由于 loss.backward() 的 gradient 会累加，所以每一个 batch 后需要归零
            outputs = model(inputs)  # 模型输入Input，输出output
            outputs = outputs.squeeze()  # 去掉最外面的 dimension，好让 outputs 可以丢进 loss()
            batch_loss = loss(outputs, labels)  # 计算模型此时的 training loss
            batch_loss.backward()  # 计算 loss 的 gradient
            optimizer.step()  # 更新模型参数

            accuracy = evaluation(outputs, labels)  # 计算模型此时的 training accuracy
            total_acc += (accuracy / batch_size)
            total_loss += batch_loss.item()
        print('Epoch | {}/{}'.format(epoch + 1, n_epoch))
        print('Train | Loss:{:.5f} Acc: {:.3f}'.format(total_loss / t_batch, total_acc / t_batch * 100))

        # validation
        model.eval()  # 将 model 的模式设为 eval，这样 model 的参数就会被固定住
        with torch.no_grad():
            total_loss, total_acc = 0, 0

            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.long)  # 因为 device 为 "cuda"，将 inputs 转成 torch.cuda.LongTensor
                labels = labels.to(device,
                                   dtype=torch.float)  # 因为 device 为 "cuda"，将 labels 转成 torch.cuda.FloatTensor，loss()需要float

                outputs = model(inputs)  # 模型输入Input，输出output
                outputs = outputs.squeeze()  # 去掉最外面的 dimension，好让 outputs 可以丢进 loss()
                batch_loss = loss(outputs, labels)  # 计算模型此时的 training loss
                accuracy = evaluation(outputs, labels)  # 计算模型此时的 training accuracy
                total_acc += (accuracy / batch_size)
                total_loss += batch_loss.item()

            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss / v_batch, total_acc / v_batch * 100))
            if total_acc > best_acc:
                # 如果 validation 的结果优于之前所有的結果，就把当下的模型保存下来，用于之后的testing
                best_acc = total_acc
                torch.save(model, "ckpt.model")
        print('-----------------------------------------------')


def testing(batch_size, test_loader, model, device):
    model.eval()  # 将 model 的模式设为 eval，这样 model 的参数就会被固定住
    ret_output = []  # 返回的output
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            outputs[outputs >= 0.5] = 1  # 大于等于0.5为正面
            outputs[outputs < 0.5] = 0  # 小于0.5为负面
            ret_output += outputs.int().tolist()

    return ret_output
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.autograd import Variable

def main():
   

# 通过 torch.cuda.is_available() 的值判断是否可以使用 GPU ，如果可以的话 device 就设为 "cuda"，没有的话就设为 "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义句子长度、要不要固定 embedding、batch 大小、要训练几个 epoch、 学习率的值、 w2v的路径
sen_len = 20
fix_embedding = True # fix embedding during training
batch_size = 128
epoch = 10
lr = 0.001
w2v_path = 'w2v_all.model' 

print("loading data ...") # 读取 'training_label.txt'  'training_nolabel.txt' 
train_x, y = load_training_data('training_label.txt')
train_x_no_label = load_training_data('training_nolabel.txt')

# 对 input 跟 labels 做预处理
preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)

train_x = preprocess.sentence_word2idx()
y = preprocess.labels_to_tensor(y)
#将train_x y变为numpy类型
train_x=train_x.numpy()
y=y.numpy()

# 定义模型
model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=150, num_layers=1, dropout=0.5, fix_embedding=fix_embedding)
model = model.to(device) # device为 "cuda"，model 使用 GPU 来训练（inputs 也需要是 cuda tensor）

# 把 data 分为 training data 和 validation data（将一部分 training data 作为 validation data）
X_train, X_val, y_train, y_val = train_test_split(train_x, y, test_size = 0.1, random_state = 1, stratify = y)
print('Train | Len:{} \nValid | Len:{}'.format(len(y_train), len(y_val)))

# 把 data 做成 dataset 供 dataloader 取用
train_dataset = TwitterDataset(X=X_train, y=y_train)
val_dataset = TwitterDataset(X=X_val, y=y_val)

# 把 data 转成 batch of tensors
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 0) 
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = 0)

# 开始训练
training(batch_size, epoch, lr, train_loader, val_loader, model, device)

if __name__=='__main__':
    main()