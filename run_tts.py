import os
import sys

import socket
import threading

import io
from scipy.io.wavfile import write


import torch
device = torch.device("cuda:1")

# define E2E-TTS model
from argparse import Namespace
from os.path import join
from espnet.espnet.asr.asr_utils import get_model_conf
from espnet.espnet.asr.asr_utils import torch_load
from espnet.espnet.utils.dynamic_import import dynamic_import
# define neural vocoder
import yaml
import parallel_wavegan.models

from espnet.espnet.transform.cmvn import CMVN
from parallel_wavegan.layers import PQMF
from sklearn.preprocessing import StandardScaler
from unidecode import unidecode
import h5py

import re
from espnet.espnet.utils.cli_readers import file_reader_helper


#------------------------sentimental-----------------------#
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook


from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

import pandas as pd
from pandas import DataFrame as df


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=4,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device), return_dict=False)
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


device_emo = torch.device("cuda:1")
bertmodel, vocab = get_pytorch_kobert_model()

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)



max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5


# 위에서 설정한 tok, max_len, batch_size, device를 그대로 입력
# comment : 예측하고자 하는 텍스트 데이터 리스트
def getSentimentValue(text, tok, max_len, batch_size, device):
  inputText = list()
  inputText.append([text, 4])
  
  test_set = BERTDataset(inputText, 0, 1, tok, max_len, True, False) 
  test_input = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=5)
  
  for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_input):
    token_ids = token_ids.long().to(device)
    segment_ids = segment_ids.long().to(device)
    valid_length= valid_length 
    out = model_emo(token_ids, valid_length, segment_ids)

    emo = torch.argmax(out)
    
  return emo # 텍스트 데이터에 1대1 매칭되는 감성값 리스트 반환


model_emo = torch.load('/home/dcl2003/Sentimental/model2.pt')
model_emo.eval()

print("sentimental load")

#-------------------------------------------------------------#


trans_type = "char"
dict_path = "downloads/ptFastSpeech/data/lang_1char/EMO_char_train_no_dev_units.txt"
model_path = "downloads/ptFastSpeech/exp/EMO_char_train_no_dev_pytorch_train_fastspeech.sgst2.spkid/results/model.loss.best"
cmvn_path = "downloads/ptFastSpeech/data/EMO_char_train_no_dev/cmvn.ark"

# set path
vocoder_path = "downloads/ptParallelWavegan/checkpoint-255000steps.pkl"
vocoder_conf = "downloads/ptParallelWavegan/config.yml"
vocoder_stat =  "downloads/ptParallelWavegan/stats.h5"

# add path

sys.path.append("espnet")
sys.path.append("downloads/ptFastSpeech")


idim, odim, train_args = get_model_conf(model_path)
model_class = dynamic_import(train_args.model_module)
model = model_class(idim, odim, train_args)
torch_load(model_path, model)
model = model.eval().to(device)
inference_args = Namespace(**{
    "threshold": 0.5,"minlenratio": 0.0, "maxlenratio": 10.0,
    # Only for Tacotron 2
    "use_attention_constraint": True, "backward_window": 1,"forward_window":3,
    # Only for fastspeech (lower than 1.0 is faster speech, higher than 1.0 is slower speech)
    "fastspeech_alpha": 1.0,
    })

with open(vocoder_conf) as f:
    config = yaml.load(f, Loader=yaml.Loader)
vocoder_class = config.get("generator_type", "ParallelWaveGANGenerator")
vocoder = getattr(parallel_wavegan.models, vocoder_class)(**config["generator_params"])
vocoder.load_state_dict(torch.load(vocoder_path, map_location="cpu")["model"]["generator"])
vocoder.remove_weight_norm()
vocoder = vocoder.eval().to(device)
if config["generator_params"]["out_channels"] > 1:
    pqmf = PQMF(config["generator_params"]["out_channels"]).to(device)


cmvn = CMVN(cmvn_path, norm_means=True, norm_vars=True, reverse=True)

scaler = StandardScaler()
with h5py.File(vocoder_stat, "r") as f:
    scaler.mean_ = f["mean"][()]
    scaler.scale_ = f['scale'][()]
scaler.n_features_in_ = scaler.mean_.shape[0]
# define text frontend
with open(dict_path) as f:
    lines = f.readlines()
lines = [line.replace("\n", "").split(" ") for line in lines]
char_to_id = {c: int(i) for c, i in lines}

class Text2Grp(object):
    def __init__(self, fn_grptable):
        self.grptable = grptable = self.loadTABLE(fn_grptable)
        self.INITIALS = self.grptable[0].split(' ')
        self.G_INITIALS = grptable[1].split(' ')
        self.MEDIALS = grptable[2].split(' ')
        self.G_MEDIALS = grptable[3].split(' ')
        self.FINALS = grptable[4].split(' ')
        self.G_FINALS = grptable[5].split(' ')
        self.SPECIALS = grptable[6].split(' ')
        self.SPECIALS.append(' ')
        self.CHARACTERS = self.INITIALS + self.MEDIALS + self.FINALS + self.SPECIALS
        self.flag_specials = False

        if len(self.INITIALS) != len(self.G_INITIALS):
            print("Error: character_INITIALS and grapheme_INITIALS length mismatch")
            sys.exit(1)
        if len(self.MEDIALS) != len(self.G_MEDIALS):
            print("Error: character_MEDIALS and grapheme_MEDIALS length mismatch")
            sys.exit(1)
        if len(self.FINALS) != len(self.G_FINALS):
            print("Error: character_FINALS and grapheme_FINALS length mismatch")
            sys.exit(1)

    def loadTABLE(self, fn_grptable):
        with open(fn_grptable, 'r', encoding='utf8') as fid:
            grptable = fid.read().split('\n')
        if len(grptable) != 7:
            print("Invalid table format")
            sys.exit(1)
        return grptable

    def check_syllable(self, x):
        return 0xAC00 <= ord(x) <= 0xD7A3


    def split_syllable_char(self, x):
        if len(x) != 1:
            raise ValueError("Input string must have exactly one character.")

        if not self.check_syllable(x):
            raise ValueError(
                "Input string does not contain a valid Korean character.")

        diff = ord(x) - 0xAC00
        _m = diff % 28
        _d = (diff - _m) // 28

        initial_index = _d // 21
        medial_index = _d % 21
        final_index = _m

        if not final_index:
            result_cha = (self.INITIALS[initial_index], self.MEDIALS[medial_index])
            result_grp = (self.G_INITIALS[initial_index], self.G_MEDIALS[medial_index], 'X')
        else:
            result_cha = (
                self.INITIALS[initial_index], self.MEDIALS[medial_index],
                self.FINALS[final_index - 1])
            result_grp = (
                self.G_INITIALS[initial_index], self.G_MEDIALS[medial_index],
                self.G_FINALS[final_index - 1])

        return result_cha, result_grp

    def split_syllables(self, string):

        new_chracter = ""
        new_grapheme = ""
        for c in string:
            if not self.check_syllable(c):
                if (c in self.SPECIALS) or (c == ' '):
                #if (c in self.SPECIALS):
                    new_c = c
                    new_g = c
                else:
                    new_c = ''
                    new_g = ''
                    self.flag_specials = True

            else:
                [c_sent, g_sent] = self.split_syllable_char(c)
                new_c = "".join(c_sent)
                new_g = "".join(g_sent)
            new_chracter += new_c
            new_grapheme += new_g

        return new_chracter, new_grapheme

def custom_english_cleaners(text):
    _whitespace_re = re.compile(r'\s+')
    '''Custom pipeline for English text, including number and abbreviation expansion.'''
    text = unidecode(text)
    text = re.sub(_whitespace_re, ' ', text)
    return text

text2grp = Text2Grp("downloads/ptFastSpeech/data/grp/table.txt")

def frontend(text):
    """Clean text and then convert to id sequence."""
    _, grp = text2grp.split_syllables(text)
    text = custom_english_cleaners(grp)
    #print(f"Cleaned text: {text}")
    charseq = list(text)
    if not charseq[-1] in [',', '.',' !', '?']:
        charseq += '.'
    idseq = []
    for c in charseq:
        if c.isspace():
            idseq += [char_to_id["<space>"]]
        elif c not in char_to_id.keys():
            idseq += [char_to_id["<unk>"]]
        else:
            idseq += [char_to_id[c]]
    idseq += [idim - 1]  # <eos>
    return torch.LongTensor(idseq).view(-1).to(device)

def recv():
    HOST = '114.70.22.237'
    PORT = 5052

    client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    client_socket.connect((HOST,PORT))
    print("connect success")
    while True:
        try:
            global recv_data
            recv = client_socket.recv(1024)

            if len(recv):
                # feel_sen = recv.decode('utf8')
                recv_data.append(recv.decode('utf8'))
        except:
            print("연결이 해제되었습니다.")
            break


def send():
    global recv_data
    global stemb_dict_f
    global stemb_dict_m
    global female_spk_list
    global male_spk_list
    HOST = '114.70.22.237'
    PORT2 = 5053

    server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server_socket.bind((HOST,PORT2))

    server_socket.listen()
    my_client_socket, addr = server_socket.accept()
    print("accept success")
    while True: 
        try:
            if len(recv_data):
                print("test recv length:",len(recv_data))
                input_text = recv_data.pop(0)
                if(input_text[0]=="M"):
                    input_speaker = "5"
                    input_weight=input_text[2:5]
                    input_text = input_text[6:]
                    print(input_text)
                elif(input_text[0]=="W"):
                    input_speaker = "0"
                    input_weight=input_text[2:5]
                    input_text = input_text[6:]
                    print(input_text)
                else:
                    continue


                emotion = getSentimentValue(input_text, tok, max_len, batch_size, device_emo)
                if emotion == 0:
                    input_emotion ='neutral'
                elif emotion == 1:
                    input_emotion ='happy'
                elif emotion == 2:
                    input_emotion ='angry'
                elif emotion == 3:
                    input_emotion ='sad'
                print("-------------------------------------")
                print("Get Sentiment Value: ", input_emotion)
                print("-------------------------------------")

                pad_fn = torch.nn.ReplicationPad1d(
                    config["generator_params"].get("aux_context_window", 0))
                use_noise_input = vocoder_class == "ParallelWaveGANGenerator"

                # eval시에는 no_grad
                with torch.no_grad():
                    start = time.time()
                    if input_speaker in female_spk_list:
                        stemb_dict_in = stemb_dict_f
                    elif input_speaker in male_spk_list:
                        stemb_dict_in = stemb_dict_m
                    spemb = torch.LongTensor([int(input_speaker)]).view(-1).to(device)
                    stemb = torch.FloatTensor(stemb_dict_in[input_emotion]).view(-1).to(device) * float(input_weight)
                    x = frontend(input_text)
                    c, _, _ = model.inference(x, None, inference_args, spemb=spemb, stemb=stemb)
                    xx_denorm = cmvn(c.cpu().numpy())
                    c = torch.FloatTensor(scaler.transform(xx_denorm))
                    c = pad_fn(c.unsqueeze(0).transpose(2, 1)).to(device)
                    xx = (c,)
                    if use_noise_input:
                        z_size = (1, 1, (c.size(2) - sum(pad_fn.padding)) * config["hop_size"])
                        z = torch.randn(z_size).to(device)
                        xx = (z,) + xx
                    if config["generator_params"]["out_channels"] == 1:
                        y = vocoder(*xx).view(-1)
                    else:
                        y = pqmf.synthesis(vocoder(*xx)).view(-1) 

                # text.wav가 있다고 가정합시다.
                with open("01.wav", mode='wb') as f:
                    write(f, config["sampling_rate"], y.view(-1).cpu().numpy())
                
                file_size = os.path.getsize("01.wav")
                print(file_size)
                if file_size > 0:
                    with open("01.wav", mode="rb") as f:
                        my_client_socket.sendall((file_size).to_bytes(length=8, byteorder="big"))
                        my_client_socket.sendfile(f)
                
        except:
            print("연결이 해제되었습니다.")
            exit()
            break

emotion_scp = 'scp:downloads/ptFastSpeech/exp/EMO_char_train_no_dev_pytorch_train_fastspeech.sgst2.spkid/outputs_model.loss.best_decode_stemb1.0/EMO_char_train_no_dev/emotion.scp'

stemb_dict_f = {}
stemb_dict_m = {}
for idx, (utt_id, stemb1) in enumerate(file_reader_helper(emotion_scp, 'mat'), 1):
    if utt_id[0] == 'f':
        stemb_dict_f[utt_id[2:]] = stemb1
    elif utt_id[0] == 'm':
        stemb_dict_m[utt_id[2:]] = stemb1

female_spk_list = ['0', '1', '2', '3', '4', '10' , '11', '12', '13', '14']
male_spk_list = ['5', '6', '7', '8', '9', '15', '16', '17', '18', '19']
spemb_list = female_spk_list + male_spk_list



print("Now ready to synthesize!")




#-------------이부분만 반복--------------#
import time


HOST = '114.70.22.237'
PORT = 5052
PORT2 = 5053
recv_data=[]

print("화자 번호를 입력해주세요. *Option* 여자: 0~4, 10~14 / 남자: 5~9, 15~19")
input_speaker = "0" # 채팅서버에서 설정할 수 있게 하면 좋고
# print("감정을 입력해주세요. *Option* neutral, happy, sad, angry")
# input_emotion = "sad"   # 이건 sentimental에서 받아와야하고
print("감정 세기를 입력해주세요. *Option* 0.5(약하게), 1.0(적당하게), 2.0(세게)")
input_weight = "1.0"    # 값은 1.0으로 통일


# client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
# client_socket.connect((HOST,PORT))

# server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
# server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# server_socket.bind((HOST,PORT2))

# server_socket.listen()
# my_client_socket, addr = server_socket.accept()

# thread1 = threading.Thread(target = recv, args=(client_socket, ))
thread1 = threading.Thread(target = recv, args=( ))
thread1.start()

# thread2 = threading.Thread(target = send, args=(my_client_socket, ))
thread2 = threading.Thread(target = send, args=( ))
thread2.start()

