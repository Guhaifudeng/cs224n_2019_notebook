#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn
from vocab import VocabEntry
import torch.nn.functional as F
class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab:VocabEntry=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder,self).__init__()
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        self.target_vocab = target_vocab
        char_vocab_size = len(self.target_vocab.char2id)
        pad_token_idex = self.target_vocab.char2id['<pad>']
        self.decoderCharEmb = nn.Embedding(char_vocab_size,char_embedding_size,padding_idx=pad_token_idex)
        self.charDecoder = nn.LSTM(input_size=char_embedding_size,hidden_size=hidden_size)
        self.char_output_projection = nn.Linear(in_features=hidden_size,out_features=char_vocab_size)
        ### END YOUR CODE
        print('char_vocab_size',len(self.target_vocab.char2id))
        print(self.target_vocab.char2id)

    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        x = self.decoderCharEmb(input) #(length,batch) -> (length,batch,50)
        dec_hiddens,(last_hidden,last_cell)=self.charDecoder(input= x,hx=dec_hidden) #(length,batch,50) -> (length,batch,hidden*1)
        # For every timestep t 2 f1; : : : ; ng we compute scores (also called logits) st
        s_t = self.char_output_projection(dec_hiddens)
        dec_hidden = (last_hidden,last_cell)
        ### END YOUR CODE
        return s_t,dec_hidden


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        #Chop of the <END> token for max
        char_sequence_no_end = char_sequence[:-1]
        s_t,dec_hidden = self.forward(char_sequence_no_end,dec_hidden) #->(length,batch,char_vocab_size)
        P = F.log_softmax(s_t , dim=-1)
        # print('P' , P.size())
        # print('target_padded_chars' , char_sequence.size())

        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (char_sequence != self.target_vocab.char2id['<pad>']).float()

        # Compute log probability of generating true target words
        target_gold_chars_log_prob = torch.gather(P , index=char_sequence[1:].unsqueeze(-1) , dim=-1).squeeze(-1) * target_masks[1:]
        # print(target_gold_chars_log_prob.size())
        scores = target_gold_chars_log_prob.sum()
        return -scores
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """
        '''
        Input: Initial states h0; c0
        Output: output word generated by the character decoder (doesn’t contain <START> or <END>)
        1: procedure decode greedy
        2: output word []
        3: current char <START>
        4: for t = 0; 1; :::; max length − 1 do
        5: ht+1; ct+1 CharDecoder(current char; ht; ct) . use last predicted character as input
        6: st+1 Wdecht+1 + bdec . compute scores
        7: pt+1 softmax(st+1) . compute probabilities
        8: current char argmaxc pt+1(c) . the most likely next character
        9: if current char=<END> then
        10: break
        11: output word output word + [current char] . append this character to output word
        12: return output word
        '''
        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        #(1,1)


        dec_hidden = initialStates
        batch_size = initialStates[0].shape[1]
        cur_batch_char = ['{'] * batch_size
        finished_flags = [False] * batch_size
        output_word = [''] * batch_size
        for t in range(0,max_length):

            char_indice_t = torch.tensor([self.target_vocab.char2id.get(cur_char,0) for cur_char in cur_batch_char],dtype=torch.long,device=device,requires_grad=False) # (5)
            char_indice_t = char_indice_t.unsqueeze(0)
            # print(char_indice_t.size())
            s_t , dec_hidden = self.forward(char_indice_t,dec_hidden=dec_hidden)
            s_t = s_t.squeeze(0)
            # print(s_t.size())
            # print(dec_hidden)
            next_char_indices =torch.argmax(F.softmax(s_t),-1)
            cur_batch_char = []
            # print(next_char_indices)
            for i in range(0,batch_size):
                next_char_indice = next_char_indices[i]
                indice = next_char_indice.item()
                # print(type(indice))
                cur_batch_char.append(self.target_vocab.id2char.get(indice))
                # print(cur_batch_char)
                if next_char_indice == self.target_vocab.end_of_word or next_char_indice == self.target_vocab.end_of_word:
                    finished_flags[i] = True
                if cur_batch_char[i] == '<pad>':
                    finished_flags[i] = True
                if next_char_indice == self.target_vocab.char_unk:
                    finished_flags[i] = True
                if not finished_flags[i]:
                    output_word[i] += cur_batch_char[i]
        # print(output_word)
        return output_word


        ### END YOUR CODE

