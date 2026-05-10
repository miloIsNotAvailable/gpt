import torch
from torch import zeros, arange, exp, sin, cos, triu, tensor, full, __version__, long, softmax, argmax, cat
from torch.nn import Module, Embedding, MultiheadAttention, LayerNorm, Linear, ReLU, Dropout, ModuleList, CrossEntropyLoss
from math import sqrt, log
import pandas
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
import sys

# print(f"GPU: {torch.cuda.is_available()}")

# tokenizer = get_tokenizer(
#     "spacy",
#     language="pl_core_news_sm"
# )

# df = pandas.read_csv( "./out11.csv" )
# for line in df["lead"][:10]:
#     line: str = line.strip()
#     print(tokenizer( line ))

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
tokenizer.pad_token = tokenizer.eos_token

class TextDataset(IterableDataset):

    def __init__(self, filename, block_size=128):
        self.filename = filename
        self.block_size = block_size
        self.df = pandas.read_csv(self.filename)

    def __iter__(self):

        for line in self.df["lead"].dropna()[:10]:

            text = str(line).strip()

            tokens = tokenizer(
                text,
                add_special_tokens=True
            )["input_ids"]

            for i in range(0, len(tokens), self.block_size):

                chunk = tokens[i:i + self.block_size]

                mask = [1] * len( chunk )

                if len(chunk) < self.block_size:
                    
                    pad_len = self.block_size - len(chunk)

                    chunk += [tokenizer.eos_token_id] * pad_len
                    
                    mask += [0] * pad_len

                yield (
                    tensor(chunk, dtype=long),
                    tensor(mask, dtype=long)
                )

dataset = TextDataset( "./out11.csv" )

loader = DataLoader(
    dataset,
    batch_size=4

)

# for batch in loader:
#     input_ids, mask = batch

#     print(input_ids.shape)
#     print(mask.shape)

#     print(input_ids.dtype)
#     break

class InputEmbeddings(Module):
    def __init__( self, vocab_size: int, d_model: int ):
        super().__init__()
        self.embedding = Embedding(num_embeddings=vocab_size, 
                                   embedding_dim=d_model)
        
        self.scale = sqrt(d_model)
    
    def forward( self, x ):
        return self.embedding( x ) * self.scale
    
class PositionalEncoding(Module):
    def __init__(self, d_model: int, max_seq_length: int):
        super().__init__()
        pe = zeros( size=(max_seq_length, d_model) )
        pos = arange( start=0, 
                            end=max_seq_length, 
                            dtype=float ).reshape( shape=(max_seq_length, 1) )
        div_term = exp(-arange(0, d_model, 2).float() * log(10000.0) / d_model)
        
        pe[:, 0::2] = sin(pos * div_term)
        pe[:, 1::2] = cos(pos * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
            return x + self.pe[:, :x.size(1)]

vocab_size = tokenizer.vocab_size
d_model = 512
max_seq_length = 128

# embedding_layer = InputEmbeddings(vocab_size, d_model)
# input_tokens, mask = next(iter(loader))
# # print( f"{dataset.tokenizer.decode( input_tokens )}: {input_tokens}" )
# embedded_tokens = embedding_layer(input_tokens)

# pos_encoding_layer = PositionalEncoding(d_model, max_seq_length)
# position_encoded_tokens = pos_encoding_layer(embedded_tokens)

class CausalSelfAttention( Module ):
    def __init__( self, num_heads, d_model, seq_len, dropout_rate=0.1 ):
        super().__init__()
        self.mha = MultiheadAttention( num_heads=num_heads, 
                                       embed_dim=d_model, batch_first=True )
        self.norm = LayerNorm( d_model )
        self.dropout=Dropout( dropout_rate )

    def forward( self, x ):

        T = x.size(1)

        attn_mask = triu(
            full((T, T), float('-inf'), device=x.device),
            diagonal=1
        )

        out, attn_weights = self.mha( query=x, 
                                        key=x, 
                                        value=x, 
                                        attn_mask=attn_mask,
                                        is_causal=True )
        
        out = x + out
        out = self.norm( out )

        out = self.dropout( out )

        return out

class CrossSelfAttention( Module ):
    def __init__( self, num_heads, d_model, seq_len, dropout_rate=.1 ):
        super().__init__()
        self.mha = MultiheadAttention( num_heads=num_heads, 
                                       embed_dim=d_model )
        self.norm = LayerNorm( d_model )

        self.dropout=Dropout( dropout_rate )

    def forward( self, x, context ):


        out, attn_weights = self.mha( query=x, 
                                      key=context, 
                                    value=context )
        
        out = x + out
        out = self.norm( out )
        out = self.dropout( out )

        return out

class FeedForward(Module):
    def __init__(self, d_model, d_ff, dropout_rate=.1):
        super().__init__()
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        self.relu = ReLU()
        self.dropout = Dropout( dropout_rate )
    def forward(self, x):
        return self.dropout(self.fc2(self.relu(self.fc1(x))))

# csa = CausalSelfAttention( num_heads=1, 
#                            d_model=d_model,
#                            seq_len=max_seq_length )

# sa = CrossSelfAttention( num_heads=1, 
#                            d_model=d_model,
#                            seq_len=max_seq_length )


# ff = FeedForward( d_model=d_model, d_ff=2048 )

class Decoder( Module ):
    def __init__( self, num_heads, d_model, seq_len, d_ff, dropout_rate=0.1 ):
        super().__init__()
        self.causal = CausalSelfAttention( num_heads=num_heads, 
                                           d_model=d_model,
                                           seq_len=seq_len,
                                           dropout_rate=dropout_rate )
        
        self.cross = CrossSelfAttention( num_heads=num_heads, 
                                         d_model=d_model,
                                         seq_len=seq_len,
                                         dropout_rate=dropout_rate )
        
        self.feed = FeedForward( d_model=d_model, d_ff=d_ff )
        self.dropout = Dropout( dropout_rate ) 
        self.norm1 = LayerNorm( d_model )
        self.norm2 = LayerNorm( d_model )

    def forward( self, x ):

        out = self.causal( x )

        out = self.cross( out, out )

        out = self.feed( out )

        x = x + out
        # out = self.norm1( out )
        # x = self.norm2( x )
        # x = self.dropout( x )  
        
        return x


class GPT( Module ):
    def __init__( self, num_heads, d_model, seq_len, vocab_size, d_ff, n_layers, dropout_rate=0.1 ):
        super().__init__()
        # self.decoder = Decoder( num_heads=num_heads,
        #                         d_model=d_model,
        #                         seq_len=seq_len,
        #                         d_ff=d_ff,
        #                         dropout_rate=dropout_rate )

        self.positional_encoding = PositionalEncoding( d_model, seq_len )
        self.embedding = InputEmbeddings( vocab_size, d_model )

        self.layers = ModuleList([
            Decoder(num_heads, d_model, seq_len, d_ff, dropout_rate)
            for _ in range(n_layers)
        ])

        self.linear = Linear( d_model, vocab_size )

    def forward( self, x ):
        
        x = self.embedding( x )
        x = self.positional_encoding( x )
        
        for layer in self.layers:
            x = layer( x )

        x = self.linear( x )

        return x

# e = csa( position_encoded_tokens )
# e = sa( e, e )
# e = ff( e )

# decoder = Decoder( num_heads=1,
#                    d_model=d_model,
#                    seq_len=max_seq_length,
#                    vocab_size=vocab_size,
#                    d_ff=2048,
#                    dropout_rate=.1 ) 

# decoder( input_tokens )
input_tokens, mask = next(iter(loader))
gpt = GPT( num_heads=2,
           d_model=d_model,
           seq_len=max_seq_length,
           vocab_size=vocab_size,
           d_ff=2048,
           dropout_rate=.1,
           n_layers=6 )

# x = input_tokens[:, :-1]
# test = input_tokens[:, 1:]

def generate( context ):

    for i in range( max_seq_length ):
        out = gpt( context[ :, -max_seq_length: ] )
        out = out[:, -1, :] 
        out = softmax( input=out, dim=-1 )
        out = argmax(out, -1)
        # out = torch.multinomial(out, num_samples=1)
        out = out.unsqueeze(1)

        # print( out )

        context = cat([context, out], dim=1)

    context = context[ 0 ]
    context = context[ context != 2 ]
    return tokenizer.decode( context )


# gpt.train()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# gpt = gpt.to(device)

optimizer = torch.optim.AdamW(gpt.parameters(), lr=3e-4)

crossentropy = torch.nn.CrossEntropyLoss( ignore_index=2 )

for epoch in range( 1 ):

    running_loss = 0.
    last_loss = 0.
    print( f"epoch {epoch}", flush=True )
    for i, data in enumerate(loader):
        
        x, mask = data
        
        # x.to( device )
        # mask.to( device )

        train = x[:, :-1]
        test = x[:, 1:]

        optimizer.zero_grad()
        
        # print( train[0], test[0] )

        logits = gpt( train )

        logits = logits.reshape(-1, vocab_size)
        test = test.reshape( -1 )

        loss = crossentropy( logits, test )
        loss.backward()

        # print( loss.item() )

        optimizer.step()

        running_loss += loss.item()
        # if i % 4 == 0:
        last_loss = running_loss
        print(f'  batch {i + 1} loss: {last_loss}', flush=True)

        # tb_x = epoch * len(loader) + i + 1
        # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
        # print( last_loss, tb_x )
        running_loss = 0.
            
        # break

tokens = tokenizer.encode("W niedzielę Komenda Miejska Policji")
context = tensor([tokens], dtype=long)

print( generate( context ) )

torch.save(gpt.state_dict(), "gpt.pt")