import torch
# from torch import zeros, arange, exp, sin, cos, triu, tensor, full, __version__, long, softmax, argmax, cat
from torch.nn import Module, Embedding, LayerNorm, Linear, ReLU, Dropout, ModuleList
from math import sqrt, log
import pandas
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, Adafactor
# import csv
# from itertools import islice
# from torch.nn.attention.flex_attention import flex_attention, and_masks, create_block_mask
from torch.nn.functional import scaled_dot_product_attention
# print(f"GPU: {torch.cuda.is_available()}")

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

class TextDataset(Dataset):

    def __init__(self, filename, block_size=128):
        self.df = pandas.read_csv(filename)
        # self.tokens = [ tokenizer(t, add_special_tokens=True)["input_ids"]
        #                 for t in self.df["lead"].dropna()[:10] ]
        self.block_size = block_size

    def __len__( self ):
        return 50

    def __getitem__( self, idx ):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        lead = self.df.iloc[idx % 50, 2] 

        tokens = tokenizer(lead, add_special_tokens=True, max_length=self.block_size)["input_ids"]

        l = len( tokens )

        tokens = torch.tensor( tokens )
        tokens = torch.cat( [ tokens, torch.full( (self.block_size - l,), tokenizer.pad_token_id ) ] )

        return tokens

# df = pandas.read_csv( "out11.csv" )

# sentences = df["lead"].dropna()[:3]
# # print( sentences.to_list() )
# tokens = tokenizer(sentences.to_list(), return_attention_mask=False, add_special_tokens=False)["input_ids"]

# tokens = [ t for s in tokens for t in s + [tokenizer.eos_token_id]  ]

# tokens = torch.tensor(tokens)

# def get_attention_mask_for_packed_sequence(x, token_id, eos: bool = True):
#     # store sequence length in variable for easier readability
#     T = tokens.size(0)
#     # get indices of all EOS tokens
#     eos_indices = (tokens == tokenizer.eos_token_id).nonzero().squeeze()
#     # from indices, get length of each sequence
#     reps = torch.cat([eos_indices[[0]]+1, eos_indices[1:] - eos_indices[:-1]])
#     # repeat each eos index n times along dimension 1 (n is the number of tokens in the sequence)
#     repeated_idx = torch.repeat_interleave(eos_indices, reps).view(1,-1).expand(T, -1)
#     # create tensor with all indices from 0 to T-1 repeated T times along dimesion 1
#     mask_indices = torch.arange(T).view(-1,1).expand(-1, T)
#     # create causal mask and additionally mask out all tokens from preceeding sequences
#     mask = torch.ones(T, T, dtype=torch.bool).tril().expand(-1, -1)
#     mask.masked_fill_(mask_indices > repeated_idx, False)
#     return mask

# print(get_attention_mask_for_packed_sequence(tokens, tokenizer.eos_token_id).shape)

max_seq_length = 64

dataset = TextDataset( filename="./out11.csv", block_size=max_seq_length )

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

for i, batch in enumerate(loader):
    # input_ids = batch
    print(tokenizer.decode( batch[0] ), batch.shape)
    if i == 3:
        break

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
        pe = torch.zeros( size=(max_seq_length, d_model) )
        pos = torch.arange( start=0, 
                            end=max_seq_length, 
                            dtype=float ).reshape( shape=(max_seq_length, 1) )
        div_term = torch.exp(-torch.arange(0, d_model, 2).float() * log(10000.0) / d_model)
        
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
            return x + self.pe[:, :x.size(1)]

vocab_size = tokenizer.vocab_size
d_model = 256

# embedding_layer = InputEmbeddings(vocab_size, d_model)
# input_tokens, mask = next(iter(loader))
# # print( f"{dataset.tokenizer.decode( input_tokens )}: {input_tokens}" )
# embedded_tokens = embedding_layer(input_tokens)

# pos_encoding_layer = PositionalEncoding(d_model, max_seq_length)
# position_encoded_tokens = pos_encoding_layer(embedded_tokens)

# SLIDING_WINDOW = 1024
# def sliding_window_causal(b, h, q_idx, kv_idx):
#     causal_mask = q_idx >= kv_idx
#     window_mask = q_idx - kv_idx <= SLIDING_WINDOW 
#     return causal_mask & window_mask

# def sliding_window(b, h, q_idx, kv_idx):
#     return q_idx - kv_idx <= SLIDING_WINDOW

# class FlexAttention( Module ):
#     def __init__( self, q, k, v ):
#         self.q = q
#         self.k = k
#         self.v = v
    
#     def forward( self, x ):

def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

class CausalSelfAttention( Module ):
    def __init__( self, num_heads, d_model, seq_len, dropout_rate=0.1 ):
        super().__init__()
        # self.mha = MultiheadAttention( num_heads=num_heads, 
        #                                embed_dim=d_model, batch_first=True )
        self.norm = LayerNorm( d_model )
        self.dropout=Dropout( dropout_rate )
        self.num_heads = num_heads
        self.embed_dim = d_model
        self.head_dim = self.embed_dim // self.num_heads

        # self.query = Linear(self.embed_dim, self.embed_dim)
        # self.key = Linear(self.embed_dim, self.embed_dim)
        # self.value = Linear(self.embed_dim, self.embed_dim)
        
        self.qkv = Linear( self.embed_dim, 3 * self.embed_dim )

        self.fc_out = Linear(self.embed_dim, self.embed_dim)

    def forward( self, x, padding_mask=None ):

        # B, S, D = x.shape

        # # T = x.size(1)

        # attn_mask = triu(
        #     torch.ones((S, S), device=x.device, dtype=torch.bool),
        #     diagonal=1
        # )


        # # print( padding_mask )

        # out, _ = self.mha(  query=x, 
        #                     key=x, 
        #                     value=x, 
        #                     attn_mask=attn_mask,
        #                     key_padding_mask=padding_mask,
        #                     is_causal=True,
        #                     need_weights=False )
        

        B, S, D = x.shape

        # Q = self.query( x )
        # K = self.key( x )
        # V = self.value( x )

        # qkv = self.qkv( x )
        # Q, K, V = qkv.chunk( 3, dim=-1 )

        # Q = Q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        # K = K.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        # V = V.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # qkv = self.qkv( x.reshape( B, self.num_heads, S, self.head_dim ) )
        # Q, K, V = qkv.chunk( 3, dim=-1 )

        qkv = self.qkv( x )

        qkv = qkv.view( B, S, 3, self.num_heads, self.head_dim )
        Q, K, V = qkv.unbind( dim=2 )

        Q = Q.transpose( 1, 2 )
        K = K.transpose( 1, 2 )
        V = V.transpose( 1, 2 )

        out = scaled_dot_product_attention( query=Q, 
                                            key=K, 
                                            value=V, 
                                            is_causal=True )
                
        out = out.transpose(1, 2)
        out = out.reshape(B, S, D)

        out = self.fc_out( out )

        # print( out1.shape, out.shape )
        # print( out1, out )

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

        self.feed = FeedForward( d_model=d_model, d_ff=d_ff )
        self.dropout = Dropout( dropout_rate ) 
        self.norm1 = LayerNorm( d_model )
        # self.norm2 = LayerNorm( d_model )

    def forward( self, x, padding_mask=None ):

        out = self.causal( x, padding_mask )

        # out = self.cross( out, out )

        # out = self.norm1( out )
        out = self.feed( out )

        x = x + out
        x = self.norm1( x )
        x = self.dropout( x )  
        
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
        
        # padding_mask = (x == tokenizer.pad_token_id)
        # print( padding_mask.shape )

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
# input_tokens, mask = next(iter(loader))
gpt = GPT( num_heads=1,
           d_model=d_model,
           seq_len=max_seq_length,
           vocab_size=vocab_size,
           d_ff=1024,
           dropout_rate=.1,
           n_layers=6 )

# x = input_tokens[:, :-1]
# test = input_tokens[:, 1:]

def generate( context ):

    for i in range( 128 ):
        out = gpt( context[ :, -max_seq_length: ] )
        out = out[:, -1, :] 
        out = torch.softmax( input=out, dim=-1 )
        out = torch.argmax(out, -1)
        # out = torch.multinomial(out, num_samples=1)
        out = out.unsqueeze(1)

        # print( out )

        context = torch.cat([context, out], dim=1)

        if out == tokenizer.eos_token_id:
            break

    context = context[ 0 ]
    context = context[ context != 2 ]
    return tokenizer.decode( context )


# gpt.load_state_dict(torch.load("gpt.pt"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt = gpt.to(device)

# optimizer = torch.optim.AdamW(gpt.parameters(), lr=3e-4)
# optimizer = torch.optim.SGD(gpt.parameters(), lr=1e-3)

# optimizer = torch.optim.AdamW(
#     gpt.parameters(),
#     lr=3e-4,
#     fused=True
# )

optimizer = Adafactor( gpt.parameters() )

crossentropy = torch.nn.CrossEntropyLoss( ignore_index=tokenizer.pad_token_id )

# accumulation_steps = 3
# num_epochs = 100

# for epoch in range( num_epochs ):

running_loss = 0.
last_loss = 0.
# print( f"epoch {epoch}", flush=True )

# optimizer.zero_grad()

scaler = torch.amp.GradScaler( device="cuda" )

def run_epoch( epoch ):

    gpt.train()

    total_loss = 0.
    num_batches = len(dataset) // 4

    for i, data in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)

        train = data[:, :-1]
        test = data[:, 1:]

        train, test = train.to(device), test.to(device)

        with torch.autocast("cuda", dtype=torch.float16):
            logits = gpt(train)

            logits = logits.reshape(-1, vocab_size)
            test = test.reshape( -1 )

            loss = crossentropy(logits, test)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # if i % 16 == 0:
        #     scaler.step(optimizer)
        #     optimizer.zero_grad(set_to_none=True)
        #     scaler.update()

        total_loss += loss.item()
    print( f"epoch {epoch} avg loss: { total_loss / num_batches }" )

for epoch in range( 100 ):
    run_epoch( epoch )

# gpt.train( mode=True )

tokens = tokenizer.encode("W niedzielę Komenda Miejska Policji")
context = torch.tensor( [tokens] )
context = context.to( device )

print( generate( context ) )

# for i, data in enumerate(loader):
    
#     x = data
            
#     # print( x )

#     # x = x.to(device)
#     # mask = mask.to(device)
    
#     train = x[:, :-1]
#     test = x[:, 1:]

#     train, test = train.to(device), test.to(device)

#     optimizer.zero_grad( set_to_none=True )
    
#     # print( train[0], test[0] )

#     logits = gpt( train )

#     logits = logits.reshape(-1, vocab_size)
#     test = test.reshape( -1 )

#     loss = crossentropy( logits, test )
    # loss.backward()
    
    # optimizer.step()

#     # print( loss.item() )

# #     if (i + 1) % accumulation_steps == 0:
# #         optimizer.step()
# #         optimizer.zero_grad()

    # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

        # if i % 4 == 1:
        #     running_loss += loss.item()
        #     last_loss = running_loss
        #     print(f'  batch {i + 1} loss: {last_loss}', flush=True)

        # tb_x = epoch * len(loader) + i + 1
        # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
        # print( last_loss, tb_x )
        # running_loss = 0.
            
        # break

# tokens = tokenizer.encode("W niedzielę Komenda Miejska Policji")
# context = torch.tensor( [tokens] )
# context = context.to( device )

# print( generate( context ) )

# torch.save(gpt.state_dict(), "gpt.pt")