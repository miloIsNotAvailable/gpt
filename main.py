import torch
# from torch import zeros, arange, exp, sin, cos, triu, tensor, full, __version__, long, softmax, argmax, cat
from torch.nn import Module, Embedding, LayerNorm, Linear, ReLU, Dropout, ModuleList, GELU
from math import sqrt, log
import pandas
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoTokenizer, Adafactor
# import csv
# from itertools import islice
# from torch.nn.attention.flex_attention import flex_attention, and_masks, create_block_mask
from torch.nn.functional import scaled_dot_product_attention
# print(f"GPU: {torch.cuda.is_available()}")
import time

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def block_causal_mask(x, eos_token_id):

    device = x.device

    B, T = x.shape

    eos_idx = (x.view(-1) == eos_token_id).nonzero(as_tuple=True)[0] + True

    eos_idx_expanded = torch.cat([
        eos_idx,
        torch.arange(0, B*T+1, T, device=device)
    ]).unique().sort()[0]

    normalized_idx = eos_idx_expanded - (eos_idx_expanded // T) * T

    normalized_idx = torch.where(
        normalized_idx == 0,
        torch.tensor(T, device=device),
        normalized_idx
    )

    reps = normalized_idx[1:] - normalized_idx[:-1]

    reps = torch.where(
        reps < 1,
        normalized_idx[1:],
        reps
    )

    repeated_idx = torch.repeat_interleave(
        normalized_idx[1:],
        reps
    ).view(B,1,T).expand(-1,T,-1)

    mask_indices = torch.arange(
        T,
        device=device
    ).view(1,-1,1).expand(B,-1,T)

    mask = torch.ones(
        T,
        T,
        dtype=torch.bool,
        device=device
    ).tril().expand(B,-1,-1)

    mask = mask.masked_fill(mask_indices >= repeated_idx, False)

    return mask

class PackedDataset( IterableDataset ):
    def __init__(self, filename, block_size=64):

        self.df = pandas.read_csv(filename)
        self.block_size = block_size
        self.word_set = set()

    def __len__( self ):
        # return len( self.df )
        return 50

    def stream_df_tokens( self ):
        for i in range( len( self ) ):
            lead = self.df.iloc[i, 2]  
            tokenized = tokenizer(lead, add_special_tokens=False)["input_ids"]

            tokenized = tokenized + [ tokenizer.eos_token_id ]

            for t in tokenized:
                self.word_set.add( t )
                yield t


    def __iter__( self ):
        buffer = []
        for token in self.stream_df_tokens():
            
            buffer += [ token ] 
            if len( buffer ) == self.block_size: 
                yield torch.tensor( buffer )
                buffer = []


class TextDataset(Dataset):

    def __init__(self, filename, block_size=128):
        self.df = pandas.read_csv(filename)
        # self.tokens = [ tokenizer(t, add_special_tokens=True)["input_ids"]
        #                 for t in self.df["lead"].dropna()[:10] ]
        self.block_size = block_size

    def __len__( self ):
        return 1000

    def __getitem__( self, idx ):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        lead = self.df.iloc[idx % 1000, 2] 

        tokens = tokenizer(lead, add_special_tokens=True, max_length=self.block_size)["input_ids"]

        l = len( tokens )

        tokens = torch.tensor( tokens )
        tokens = torch.cat( [ tokens, torch.full( (self.block_size - l,), tokenizer.pad_token_id ) ] )

        return tokens


# sentences = df["lead"].dropna()[:3]
# # print( sentences.to_list() )
# tokens = tokenizer(sentences.to_list(), return_attention_mask=False, add_special_tokens=False)["input_ids"]

# tokens = [ t for s in tokens for t in s + [tokenizer.eos_token_id]  ]

# tokens = torch.tensor(tokens)

# print(get_attention_mask_for_packed_sequence(tokens, tokenizer.eos_token_id).shape)

max_seq_length = 64

# dataset = TextDataset( filename="./out11.csv", block_size=max_seq_length )
dataset = PackedDataset( filename="./out11.csv", block_size=max_seq_length )

df = pandas.read_csv( "out11.csv" )
word_set = set()
# count = 0

for i in range( len( df ) ):
    lead = df.iloc[i, 2]  
    tokenized = tokenizer(lead, add_special_tokens=False)["input_ids"]

    tokenized = tokenized + [ tokenizer.eos_token_id ]

    for t in tokenized:
        word_set.add( t )
        # count += 1

print( f"unqiue tokens in ds: { len( word_set ) }" )

loader = DataLoader(
    dataset,
    batch_size=4,
    # shuffle=True
)

for i, batch in enumerate(loader):
    # input_ids = batch
    for j in range( 3 ):
        print(tokenizer.decode( batch[j] ))
    
    break

for i, batch in enumerate(loader):
    # input_ids = batch
    print( block_causal_mask(batch, eos_token_id=tokenizer.eos_token_id) )
    # for j in range( 3 ):
        # eos_token_indices = (batch[j] == tokenizer.eos_token_id).nonzero()
        # print( block_causal_mask(batch[j], eos_token_id=tokenizer.eos_token_id) )
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

        # print( x.shape )

        # mask = block_causal_mask(x, eos_token_id=tokenizer.eos_token_id)
        # mask = mask.unsqueeze(1)
        # mask = mask.expand(-1, 4, -1, -1)

        # with torch.nn.attention.sdpa_kernel( [
        #     # torch.backends.cuda.enable_flash_sdp(),
        #     torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
        #     torch.nn.attention.SDPBackend.FLASH_ATTENTION,
        # ] ):

        # padding_mask = padding_mask.unsqueeze(1)
        # print( x.shape, padding_mask.shape )
        padding_mask = padding_mask[:, None, :, :]

        out = scaled_dot_product_attention( query=Q, 
                                            key=K, 
                                            value=V, 
                                            attn_mask=padding_mask,
                                            is_causal=False )
                
        out = out.transpose(1, 2)
        out = out.reshape(B, S, D)

        out = self.fc_out( out )

        # print( out1.shape, out.shape )
        # print( out1, out )

        # out = x + out
        # out = self.norm( out )

        out = self.dropout( out )

        return out

class FeedForward(Module):
    def __init__(self, d_model, d_ff, dropout_rate=.1):
        super().__init__()
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        self.relu = GELU()
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
        self.norm2 = LayerNorm( d_model )

    def forward( self, x, padding_mask=None ):

        # out = self.causal( x, padding_mask )

        # x = x + self.feed( out )
        # x = self.norm1( x )
        x = self.dropout( x )  
        
        x = x + self.causal( self.norm1(x), padding_mask )
        x = x + self.feed(self.norm2(x))
        # x = self.dropout(x)

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

        self.dropout = Dropout( .1 )

        self.norm = LayerNorm( d_model )

    def forward( self, x ):
        
        # padding_mask = (x == tokenizer.pad_token_id)
        # print( padding_mask.shape )

        mask = block_causal_mask(x, eos_token_id=tokenizer.eos_token_id)

        x = self.embedding( x )
        x = self.positional_encoding( x )
        
        x = self.dropout( x )

        for layer in self.layers:
            x = layer( x, mask )

        x = self.norm( x )
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
gpt = GPT( num_heads=4,
           d_model=d_model,
           seq_len=max_seq_length,
           vocab_size=vocab_size,
           d_ff=1024,
           dropout_rate=.1,
           n_layers=6 )

# x = input_tokens[:, :-1]
# test = input_tokens[:, 1:]

def generate(context, temperature=0.8, top_k=40):

    for _ in range(512):

        logits = gpt(context[:, -max_seq_length:])

        logits = logits[:, -1, :]

        values, indices = torch.topk(logits, top_k)

        probs = torch.softmax(values / temperature, dim=-1)

        sampled = torch.multinomial(probs, 1)

        out = indices.gather(-1, sampled)

        context = torch.cat([context, out], dim=1)

        if out.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(context[0])


# gpt.load_state_dict(torch.load("gpt_10k_104.pt"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt = gpt.to(device)

# optimizer = torch.optim.AdamW(gpt.parameters(), lr=3e-4)
# optimizer = torch.optim.SGD(gpt.parameters(), lr=1e-3)

# optimizer = torch.optim.AdamW(
#     gpt.parameters(),
#     lr=3e-4,
#     fused=True
# )

optimizer = Adafactor(  gpt.parameters(),
                        lr=1e-3,
                        scale_parameter=True,
                        relative_step=False,
                        warmup_init=False )

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
    start = time.time()

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
    print( f"epoch {epoch} avg loss: { total_loss / num_batches } time: {time.time() - start}" )

for epoch in range( 25 ):
    run_epoch( epoch )

# gpt.train( mode=False )

proompts = [ "Trwają poszkiwania", 
             "W Niedziele", 
             "Zaginął", 
             "Zaginiona",
             "Zaginęła",
             "W niedzielę Komenda Miejska Policji",
             "Zobacz" 
             ]


for i in proompts:
    tokens = tokenizer.encode(i, add_special_tokens=False)
    # tokens = tokens + [tokenizer.eos_token_id]
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