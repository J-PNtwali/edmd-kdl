import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
import time


torch.set_default_dtype(torch.float64)


#=================================== Deep Koopman Network ===========================
class DeepKoopmanNet(nn.Module):
    def __init__(self, 
                encoder_net:nn.Module,
                decoder_net:nn.Module=None,
                fixed_features_fn:callable=None,
                type="operator"
                ):
        ''' 
        Deep Koopman Network module to be optimized.

        Arg(s):
            encoder_net: encoder net with output dim equal to the number of learnable features
            
            decoder_net: decoder net mapping from the embedding space to the state-space. can be linear or nonlinear.

            fixed_features_fn: fixed dictionary elements, for eg the full-state observable when decoder_net=None.
                Ignored if decoder_net is not None

            type: type of Koopman framework, either "operator" or "generator". Default: "operator"
        '''
        super(DeepKoopmanNet, self).__init__()


        # autoencoder
        self.encoder_net = encoder_net   
        # ensure either the decoder or fixed features are specified
        if decoder_net is None and fixed_features_fn is None:
            raise ValueError("Either of decoder_net or fixed_features_fn must be specified")     
        self.decoder_net = decoder_net
        self.fixed_features_fn = None if decoder_net else fixed_features_fn 
        # dictionary size
        self.dict_size = encoder_net.embed_dim + (fixed_features_fn(torch.zeros(1, encoder_net.input_dim)).shape[1] if fixed_features_fn else 0)
        # EDMD matrix
        self.K = nn.Parameter(torch.zeros((self.dict_size, self.dict_size), dtype=torch.float64), requires_grad=True)
    
    def forward(self, x):
        psi_x = self.encoder_net(x)

        if self.fixed_features_fn:
            psi_x = torch.cat([self.fixed_features_fn(x), psi_x], dim=1)
            xhat = None
        else:
            xhat = self.decoder_net(psi_x)

        return psi_x, xhat
    
    def optimize(self, 
        dataloader, 
        num_epochs:int,
        steps_til_summary:int,
        lr:float=1e-4,
        max_lr:float = 1e-3,
        weight_decay:float=1e-6,
        beta:float = 1.,
        delta:float = 0.,
        rcond:float=None,
        device="cpu"
        ):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.dataloader = dataloader
        self.device = device
        self.delta = delta
        self.rcond = rcond

        losses = deepkoopman_optimizer(self, dataloader, optimizer, num_epochs=num_epochs, steps_til_summary=steps_til_summary,
                                        delta=delta, beta=beta, rcond=rcond, device=device, max_lr=max_lr)
        #self._assemble_edmd_matrix_()
        
        return losses
    
    def _assemble_edmd_matrix_(self):
        with torch.no_grad():
            for b, (X, Y) in enumerate(self.dataloader):
                start_time = time.time()
                X, Y = X.to(self.device), Y.to(self.device)
                
                # Koopman network forward pass -- note that the matrices are transposed
                PsiX, _ = self(X)
                PsiY, _ = self(Y)
                PsiX, PsiY = PsiX.T, PsiY.T

                # calculate covariance matrices
                if b == 0:
                    PsiXX = PsiX @ PsiX.T
                    PsiYX = PsiY @ PsiX.T
                else:
                    PsiXX += PsiX @ PsiX.T
                    PsiYX += PsiY @ PsiX.T
            
            self.K = torch.linalg.lstsq(PsiXX.T + self.delta * torch.eye(PsiXX.shape[0], device=self.device, dtype=PsiXX.dtype),
                                    PsiYX.T, rcond=self.rcond).solution.T

        
class EncoderNet(nn.Module):
    def __init__(self, input_dim:int, embed_dim:int, hidden_dims:list=[], activation=nn.ELU(), as_complex=False):
        ''' 
        Encoder network for deep Koopman.

        Arg(s):
            (*) input_dim: state-space dimension
            (*) embed_dim: embedding space dimension, i.e. number of dictionary functions
            (*) hidden_dims: list of dimensions for hidden layers
            (*) activation: activation function. Default: nn.ELU()
        '''
        super(EncoderNet, self).__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.as_complex = as_complex
        # If complex, the output will be of shape (batch_size, embed_dim, 2) where the last dimension is for real and imaginary parts
        
        # I/O and hidden layers
        if len(hidden_dims) == 0:
            self.encoder = nn.Linear(input_dim, embed_dim)
        else:
            layers = [nn.Linear(input_dim, hidden_dims[0]), activation]
            for i in range(len(hidden_dims) - 1):
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                layers.append(activation)
            # output the real and imaginary parts    
            layers.append(nn.Linear(hidden_dims[-1], embed_dim * 2) if as_complex else nn.Linear(hidden_dims[-1], embed_dim))
            self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        psi_x = self.encoder(x).view(-1, self.embed_dim, 2) if self.as_complex else self.encoder(x).view(-1, self.embed_dim)
        return psi_x  


class DecoderNet(nn.Module):
    def __init__(self, input_dim:int, embed_dim:int, hidden_dims:list=[], activation=nn.ELU(), as_complex=True):
        ''' 
        Decoder net for deep Koopman

        Arg(s):
            (*) input_dim: state-space dimension
            (*) embed_dim: embedding space dimension, i.e. number of eigen-functions
            (*) hidden_dims: list of dimensions for hidden layers
            (*) activation: activation function. Default: nn.ELU()
        '''
        super(DecoderNet, self).__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.as_complex = as_complex
        
        # I/O and hidden layers
        if len(hidden_dims) == 0:
            self.decoder = nn.Linear(embed_dim * 2 if as_complex else embed_dim, input_dim)
        else:
            layers = [nn.Linear(embed_dim * 2 if as_complex else embed_dim, hidden_dims[0]), activation] 
            for i in range(len(hidden_dims) - 1):
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
                layers.append(activation)
            # outputs are real valued  
            layers.append(nn.Linear(hidden_dims[-1], input_dim))
            self.decoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.decoder(x)


def deepkoopman_optimizer(
        koopman_net: DeepKoopmanNet,
        train_dataloader,
        optimizer,
        num_epochs:int,
        steps_til_summary:int,
        max_lr:float = 1e-2,
        beta:float = 1.,
        delta:float = 0.,
        rcond:float=None,
        device="cpu"
    ):
    # Ensure models are on the correct device
    koopman_net = koopman_net.to(device)
    # Learning rate schedule
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=max_lr, 
                                                    steps_per_epoch=len(train_dataloader),
                                                    epochs=num_epochs
                                                )
    loss_fn = torch.nn.MSELoss()

    with tqdm(total=len(train_dataloader) * num_epochs) as pbar:
        losses = []
        
        for epoch in range(num_epochs):
            loss = 0.0

            for _, (X, Y) in enumerate(train_dataloader):
                start_time = time.time()
                X, Y = X.to(device), Y.to(device)
                
                # Koopman network forward pass -- note that the matrices are transposed
                PsiX, Xhat = koopman_net(X)
                PsiY, Yhat = koopman_net(Y)

                #=================          losses   ================================       
                # calculate the EDMD matrix
                PsiXX = PsiX.T @ PsiX
                PsiYX = PsiY.T @ PsiX
                #K = torch.linalg.lstsq(PsiXX.T + delta * torch.eye(PsiXX.shape[0], device=device, dtype=PsiXX.dtype), PsiYX.T, rcond=rcond).solution.T
                
                loss_batch =  torch.norm(PsiY.T - koopman_net.K @ PsiX.T, p='fro')**2 #loss_fn(PsiY.T, K @ PsiX.T)
                
                # full-state observable
                if koopman_net.decoder_net:
                    loss_recon = 0.5 * torch.norm(X - Xhat, p='fro')  + 0.5 * torch.norm(Y - Yhat, p='fro')
                    loss_batch += beta * loss_recon
               
                loss += loss_batch 

                # Backpropagation
                optimizer.zero_grad(set_to_none=True)
                loss_batch.backward()
                optimizer.step()
                pbar.update(1)
                scheduler.step()
            
            # Record losses
            losses.append(loss.item())

            # Training verbose
            if not (epoch + 1) % steps_til_summary:
                tqdm.write(
                    f"Epoch {epoch} | Loss: {loss.item():.2e} | iteration time {time.time()-start_time:.2e}\n"
                )
    
    return losses