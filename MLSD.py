import torch
from math import prod
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float64

# structure constrant of SU2 * SU2 basis
t1 = torch.zeros(6,6,6).to(dtype).to(device)
t1[0,1,2] = 1
t1[2,0,1] = 1
t1[1,2,0] = 1
t1[0,2,1] = -1
t1[1,0,2] = -1
t1[2,1,0] = -1
t1[3,4,5] = 1
t1[5,3,4] = 1
t1[4,5,3] = 1
t1[3,5,4] = -1
t1[4,3,5] = -1
t1[5,4,3] = -1

# structure constrant of rotation basis
t2 = torch.zeros(6,6,6).to(dtype).to(device)
t2[5,2,4] = 1
t2[5,1,3] = 1
t2[4,0,3] = 1
t2[2,0,1] = 1
t2 = t2 - t2.permute(0,2,1) + t2.permute(1,2,0) - t2.permute(1,0,2) + t2.permute(2,0,1) - t2.permute(2,1,0)
    
def energy_kep(xps):
    "ground truth energy function of Kepler problem"
    return ((xps[:,3:]**2).sum(-1) / 2 - 1 / torch.norm(xps[:,:3], dim=-1)).view(-1, 1) # (..., 1)

def energy_ho(xps):
    "ground truth energy function of harmonic oscillator problem"
    return (xps**2).sum(-1, keepdims=True)/2 # (..., 1)

# angular momentum xps -> (..., x y z px py pz)
def L(xps):
    return torch.linalg.cross(xps[:,:3], xps[:,3:]) # (..., 3)

def L1(xps):
    return L(xps)[:,0].view(-1,1)
def L2(xps):
    return L(xps)[:,1].view(-1,1)
def L3(xps):
    return L(xps)[:,2].view(-1,1)

# LRL vector
def A(xps):
    Ls = L(xps) # (..., 3)
    return torch.linalg.cross(xps[:,3:], Ls) - xps[:,:3]/torch.norm(xps[:,:3], dim=-1, keepdim=True) # (..., 3)

def A1(xps):
    return A(xps)[:,0].view(-1,1)
def A2(xps):
    return A(xps)[:,1].view(-1,1)
def A3(xps):
    return A(xps)[:,2].view(-1,1)

# L + A/sqrt(-2E)
def S1(xps):
    return (1/2)*(L(xps) + A(xps)/(-energy_kep(xps)*2).sqrt())[:,0].view(-1,1) # (..., 1)
def S2(xps):
    return (1/2)*(L(xps) + A(xps)/(-energy_kep(xps)*2).sqrt())[:,1].view(-1,1) # (..., 1)
def S3(xps):
    return (1/2)*(L(xps) + A(xps)/(-energy_kep(xps)*2).sqrt())[:,2].view(-1,1) # (..., 1)

# L - A/sqrt(-2E)
def T1(xps):
    return (1/2)*(L(xps) - A(xps)/(-energy_kep(xps)*2).sqrt())[:,0].view(-1,1) # (..., 1)
def T2(xps):
    return (1/2)*(L(xps) - A(xps)/(-energy_kep(xps)*2).sqrt())[:,1].view(-1,1) # (..., 1)
def T3(xps):
    return (1/2)*(L(xps) - A(xps)/(-energy_kep(xps)*2).sqrt())[:,2].view(-1,1) # (..., 1)

def potential(data, tgt):
    "potential function for HMC"
    return ((energy_kep(data) - tgt)**2).view(-1)

class HMCSampler():
    "Hamiltonian Monte Carlo sampler for data preparation"
    def __init__(self, energy):
        self.energy = energy # energy function

    def grad_energy(self, x, tgt):
        "gradient of energy function"
        with torch.enable_grad():
            x.requires_grad_(True)
            total_energy = self.energy(x, tgt).sum()
            grad_energy = torch.autograd.grad(total_energy, x)[0]
        x.requires_grad_(False)
        return grad_energy

    def leap_frog(self, x0, p0, tgt, dt=0.01, traj_len=32):
        with torch.no_grad():
            x, p = x0, p0
            p = p - 0.5 * dt * self.grad_energy(x, tgt)
            x = x + dt * p
            for t in range(traj_len):
                p = p - dt * self.grad_energy(x, tgt)
                x = x + dt * p
            p = p - 0.5 * dt * self.grad_energy(x, tgt)
        return x, p

    def hamiltonian(self, x, p, tgt):
        V = self.energy(x, tgt)
        K = (p ** 2).sum(-1) / 2
        return K + V

    def step(self, x0, tgt, **kwargs):
        p0 = torch.randn_like(x0)
        H0 = self.hamiltonian(x0, p0, tgt)
        x, p = self.leap_frog(x0, p0, tgt, **kwargs)
        H = self.hamiltonian(x, p, tgt)
        prob_accept = torch.exp(H0 - H)
        mask = prob_accept > torch.rand_like(prob_accept)
        x = torch.where(mask[...,None], x, x0)
        return x

    def update(self, x, tgt, steps=1, **kwargs):
        for _ in range(steps):
            x = self.step(x, tgt, **kwargs)
        return x
    
def get_data(N=100, tgt=-5, steps=100):
    sampler = HMCSampler(potential)
    init_data = torch.randn((N, 6), device=device, dtype=dtype)
    init_data = sampler.update(init_data, tgt, steps)
    init_data = init_data[energy_kep(init_data).view(-1) < 0]
    return init_data.requires_grad_(True)

class NN(torch.nn.Module):
    "Linear feedforward neural network"
    def __init__(self, input_size, output_size, dims, dropout_prob=0., meta_func=[]):
        super().__init__()
        self.input_size, self.output_size = input_size, output_size
        self.meta_func = meta_func
        self.dims = dims
        input_size += len(meta_func)
        layers = []
        for d in dims:
            layers.append(torch.nn.Linear(input_size, d))
            #layers.append(torch.nn.ReLU())
            layers.append(torch.nn.SiLU())
            layers.append(torch.nn.Dropout(dropout_prob))
            input_size = d
        layers.append(torch.nn.Linear(dims[-1], output_size))
        self.ffn = torch.nn.Sequential(*layers)
        self.to(dtype)

    def forward(self, x):
        if len(self.meta_func) > 0:
            meta = torch.cat([f(x) for f in self.meta_func], dim=-1).to(x.device, x.dtype)
            x = torch.cat([x, meta], dim=-1)
        return self.ffn(x)
    
class M_net(torch.nn.Module):
    "Quadratic neural network for observables"
    def __init__(self, input_size, output_size, **kwargs):
        super().__init__()
        self.input_size, self.output_size = input_size, output_size
        self.G = torch.nn.Parameter(torch.randn(int(input_size),int(input_size)).to(dtype))

    def forward(self, x):
        # x (...,6)
        return torch.bmm(torch.bmm(x.unsqueeze(1), self.G[None,...].expand(x.shape[0],-1,-1)),x.unsqueeze(-1)).view(-1,1)
    
class Observable(torch.nn.Module):
    "Wrapper for observable functions"
    def __init__(self, func):
        super().__init__()
        self.d_ph = func.input_size # dim phase
        self.func = func
        
    # xps -> (..., x_dim + pdim)
    def forward(self, xps):
        return self.func(xps) # (..., 1)
    
def grad(func, xps):
    xps.requires_grad_(True)
    grad = torch.autograd.grad(func(xps).sum(), xps, create_graph=True)[0] # (..., phase dim)
    return grad

class MLSD(torch.nn.Module):
    
    def __init__(self, phase_dim, lie_dim, base_NN, energy_func=None, obs_list=None, **kwargs):
        super().__init__()
        self.phase_dim, self.lie_dim = phase_dim, lie_dim
        self.energy_func = Observable(base_NN(input_size=phase_dim, output_size=1, **kwargs)) if energy_func is None else energy_func
        if obs_list is not None:
            self.obs = obs_list
        else:
            self.obs = torch.nn.ModuleList([Observable(base_NN(input_size=phase_dim, output_size=1, **kwargs)) for _ in range(lie_dim)])
        self.J = torch.kron(torch.tensor([[0.,1.],[-1.,0.]]), torch.eye(int(self.phase_dim/2), int(self.phase_dim/2))).to(device).to(dtype) # (phase dim, phase dim)
        self.f_para = torch.nn.Parameter(torch.randn(3*(lie_dim, ), device=device, dtype=dtype).to(device))
        
    # output anti-symmetric structure constant
    @property
    def f(self):
        return self.f_para - self.f_para.permute(0,2,1) + self.f_para.permute(1,2,0) - self.f_para.permute(1,0,2) + self.f_para.permute(2,0,1) - self.f_para.permute(2,1,0)
        #return -t1
    
    # output symmetric Killing form
    @property
    def B(self):
        return torch.tensordot(self.f, self.f, dims=([1,2],[2,1]))
        
    # output Poisson bracket
    def PB(self, obs1, obs2):
        def out_func(xps):
            vec1 = grad(obs1, xps) # (..., phase dim)
            vec2 = grad(obs2, xps) @ self.J # (..., phase dim)
            return (vec1 * vec2).sum(-1) # (...)
        return out_func # (...)
    
    # output Jacobi identity
    def Jacob(self, obs1, obs2, obs3):
        def out_func(xps):
            return self.PB(obs1, self.PB(obs2, obs3))(xps) + self.PB(obs3, self.PB(obs1, obs2))(xps) + self.PB(obs2, self.PB(obs3, obs1))(xps)
        return out_func
    
    # output values of observables
    def obs_vals(self, xps):
        return torch.cat([obs(xps) for obs in self.obs], -1).view(-1, self.lie_dim) # (..., lie dim)
    
    # output loss function
    def loss(self, xps):
        loss_con = sum(self.PB(self.energy_func, obs)(xps)**2 for obs in self.obs) # (...)
        loss_lie = 0.
        right = torch.tensordot(self.obs_vals(xps), self.f, dims=([1],[0])) # (..., lie dim, lie dim)
        for i in range(self.lie_dim):
            for j in range(i+1, self.lie_dim):
                loss_lie += (self.PB(self.obs[i], self.obs[j])(xps) - right[:,i,j])**2
        return (loss_con.mean() + loss_lie.mean())/self.lie_dim**2

    def entropy(self, xps, eps=1e-10):
        M = torch.cat([grad(obs, xps).unsqueeze(1) for obs in self.obs]+[(grad(self.energy_func, xps)@ self.J).unsqueeze(1)], 1) # (..., lie dim + 1, phase dim)
        M = M/torch.norm(M, dim=-1, keepdim=True)
        Gram = M @ M.mT # (..., lie dim + 1, lie dim + 1)
        eig = torch.linalg.eigh(Gram)[0] # (..., lie dim + 1)
        eig = eig + eps
        norm_eig = eig/ torch.sum(eig, dim=-1, keepdim=True)
        entropy = (-norm_eig * torch.log(norm_eig)).sum(-1)
        return entropy.mean()