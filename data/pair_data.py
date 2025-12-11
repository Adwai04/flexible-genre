from torch.utils.data import Dataset
import torch
import numpy as np
import matplotlib.pyplot as plt


class PairData(Dataset):
    def __init__(self, X, y, ystar, k=2, dist_p=2):
        self.X = X
        self.y = y
        self.ystar = ystar

        self.pair_idxs = None
        self.k = k
        self.dist_p = dist_p
        self.create_pairs()

    def create_pairs(self):
        D = torch.cdist(self.X, self.X, p=self.dist_p)
        # TODO: do we need to avoid self-pairing?
        D = D + torch.eye(D.shape[0]) * 1e10  # avoid self-pairing

        # TODO: should only output good labels?
        good_idxs = torch.where(self.y == self.ystar)[0]
        self.pair_idxs = torch.topk(D[:, good_idxs], k=self.k, largest=False)[1]
        self.pair_idxs = good_idxs[self.pair_idxs]
        pass

    def __len__(self):
        return len(self.X) * self.k

    def __getitem__(self, idx):
        # pair_id = self.pair_idxs[idx][torch.randperm(len(self.pair_idxs[idx]))[0]]
        idx, pair_id = idx // self.k, self.pair_idxs[idx // self.k][idx % self.k]
        return {
            "x": self.X[idx],
            "y": self.y[idx],
            "pair_x": self.X[pair_id],
            "pair_y": self.y[pair_id],
        }


class PairDatav2(Dataset):
    def __init__(self, X, y, ystar, k=2, dist_p=2):
        self.X = X
        self.y = y
        self.ystar = ystar
        self.goodX = X[y == ystar]
        self.goody = y[y == ystar]
        self.pair_idxs = None
        self.k = k
        self.dist_p = dist_p
        self.create_pairs()

    def create_pairs(self):
        # allows for self pairing, it's ok
        D = torch.cdist(self.goodX, self.X, p=self.dist_p)
        self.pair_idxs = torch.topk(D, k=self.k, largest=False)[1]

    def __len__(self):
        return len(self.goodX) * self.k

    def __getitem__(self, idx):
        # pair_id = self.pair_idxs[idx][torch.randperm(len(self.pair_idxs[idx]))[0]]
        idx, pair_idx = idx // self.k, self.pair_idxs[idx // self.k][idx % self.k]
        return {
            "x": self.X[pair_idx],
            "y": self.y[pair_idx],
            "pair_x": self.goodX[idx],
            "pair_y": self.goody[idx],
        }


class WeightedPairData(Dataset):
    def __init__(self, X, y, ystar, lambda_=0.9, k=2, p=2):
        src_labels = [1 - ystar]
        tgt_labels = [ystar]

        if len(src_labels) == 2:
            self.src_X = X
            self.src_y = y
        else:
            self.src_X = X[y == src_labels[0]]
            self.src_y = y[y == src_labels[0]]

        if len(tgt_labels) == 2:
            self.tgt_X = X
            self.tgt_y = y
        else:
            self.tgt_X = X[y == tgt_labels[0]]
            self.tgt_y = y[y == tgt_labels[0]]

        self.ystar = ystar

        self.pair_idxs = None
        self.pair_dist = None  # will be used for sampling

        self.k = k
        self.p = p
        self.lambda_ = lambda_
        self.create_pairs()

    def create_pairs(self):
        D = torch.cdist(self.src_X, self.tgt_X, p=self.p)
        topk_out = torch.topk(D, k=self.k, largest=False)
        pair_dist_exp = torch.exp(-self.lambda_ * topk_out[0])

        self.pair_prob = pair_dist_exp / pair_dist_exp.sum(dim=-1, keepdim=True)
        self.pair_idxs = topk_out[1]

    def __len__(self):
        return len(self.src_X) * self.k

    def __getitem__(self, idx):

        idx, pair_idx, wght_idx = (
            idx // self.k,
            self.pair_idxs[idx // self.k][idx % self.k],
            idx % self.k,
        )

        return {
            "x": self.src_X[idx],
            "y": self.src_y[idx],
            "pair_x": self.tgt_X[pair_idx],
            "pair_y": self.tgt_y[pair_idx],
            "weight": self.pair_prob[idx][wght_idx],
        }


# TODO: Deprecate StochasticPairData
class StochasticPairData(Dataset):
    def __init__(self, X, y, ystar, lambda_=0.9, k=2, p=2):
        src_labels = [1 - ystar]
        tgt_labels = [ystar]

        if len(src_labels) == 2:
            self.src_X = X
            self.src_y = y
        else:
            self.src_X = X[y == src_labels[0]]
            self.src_y = y[y == src_labels[0]]

        if len(tgt_labels) == 2:
            self.tgt_X = X
            self.tgt_y = y
        else:
            self.tgt_X = X[y == tgt_labels[0]]
            self.tgt_y = y[y == tgt_labels[0]]

        self.ystar = ystar

        self.pair_idxs = None
        self.pair_dist = None  # will be used for sampling

        self.k = k
        self.p = p
        self.lambda_ = lambda_
        self.create_pairs()

    def create_pairs(self):
        D = torch.cdist(self.src_X, self.tgt_X, p=self.p)

        if self.k > D.shape[1]:
            topk_out = torch.topk(D, k=D.shape[1], largest=False)
            self.k = D.shape[1]
        else:
            topk_out = torch.topk(D, k=self.k, largest=False)

        pair_dist_exp = torch.exp(-self.lambda_ * topk_out[0])

        self.pair_prob = pair_dist_exp / pair_dist_exp.sum(dim=-1, keepdim=True)
        self.pair_idxs = topk_out[1]

    def __len__(self):
        return len(self.src_X)

    def __getitem__(self, idx):

        sampled_idx = np.random.choice(
            np.arange(0, self.k), p=self.pair_prob[idx].cpu().numpy()
        )
        pair_id = self.pair_idxs[idx][sampled_idx]
        return {
            "x": self.src_X[idx],
            "y": self.src_y[idx],
            "pair_x": self.tgt_X[pair_id],
            "pair_y": self.tgt_y[pair_id],
        }


class StochasticPairs(Dataset):

    def __init__(self, src_X, tgt_X, src_y, tgt_y, lambda_=0.9, k=2, p=2):
        self.src_X = src_X
        self.src_y = src_y
        self.tgt_X = tgt_X
        self.tgt_y = tgt_y

        self.pair_idxs = None
        self.pair_dist = None  # will be used for sampling

        self.k = k
        self.p = p
        self.lambda_ = lambda_
        self.create_pairs()

    def create_pairs(self):
        D = torch.cdist(self.src_X, self.tgt_X, p=self.p)

        if self.k > D.shape[1]:
            topk_out = torch.topk(D, k=D.shape[1], largest=False)
            self.k = D.shape[1]
        else:
            topk_out = torch.topk(D, k=self.k, largest=False)

        pair_dist_exp = torch.exp(-self.lambda_ * topk_out[0])

        self.pair_prob = pair_dist_exp / pair_dist_exp.sum(dim=-1, keepdim=True)
        self.pair_idxs = topk_out[1]

    def __len__(self):
        return len(self.src_X)

    def __getitem__(self, idx):

        sampled_idx = np.random.choice(
            np.arange(0, self.k), p=self.pair_prob[idx].cpu().numpy()
        )
        pair_id = self.pair_idxs[idx][sampled_idx]
        return {
            "x": self.src_X[idx],
            "y": self.src_y[idx],
            "pair_x": self.tgt_X[pair_id],
            "pair_y": self.tgt_y[pair_id],
        }


class StochasticPairsNegSamp(Dataset):
    def __init__(self, src_X, tgt_X, src_y, tgt_y,num_neg,lambda_, k, p):
        self.src_X = src_X
        self.src_y = src_y
        self.tgt_X = tgt_X
        self.tgt_y = tgt_y

        self.pair_idxs = None
        self.pair_dist = None  # will be used for sampling

        self.k = k
        self.p = p
        self.lambda_ = lambda_
        self.num_neg = num_neg
        self.tgt_len = len(self.tgt_X)
        self.create_pairs()

    def create_pairs(self):
        D = torch.cdist(self.src_X, self.tgt_X, p=self.p)

        if self.k > D.shape[1]:
            topk_out = torch.topk(D, k=D.shape[1], largest=False)
            self.k = D.shape[1]
        else:
            topk_out = torch.topk(D, k=self.k, largest=False)

        pair_dist_exp = torch.exp(-self.lambda_ * topk_out[0])

        self.pair_prob = pair_dist_exp / pair_dist_exp.sum(dim=-1, keepdim=True)
        self.pair_idxs = topk_out[1]

    def __len__(self):
        return len(self.src_X)

    def __getitem__(self, idx):

        sampled_idx = np.random.choice(
            np.arange(0, self.k), p=self.pair_prob[idx].cpu().numpy()
        )
        pair_id = self.pair_idxs[idx][sampled_idx]
        neg_pair_id  = list(np.random.choice(self.tgt_len, self.num_neg, replace=False)) # multiple, for each example we sample num_neg negative pairs
        return {
            "x": self.src_X[idx],
            "y": self.src_y[idx],
            "pair_x": self.tgt_X[pair_id],
            "pair_y": self.tgt_y[pair_id],
            "neg_pair_x" : self.tgt_X[neg_pair_id]
        }
    

class StochasticPairsImmut(Dataset):

    def __init__(self, src_X, tgt_X, src_y, tgt_y, immutable_mask,lambda_=0.9, k=2, p=2, w=[1.0, 1.0]):
        self.src_X = src_X
        self.src_y = src_y
        self.tgt_X = tgt_X
        self.tgt_y = tgt_y

        self.pair_idxs = None
        self.pair_dist = None  # will be used for sampling

        self.k = k
        self.p = p
        self.lambda_ = lambda_

        if torch.all(torch.tensor(immutable_mask)):
            self.immutable_mask = None
        else:
            self.immutable_mask = immutable_mask
        self.create_pairs(w)

    def create_pairs(self, cost_weights):
        # D = torch.cdist(self.src_X, self.tgt_X, p=self.p)
        cost_weights = torch.tensor(cost_weights)
        cost_weights /= torch.sum(cost_weights)
        diff = (self.src_X[:, None, :] - self.tgt_X[None, :, :]).abs()  # shape: [N_src, N_tgt, D]
        D = ((diff ** self.p) * cost_weights[None, None, :]).sum(dim=-1) ** (1 / self.p)

        expD = torch.exp(-self.lambda_ * D)

        if self.immutable_mask is not None:
            immut_dist = 1 - 1.0*(torch.cdist(self.src_X[:,self.immutable_mask], self.tgt_X[:,self.immutable_mask], p=self.p) > 0)
            expD = immut_dist*expD

        if self.k > D.shape[1]:
            topk_out = torch.topk(expD, k=D.shape[1], largest=True)
            self.k = D.shape[1]
        else:
            topk_out = torch.topk(expD, k=self.k, largest=True)

        pair_dist_exp = topk_out[0]

        self.pair_prob = pair_dist_exp / pair_dist_exp.sum(dim=-1, keepdim=True)
        self.pair_idxs = topk_out[1]

    def __len__(self):
        return len(self.src_X)

    def __getitem__(self, idx):

        sampled_idx = np.random.choice(
            np.arange(0, self.k), p=self.pair_prob[idx].cpu().numpy()
        )
        pair_id = self.pair_idxs[idx][sampled_idx]
        return {
            "x": self.src_X[idx],
            "y": self.src_y[idx],
            "pair_x": self.tgt_X[pair_id],
            "pair_y": self.tgt_y[pair_id],
        }


class StochasticPairsImmutFlexible(Dataset):

    def __init__(self, src_X, tgt_X, src_y, tgt_y,lambda_=0.9, num_samples=10000, n_bins=50, epsilon=1e-13):
        N, D = src_X.shape
        self.src_X = src_X
        self.src_y = src_y
        self.tgt_X = tgt_X
        self.tgt_y = tgt_y
        self.num_samples = num_samples
        self.priors = torch.zeros(D, n_bins)
        self.dim = D
        self.epsilon = epsilon
        self.bin_centers = None
        
        #Counting based priors (Naive Bayes fuj)
        bins = torch.linspace(0, 1, n_bins + 1)
        self.bin_centers = 0.5 * (bins[:-1] + bins[1:])
        self.bin_width = bins[1] - bins[0]

        for j in range(D):
            idx = torch.bucketize(src_X[:, j], bins, right=False) - 1
            idx = torch.clamp(idx, 0, n_bins - 1)
            counts = torch.bincount(idx, minlength=n_bins).float()
            self.priors[j] = counts / counts.sum()

        # self.plot_and_save_priors_heatmap()

        self.lambda_ = lambda_
        self.pair_X = None
        self.pair_r = None
        self.pair_a = None
        self.create_pairs()
        
    def create_pairs(self):
        bin_width = self.bin_width
    
        pairs_x, pairs_r, pairs_a = [], [], []
    
        for _ in range(self.num_samples):
            idx = torch.randint(0, self.tgt_X.shape[0], (1,))
            x_plus = self.tgt_X[idx].squeeze(0) 
            a = torch.rand(self.dim)
            a = a / a.sum()
    
            r = torch.zeros(self.dim)
    
            for j in range(self.dim):
                cj = a[j] * torch.abs(x_plus[j] - self.bin_centers)
                w = (torch.exp(- self.lambda_ * cj) + self.epsilon) * self.priors[j]
                p = w / w.sum()
                u_idx = torch.multinomial(p, 1).item()
                r_j = self.bin_centers[u_idx]
                noise = torch.randn(1) * (bin_width / 8)
                r[j] = r_j + noise
    
            pairs_x.append(x_plus)
            pairs_r.append(r)
            pairs_a.append(a)
    
        self.pair_X = torch.stack(pairs_x)
        self.pair_r = torch.stack(pairs_r)
        self.pair_a = torch.stack(pairs_a)

    def __getitem__(self, idx):
        return {
            "pair_x": self.pair_X[idx],
            "x": self.pair_r[idx],
            "weights": self.pair_a[idx],
            "y": 0,
            "pair_y" : 1,
            
        }

    def __len__(self):
        return self.num_samples
    
    def plot_and_save_priors_heatmap(self, save_path="priors_heatmap.png"):
        """
        Draw and save a heatmap of self.priors tensor.
        Each row corresponds to a dimension and each column to a bin.
        """
        priors = self.priors.detach().cpu()  # ensure it's on CPU and detached from autograd

        plt.figure(figsize=(8, 5))
        im = plt.imshow(priors, cmap="viridis", aspect="auto", origin="lower")
        plt.colorbar(im, label="Prior value")
        plt.xlabel("Bins")
        plt.ylabel("Dimensions")
        plt.title("Heatmap of self.priors")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"✅ Heatmap saved to {save_path}")