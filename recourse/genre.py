import torch

class GenRe:
    def __init__(self, pair_model, temp, sigma, best_k, ann_clf,ystar, cat_mask):
        self.pair_model = pair_model.eval()
        self.temp = temp
        self.sigma = sigma 
        # assert 0 <sigma < 1/pair_model.n_bins
        self.best_k = best_k
        self.model = ann_clf.eval()
        self.ystar = ystar
        self.cat_mask = cat_mask

    @torch.no_grad   
    def __call__(self,xf_r):
        DEVICE = xf_r.device
        yf_r = torch.ones(xf_r.shape[0]).to(DEVICE)*self.ystar
        self.pair_model = self.pair_model.to(DEVICE)
        self.model = self.model.to(DEVICE)

        sampled_list = []
        for i in range(self.best_k):
            sample_xcf = self.pair_model._sample(xf_r, yf_r, y=yf_r*0 + self.ystar, temp=self.temp, sigma=self.sigma)
            # project categorical variables
            sample_xcf[:,self.cat_mask] = torch.round(sample_xcf[:,self.cat_mask])
            if self.best_k ==1:
                return sample_xcf
            sampled_list.append(sample_xcf.detach().unsqueeze(1))

        sample_concat = torch.concat(sampled_list,dim=1)
        sample_predictions = self.model(sample_concat).squeeze() 
        best_sample_idx = sample_predictions.argmax(dim=1)
        return sample_concat[torch.arange(xf_r.shape[0]),best_sample_idx,:]
    

class FlexibleGenRe:
    def __init__(self, pair_model, temp, sigma, best_k, ann_clf,ystar, cat_mask):
        self.pair_model = pair_model.eval()
        self.temp = temp
        self.sigma = sigma 
        # assert 0 <sigma < 1/pair_model.n_bins
        self.best_k = best_k
        self.model = ann_clf.eval()
        self.ystar = ystar
        self.cat_mask = cat_mask

    @torch.no_grad   
    def __call__(self,xf_r, w):
        DEVICE = xf_r.device
        w = torch.tensor(w)
        w = w/torch.sum(w)
        w = w.expand(len(xf_r), -1) #broadcasting for each input
        yf_r = torch.ones(xf_r.shape[0]).to(DEVICE)*self.ystar
        self.pair_model = self.pair_model.to(DEVICE)
        self.model = self.model.to(DEVICE)

        sampled_list = []
        for i in range(self.best_k):
            sample_xcf = self.pair_model._sample(xf_r, yf_r, y=yf_r*0 + self.ystar, weights=w, temp=self.temp, sigma=self.sigma)
            # project categorical variables
            sample_xcf[:,self.cat_mask] = torch.round(sample_xcf[:,self.cat_mask])
            if self.best_k ==1:
                return sample_xcf
            sampled_list.append(sample_xcf.detach().unsqueeze(1))

        sample_concat = torch.concat(sampled_list,dim=1)
        sample_predictions = self.model(sample_concat).squeeze() 
        best_sample_idx = sample_predictions.argmax(dim=1)
        return sample_concat[torch.arange(xf_r.shape[0]),best_sample_idx,:]
    
    @torch.no_grad   
    def sample(self,xf_r, w):
        DEVICE = xf_r.device
        w = torch.tensor(w)
        w = w/torch.sum(w)
        w = w.expand(len(xf_r), -1) #broadcasting for each input
        yf_r = torch.ones(xf_r.shape[0]).to(DEVICE)*self.ystar
        self.pair_model = self.pair_model.to(DEVICE)
        self.model = self.model.to(DEVICE)

        sampled_list = []
        for i in range(self.best_k):
            sample_xcf = self.pair_model._sample(xf_r, yf_r, y=yf_r*0 + self.ystar, weights=w, temp=self.temp, sigma=self.sigma)
            # project categorical variables
            sample_xcf[:,self.cat_mask] = torch.round(sample_xcf[:,self.cat_mask])
            if self.best_k ==1:
                return sample_xcf
            sampled_list.append(sample_xcf.detach().unsqueeze(1))

        sample_concat = torch.concat(sampled_list,dim=1)
        return sample_concat


class OldFlexibleGenRe:
    def __init__(self, pair_models, temp, sigma, best_k, ann_clf,ystar, cat_mask):
        self.pair_models = [pair_model.eval() for pair_model in pair_models]
        self.temp = temp
        self.sigma = sigma 
        # assert 0 <sigma < 1/pair_model.n_bins
        self.best_k = best_k
        self.model = ann_clf.eval()
        self.ystar = ystar
        self.cat_mask = cat_mask

    @torch.no_grad   
    def __call__(self,xf_r, weights=[1.0, 0.0], gamma=1e-7):
        #######
        weights = torch.tensor(weights)
        weights = weights/torch.sum(weights)
        #######
        DEVICE = xf_r.device
        yf_r = torch.ones(xf_r.shape[0]).to(DEVICE)*self.ystar
        # self.pair_model = self.pair_model.to(DEVICE)
        pair_model_1 = self.pair_models[0].to(DEVICE)
        pair_model_2 = self.pair_models[1].to(DEVICE)
        self.model = self.model.to(DEVICE)

        sampled_list = []
        for i in range(self.best_k):
            pair_model_1.eval()
            pair_model_2.eval()
            temp=self.temp
            sigma=self.sigma
            xf = xf_r
            yf = yf_r.view(-1, 1)
            y = (yf_r*0 + self.ystar).view(-1,1)
            memory_1 = pair_model_1._encode(xf, yf)
            memory_2 = pair_model_2._encode(xf, yf)
            batch_size = xf.shape[0]
            hist = torch.zeros((batch_size,0), device=xf.device) # initialize empty features, will be concatenated with y
            for i in range(pair_model_1.num_inputs):
                mask_1 = pair_model_1.generate_square_subsequent_mask(i+1)
                mask_2 = pair_model_2.generate_square_subsequent_mask(i+1)
                logits_1 = temp*pair_model_1._decode(hist, mask_1, y, memory_1) 
                probs_1 = logits_1.softmax(dim=-1) 
                logits_2 = temp*pair_model_2._decode(hist, mask_2, y, memory_2) 
                probs_2 = logits_2.softmax(dim=-1) 

                # if i == 0:
                probs_1 = probs_1 + gamma
                probs_1 = probs_1 / torch.sum(probs_1)
                probs_2 = probs_2 + gamma
                probs_2 = probs_2 / torch.sum(probs_2)

                probs = (probs_1**weights[0])*(probs_2**weights[1])

                sampled_idx = torch.multinomial(probs,1) # get bin index
                bin_val = pair_model_1.bins[sampled_idx] # get bin value for each batch element
                samp_val = bin_val + sigma*torch.randn_like(bin_val) # add sizzle to bin value
                hist = torch.cat((hist, samp_val), dim=1)

            sample_xcf = hist

            # sample_xcf = self.pair_model._sample(xf_r, yf_r, y=yf_r*0 + self.ystar, temp=self.temp, sigma=self.sigma)
            # project categorical variables
            sample_xcf[:,self.cat_mask] = torch.round(sample_xcf[:,self.cat_mask])
            if self.best_k ==1:
                return sample_xcf
            sampled_list.append(sample_xcf.detach().unsqueeze(1))

        sample_concat = torch.concat(sampled_list,dim=1)
        sample_predictions = self.model(sample_concat).squeeze() 
        best_sample_idx = sample_predictions.argmax(dim=1)
        return sample_concat[torch.arange(xf_r.shape[0]),best_sample_idx,:]
    

    def _sample(self, xf, yf, y, sigma=0.0, temp=1.0, n_samples=1):

        self.eval()
        yf = yf.view(-1, 1)
        y = y.view(-1, 1)
        memory = self._encode(xf, yf)  # encode once
        batch_size = xf.shape[0]

        all_hist = []

        for _ in range(n_samples):
            hist = torch.zeros((batch_size, 0), device=xf.device)
            for i in range(self.num_inputs):
                mask = self.generate_square_subsequent_mask(i + 1)
                logits = temp * self._decode(hist, mask, y, memory)  # unnormalized log probability
                probs = logits.softmax(dim=-1)

                sampled_idx = torch.multinomial(probs, 1)  # get bin index
                bin_val = self.bins[sampled_idx]            # get bin value
                samp_val = bin_val + sigma * torch.randn_like(bin_val)  # add noise

                hist = torch.cat((hist, samp_val), dim=1)
            all_hist.append(hist)

        # stack all samples into one tensor → (batch_size, n_samples, num_inputs)
        hist = torch.stack(all_hist, dim=1)

        return hist

    def old_sample(self,xf,yf,y,sigma = 0.0, temp=1.0):

        self.eval()
        yf = yf.view(-1,1)
        y = y.view(-1,1)
        memory = self._encode(xf, yf) # do this once
        batch_size = xf.shape[0]
        hist = torch.zeros((batch_size,0), device=xf.device) # initialize empty features, will be concatenated with y
        probs_hist = torch.zeros((batch_size,0, self.n_bins), device=xf.device) # initialize empty features, will be concatenated with y

        for i in range(self.num_inputs):
            mask = self.generate_square_subsequent_mask(i+1)
            logits = temp*self._decode(hist, mask, y, memory)  # gives unnormalised log probability
            probs = logits.softmax(dim=-1) # get probability
            
            sampled_idx = torch.multinomial(probs,1) # get bin index
            bin_val = self.bins[sampled_idx] # get bin value for each batch element
            samp_val = bin_val + sigma*torch.randn_like(bin_val) # add sizzle to bin value
            hist = torch.cat((hist, samp_val), dim=1)
            probs_hist = torch.cat((probs_hist, probs.unsqueeze(1)), dim=1)
        return hist
    