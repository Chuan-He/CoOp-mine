import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.data_manager import build_data_loader

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import math
from collections import defaultdict
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
from dassl.utils import read_image
from dassl.data import DataManager
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
import time
import numpy as np
import datetime
from tqdm import tqdm
import logging
from torch.autograd import grad

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

# Feature selection part

class FeatureSelector(nn.Module):
    def __init__(self, input_dim, sigma):
        super(FeatureSelector, self).__init__()
        self.mu = torch.nn.Parameter(torch.randn(input_dim, ), requires_grad=True)
        #self.noise = torch.randn(self.mu.size())
        self.sigma = sigma
        self.input_dim = input_dim

    def renew(self):
        self.mu = torch.nn.Parameter(0.00 * torch.randn(self.input_dim, ), requires_grad=True)
        #self.noise = torch.randn(self.mu.size())

    def forward(self, prev_x):
        #z = self.mu + self.sigma * self.noise.normal_() * self.training
        z = self.mu
        stochastic_gate, n_gate = self.hard_sigmoid(z)
        new_x = prev_x * stochastic_gate
        neg_x = prev_x * n_gate
        return new_x, neg_x

    def hard_sigmoid(self, x):
        return torch.clamp(x + 0.5, 0.0, 1.0),torch.clamp(0.5 - x, 0.0, 1.0)

    def regularizer(self, x):
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    def _apply(self, fn):
        super(FeatureSelector, self)._apply(fn)
        #self.noise = fn(self.noise)
        return self


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

class GradientReversalLayer(torch.autograd.Function):
    """
    Implement the gradient reversal layer for the convenience of domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """
    @staticmethod
    def forward(self, inputs):
        return inputs

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        grad_input = -grad_input
        return grad_input

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.input_dim = 512
        self.feature_selector = FeatureSelector(input_dim=self.input_dim, sigma=0.1)
        #self.mu = torch.nn.Parameter(torch.randn(self.input_dim, ), requires_grad=True)
        #self.noise = torch.randn(self.mu.size())

        self.num_hidden_layers = 3
        self.num_neurons = [self.input_dim] + [1000,500,100]
        self.num_domains = 3
        # Parameters of hidden, fully-connected layers, feature learning component.
        self.hiddens = nn.ModuleList([nn.Linear(self.num_neurons[i], self.num_neurons[i+1])
                                      for i in range(self.num_hidden_layers)])
        # Parameter of the final softmax classification layer.
        # self.softmax = nn.Linear(self.num_neurons[-1], 3)
        # Parameter of the domain classification layer, multiple sources single target domain adaptation.
        # self.domains = nn.ModuleList([nn.Linear(self.num_neurons[-1], 2) for _ in range(self.num_domains)])
        self.domain_classi = nn.Linear(self.num_neurons[-1], 2)
        # Gradient reversal layer.
        self.grls = [GradientReversalLayer() for _ in range(self.num_domains)]
    '''
    def feature_select(self,x):
        #self.stochastic_gate = torch.round(torch.clamp(self.mu + 0.5, 0.0, 1.0))
        self.stochastic_gate = torch.clamp(self.mu + 0.5, 0.0, 1.0)
        new_x = x * self.stochastic_gate
        return new_x
    '''
 
    def forward(self, xs):
        
        sinputs = xs

        image_features, p_image_features, n_image_features= [], [], []

        for i in range(self.num_domains):
            image_features.append(self.image_encoder(sinputs[i]))
        for i in range(self.num_domains):
            pf, nf = self.feature_selector(image_features[i])
            p_image_features.append(pf)
            n_image_features.append(nf)

        tokenized_prompts = self.tokenized_prompts
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)

        for i in range(self.num_domains):
            p_image_features[i] = p_image_features[i] / p_image_features[i].norm(dim=-1, keepdim=True)
            n_image_features[i] = n_image_features[i] / n_image_features[i].norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits, nlogits = [], []
        for i in range(self.num_domains):
            logits.append(logit_scale * p_image_features[i] @ text_features.t())
            nlogits.append(logit_scale * self.grls[i].apply(n_image_features[i]) @ text_features.t())
            #nlogits.append(logit_scale * n_image_features[i] @ text_features.t())

        p_relus, n_relus = [],[]
        for i in range(self.num_domains):
            p_relu = p_image_features[i]
            n_relu = n_image_features[i]
            for hidden in self.hiddens:
                p_relu = F.relu(hidden(p_relu))
                n_relu = F.relu(hidden(n_relu))
            p_relus.append(p_relu)
            n_relus.append(n_relu)

        # Classification probabilities on k source domains.
        #logprobs = []
        #for i in range(self.num_domains):
            #logprobs.append(F.log_softmax(self.softmax(sh_relu[i]), dim=1))
        # Domain classification accuracies.
        in_domains, out_domains = [],[]
        for i in range(self.num_domains):
            d1 = F.log_softmax(self.domain_classi(p_relus[i]), dim=1)
            #y = y.view(sh_relu[i].shape[0],-1)
            d2 = F.log_softmax(self.domain_classi(n_relus[i]), dim=1)
            in_domains.append(d1)
            out_domains.append(d2)
        '''
        pull_domains, push_domains = [], []
        for i in range(self.num_domains):
            for j in range(self.num_domains):
                if(j!=i):
                    pull_domains.append(F.log_softmax(self.domains[i](p_relus[j]), dim=1))
                
        for i in range(self.num_domains):
            for j in range(self.num_domains):
                if(j!=i):
                    #pull_domains.append(F.log_softmax(self.domains[i](p_relus[j]), dim=1))
                    push_domains.append(F.log_softmax(self.domains[i](n_relus[j]), dim=1))
        '''
        return logits, nlogits, in_domains, out_domains
    
    @torch.no_grad()
    def predict(self, x):
        tokenized_prompts = self.tokenized_prompts
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        image_feature,_ = self.feature_selector(self.image_encoder(x))
        
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_feature= image_feature / image_feature.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_feature @ text_features.t()
        return logits
    
    '''
    def model_inference(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits
    '''
@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    
    def build_data_loader(self):
        dm = DataManager(self.cfg)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm
    """

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loaders, train_loader_iters = [], []
        for i in range(self.num_domains):
            len_train_loaders.append(len(self.train_loaders[i]))
            train_loader_iters.append(iter(self.train_loaders[i]))

        self.num_batches = max(len_train_loaders)

        #val_loader = self.val_loader
        #val_loader_iter = iter(val_loader)

        end = time.time()

        #batch_size = self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        
        for self.batch_idx in range(self.num_batches):
            batches = []
            for i in range(self.num_domains):
                try:
                    batches.append(next(train_loader_iters[i]))
                except StopIteration:
                    train_loader_iters[i] = iter(self.train_loaders[i])
                    batches.append(next(train_loader_iters[i]))
            '''
            try:
                val_batch = next(val_loader_iter)
            except StopIteration:
                val_loader_iter = iter(val_loader)
                val_batch = next(val_loader_iter)
            '''              
            data_time.update(time.time() - end)

            loss_summary = self.forward_backward(batches)

            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

    def split_dataset_by_domain(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into domain-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.domain].append(item)

        return output

    def get_logger(filename):
        # Logging configuration: set the basic configuration of the logging system
        log_formatter = logging.Formatter(fmt='%(asctime)s [%(processName)s, %(process)s] [%(levelname)-5.5s]  %(message)s',
                                      datefmt='%m-%d %H:%M')
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        # File logger
        file_handler = logging.FileHandler("{}.log".format(filename))
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        # Stderr logger
        std_handler = logging.StreamHandler(sys.stdout)
        std_handler.setFormatter(log_formatter)
        std_handler.setLevel(logging.DEBUG)
        logger.addHandler(std_handler)
        return logger
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self.preload(cfg)

    def preload(self, cfg):
        data_source = self.dm.dataset.train_x
        #data_target = self.dm.dataset.train_u
        self.domain_dataset = self.split_dataset_by_domain(data_source)
        self.num_domains = 3
        cfg.DATALOADER.NUM_WORKERS = 0
        self.batch_size = self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        
        self.train_loaders = []
        for i in range(self.num_domains):
            self.train_loaders.append(build_data_loader(cfg=cfg,batch_size=self.batch_size,tfm=transforms.Compose([transforms.Resize([224, 224]),transforms.ToTensor()]),data_source=self.domain_dataset[i]))
        #self.train_loaders.append(build_data_loader(cfg=cfg,batch_size=self.batch_size,tfm=transforms.Compose([transforms.Resize([224, 224]),transforms.ToTensor()]),data_source=data_target))
        #self.train_loader1 = build_data_loader(cfg=cfg,batch_size=batch_size,tfm=transforms.Compose([transforms.Resize([224, 224]),transforms.ToTensor()]),data_source=self.domain_dataset[1])
        #self.train_loader2 = build_data_loader(cfg=cfg,batch_size=batch_size,tfm=transforms.Compose([transforms.Resize([224, 224]),transforms.ToTensor()]),data_source=self.domain_dataset[2])

        #self.model.requires_grad_(True)
        #self.model.prompt_learner.requires_grad_(True)
    '''
    def before_epoch(self):
        super().before_epoch()

        if(self.epoch%2 == 1):
            self.model.feature_selector.requires_grad_(True)
            self.model.domains.requires_grad_(False)
            self.model.prompt_learner.requires_grad_(True)
            self._optims["CustomCLIP"] = self.optim
            self._scheds["CustomCLIP"] = self.sched
        else:
            self.model.feature_selector.requires_grad_(True)
            self.model.domains.requires_grad_(True)
            self.model.prompt_learner.requires_grad_(False)
            self._optims["CustomCLIP"] = self.optim1
            self._scheds["CustomCLIP"] = self.sched1
    '''
    def before_epoch(self):
        super().before_epoch()
        print(f"{self.model.feature_selector.mu}")
        #print(f"{self.model.feature_selector.stochastic_gate}")

    def forward_backward(self, batches):
        device = "cuda" 
        #gamma = 10.0
        #coeff0 = 0.01
        #coeff1 = 1
        #coeff2 = 1
        #mode = "else"
        b = self.batch_size
        n = self.num_domains
        xs = []
        ys = []
        for i in range(self.num_domains):
            xs.append(batches[i]['img'].to(device))
            ys.append(batches[i]['label'].to(device))
        #xt = val_batch['img'].to(device)
        #yt = val_batch['label'].to(device)

        # Training phase.
        
        p_labels = torch.ones(b, requires_grad=False).type(torch.LongTensor).to(device)
        #zerolabels = torch.zeros(b, requires_grad=False).type(torch.LongTensor).to(device)
        #pull_labels = torch.ones(b, requires_grad=False).type(torch.LongTensor).to(device)
        n_labels = torch.zeros(b, requires_grad=False).type(torch.LongTensor).to(device)
        #for j in range(self.num_domains):
            #xs[j] = torch.tensor(xs[j].clone().detach(), requires_grad=False).to(device)
            #ys[j] = torch.tensor(ys[j].clone().detach(), requires_grad=False).to(device)
            #tinputs = torch.tensor(tinputs, requires_grad=False).to(device)
        self._optims["CustomCLIP"].zero_grad()

        logits, nlogits, in_domains, out_domains = self.model(xs)

        domain_loss1 = 0
        for i in range(n):
            domain_loss1 += F.nll_loss(in_domains[i], p_labels)

        domain_loss2 = 0
        for i in range(n):
            domain_loss2 += F.nll_loss(out_domains[i], n_labels)

        '''
        for j in range(self.num_domains):
            losses = torch.stack([F.cross_entropy(logits[j], ys[j].to(device)) for j in range(self.num_domains)])
            #omain_losses = torch.stack([F.nll_loss(sdomains[j], slabels) for j in range(self.num_domains)])
            domain_losses = torch.stack([F.nll_loss(sdomains[j], slabels) +
                        F.nll_loss(nsdomains[j][0], nslabels) + F.nll_loss(nsdomains[j][1], nslabels) for j in range(self.num_domains)])
            # Different final loss function depending on different training modes.
            if mode == "maxmin":
                loss = torch.max(losses) + mu * torch.min(domain_losses)
            elif mode == "dynamic":
                loss = torch.log(torch.sum(torch.exp(gamma * (losses + mu * domain_losses)))) / gamma
            else:
                #raise ValueError("No support for the training mode on madnNet: {}.".format(mode))
                loss = torch.min(losses + mu * domain_losses)
        '''
        class_loss1, class_loss2 = 0, 0
        for j in range(n):
            class_loss1 += F.cross_entropy(logits[j], ys[j])
            class_loss2 += F.cross_entropy(nlogits[j], ys[j])
            #domain_losses = torch.stack([F.nll_loss(sdomains[j], slabels) for j in range(self.num_domains)])
            #domain_losses = torch.stack([F.nll_loss(sdomains[j], slabels) +
                        #F.nll_loss(nsdomains[j][0], nslabels) + F.nll_loss(nsdomains[j][1], nslabels) for j in range(self.num_domains)])

        #losses_avg = losses / n
        losses = class_loss1  + 0.01 * class_loss2 + 0.01 * (domain_loss1 + domain_loss2)
        
        #feature_selector = self.model.feature_selector
        # regulizer for feature selector
        #reg = torch.sum(feature_selector.regularizer((feature_selector.mu + 0.5) / feature_selector.sigma))
        #reg = (reg-1.0)**2
        #gradient = grad(losses, self.model.feature_selector.parameters(), create_graph=True)[0].reshape(-1)
        #penalty = torch.tensor(np.zeros(self.model.feature_selector.input_dim, dtype=np.float32))
        #penalty = torch.pow((gradient.norm(2, dim=0) - 1), 2).mean()
        '''
        mu = self.model.feature_selector.mu
        sigma = self.model.feature_selector.sigma
        
        penalty_detach = torch.sum(penalty.reshape(mu.shape)*(mu+0.5))
        reg = torch.sum(self.reg((mu + 0.5) / sigma))
        reg = (reg-1.0)**2
        total_loss = losses_avg + self.alpha * (penalty_detach)
        total_loss = total_loss + self.lam * reg
        '''
        #losses += 0.00001 * reg

        losses.backward()
        self._optims["CustomCLIP"].step()

        acc = 0
        for j in range(n):
            acc += compute_accuracy(logits[j], ys[j])[0].item()
        acc /= n
                
        loss_summary = {
            "loss": losses.item(),
            "acc": acc,
        }

        return loss_summary
    

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "logit_scale" in name:
                param.requires_grad_(False)
            if "image_encoder" in name:
                param.requires_grad_(False)
            if "text_encoder" in name:
                param.requires_grad_(False)
        
        self.model.feature_selector.requires_grad_(True)
        self.model.domain_classi.requires_grad_(True)
        self.model.prompt_learner.requires_grad_(True)
        #if cfg.MODEL.INIT_WEIGHTS:
            #load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        #cfg.OPTIM.WARMUP_CONS_LR = 2e-3
        #cfg.OPTIM.LR = 0.02
        #self.optim1 = build_optimizer(self.model, cfg.OPTIM)
        #self.sched1 = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("CustomCLIP", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)


    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]


    '''
    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model_predict(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
    '''
    @torch.no_grad()
    def model_inference(self, x):
        logits = self.model.predict(x)
        return logits