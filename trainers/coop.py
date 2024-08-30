import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX, TrainerXU
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint, count_num_param
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.modeling import build_head
from dassl.modeling.ops import ReverseGrad

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from datasets.domainbed import *
from utils.Logger import Logger
from utils.tools import *
from optimizer.optimizer_helper import get_optim_and_scheduler

_tokenizer = _Tokenizer()

class Masker(nn.Module):
    def __init__(self, in_dim=2048, num_classes=2048, middle =8192, k = 1024):
        super(Masker, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.k = k

        self.layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_dim, middle),
            nn.BatchNorm1d(middle, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(middle, middle),
            nn.BatchNorm1d(middle, affine=True),
            nn.ReLU(inplace=True),
            nn.Linear(middle, num_classes))

        self.bn = nn.BatchNorm1d(num_classes, affine=False)

    def forward(self, f):
       mask = self.bn(self.layers(f))
       z = torch.zeros_like(mask)
       for _ in range(self.k):
           mask = F.gumbel_softmax(mask, dim=1, tau=0.5, hard=False)
           z = torch.maximum(mask,z)
       return z


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


class ClassifierCLIP(nn.Module):
    def __init__(self, text_features, clip_model):
        super().__init__()
        #self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        #dim = clip_model.visual.output_dim
        #self.dtype = clip_model.dtype
        
        #self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        #self.image_encoder = clip_model.visual
        self.text_features = text_features
        self.logit_scale = clip_model.logit_scale

        
    def forward(self, features):
        image_features = features

        #prompts = self.prompt_learner()
        #tokenized_prompts = self.tokenized_prompts
        #text_features = self.text_encoder(prompts, tokenized_prompts)

        #neg_ifeatures = image_features - pos_ifeatures
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # norm_neg_ifeatures = neg_ifeatures / neg_ifeatures.norm(dim=-1, keepdim=True)
        text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()     
        # logits1 = logit_scale * norm_neg_ifeatures @ norm_text_features.t()

        return logits

@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    
    '''
    def __init__(self, cfg, clip_model):
        super().__init__(cfg)
        
        dim = clip_model.visual.output_dim
        self.dtype = clip_model.dtype
        self.logit_scale = clip_model.logit_scale

        self.masker = Masker(in_dim=dim, num_classes = dim,middle = 4*dim,k=1228).to("cuda")
        self.text_encoder = TextEncoder(clip_model)

        classnames = self.dm.dataset.classnames
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        #self.encoder = clip_model.visual
        self.image_encoder = clip_model.visual
        with torch.no_grad():
            self.text_encoder = TextEncoder(clip_model)
            tokenized_prompts = self.tokenized_prompts
            prompts = self.prompt_learner()
            text_features = self.text_encoder(prompts, tokenized_prompts)

        self.classifier = ClassifierCLIP(cfg, text_features, clip_model)
        self.classifier_ad = ClassifierCLIP(cfg, text_features, clip_model)
        #self.encoder.eval()
        for name, param in self.text_encoder.named_parameters():
            param.requires_grad_(False)
        self.classifier.eval()
        self.masker.eval()
        self.classifier_ad.eval()
    '''
       
    def run_epoch(self):
        criterion = nn.CrossEntropyLoss()

        # turn on train mode
        for name, param in self.image_encoder.named_parameters():
            param.requires_grad_(True)
        self.classifier.train()
        self.classifier_ad.train()
        self.masker.train()

        for it, (batch, label, domain) in enumerate(self.train_loader):

            # preprocessing
            batch = torch.cat(batch, dim=0).to(self.device)
            labels = torch.cat(label, dim=0).to(self.device)
            if self.args.target in pacs_dataset:
                labels -= 1
            # zero grad
            self.encoder_optim.zero_grad()
            self.classifier_optim.zero_grad()
            self.classifier_ad_optim.zero_grad()
            self.masker_optim.zero_grad()

            # forward
            loss_dict = {}
            correct_dict = {}
            num_samples_dict = {}
            total_loss = 0.0

            ## --------------------------step 1 : update G and C -----------------------------------
            features = self.image_encoder(batch)
            masks_sup = self.masker(features.detach())
            masks_inf = torch.ones_like(masks_sup) - masks_sup
            self.current_epoch = self.epoch
            if self.current_epoch <= 5:
                masks_sup = torch.ones_like(features.detach())
                masks_inf = torch.ones_like(features.detach())
            features_sup = features * masks_sup
            features_inf = features * masks_inf
            scores_sup = self.classifier(features_sup)
            scores_inf = self.classifier_ad(features_inf)

            assert batch.size(0) % 2 == 0
            split_idx = int(batch.size(0) / 2)
            features_ori, features_aug = torch.split(features, split_idx)
            assert features_ori.size(0) == features_aug.size(0)

            # classification loss for sup feature
            loss_cls_sup = criterion(scores_sup, labels)
            loss_dict["sup"] = loss_cls_sup.item()
            correct_dict["sup"] = calculate_correct(scores_sup, labels)
            num_samples_dict["sup"] = int(scores_sup.size(0))

            # classification loss for inf feature
            loss_cls_inf = criterion(scores_inf, labels)
            loss_dict["inf"] = loss_cls_inf.item()
            correct_dict["inf"] = calculate_correct(scores_inf, labels)
            num_samples_dict["inf"] = int(scores_inf.size(0))

            # factorization loss for features between ori and aug
            loss_fac = factorization_loss(features_ori,features_aug)
            loss_dict["fac"] = loss_fac.item()

            # get consistency weight
            const_weight = get_current_consistency_weight(epoch=self.current_epoch,
                                                          weight=self.config["lam_const"],
                                                          rampup_length=self.config["warmup_epoch"],
                                                          rampup_type=self.config["warmup_type"])

            # calculate total loss
            total_loss = 0.5*loss_cls_sup + 0.5*loss_cls_inf + const_weight*loss_fac
            loss_dict["total"] = total_loss.item()

            # backward
            total_loss.backward()

            # update
            self.encoder_optim.step()
            self.classifier_optim.step()
            self.classifier_ad_optim.step()


            ## ---------------------------------- step2: update masker------------------------------
            self.masker_optim.zero_grad()
            features = self.image_encoder(batch)
            masks_sup = self.masker(features.detach())
            masks_inf = torch.ones_like(masks_sup) - masks_sup
            features_sup = features * masks_sup
            features_inf = features * masks_inf
            scores_sup = self.classifier(features_sup)
            scores_inf = self.classifier_ad(features_inf)

            loss_cls_sup = criterion(scores_sup, labels)
            loss_cls_inf = criterion(scores_inf, labels)
            total_loss = 0.5*loss_cls_sup - 0.5*loss_cls_inf
            total_loss.backward()
            self.masker_optim.step()

            self.global_step += 1

            # record
            self.logger.log(
                it=it,
                iters=len(self.train_loader),
                losses=loss_dict,
                samples_right=correct_dict,
                total_samples=num_samples_dict
            )

        # turn on eval mode
        for name, param in self.image_encoder.named_parameters():
            param.requires_grad_(False)
        self.classifier.eval()
        self.masker.eval()
        self.classifier_ad.eval()

        # evaluation
        with torch.no_grad():
            for phase, loader in self.eval_loader.items():
                total = len(loader.dataset)
                class_correct = self.do_eval(loader)
                class_acc = float(class_correct) / total
                self.logger.log_test(phase, {'class': class_acc})
                self.results[phase][self.current_epoch] = class_acc

            # save from best model
            if self.results['test'][self.current_epoch] >= self.best_acc:
                self.best_acc = self.results['test'][self.current_epoch]
                self.best_epoch = self.current_epoch + 1
                self.logger.save_best_model(self.encoder, self.classifier, self.best_acc)
    

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def preload(self, args, config):
        self.config = config
        self.args = args
        self.global_step = 0
        # dataloaders
        self.train_loader = get_fourier_train_dataloader(args=self.args, config=self.config)
        self.val_loader = get_val_dataloader(args=self.args, config=self.config)
        self.test_loader = get_test_loader(args=self.args, config=self.config)
        self.eval_loader = {'val': self.val_loader, 'test': self.test_loader}

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        dim = clip_model.visual.output_dim
        self.dtype = clip_model.dtype
        self.logit_scale = clip_model.logit_scale

        self.masker = Masker(in_dim=dim, num_classes = dim,middle = 4*dim,k=1228).to("cuda")
        self.text_encoder = TextEncoder(clip_model).to("cuda")

        self.prompt_learner = PromptLearner(cfg, classnames, clip_model).to("cuda")
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        #self.encoder = clip_model.visual
        self.image_encoder = clip_model.visual.to("cuda")
        with torch.no_grad():
            self.text_encoder = TextEncoder(clip_model)
            tokenized_prompts = self.tokenized_prompts
            prompts = self.prompt_learner()
            text_features = self.text_encoder(prompts, tokenized_prompts)

        self.classifier = ClassifierCLIP(text_features.detach(), clip_model).to("cuda")
        self.classifier_ad = ClassifierCLIP(text_features.detach(), clip_model).to("cuda")
        
        #if cfg.MODEL.INIT_WEIGHTS:
            #load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        # NOTE: only give prompt_learner to the optimizer
        #self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        #self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        #self.register_model("ClassifierCLIP", self.model, self.optim, self.sched)
    
        #self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in 
        self.register_model("CustomCLIP")
        #self.encoder.eval()
        for name, param in self.text_encoder.named_parameters():
            param.requires_grad_(False)
        self.classifier.eval()
        self.masker.eval()
        self.classifier_ad.eval()

        config = __import__("ResNet50", fromlist=[""]).config
        self.config = config
        # optimizers
        self.encoder_optim, self.encoder_sched = \
            get_optim_and_scheduler(self.image_encoder, self.config["optimizer"]["encoder_optimizer"])
        self.classifier_optim, self.classifier_sched = \
            get_optim_and_scheduler(self.classifier, self.config["optimizer"]["classifier_optimizer"])
        self.classifier_ad_optim, self.classifier_ad_sched = \
            get_optim_and_scheduler(self.classifier_ad, self.config["optimizer"]["classifier_optimizer"])
        self.masker_optim, self.masker_sched = \
            get_optim_and_scheduler(self.masker, self.config["optimizer"]["classifier_optimizer"])
      
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            #self.model = nn.DataParallel(self.model)

    #def before_epoch(self):

    def train(self):
        self.logger = Logger(self.args, self.config, update_frequency=30)
        self.logger.save_config()

        self.epochs = self.config["epoch"]
        self.results = {"val": torch.zeros(self.epochs), "test": torch.zeros(self.epochs)}

        self.best_acc = 0
        self.best_epoch = 0

        for self.current_epoch in range(self.epochs):

            # step schedulers
            self.encoder_sched.step()
            self.classifier_sched.step()

            self.logger.new_epoch([group["lr"] for group in self.encoder_optim.param_groups])
            self.run_epoch()
            self.logger.finish_epoch()

        # save from best model
        val_res = self.results['val']
        test_res = self.results['test']
        self.logger.save_best_acc(val_res, test_res, self.best_acc, self.best_epoch - 1)

        return self.logger