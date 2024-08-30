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
import datetime
from tqdm import tqdm

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
        self.ctx1 = nn.Parameter(ctx_vectors.clone().detach())
        self.ctx2 = nn.Parameter(ctx_vectors.clone().detach())

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

    def forward(self, domain):
        if(domain == 0):
            ctx = self.ctx
        if(domain == 1):
            ctx = self.ctx1
        if(domain == 2):
            ctx = self.ctx2
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


# Feature selection part
class FeatureSelector(nn.Module):
    def __init__(self, input_dim, sigma):
        super(FeatureSelector, self).__init__()
        self.mu = torch.nn.Parameter(0.00 * torch.randn(input_dim, ), requires_grad=True)
        self.noise = torch.randn(self.mu.size())
        self.sigma = sigma
        self.input_dim = input_dim

    def renew(self):
        self.mu = torch.nn.Parameter(0.00 * torch.randn(self.input_dim, ), requires_grad=True)
        self.noise = torch.randn(self.mu.size())

    def forward(self, prev_x):
        z = self.mu + self.sigma * self.noise.normal_() * self.training
        stochastic_gate = self.hard_sigmoid(z)
        new_x = prev_x * stochastic_gate
        return new_x

    def hard_sigmoid(self, x):
        return torch.clamp(x + 0.5, 0.0, 1.0)

    def regularizer(self, x):
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    def _apply(self, fn):
        super(FeatureSelector, self)._apply(fn)
        self.noise = fn(self.noise)
        return self

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        input_dim = clip_model.visual.output_dim
        self.feature_selector = FeatureSelector(input_dim, sigma=1.0)

    def forward(self, image, domain=3):
        image_features = self.image_encoder(image)
        tokenized_prompts = self.tokenized_prompts

        if(domain < 3):
            prompts = self.prompt_learner(domain)
        if(domain == 3):
            prompts = self.prompt_learner(domain=0)
            prompts1 = self.prompt_learner(domain=1)
            prompts2 = self.prompt_learner(domain=2)
  
        text_features = self.text_encoder(prompts, tokenized_prompts)
        if(domain == 3):      
            text_features1 = self.text_encoder(prompts1, tokenized_prompts)
            text_features2 = self.text_encoder(prompts2, tokenized_prompts)

        # mark
        if(domain == 3):
            image_features = self.feature_selector(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        if(domain == 3):
            text_features1 = text_features1 / text_features1.norm(dim=-1, keepdim=True)
            text_features2 = text_features2 / text_features2.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        if(domain == 3):
            logits1 = logit_scale * image_features @ text_features1.t()
            logits2 = logit_scale * image_features @ text_features2.t()

        if(domain < 3):
            return logits
        if(domain == 3):
            return logits, logits1, logits2

   
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
        losses1 = MetricMeter()
        losses2 = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Decide to iterate over labeled or unlabeled dataset
        len_train_loader = len(self.train_loader)
        len_train_loader1 = len(self.train_loader1)
        len_train_loader2 = len(self.train_loader2)
        self.num_batches = max(len_train_loader, len_train_loader1, len_train_loader2)

        train_loader_iter = iter(self.train_loader)
        train_loader_iter1 = iter(self.train_loader1)
        train_loader_iter2 = iter(self.train_loader2)

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            try:
                batch_x = next(train_loader_iter)
            except StopIteration:
                train_loader_iter = iter(self.train_loader)
                batch_x = next(train_loader_iter)
            
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            try:
                batch_x1 = next(train_loader_iter1)
            except StopIteration:
                train_loader_iter1 = iter(self.train_loader1)
                batch_x1 = next(train_loader_iter1)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x1)
            batch_time.update(time.time() - end)
            losses1.update(loss_summary)

            try:
                batch_x2 = next(train_loader_iter2)
            except StopIteration:
                train_loader_iter2 = iter(self.train_loader2)
                batch_x2 = next(train_loader_iter2)

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch_x2)
            batch_time.update(time.time() - end)
            losses2.update(loss_summary)

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

    #def __init__(self, cfg):
        #super().__init__(cfg)
        #self.pretrain(cfg)
        
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
            output = self.model(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
        
        
    def pretrain(self, cfg, epochs=5):
        data_source = self.dm.dataset.train_x
        self.domain_dataset = self.split_dataset_by_domain(data_source)
        cfg.DATALOADER.NUM_WORKERS = 0
        self.train_loader = build_data_loader(cfg=cfg,batch_size=64,tfm=transforms.Compose([transforms.Resize([224, 224]),transforms.ToTensor()]),data_source=self.domain_dataset[0])
        self.train_loader1 = build_data_loader(cfg=cfg,batch_size=64,tfm=transforms.Compose([transforms.Resize([224, 224]),transforms.ToTensor()]),data_source=self.domain_dataset[1])
        self.train_loader2 = build_data_loader(cfg=cfg,batch_size=64,tfm=transforms.Compose([transforms.Resize([224, 224]),transforms.ToTensor()]),data_source=self.domain_dataset[2])
        pre_optimizer = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        pre_sched = build_lr_scheduler(pre_optimizer, cfg.OPTIM)
        pre_optimizer1 = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        pre_sched1 = build_lr_scheduler(pre_optimizer1, cfg.OPTIM)
        pre_optimizer2 = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        pre_sched2 = build_lr_scheduler(pre_optimizer2, cfg.OPTIM)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        self.model.feature_selector.requires_grad_(False)
        #self.num_batches = 5
        # pass 1
        self.model.prompt_learner.requires_grad_(True)
        #self.model.prompt_learner1.requires_grad_(False)
        #self.model.prompt_learner2.requires_grad_(False)
        #train_loader_x_iter = iter(self.train_loader_x)
        #b = next(train_loader_x_iter)
        for _ in range(epochs):
            for batch_idx, batch in enumerate(self.train_loader):
                X = batch["img"].to(self.device)
                y = batch["label"].to(self.device)
                pre_optimizer.zero_grad()
                logits= self.model.forward(X, domain=0)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                pre_optimizer.step()
        
        for _ in range(epochs):
            for batch_idx, batch in enumerate(self.train_loader1):
                X = batch["img"].to(self.device)
                y = batch["label"].to(self.device)
                pre_optimizer1.zero_grad()
                logits = self.model.forward(X, domain=1)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                pre_optimizer1.step()

        for _ in range(epochs):
            for batch_idx, batch in enumerate(self.train_loader2):
                X = batch["img"].to(self.device)
                y = batch["label"].to(self.device)
                pre_optimizer2.zero_grad()
                logits = self.model.forward(X, domain=2)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                pre_optimizer2.step()

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

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
            if "feature_selector" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.feature_selector, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("CustomCLIP", self.model, self.optim, self.sched)

        #self.optim1 = build_optimizer(self.model.feature_selector, cfg.OPTIM)
        #self.sched1 = build_lr_scheduler(self.optim1, cfg.OPTIM)
        #self.register_model("CustomCLIP1", self.model, self.optim1, self.sched1)

        #self.optim2 = build_optimizer(self.model.feature_selector, cfg.OPTIM)
        #self.sched2 = build_lr_scheduler(self.optim2, cfg.OPTIM)
        #self.register_model("CustomCLIP2", self.model, self.optim2, self.sched2)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def before_epoch(self):
        super().before_epoch()
        for name, param in self.model.named_parameters():
            if "feature_selector" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

    def after_epoch(self):
        super().after_epoch()
        print(f"{self.model.feature_selector.mu}")

    def forward_backward(self, batch):
        image = batch["img"].to(self.device)
        #image1 = batch1["img"].to(self.device)
        #image2 = batch2["img"].to(self.device)
        label = batch["label"].to(self.device)
        #label1 = batch1["label"].to(self.device)
        #label2 = batch2["label"].to(self.device)

        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output, output1, output2 = self.model(image, domain = 3)
                loss = F.cross_entropy(output, label) + F.cross_entropy(output1, label) + F.cross_entropy(output2, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output, output1, output2 = self.model(image, domain = 3)
            loss = F.cross_entropy(output, label) + F.cross_entropy(output1, label) + F.cross_entropy(output2, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
            "acc1": compute_accuracy(output1, label)[0].item(),
            "acc2": compute_accuracy(output2, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        #input = batch[0]
        label = batch["label"]
        #label = batch[1]
        #domain = batch["domain"]
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