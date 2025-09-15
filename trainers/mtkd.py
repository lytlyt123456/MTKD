import math
import os.path as osp
import types

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

import clip.model as m
from clip.model import CLIP
from clip import clip
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ProjectLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=output_dim,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=output_dim, out_channels=output_dim,
                               kernel_size=1, bias=False)
    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.conv2(self.relu(self.bn1(self.conv1(x))))
        x = x.squeeze(-1).squeeze(-1)
        return x


def load_clip_to_cpu_teacher(cfg):
    backbone_name = cfg.TRAINER.MTKD.TEACHER_NAME

    model_path = ''
    if backbone_name == 'ViT-L/14':
        model_path = './clip/ViT-L-14.pt'
    elif backbone_name == 'ViT-B/16':
        model_path = './clip/ViT-B-16.pt'
    elif backbone_name == 'ViT-B/32':
        model_path = './clip/ViT-B-32.pt'
    else:
        print(f'{backbone_name} is a wrong name of CLIP model.')

    try:
        model0 = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    except RuntimeError:
        model0 = None
        state_dict = torch.load(model_path, map_location='cpu')

    design_details = {
        'prompt_length' : 4,
        'prompt_depth' : 9,
        'trainers' : 'IVLP'
    }

    model = m.build_model(state_dict or model0.state_dict(), design_details)
    return model


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME

    model_path = ''
    if backbone_name == 'ViT-L/14':
        model_path = './clip/ViT-L-14.pt'
    elif backbone_name == 'ViT-B/16':
        model_path = './clip/ViT-B-16.pt'
    elif backbone_name == 'ViT-B/32':
        model_path = './clip/ViT-B-32.pt'
    else:
        print(f'{backbone_name} is a wrong name of CLIP model.')

    try:
        model0 = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    except RuntimeError:
        model0 = None
        state_dict = torch.load(model_path, map_location='cpu')

    design_details = {
        'prompt_length' : cfg.TRAINER.MTKD.PROMPT_LENGTH,
        'prompt_depth' : cfg.TRAINER.MTKD.PROMPT_DEPTH,
        'trainers' : 'IVLP'
    }

    model = m.build_model(state_dict or model0.state_dict(), design_details)
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model: CLIP):
        super().__init__()
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, embedding_texts_with_prompt, tokenized_texts):
        x = embedding_texts_with_prompt.to(device=device, dtype=self.dtype)
        x = x + self.positional_embedding[None, :, :].type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = self.ln_final(x)
        x = x.permute(1, 0, 2)
        x = x[torch.arange(0, x.shape[0]), tokenized_texts.argmax(dim=-1)] @ self.text_projection.type(self.dtype)
        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, class_names, clip_model : CLIP):
        super().__init__()

        ctx_init = cfg.TRAINER.MTKD.CTX_INIT # 'a photo of a'
        ctx_init = ctx_init.replace('_', ' ')
        ctx_length = cfg.TRAINER.MTKD.PROMPT_LENGTH
        tokenized_ctx_init = clip.tokenize(ctx_init) # [<SOS>, a, photo, of, a, <EOS>]
        with torch.no_grad():
            embedding_ctx_init = clip_model.token_embedding(tokenized_ctx_init)
        self.ctx = nn.Parameter(embedding_ctx_init[0, 1 : 1 + ctx_length, :]) # [4, word_dim]
        texts = [ctx_init + ' ' + name.replace('_', ' ') + '.' for name in class_names]
        self.tokenized_prompts = clip.tokenize(texts) # [B, N]
        with torch.no_grad():
            embedding_texts = clip_model.token_embedding(self.tokenized_prompts) # [B, N, C]
        self.register_buffer('token_prefix', embedding_texts[: math.ceil(embedding_texts.shape[0] / 2), : 1, :])
        self.register_buffer('token_suffix', embedding_texts[: math.ceil(embedding_texts.shape[0] / 2), 1 + ctx_length :, :])
        self.register_buffer('token_prefix2', embedding_texts[math.ceil(embedding_texts.shape[0] / 2) :, : 1, :])
        self.register_buffer('token_suffix2', embedding_texts[math.ceil(embedding_texts.shape[0] / 2) :, 1 + ctx_length :, :])

    def forward(self):
        token_prefix = torch.cat([self.token_prefix, self.token_prefix2], dim=0)
        token_suffix = torch.cat([self.token_suffix, self.token_suffix2], dim=0)
        embedding_texts_with_prompt = torch.cat([
            token_prefix,
            self.ctx.expand(token_prefix.shape[0], -1, -1),
            token_suffix
        ], dim=1)
        return embedding_texts_with_prompt


class CustomCLIP_teacher(nn.Module):
    def __init__(self, cfg, class_names, clip_model : CLIP):
        super().__init__()

        self.prompt_learner = VLPromptLearner(cfg, class_names, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(clip_model)
        self.image_encoder = clip_model.visual
        self.dtype = clip_model.dtype
        self.logit_scale = clip_model.logit_scale

    def forward(self, images):
        embedding_texts_with_prompt = self.prompt_learner().to(device=device, dtype=self.dtype)
        tokenized_texts = self.tokenized_prompts
        text_features = self.text_encoder(embedding_texts_with_prompt, tokenized_texts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = self.image_encoder(images.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return image_features, text_features, logits


class CustomCLIP(nn.Module):
    def __init__(self, clip_model : CLIP):
        super().__init__()

        self.image_encoder = clip_model.visual
        self.project_layer_end_model = ProjectLayer(512, 768)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, images):
        logit_scale = self.logit_scale.exp()

        image_features = self.image_encoder(images.type(self.dtype))
        image_features = self.project_layer_end_model(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        return image_features, logit_scale


class Adapter(nn.Module):
    def __init__(self, input_dim, reduction=24, residual_ratio=0.2):
        super(Adapter, self).__init__()
        self.residual_ratio = residual_ratio
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // reduction, input_dim, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.type(torch.float32)
        a = self.fc(x)
        x = self.residual_ratio * a + (1 - self.residual_ratio) * x
        return x.type(torch.float16)


"""
train.py -> main():

cfg = setup_cfg(args) # args from command line input
mtkd = MTKD(cfg) -> {
    self.build_data_loader() -> self.dm(DataManager)
    self.build_model() -> self.model(load_state_dict, requires_grad), self.optim, self.sched
}
--- only MTKD ---
if args.second_phase:
    mtkd.load_model(directory, None)
    mtkd.model_add_adapter()
-----------------
mtkd.train() -> {
    for epoch in range(num_epochs):
        for batch_idx, batch in self.dm.train_loader:
            self.forward_backward(batch)
        result = test('val')
        if result > best_result:
            save_model()
            best_result = result
    if cfg.TEST.FINAL_MODEL == 'best_val':
        load_model() # 'model-best.pth.tar'
    test('test')
}
"""


@TRAINER_REGISTRY.register()
class MTKD(TrainerX):
    def build_model(self):
        cfg = self.cfg

        student_model = load_clip_to_cpu(cfg)
        teacher_model = load_clip_to_cpu_teacher(cfg)

        class_names = self.dm.dataset.classnames
        self.n_cls = len(class_names)

        self.model = CustomCLIP(student_model)
        self.model_teacher = CustomCLIP_teacher(cfg, class_names, teacher_model)
        self.model = self.model.to(device)
        m.convert_weights(self.model)
        self.model_teacher = self.model_teacher.to(device)
        m.convert_weights(self.model_teacher)

        model_path = f'./teacher_model/{str(cfg.DATASET.NAME)}/VLPromptLearner/model-best.pth.tar'
        checkpoint = load_checkpoint(model_path)
        state_dict = checkpoint['state_dict']
        self.model_teacher.load_state_dict(state_dict, strict=False)
        self.model_teacher.eval()

        for name, param in self.model.named_parameters():
            if 'project_layer_end_model' in name:
                param.requires_grad_(True)
            elif ('initial_prompt' in name) or ('new_prompt' in name):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        train_module_list = nn.ModuleList([self.model])
        self.optim = build_optimizer(train_module_list, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model('VLPromptLearner', self.model, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        images, labels = self.parse_batch_train(batch)
        with torch.no_grad():
            tea_image_features, tea_text_features, tea_logits = self.model_teacher(images)
        image_features, logit_scale = self.model(images)
        stu_logits = logit_scale * image_features @ tea_text_features.t()
        tea_logits = tea_logits.type(torch.float32)
        stu_logits = stu_logits.type(torch.float32)

        if self.cfg.TRAINER.MTKD.SECOND_PHASE:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(stu_logits, labels)

        else:
            def ins_loss(logits_tea, logits_stu, t):
                predict_tea = F.softmax(logits_tea / t, dim=1)
                log_predict_stu = F.log_softmax(logits_stu / t, dim=1)
                loss_ins = t ** 2 * F.kl_div(log_predict_stu, predict_tea, reduction='none').sum(dim=1).mean()
                return loss_ins

            def batch_loss(logits_tea, logits_stu, t):
                batch_size = logits_stu.shape[0]
                predict_tea = F.softmax(logits_tea / t, dim=1)
                predict_stu = F.softmax(logits_stu / t, dim=1)
                matrix_tea = predict_tea @ predict_tea.t()
                matrix_stu = predict_stu @ predict_stu.t()
                loss_batch = ((matrix_tea - matrix_stu) ** 2).sum() / batch_size
                return loss_batch

            def class_loss(logits_tea, logits_stu, t):
                class_num = logits_stu.shape[1]
                predict_tea = F.softmax(logits_tea / t, dim=1)
                predict_stu = F.softmax(logits_stu / t, dim=1)
                matrix_tea = predict_tea.t() @ predict_tea
                matrix_stu = predict_stu.t() @ predict_stu
                loss_class = ((matrix_tea - matrix_stu) ** 2).sum() / class_num
                return loss_class

            loss_ins = ins_loss(tea_logits, stu_logits, self.cfg.TRAINER.MTKD.TEMPERATURE)
            loss_batch = batch_loss(tea_logits, stu_logits, self.cfg.TRAINER.MTKD.TEMPERATURE)
            loss_class = class_loss(tea_logits, stu_logits, self.cfg.TRAINER.MTKD.TEMPERATURE)
            for t in [2, 3, 5, 6]:
                loss_ins += ins_loss(tea_logits, stu_logits, t)
                loss_batch += batch_loss(tea_logits, stu_logits, t)
                loss_class += class_loss(tea_logits, stu_logits, t)
            loss = self.cfg.TRAINER.MTKD.KD_WEIGHT * (loss_ins + loss_batch + loss_class)

            optim = self.optim
            optim.zero_grad()
            loss.backward()
            optim.step()
            if (self.batch_idx + 1) == self.num_batches:
                self.update_lr()

            return {'loss' : loss.item()}

    def parse_batch_train(self, batch):
        images = batch['img']
        labels = batch['label']
        images = images.to(device)
        labels = labels.to(device)
        return images, labels

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
            images, labels = self.parse_batch_test(batch)
            with torch.no_grad():
                tea_image_features, tea_text_features, tea_logits = self.model_teacher(images)
                stu_image_features, logit_scale = self.model(images)
            if split == 'train' or split == 'val':
                stu_logits = logit_scale * stu_image_features @ (tea_text_features[: math.ceil(self.n_cls / 2), :]).t()
            else:
                stu_logits = logit_scale * stu_image_features @ (tea_text_features[math.ceil(self.n_cls / 2) :, :]).t()
            self.evaluator.process(stu_logits, labels)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f'{split}/{k}'
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def model_add_adapter(self):
        adapter = Adapter(768, self.cfg.TRAINER.MTKD.REDUCTION, self.cfg.TRAINER.MTKD.RESIDUAL_RATIO)
        self.model.adapter = adapter.to(device)
        self.model.forward = types.MethodType(new_forward, self.model)
        for name, param in self.model.named_parameters():
            if 'adapter' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        self.optim = build_optimizer(self.model, self.cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, self.cfg.OPTIM)
        self.register_model('VLPromptLearner', self.model, self.optim, self.sched)


def new_forward(self, images):
    logit_scale = self.logit_scale.exp()

    image_features = self.image_encoder(images.type(self.dtype))
    image_features = self.project_layer_end_model(image_features)
    image_features = self.adapter(image_features)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features, logit_scale
