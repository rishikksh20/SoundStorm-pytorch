import os
import random
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn.functional as F
from SoundStorm import SoundStorm
from dataset import get_tts_dataset
from lr_schedule import WarmupCosineLRSchedule
from torch.utils.tensorboard import SummaryWriter
from encodec import EncodecModel


def topk_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = (pred == target.unsqueeze(dim=0)).expand_as(pred)

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

    return res


class TrainTransformer:
    def __init__(self, args):
        encodec_ = EncodecModel.encodec_model_24khz()
        encodec_.normalize = False
        encodec_.set_target_bandwidth(6.0)
        encodec_ = encodec_.to(device=args.device)

        self.model = SoundStorm(encodec=encodec_).to(device=args.device)
        self.optim = torch.optim.AdamW(
            self.model.parameters(), lr=2e-4, betas=(0.8, 0.99), eps=0.000000001
        )
        self.lr_schedule = WarmupCosineLRSchedule(
            self.optim,
            init_lr=0.000001,
            peak_lr=0.0002,
            end_lr=0.00001,
            warmup_steps=2000,
            total_steps=40000,
        )

        if args.run_name:
            self.logger = SummaryWriter(f"./runs/{args.run_name}")
        else:
            self.logger = SummaryWriter()
        self.train(args)

    def train(self, args):
        train_dataset, valid_dataset = get_tts_dataset(
            args.spath, args.epath, args.batch_size
        )
        len_train_dataset = len(train_dataset)
        step = 0
        start_from_epoch = 0
        if args.chkpt is not None:
            checkpoint = torch.load(args.chkpt)
            start_from_epoch = checkpoint["epoch"]

        self.lr_schedule = torch.optim.lr_scheduler.ExponentialLR(
            self.optim, gamma=0.999875, last_epoch=start_from_epoch - 1
        )
        if args.chkpt is not None:
            print("Loading checkpoints")
            self.model.load_state_dict(checkpoint["model"])
            self.optim.load_state_dict(checkpoint["optim"])
            self.lr_schedule.load_state_dict(checkpoint["schedular"])
            step = checkpoint["step"]
            args.start_from_epoch = checkpoint["epoch"]

        self.model.train()
        for epoch in range(args.start_from_epoch + 1, args.epochs + 1):
            print(f"Epoch {epoch}:")
            with tqdm(range(len(train_dataset))) as pbar:
                for i, (cond, codes, ids) in zip(pbar, train_dataset):
                    start = random.randint(0, 999)
                    codes = codes.cuda()  # [B, 8, 2250]
                    cond = cond.cuda()  # [B, 1500]

                    b, n = cond.shape
                    cond = cond.reshape(b, n // 2, 2)
                    cond = cond.repeat_interleave(2, -1)[:, :, :3]
                    cond[:, :, 1] = 1025
                    cond = cond.reshape(b, n // 2 * 3)
                    assert cond.shape[-1] == codes.shape[-1]

                    loss, logit, target = self.model(
                        cond.long()[:, start : start + 375],
                        codes.long()[:, :, start : start + 375],
                    )
                    # loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                    loss.backward()

                    ### Calculate accuracy:
                    mask = target.eq(1025)
                    maske_target = target[~mask]
                    masked_logits = logit[~mask]

                    masked_token_prediction = torch.argmax(masked_logits, dim=-1)
                    token_correct = (masked_token_prediction == maske_target).sum()
                    token_total = maske_target.shape[0]
                    token_accuracy = token_correct / token_total

                    topk = topk_accuracy(masked_logits, maske_target, topk=(1, 10))

                    if step % args.accum_grad == 0:
                        self.optim.step()
                        self.lr_schedule.step()
                        self.optim.zero_grad()
                    step += 1
                    pbar.set_postfix(
                        Top10k=topk[1].cpu().detach().numpy().item(),
                        Top1k=topk[0].cpu().detach().numpy().item(),
                        Transformer_Loss=np.round(
                            loss.cpu().detach().numpy().item(), 4
                        ),
                    )
                    pbar.update(0)
                    self.logger.add_scalar(
                        "Cross Entropy Loss",
                        np.round(loss.cpu().detach().numpy().item(), 4),
                        step,
                    )
                    self.logger.add_scalar(
                        "Accuracy", token_accuracy.cpu().detach().numpy().item(), step
                    )
                    self.logger.add_scalar(
                        "Top10K", topk[1].cpu().detach().numpy().item(), step
                    )
                    self.logger.add_scalar(
                        "Top1K", topk[0].cpu().detach().numpy().item(), step
                    )
                    if step % args.ckpt_interval == 0:
                        torch.save(
                            {
                                "model": self.model.state_dict(),
                                "optim": self.optim.state_dict(),
                                "schedular": self.lr_schedule.state_dict(),
                                "step": step,
                                "epoch": epoch + 1,
                            },
                            os.path.join("checkpoints", f"transformer_step_{step}.pt"),
                        )

                    if step % args.validation_step == 0:
                        self.validation_step(valid_dataset, step)

    def validation_step(self, valid_set, steps):
        self.model.eval()
        accuracy = 0
        avg_loss = 0
        top10k = 0
        with tqdm(range(len(valid_set))) as pbar:
            for i, (cond, codes, ids) in zip(pbar, valid_set):
                codes = codes.cuda()[:, :, :750]  # [B, 8, 2250]
                cond = cond.cuda()[:, :500]  # [B, 1500]

                b, n = cond.shape
                cond = cond.reshape(b, n // 2, 2)
                cond = cond.repeat_interleave(2, -1)[:, :, :3]
                cond[:, :, 1] = 1025
                cond = cond.reshape(b, n // 2 * 3)
                assert cond.shape[-1] == codes.shape[-1]
                with torch.no_grad():
                    loss, logit, target = self.model(
                        cond.long()[:, :750], codes.long()[:, :, :750]
                    )
                # loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                avg_loss = avg_loss + loss
                ### Calculate accuracy:
                mask = target.eq(1025)
                maske_target = target[~mask]
                masked_logits = logit[~mask]

                masked_token_prediction = torch.argmax(masked_logits, dim=-1)
                token_correct = (masked_token_prediction == maske_target).sum()
                token_total = maske_target.shape[0]
                token_accuracy = token_correct / token_total

                topk = topk_accuracy(masked_logits, maske_target, topk=(1, 10))
                top10k = top10k + topk[-1]
                accuracy = accuracy + token_accuracy
                pbar.set_postfix(
                    Accuracy=token_accuracy.cpu().detach().numpy().item(),
                    Transformer_Loss=np.round(loss.cpu().detach().numpy().item(), 4),
                )
                pbar.update(0)

        self.logger.add_scalar(
            "Infer loss:",
            np.round((avg_loss.cpu().detach().numpy().item()) / len(valid_set), 4),
            steps,
        )
        self.logger.add_scalar(
            "Infer Accuracy",
            accuracy.cpu().detach().numpy().item() / len(valid_set),
            steps,
        )
        self.logger.add_scalar(
            "Infer Top10k", top10k.cpu().detach().numpy().item() / len(valid_set), steps
        )
        self.model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--nq", type=int, default=8, help="Number of quantizer.")
    parser.add_argument(
        "--spath ",
        type=str,
        default="./data/whisperspeech/whisperspeech/librilight/stoks/",
        help="Path to data.",
    )
    parser.add_argument(
        "--epath ",
        type=str,
        default="./data/whisperspeech/whisperspeech/librilight/encodec-6kbps/",
        help="Path to data.",
    )
    parser.add_argument(
        "--device", type=str, default="mps", help="Which device the training is on."
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training."
    )
    parser.add_argument(
        "--accum-grad", type=int, default=10, help="Number for gradient accumulation."
    )
    parser.add_argument(
        "--epochs", type=int, default=300, help="Number of epochs to train."
    )
    parser.add_argument(
        "--start-from-epoch", type=int, default=0, help="Number of epochs to train."
    )
    parser.add_argument(
        "--ckpt-interval", type=int, default=5000, help="Number of epochs to train."
    )
    parser.add_argument(
        "--validation_step", type=int, default=1000, help="Number of epochs to train."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate."
    )
    parser.add_argument(
        "--chkpt", type=str, default=None, help="checkpoint path to load"
    )

    parser.add_argument(
        "--n-layers", type=int, default=12, help="Number of layers of transformer."
    )
    parser.add_argument(
        "--dim", type=int, default=768, help="Dimension of transformer."
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=3072, help="Dimension of transformer."
    )

    args = parser.parse_args()
    args.run_name = "tests3"
    args.checkpoint_path = r".\checkpoints"
    args.n_layers = 24
    args.dim = 768
    args.hidden_dim = 3072
    args.batch_size = 16
    args.accum_grad = 4
    args.epochs = 1000

    args.start_from_epoch = 0

    # args.spath = "../../../data/whisperspeech/whisperspeech/librilight/stoks/"
    # args.epath = "../../../data/whisperspeech/whisperspeech/librilight/encodec-6kbps/"

    train_transformer = TrainTransformer(args)
