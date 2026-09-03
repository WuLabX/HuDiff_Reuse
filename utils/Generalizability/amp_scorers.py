"""Differentiable activity guidance and independent AMP evaluation.

The four activity predictors are selectable at runtime:
``pepnet``, ``amppred_mfa``, ``iamp_attenpred`` and ``unidl4biopep``.
Exactly one activity predictor is used by ``combined_guidance_loss``.  Any
other loaded predictors are evaluation-only and never contribute to that loss.
HemoPI2 is a separate, optional hemolysis objective/evaluator.

Soft-input adapters
-------------------
PepNet uses expected one-hot/physicochemical features (its unavailable
precomputed embedding feature is zero-filled). AMPpred-MFA uses differentiable
expected DDE features and expected vocabulary embeddings. iAMP-Attenpred and
UniDL4BioPep use the expected ESM2 token embedding, followed by the frozen ESM2
encoder and their frozen published/repository heads.
"""

import json
import os
import pickle
import sys
from typing import Iterable, List, Optional, Sequence, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


HUDIFF_AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
PEPNET_AA_ORDER = list("ALRKNMDFCPQSETGWHYIV")
HUDIFF_TO_PEPNET = [PEPNET_AA_ORDER.index(aa) for aa in HUDIFF_AA_ORDER]
ACTIVITY_SCORERS = (
    "pepnet",
    "amppred_mfa",
    "iamp_attenpred",
    "unidl4biopep",
)
SCORER_ALIASES = {
    "amppred": "amppred_mfa",
    "amppred-mfa": "amppred_mfa",
    "iamp": "iamp_attenpred",
    "iamp-attenpred": "iamp_attenpred",
    "unidl": "unidl4biopep",
}
_PERM_TENSOR = None


def normalize_scorer_name(name: str) -> str:
    name = name.strip().lower()
    return SCORER_ALIASES.get(name, name)


def parse_scorer_names(value: str) -> Set[str]:
    if not value:
        return set()
    names = {normalize_scorer_name(part) for part in value.split(",") if part.strip()}
    if "none" in names:
        return set()
    unknown = names - set(ACTIVITY_SCORERS) - {"hemopi2"}
    if unknown:
        raise ValueError(
            f"Unknown scorer(s): {sorted(unknown)}. "
            f"Expected {', '.join(ACTIVITY_SCORERS)}, hemopi2, or none."
        )
    return names


def resolve_independent_scorers(guidance_scorer: str, eval_arg: str = "auto") -> List[str]:
    """Resolve evaluation-only activity predictors.

    ``auto`` means all three activity predictors other than the selected
    guidance predictor. Explicit lists are also accepted, but the guidance
    predictor is rejected to preserve independence by construction.
    """
    guidance = normalize_scorer_name(guidance_scorer)
    if guidance not in ACTIVITY_SCORERS:
        raise ValueError(f"guidance_scorer must be one of {ACTIVITY_SCORERS}")
    if eval_arg.strip().lower() == "auto":
        return [name for name in ACTIVITY_SCORERS if name != guidance]
    requested = parse_scorer_names(eval_arg) - {"hemopi2"}
    if guidance in requested:
        raise ValueError(
            f"{guidance} is the guidance predictor and cannot also be an independent evaluator"
        )
    return [name for name in ACTIVITY_SCORERS if name in requested]


def _get_perm_tensor(device):
    global _PERM_TENSOR
    if _PERM_TENSOR is None or _PERM_TENSOR.device != device:
        _PERM_TENSOR = torch.tensor(HUDIFF_TO_PEPNET, dtype=torch.long, device=device)
    return _PERM_TENSOR


def _build_props_matrix(props_pkl: str) -> torch.Tensor:
    with open(props_pkl, "rb") as handle:
        props = pickle.load(handle)
    return torch.tensor([props[aa] for aa in HUDIFF_AA_ORDER], dtype=torch.float32)


class IampHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(320, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.classifier(x)


class UniDLHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(1, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.15),
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.15),
        )
        self.fc = nn.Sequential(
            nn.Linear(25600, 64), nn.ReLU(), nn.Dropout(0.15), nn.Linear(64, 2),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        return self.fc(x.flatten(1))


class AMPScorerInterface:
    """Frozen scorer collection with one explicit activity guidance model."""

    def __init__(
        self,
        activity_scorers: Optional[Iterable[str]] = None,
        guidance_activity: Optional[str] = None,
        use_pepnet: Optional[bool] = None,
        use_hemopi2: bool = False,
        pepnet_root: str = "/mnt/wucy/WUCHUYA/PepNet",
        hemopi2_root: str = "/mnt/wucy/WUCHUYA/hemopi2",
        amppred_root: str = "/mnt/wucy/WUCHUYA/AMPpred-MFA",
        iamp_root: str = "/mnt/wucy/WUCHUYA/iAMP-Attenpred",
        unidl_root: str = "/mnt/wucy/WUCHUYA/UniDL4BioPep",
        pepnet_ckpt: str = None,
        device=None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if activity_scorers is None:
            activity_scorers = ["pepnet"] if use_pepnet else []
        self.activity_scorers = {normalize_scorer_name(x) for x in activity_scorers}
        unknown = self.activity_scorers - set(ACTIVITY_SCORERS)
        if unknown:
            raise ValueError(f"Unknown activity scorer(s): {sorted(unknown)}")

        self.guidance_activity = (
            normalize_scorer_name(guidance_activity) if guidance_activity else None
        )
        if self.guidance_activity and self.guidance_activity not in self.activity_scorers:
            raise ValueError("guidance_activity must be included in activity_scorers")

        self.pepnet_model = None
        self.props_mat = None
        self.theta_pep = 40
        self.amppred_model = None
        self.amppred_vocab = None
        self.amppred_aa_vocab_ids = None
        self.amppred_pad_id = None
        self.esm_activity_model = None
        self.esm_activity_alphabet = None
        self.esm_activity_aa_ids = None
        self.iamp_head = None
        self.unidl_head = None
        self.esm_model = None
        self.esm_tokenizer = None
        self.esm_aa_token_ids = None
        self.theta_esm = 40

        if "pepnet" in self.activity_scorers:
            self._load_pepnet(pepnet_root, pepnet_ckpt)
        if "amppred_mfa" in self.activity_scorers:
            self._load_amppred_mfa(amppred_root)
        if self.activity_scorers & {"iamp_attenpred", "unidl4biopep"}:
            self._load_activity_esm()
        if "iamp_attenpred" in self.activity_scorers:
            self.iamp_head = self._freeze(IampHead(), os.path.join(iamp_root, "iAMP-Attenpred.pth"))
        if "unidl4biopep" in self.activity_scorers:
            self.unidl_head = self._freeze(UniDLHead(), os.path.join(unidl_root, "unidl4biopep.pt"))
        if use_hemopi2:
            self._load_hemopi2_esm(hemopi2_root)

    def _freeze(self, model: nn.Module, checkpoint: Optional[str] = None):
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(state, strict=True)
        model.eval().to(self.device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def _load_pepnet(self, root: str, checkpoint: Optional[str]):
        import importlib.util

        spec = importlib.util.spec_from_file_location("pepnet_model", os.path.join(root, "script", "model.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        repository_default = os.path.join(
            root, "datasets", "AMP", "checkpoints", "2024_03_27_19_58_59_951",
            "model", "model_final.pth",
        )
        ckpt = checkpoint or (
            repository_default if os.path.exists(repository_default)
            else os.path.join(root, "pepnet.pth")
        )
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            state = state.get("model_state_dict", state.get("model", state))
        hidden = state["clf.0.weight"].shape[0]
        model = module.AIMP(pre_feas_dim=1024, feas_dim=34, hidden=hidden,
                            n_transformer=1, dropout=0.1)
        model.load_state_dict(state)
        self.pepnet_model = self._freeze(model)
        self.props_mat = _build_props_matrix(
            os.path.join(root, "datasets", "properties.pkl")
        ).to(self.device)

    def _load_amppred_mfa(self, root: str):
        if root not in sys.path:
            sys.path.insert(0, root)
        from AMPpred_MFA.models.AMPpred_MFA import Config, Model

        with open(os.path.join(root, "trained_model", "vocab.json")) as handle:
            vocab = json.load(handle)["token_to_idx"]
        config = Config()
        config.device = self.device
        config.config_manual_feature.device = self.device
        config.config_vocab_feature.device = self.device
        config.embed_padding_idx = vocab[config.padding_token]
        config.feature_dim = 400
        config.vocab_size = len(vocab)
        model = Model(config)
        checkpoint = os.path.join(root, "trained_model", "amppred.pth")
        self.amppred_model = self._freeze(model, checkpoint)
        self.amppred_vocab = vocab
        self.amppred_aa_vocab_ids = torch.tensor(
            [vocab.get(aa, vocab.get("<unk>", 0)) for aa in HUDIFF_AA_ORDER],
            dtype=torch.long, device=self.device,
        )
        # The upstream EasyUse encoder pads with the distinct ``<pad>`` token,
        # although the embedding layer's padding_idx is configured as
        # ``<padding>``. Preserve that repository behavior exactly.
        self.amppred_pad_id = vocab.get("<pad>", vocab.get("<unk>", 0))

    def _load_activity_esm(self):
        import esm

        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.esm_activity_model = self._freeze(model)
        self.esm_activity_alphabet = alphabet
        self.esm_activity_aa_ids = torch.tensor(
            [alphabet.get_idx(aa) for aa in HUDIFF_AA_ORDER],
            dtype=torch.long, device=self.device,
        )

    def _load_hemopi2_esm(self, root: str):
        from transformers import AutoTokenizer, EsmForSequenceClassification

        model_dir = os.path.join(root, "model")
        self.esm_tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.esm_model = self._freeze(EsmForSequenceClassification.from_pretrained(model_dir))
        self.esm_aa_token_ids = torch.tensor(
            [self.esm_tokenizer.convert_tokens_to_ids(aa) for aa in HUDIFF_AA_ORDER],
            dtype=torch.long, device=self.device,
        )

    # ------------------------------------------------------------------
    # Differentiable activity scoring
    # ------------------------------------------------------------------

    def _lengths(self, soft_probs: torch.Tensor, lengths=None) -> torch.Tensor:
        if lengths is None:
            return torch.full(
                (soft_probs.shape[0],), soft_probs.shape[1],
                dtype=torch.long, device=self.device,
            )
        return torch.as_tensor(lengths, dtype=torch.long, device=self.device).clamp(
            min=1, max=soft_probs.shape[1]
        )

    def _soft_feas_pepnet(self, soft_probs: torch.Tensor, lengths=None):
        batch, length, _ = soft_probs.shape
        sp = soft_probs.to(self.device)
        seq_lengths = self._lengths(sp, lengths)
        valid = torch.arange(length, device=self.device)[None, :] < seq_lengths[:, None]
        sp = sp * valid.unsqueeze(-1)
        theta = max(self.theta_pep, int(seq_lengths.max().item()))
        if length < theta:
            sp = F.pad(sp, (0, 0, 0, theta - length))
        else:
            sp = sp[:, :theta, :]
        onehot = sp[:, :, _get_perm_tensor(self.device)]
        props = sp @ self.props_mat
        features = torch.cat([onehot, props], dim=-1).float()
        pretrained = torch.zeros(batch, theta, 1024, device=self.device)
        return pretrained, features

    def _pepnet_score(self, soft_probs: torch.Tensor, lengths=None) -> torch.Tensor:
        pretrained, features = self._soft_feas_pepnet(soft_probs, lengths)
        return self.pepnet_model(pretrained, features).reshape(-1)

    def _soft_dde(self, soft_probs: torch.Tensor, lengths=None) -> torch.Tensor:
        sp = soft_probs.to(self.device)
        length = sp.shape[1]
        if length < 2:
            raise ValueError("AMPpred-MFA requires sequence length >= 2")
        seq_lengths = self._lengths(sp, lengths).clamp(min=2)
        pair_valid = (
            torch.arange(length - 1, device=self.device)[None, :]
            < (seq_lengths - 1)[:, None]
        )
        pair_products = torch.einsum("bla,blc->blac", sp[:, :-1], sp[:, 1:])
        observed = (pair_products * pair_valid[:, :, None, None]).sum(1)
        denominators = (seq_lengths - 1).to(sp.dtype)
        observed = observed / denominators[:, None, None]
        codons = torch.tensor(
            [4, 2, 2, 2, 2, 4, 2, 3, 2, 6, 1, 2, 4, 2, 6, 6, 4, 3, 1, 2],
            dtype=sp.dtype, device=self.device,
        )
        expected = (codons[:, None] * codons[None, :]) / (61.0 ** 2)
        variance = expected[None] * (1.0 - expected[None]) / denominators[:, None, None]
        return ((observed - expected) / variance.sqrt()).flatten(1)

    def _amppred_vocab_embeddings(self, soft_probs: torch.Tensor, lengths=None) -> torch.Tensor:
        model = self.amppred_model.net2
        sp = soft_probs.to(self.device)
        length = min(sp.shape[1], 100)
        seq_lengths = self._lengths(sp, lengths).clamp(max=100)
        aa_embeddings = model.embedding.weight[self.amppred_aa_vocab_ids]
        expected = sp[:, :length] @ aa_embeddings
        valid = torch.arange(length, device=self.device)[None, :] < seq_lengths[:, None]
        input_pad = model.embedding.weight[self.amppred_pad_id].view(1, 1, -1)
        expected = torch.where(valid.unsqueeze(-1), expected, input_pad)
        if length < 100:
            expected = torch.cat(
                [expected, input_pad.expand(sp.shape[0], 100 - length, -1)], dim=1
            )
        return expected

    def _amppred_score(self, soft_probs: torch.Tensor, lengths=None) -> torch.Tensor:
        model = self.amppred_model
        with torch.backends.cudnn.flags(enabled=False):
            out1 = model.net1([self._soft_dde(soft_probs, lengths)])
            net2 = model.net2
            out2 = self._amppred_vocab_embeddings(soft_probs, lengths)
            out2 = net2.position_encoding(out2)
            out2, _ = net2.attention(out2)
            out2, _ = net2.lstm(out2)
            out2 = net2.fc(out2.reshape(out2.size(0), -1))
            logits = model.fc(torch.cat([out1, out2], dim=1))
        return F.softmax(logits, dim=-1)[:, 1]

    def _soft_esm_mean(self, soft_probs: torch.Tensor, lengths=None) -> torch.Tensor:
        """Run ESM2 from expected AA embeddings while retaining input gradients."""
        sp = soft_probs.to(self.device)
        batch, length, _ = sp.shape
        seq_lengths = self._lengths(sp, lengths)
        model = self.esm_activity_model
        alphabet = self.esm_activity_alphabet
        table = model.embed_tokens.weight
        content = sp @ table[self.esm_activity_aa_ids]
        valid = torch.arange(length, device=self.device)[None, :] < seq_lengths[:, None]
        pad_embedding = table[alphabet.padding_idx].view(1, 1, -1)
        content = torch.where(valid.unsqueeze(-1), content, pad_embedding)
        bos = table[alphabet.cls_idx].view(1, 1, -1).expand(batch, -1, -1)
        tail = pad_embedding.expand(batch, 1, -1)
        injected = torch.cat([bos, content, tail], dim=1).clone()
        injected[torch.arange(batch, device=self.device), seq_lengths + 1] = table[alphabet.eos_idx]
        tokens = torch.full(
            (batch, length + 2), alphabet.padding_idx, dtype=torch.long, device=self.device
        )
        tokens[:, 0] = alphabet.cls_idx
        tokens[:, 1 : length + 1] = torch.where(
            valid,
            torch.full_like(valid, alphabet.get_idx("A"), dtype=torch.long),
            torch.full_like(valid, alphabet.padding_idx, dtype=torch.long),
        )
        tokens[torch.arange(batch, device=self.device), seq_lengths + 1] = alphabet.eos_idx
        handle = model.embed_tokens.register_forward_hook(lambda _m, _i, _o: injected)
        try:
            output = model(tokens, repr_layers=[6], return_contacts=False)
        finally:
            handle.remove()
        representations = output["representations"][6][:, 1 : length + 1]
        return (representations * valid.unsqueeze(-1)).sum(1) / seq_lengths[:, None]

    def activity_score(self, soft_probs: torch.Tensor, scorer_name: Optional[str] = None, lengths=None):
        name = normalize_scorer_name(scorer_name or self.guidance_activity or "")
        if not name:
            if len(self.activity_scorers) != 1:
                raise ValueError("scorer_name is required when multiple activity scorers are loaded")
            name = next(iter(self.activity_scorers))
        if name not in self.activity_scorers:
            raise ValueError(f"Activity scorer {name!r} is not loaded")
        if name == "pepnet":
            return self._pepnet_score(soft_probs, lengths)
        if name == "amppred_mfa":
            return self._amppred_score(soft_probs, lengths)
        embedding = self._soft_esm_mean(soft_probs, lengths)
        if name == "iamp_attenpred":
            return torch.sigmoid(self.iamp_head(embedding).squeeze(-1))
        return F.softmax(self.unidl_head(embedding), dim=-1)[:, 1]

    def hemolysis_score(self, soft_probs: torch.Tensor, lengths=None) -> torch.Tensor:
        if self.esm_model is None:
            raise ValueError("HemoPI2 is not loaded")
        batch, length, _ = soft_probs.shape
        sp = soft_probs.to(self.device)
        seq_lengths = self._lengths(sp, lengths)
        theta = max(self.theta_esm, int(seq_lengths.max().item()))
        if length < theta:
            sp = F.pad(sp, (0, 0, 0, theta - length))
        else:
            sp = sp[:, :theta]
        table = self.esm_model.esm.embeddings.word_embeddings.weight
        mask = (
            torch.arange(theta, device=self.device)[None, :]
            < seq_lengths[:, None]
        ).long()
        sp = sp * mask.unsqueeze(-1)
        soft_embeddings = sp @ table[self.esm_aa_token_ids]
        logits = self.esm_model(inputs_embeds=soft_embeddings, attention_mask=mask).logits
        return F.softmax(logits, dim=-1)[:, 1]

    def combined_guidance_loss(
        self,
        soft_probs: torch.Tensor,
        mode: str = "de",
        include_hemolysis: bool = True,
        lengths=None,
    ) -> torch.Tensor:
        if self.guidance_activity is None:
            raise ValueError("No guidance_activity was configured")
        loss = -self.activity_score(
            soft_probs, self.guidance_activity, lengths=lengths
        ).mean()
        if include_hemolysis and mode == "inp" and self.esm_model is not None:
            loss = loss + self.hemolysis_score(soft_probs, lengths=lengths).mean()
        return loss

    # ------------------------------------------------------------------
    # Hard-sequence evaluation (never used by the guidance loss)
    # ------------------------------------------------------------------

    def _onehot(self, sequence: str) -> torch.Tensor:
        tensor = torch.zeros(1, len(sequence), 20, device=self.device)
        for index, aa in enumerate(sequence):
            if aa not in HUDIFF_AA_ORDER:
                raise ValueError(f"Non-standard amino acid {aa!r} in {sequence!r}")
            tensor[0, index, HUDIFF_AA_ORDER.index(aa)] = 1.0
        return tensor

    def _hemopi2_hard(self, sequence: str) -> float:
        inputs = self.esm_tokenizer(
            [sequence], padding=True, truncation=False, return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        logits = self.esm_model(**inputs).logits
        return F.softmax(logits, dim=-1)[0, 1].item()

    def _amppred_hard(self, sequence: str) -> float:
        """Run AMPpred-MFA's original discrete DDE/vocabulary preprocessing."""
        from AMPpred_MFA.lib.Encoding import DDE

        dde = torch.tensor(DDE([sequence]), dtype=torch.float32, device=self.device)
        token_ids = [self.amppred_vocab.get(aa, self.amppred_vocab.get("<unk>", 0))
                     for aa in sequence[:100]]
        token_ids.extend([self.amppred_pad_id] * (100 - len(token_ids)))
        vocab = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        logits = self.amppred_model([dde, vocab])
        return F.softmax(logits, dim=-1)[0, 1].item()

    def evaluate(self, sequences: Sequence[str]):
        import pandas as pd

        rows = []
        with torch.no_grad():
            for sequence in sequences:
                row = {"sequence": sequence}
                onehot = self._onehot(sequence)
                for name in ACTIVITY_SCORERS:
                    if name in self.activity_scorers:
                        if name == "amppred_mfa":
                            row[f"{name}_score"] = self._amppred_hard(sequence)
                        else:
                            row[f"{name}_score"] = self.activity_score(onehot, name).item()
                if self.esm_model is not None:
                    row["hemolysis_score"] = self._hemopi2_hard(sequence)
                rows.append(row)
        return pd.DataFrame(rows)


def build_scorer_from_args(
    scorer_arg: str,
    pepnet_ckpt: str = None,
    guidance_activity: Optional[str] = None,
    infer_single_guidance: bool = True,
    **kwargs,
) -> Optional[AMPScorerInterface]:
    """Build a scorer collection from a comma-separated runtime argument.

    This keeps the old builder API while adding all four activity predictors.
    For gradient guidance, pass ``guidance_activity`` explicitly.
    """
    names = parse_scorer_names(scorer_arg)
    if not names:
        return None
    activity = [name for name in ACTIVITY_SCORERS if name in names]
    guidance = normalize_scorer_name(guidance_activity) if guidance_activity else None
    if guidance is None and infer_single_guidance and len(activity) == 1:
        # Backward compatibility for legacy callers that supplied one activity
        # scorer plus optional HemoPI2 through --scorer.
        guidance = activity[0]
    if guidance and guidance not in activity:
        raise ValueError("guidance_activity must appear in scorer_arg")
    return AMPScorerInterface(
        activity_scorers=activity,
        guidance_activity=guidance,
        use_hemopi2="hemopi2" in names,
        pepnet_ckpt=pepnet_ckpt,
        **kwargs,
    )
