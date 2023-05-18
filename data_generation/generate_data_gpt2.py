import argparse
from random import shuffle

import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn.functional as F
from torchtext.datasets import WikiText103
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel

nlp = spacy.load('en_core_web_sm')

def set_hooks_gpt2(model):
    """
    Only works on GPT2 from HF
    """
    final_layer = model.config.n_layer - 1

    for attr in ["activations_"]:
        if not hasattr(model, attr):
            setattr(model, attr, {})

    def get_activation(name):
        def hook(module, input, output):
            if "mlp" in name or "attn" in name or "m_coef" in name:
                if "attn" in name:
                    num_tokens = list(output[0].size())[1]
                    model.activations_[name] = output[0][:, num_tokens - 1].detach()
                elif "mlp" in name:
                    num_tokens = list(output[0].size())[0]  # [num_tokens, 3072] for values;
                    model.activations_[name] = output[0][num_tokens - 1].detach()
                elif "m_coef" in name:
                    num_tokens = list(input[0].size())[1]  # (batch, sequence, hidden_state)
                    model.activations_[name] = input[0][:, num_tokens - 1].detach()
            elif "residual" in name or "embedding" in name:
                num_tokens = list(input[0].size())[1]  # (batch, sequence, hidden_state)
                if name == "layer_residual_" + str(final_layer):
                    model.activations_[name] = model.activations_["intermediate_residual_" + str(final_layer)] + \
                                               model.activations_["mlp_" + str(final_layer)]
                else:
                    model.activations_[name] = input[0][:,
                                               num_tokens - 1].detach()  # https://github.com/huggingface/transformers/issues/7760

        return hook

    model.transformer.h[0].ln_1.register_forward_hook(get_activation("input_embedding"))

    for i in range(model.config.n_layer):
        if i != 0:
            model.transformer.h[i].ln_1.register_forward_hook(get_activation("layer_residual_" + str(i - 1)))
        model.transformer.h[i].ln_2.register_forward_hook(get_activation("intermediate_residual_" + str(i)))

        model.transformer.h[i].attn.register_forward_hook(get_activation("attn_" + str(i)))
        model.transformer.h[i].mlp.register_forward_hook(get_activation("mlp_" + str(i)))
        model.transformer.h[i].mlp.c_proj.register_forward_hook(get_activation("m_coef_" + str(i)))

    model.transformer.ln_f.register_forward_hook(get_activation("layer_residual_" + str(final_layer)))


def get_resid_predictions(model, tokenizer, tokens, TOP_K=1, start_idx=None, end_idx=None, set_mlp_0=False):
    HIDDEN_SIZE = model.config.n_embd

    layer_residual_preds = []
    intermed_residual_preds = []
    output = model(**tokens, output_hidden_states=True)

    for layer in model.activations_.keys():
        if "layer_residual" in layer or "intermediate_residual" in layer:
            normed = model.transformer.ln_f(model.activations_[layer])
            logits = torch.matmul(model.lm_head.weight, normed.T)

            probs = F.softmax(logits.T[0], dim=-1)

            probs = torch.reshape(probs, (-1,)).cpu().detach().numpy()

            assert np.abs(np.sum(probs) - 1) <= 0.01, str(np.abs(np.sum(probs) - 1)) + layer

            probs_ = []
            for index, prob in enumerate(probs):
                probs_.append((index, prob))

            top_k = sorted(probs_, key=lambda x: x[1], reverse=True)[:TOP_K]
            top_k = [(t[1].item(), t[0]) for t in top_k]

        if "layer_residual" in layer:
            layer_residual_preds.append(top_k)
        elif "intermediate_residual" in layer:
            intermed_residual_preds.append(top_k)
        for attr in ["layer_resid_preds", "intermed_residual_preds"]:
            if not hasattr(model, attr):
                setattr(model, attr, [])

        model.layer_resid_preds = layer_residual_preds
        model.intermed_residual_preds = intermed_residual_preds


def project_value_to_vocab(layer, value_idx, top_k=10):
    normed = model.transformer.ln_f(model.transformer.h[layer].mlp.c_proj.weight.data[value_idx])

    logits = torch.matmul(model.lm_head.weight, normed.T)
    probs = F.softmax(logits, dim=-1)
    probs = torch.reshape(probs, (-1,)).cpu().detach().numpy()

    probs_ = []
    for index, prob in enumerate(probs):
        probs_.append((index, prob))

    top_k = sorted(probs_, key=lambda x: x[1], reverse=True)[:top_k]
    value_preds = [(tokenizer.decode(t[0]), t[0]) for t in top_k]

    return value_preds


def get_prediction_depth(layer_resid_top_preds):
    pred_depth = len(layer_resid_top_preds)

    prev_pred = layer_resid_top_preds[-1]

    for pred in [pred for pred in reversed(layer_resid_top_preds)]:
        if pred != prev_pred:
            break
        else:
            pred_depth -= 1
            prev_pred = pred

    return pred_depth


def get_preds_and_hidden_states(wiki_text_sentences, gpt2_model, gpt2_tokenizer, random_pos=True):
    set_hooks_gpt2(gpt2_model)

    sent_to_preds = {}
    sent_to_hidden_states = {}
    idx = 0
    for sentence in tqdm(wiki_text_sentences):
        if random_pos:
            tokens_old = [token for token in sentence.split(' ')]
            sentence_old = sentence[:]
            tokens = gpt2_tokenizer(sentence, return_tensors="pt")
            tokens_to_sent = gpt2_tokenizer.tokenize(sentence)
            if len(tokens_to_sent) > 0:
                if len(tokens_to_sent) == 1:
                    random_pos = 1
                else:
                    random_pos = np.random.randint(1, len(tokens_to_sent))
                tokens = {k: v[:, :random_pos].to(device) for k, v in tokens.items()}
                tokens_to_sent = tokens_to_sent[:random_pos]
                sentence = gpt2_tokenizer.convert_tokens_to_string(tokens_to_sent)
            else:
                continue

        key = (sentence, idx)
        get_resid_predictions(gpt2_model, gpt2_tokenizer, tokens, TOP_K=30)

        if sentence not in sent_to_preds.keys():
            sent_to_preds[key] = {}
        sent_to_preds[key]["layer_resid_preds"] = gpt2_model.layer_resid_preds
        sent_to_preds[key]["intermed_residual_preds"] = gpt2_model.intermed_residual_preds
        sent_to_hidden_states[key] = {k: v.cpu() for k, v in gpt2_model.activations_.items()}
        if len(tokens_to_sent) == 1:
            sent_to_hidden_states[key]['gold_token'] = gpt2_tokenizer(sentence_old, return_tensors="pt")['input_ids'][:,
                                                       0].item()
        else:
            sent_to_hidden_states[key]['gold_token'] = gpt2_tokenizer(sentence_old, return_tensors="pt")['input_ids'][:,
                                                       random_pos].item()
        idx += 1

    return sent_to_hidden_states, sent_to_preds


def get_examples_df(prompts, model, tokenizer):
    sent_to_hidden_states, sent_to_preds = get_preds_and_hidden_states(prompts, model, tokenizer, random_pos=True)
    idx = 0
    records = []
    for key in tqdm(sent_to_preds.keys()):
        sent = key[0]
        top_coef_idx = []
        top_coef_vals = []
        top_coef_abs_idx = []
        top_coef_vals_abs = []

        rand_coef_idx = []
        rand_coef_vals = []
        rand_coef_abs_idx = []
        rand_coef_vals_abs = []

        coefs_sums = []

        residual_preds_probs = []
        residual_preds_tokens = []
        layer_preds_probs = []
        layer_preds_tokens = []
        res_vecs = []
        mlp_vecs = []
        for LAYER in range(model.config.n_layer):
            coefs_ = []
            coefs_abs_ = []
            m_coefs = sent_to_hidden_states[key]["m_coef_" + str(LAYER)].squeeze(0).cpu().numpy()
            res_vec = sent_to_hidden_states[key]["intermediate_residual_" + str(LAYER)].squeeze().cpu().numpy()
            mlp_vec = sent_to_hidden_states[key]["mlp_" + str(LAYER)].squeeze().cpu().numpy()
            res_vecs.append(res_vec)
            mlp_vecs.append(mlp_vec)
            value_norms = torch.linalg.norm(model.transformer.h[LAYER].mlp.c_proj.weight.data, dim=1).cpu()
            coefs = m_coefs * value_norms.numpy()
            coefs_abs = np.absolute(m_coefs) * value_norms.numpy()
            coefs_sums.append(coefs_abs.sum())
            for index, prob in enumerate(coefs):
                coefs_.append((index, prob))
            for index, prob in enumerate(coefs_abs):
                coefs_abs_.append((index, prob))
            top_values = sorted(coefs_, key=lambda x: x[1], reverse=True)[:30]
            c_idx, c_vals = zip(*top_values)
            top_coef_idx.append(c_idx)
            top_coef_vals.append(c_vals)

            top_values_abs = sorted(coefs_abs_, key=lambda x: x[1], reverse=True)[:30]
            c_idx_abs, c_vals_abs = zip(*top_values_abs)
            top_coef_abs_idx.append(c_idx_abs)
            top_coef_vals_abs.append(c_vals_abs)

            shuffle(coefs_)
            rand_idx, rand_vals = zip(*coefs_)
            rand_coef_idx.append(rand_idx[:30])
            rand_coef_vals.append(rand_vals[:30])
            shuffle(coefs_abs_)
            rand_idx_abs, rand_vals_abs = zip(*coefs_abs_)
            rand_coef_abs_idx.append(rand_idx_abs[:30])
            rand_coef_vals_abs.append(rand_vals_abs[:30])

            residual_p_probs, residual_p_tokens = zip(*sent_to_preds[key]['intermed_residual_preds'][LAYER])
            residual_preds_probs.append(residual_p_probs)
            residual_preds_tokens.append(residual_p_tokens)

            layer_p_probs, layer_p_tokens = zip(*sent_to_preds[key]['layer_resid_preds'][LAYER])
            layer_preds_probs.append(layer_p_probs)
            layer_preds_tokens.append(layer_p_tokens)
        gold_token = sent_to_hidden_states[key]['gold_token']
        records.append({
            "sent": sent,
            "top_coef_idx": top_coef_idx,
            "top_coef_vals": top_coef_vals,
            "top_coef_abs_idx": top_coef_abs_idx,
            "top_coef_vals_abs": top_coef_vals_abs,
            "rand_coef_idx": rand_coef_idx,
            "rand_coef_vals": rand_coef_vals,
            "rand_coef_abs_idx": rand_coef_abs_idx,
            "rand_coef_vals_abs": rand_coef_vals_abs,

            "coefs_total_sum": coefs_sums,
            "residual_preds_probs": residual_preds_probs,
            "residual_preds_tokens": residual_preds_tokens,
            "layer_preds_probs": layer_preds_probs,
            "layer_preds_tokens": layer_preds_tokens,
            "layer_mlp_vec": mlp_vecs,
            "gold_token": gold_token
        })
        idx += 1

    df = pd.DataFrame(records)
    return df


def parse_line(line):
    tokens = [
        token for token in line.split(' ')
        if token not in ['', '\n']
    ]
    if len(tokens) == 0:
        return None
    spaces = [True for _ in range(len(tokens) - 1)] + [False]
    assert len(tokens) == len(spaces), f"{len(tokens)} != {len(spaces)}"

    doc = spacy.tokens.doc.Doc(
        nlp.vocab, words=tokens, spaces=spaces)
    for name, proc in nlp.pipeline:
        doc = proc(doc)
    return [str(sent) for sent in doc.sents]


parser = argparse.ArgumentParser()
parser.add_argument(
    "--gpt2_model_name", default='gpt2', type=str, help="GPT2 model name"
)
parser.add_argument(
    "--device", default='cuda:0', type=str, help="device"
)
parser.add_argument(
    "--max_sentences", default=10000, type=int, help="max sentences to include in the data"
)
parser.add_argument(
    "--output_path", default='gpt2_df_10k.pkl', type=str, help="output pickle file path"
)

args = parser.parse_args()

pt2_model_name = 'gpt2'
device = "cuda:0" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
tokenizer = GPT2Tokenizer.from_pretrained(args.gpt2_model_name)
model = GPT2LMHeadModel.from_pretrained(args.gpt2_model_name)
model = model.to(device)
model.eval()

layer_fc2_vals = [
    model.transformer.h[layer_i].mlp.c_proj.weight.T.detach()
    for layer_i in tqdm(range(model.config.n_layer))
]

E = model.get_input_embeddings().weight.cpu().detach()

tok_to_idx = tokenizer.get_vocab()

vocab = [None] * len(tok_to_idx)
for k, v in tok_to_idx.items():
    vocab[v] = [k, 0]

wiki_text = list(WikiText103(split="valid"))
num_sentences = 0
wiki_text_sentences = []
for line in WikiText103(split="valid"):
    sentences = parse_line(line)
    if sentences is None:
        continue
    else:
        for sentence in sentences:
            if len(tokenizer.tokenize(sentence)) == 0:
                continue
            wiki_text_sentences.append(sentence)
            num_sentences += 1
shuffle(wiki_text_sentences)
df = get_examples_df(wiki_text_sentences[:args.max_sentences], model, tokenizer)
df.to_pickle(args.output_path)
