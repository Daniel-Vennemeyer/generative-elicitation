import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from lifelines import LogNormalAFTFitter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch.nn.functional as F
import math
import logging
from sentence_transformers import SentenceTransformer
import os
os.environ["HF_HOME"] = "/data/jiang/vennemdp/hf_cache"

# log everything to a file
logging.basicConfig(
    filename='llama_distribution.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
# torch.use_deterministic_algorithms(True)

test_samples =[
        "Website Name: New York Times\nTitle: In Norway, the Electric Vehicle Future Has Already Arrived\nDescription: About 80 percent of new cars sold in Norway are battery-powered. As a result, the air is cleaner, the streets are quieter and the grid hasn’t collapsed. But problems with unreliable chargers persist.",
        "Website Name: TIME Magazine\nTitle: Two Coronations, 70 Years Apart\nDescription: The Visual Parallels Between Elizabeth II and Charles III's Coronations",
        "Website Name: The Atlantic\nTitle: What Does Sentience Really Mean?\nDescription: The fact that AI isn’t alive doesn’t mean it can’t be sentient, the sociologist Jacy Reese Anthis argues.",
        "Website Name: Bon Appetit\nTitle: Bon Appetit's Best Banana Bread Recipe\nDescription: After baking 14 loaves, we aligned on this tender, ultra-moist banana bread as our collective favorite.",
        "Website Name: National Geographic\nTitle: There’s a new way to tour the Amazon rainforest: by crane\nDescription: Canopy cranes, once reserved for treetop research, let you see the “eighth continent” like an arbornaut.",
        "Website Name: Travel and Leisure\nTitle: These Airlines Have the Most Luxurious Economy Seats\nDescription: You can find good Champagne, amenity kits, hot towel service, and wider seats in the economy cabin — if you know where to look.",
        "Website Name: Popular Science\nTitle: The right amount of online scrolling could decrease your risk of dementia\nDescription: A new demographic survey indicates a potential link between regular internet usage and cognitive health in older populations.",
        "Website Name: Rolling Stone\nTitle: All of Nintendo’s Zelda Games, Ranked\nDescription: From NES to Switch, our picks for the best legends in the 35-year-old iconic adventure saga.",
        "Website Name: Sports Illustrated\nTitle: 2024 NFL Draft: The Six Players Fans Need to Know, Including Caleb Williams\nDescription: Quarterbacks will dominate the top 10, but the son of a former Hall of Fame receiver will be picked in the top five, too, along with another dominant Alabama defensive player.",
        "Website Name: Architectural Digest\nTitle: Hang On, Is Artificial Grass Actually Chic?\nDescription: Designers discuss why faux lawns are more popular than you think.",
        "Website Name: Esquire\nTitle:The Best Summer Songs of 2023\nDescription: A sunny mix of 10 hits to carry you through the season.",
        "Website Name: Sunset Magazine\nTitle: 31 Perfect Recipes for Mother’s Day Brunch.\nDescription:Make mom’s day extra-special with an elegant homemade brunch. We’ve got your menu covered, from mains to drinks to dessert.",
        "Website Name: Vogue\nTitle: 40 Minimalist Earrings to Wear Now and Forever\nDescription: From classic hoops to sculptural drops, these are the earrings you’ll never want to take off.",
        "Website Name: Runner's World\nTitle: How to Prevent Running Injuries Before They Sideline You\nDescription: All the tips you need to keep running strong, including strength and mobility moves.",
        "Website Name: Forbes\nTitle: Killing It\nDescription: How Two Failed Entrepreneurs Suddenly Began Making Millions Selling Murder Mysteries",
        "Website Name: Tricycle: The Buddhist Review\nTitle: Why We Look for Happiness in the Wrong Places\nDescription: Meditation teacher Sharon Salzberg discusses how our yearning for happiness can support us in our journey toward freedom—and why we tend to search for it in the wrong places."
    ],

class ObjectEIGCalculator:
    """
    Compute Expected Information Gain and various distributional change metrics
    over a fixed object set O using an LLM.
    """
    def __init__(self, objects, model_name="gpt2", device=None, temperature=0.6):
        self.objects = objects
        self.N = len(objects)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature

        # Load tokenizer (once)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # BitsAndBytes config for 8-bit inference
        # bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load model with remote code if necessary (required for LLaMA 3)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # quantization_config=bnb_config,
            device_map="auto",
            offload_folder="offload",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.model.eval()

        # Precompute choice tokens
        self.choice_tokens = [
            self.tokenizer.encode(str(i + 1), add_special_tokens=False)[0]
            for i in range(self.N)
        ]

        # Load semantic embedding model for QA similarity
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)

    @staticmethod
    def entropy(probs):
        return -(probs * torch.log(probs + 1e-12)).sum().item()

    @staticmethod
    def kl_divergence(p, q, eps=1e-8):
        p = p + eps
        q = q + eps
        p = p / p.sum()
        q = q / q.sum()
        return (p * (p.log() - q.log())).sum().item()

    @staticmethod
    def total_variation(p, q, eps=1e-8):
        return 0.5 * torch.abs(p - q).sum().item()

    @staticmethod
    def wasserstein(p, q):
        # 1D earth mover's distance via CDF
        cdf_p = torch.cumsum(p, dim=0)
        cdf_q = torch.cumsum(q, dim=0)
        return torch.abs(cdf_p - cdf_q).sum().item()

    @staticmethod
    def cosine_distance(p, q):
        return 1 - F.cosine_similarity(p.unsqueeze(0), q.unsqueeze(0)).item()

    @staticmethod
    def euclidean_distance(p, q):
        return torch.norm(p - q, p=2).item()

    @staticmethod
    def gini_impurity(p):
        return (p * (1 - p)).sum().item()

    @staticmethod
    def margin(p):
        vals, _ = torch.sort(p, descending=True)
        return (vals[0] - vals[1]).item()

    @staticmethod
    def reciprocal_rank(p, true_idx):
        # rank starts at 1
        order = torch.argsort(p, descending=True)
        rank = (order == true_idx).nonzero(as_tuple=True)[0].item() + 1
        return 1.0 / rank

    @staticmethod
    def ndcg(p, true_idx, k=None):
        # Normalized Discounted Cumulative Gain at rank K (or all items if K is None)
        order = torch.argsort(p, descending=True)
        if k is None:
            k = len(p)
        dcg = 0.0
        for i in range(k):
            if order[i] == true_idx:
                dcg = 1.0 / math.log2(i + 2)
                break
        idcg = 1.0  # Ideal DCG is 1 at rank 1
        return dcg / idcg

    @staticmethod
    def js_divergence(p, q):
        m = 0.5 * (p + q)
        return 0.5 * F.kl_div(p.log(), m.log(), reduction='sum').item() + 0.5 * F.kl_div(q.log(), m.log(), reduction='sum').item()

    @staticmethod
    def semantic_embedding_shift(p, q, embeddings):
        embed_p = torch.sum(p.unsqueeze(1) * embeddings, dim=0)
        embed_q = torch.sum(q.unsqueeze(1) * embeddings, dim=0)
        return F.cosine_similarity(embed_p.unsqueeze(0), embed_q.unsqueeze(0)).item()

    @staticmethod
    def entropy_constrained_margin_gain(prior, posterior):
        prior_margin = ObjectEIGCalculator.margin(prior)
        post_margin = ObjectEIGCalculator.margin(posterior)
        entropy_prior = ObjectEIGCalculator.entropy(prior)
        return (post_margin - prior_margin) / (entropy_prior + 1e-8)

    @staticmethod
    def surprisal_adjusted_belief_shift(prior, posterior, true_idx):
        delta_conf = (posterior[true_idx] - prior[true_idx]).item()
        surprisal = -torch.log(prior[true_idx] + 1e-12).item()
        return delta_conf * surprisal

    @staticmethod
    def fisher_information_approx(p, epsilon=1e-4):
        grad_estimates = []
        for i in range(len(p)):
            dp = torch.zeros_like(p)
            dp[i] = epsilon
            shifted = p + dp
            shifted = shifted / shifted.sum()  # renormalize
            grad = (torch.log(shifted + 1e-12) - torch.log(p + 1e-12)) / epsilon
            grad_estimates.append((p[i] * grad[i])**2)
        return sum(grad_estimates).item()

    @staticmethod
    def log_odds_gain(prior, posterior, true_idx):
        prior_odds = torch.log(prior[true_idx] / (1 - prior[true_idx] + 1e-12) + 1e-12)
        post_odds = torch.log(posterior[true_idx] / (1 - posterior[true_idx] + 1e-12) + 1e-12)
        return (post_odds - prior_odds).item()

    @staticmethod
    def entropy_weighted_confidence_gain(prior, posterior, true_idx):
        delta_conf = (posterior[true_idx] - prior[true_idx]).item()
        entropy = ObjectEIGCalculator.entropy(prior)
        return delta_conf * entropy

    @staticmethod
    def cross_entropy_reduction(prior, posterior, true_idx):
        ce_prior = -torch.log(prior[true_idx] + 1e-12).item()
        ce_post = -torch.log(posterior[true_idx] + 1e-12).item()
        return ce_prior - ce_post

    @staticmethod
    def kl_to_truth_reduction(prior, posterior, true_idx):
        y = torch.zeros_like(prior)
        y[true_idx] = 1.0
        kl_prior = F.kl_div(prior.log(), y, reduction='sum').item()
        kl_post = F.kl_div(posterior.log(), y, reduction='sum').item()
        return kl_prior - kl_post


    def get_distribution(self, context, style="numbered"):
        # prompt = context.strip() + "\n\nWhich one?"
        if not context:
            context = "None"
        prompt = f"Which online article do you think this person would enjoy the most given the context below? \n\n Context: {context.strip()} \n\n"
        
        if style == 'numbered':
            for idx, obj in enumerate(test_samples):
                prompt += f"\n{idx+1}) {obj}"
            prompt += "\nEnter choice number:"
        elif style == 'candidates':
            prompt += "\nBelow are the candidates for the possible candidates to be guessed:"
            for obj in test_samples:
                prompt += f"\n- {obj}"
            prompt += "\nEnter candidate name:"
        elif style == 'open':
            prompt += "\nEnter ONLY the name of your guess. Answer:"
            # prompt += "\nEnter ONLY the name of your guess (e.g 'abacus'):"

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        last = logits[0, -1, :] / self.temperature
        choice_logits = last[self.choice_tokens]
        return torch.softmax(choice_logits, dim=-1)
        return ''

    def compute_metrics(self, context, prior, object, style='numbered'):
        # prior = self.get_distribution(context)
        if prior == None:
            prior = self.get_distribution("", style=style)
        H_prior = self.entropy(prior)
        # posterior
        true_idx = self.objects.index(object)
        # posterior = prior.clone()
        # simulate conditioning by boosting true choice
        posterior = self.get_distribution(context, style=style)
        H_post = self.entropy(posterior)
        # metrics
        kl = self.kl_divergence(prior, posterior)
        tv = self.total_variation(prior, posterior)
        wd = self.wasserstein(prior, posterior)
        cos = self.cosine_distance(prior, posterior)
        euc = self.euclidean_distance(prior, posterior)
        dir_score = (posterior - prior)[true_idx].item()
        gini_drop = self.gini_impurity(prior) - self.gini_impurity(posterior)
        margin_gain = self.margin(posterior) - self.margin(prior)
        rr_change = self.reciprocal_rank(posterior, true_idx) - self.reciprocal_rank(prior, true_idx)
        ndcg_gain = self.ndcg(posterior, true_idx) - self.ndcg(prior, true_idx)
        hybrid = kl + tv
        eig = H_prior - H_post
        log_odds = self.log_odds_gain(prior, posterior, true_idx)
        ewc_gain = self.entropy_weighted_confidence_gain(prior, posterior, true_idx)
        ce_reduction = self.cross_entropy_reduction(prior, posterior, true_idx)
        kl_truth_reduction = self.kl_to_truth_reduction(prior, posterior, true_idx)
        # Compute change in surprisal of the correct answer
        surprisal_prior = -torch.log(prior[true_idx] + 1e-12).item()
        surprisal_post = -torch.log(posterior[true_idx] + 1e-12).item()
        delta_surprisal = surprisal_prior - surprisal_post  # Positive if model becomes more confident


        # Semantic Embedding-Based Metrics using SentenceTransformer
        # a = context.split('Oracle said:')[-1].strip() if 'Oracle said:' in context else ""
        # qa_text = f"Q: {context.split('Guesser said:')[-1].strip()} A: {a}"
        # sem_inputs = [qa_text, object]
        # sem_embeddings = self.embedding_model.encode(sem_inputs, convert_to_tensor=True)
        # qa_embed = sem_embeddings[0]
        # ans_embed = sem_embeddings[1]

        # sem_relevance_cosine = 1 - F.cosine_similarity(qa_embed.unsqueeze(0), ans_embed.unsqueeze(0)).item()
        # sem_relevance_euclidean = torch.norm(qa_embed - ans_embed, p=2).item()
        # sem_relevance_dot = -torch.dot(qa_embed, ans_embed).item()

        answer_texts = [f"{i+1}) {self.objects[i]}" for i in range(self.N)]
        answer_embeds = self.embedding_model.encode(answer_texts, convert_to_tensor=True)
        consensus_embed = torch.sum(posterior.unsqueeze(1) * answer_embeds, dim=0)
        true_idx = self.objects.index(object)
        true_embed = answer_embeds[true_idx]

        sem_info_cosine = 1 - F.cosine_similarity(consensus_embed.unsqueeze(0), true_embed.unsqueeze(0)).item()
        sem_info_euclidean = torch.norm(consensus_embed - true_embed, p=2).item()
        sem_info_dot = -torch.dot(consensus_embed, true_embed).item()

        js = self.js_divergence(prior, posterior)
        # semantic_embeddings = torch.randn((len(self.objects), 768)).to(self.device)
        sem_shift = self.semantic_embedding_shift(prior, posterior, answer_embeds)
        ec_margin = self.entropy_constrained_margin_gain(prior, posterior)
        sab_shift = self.surprisal_adjusted_belief_shift(prior, posterior, true_idx)
        fisher_info = self.fisher_information_approx(posterior)
        norm_kl = kl / (H_prior + 1e-8)
        norm_margin = margin_gain / (H_prior + 1e-8)
        eig_cal = (H_prior - H_post) / (H_prior + 1e-8)

        return {
            'eig': eig,
            'H_prior': H_prior,
            'H_post': H_post,
            'kl': kl,
            'wasserstein': wd,
            'tv': tv,
            'cosine': cos,
            'euclidean': euc,
            'directional': dir_score,
            'gini_drop': gini_drop,
            'margin_gain': margin_gain,
            'rr_change': rr_change,
            'ndcg_gain': ndcg_gain,
            'hybrid': hybrid,
            'js_divergence': js,
            'semantic_shift': sem_shift,
            'entropy_constrained_margin': ec_margin,
            'surprisal_adj_belief_shift': sab_shift,
            'fisher_info': fisher_info,
            'normalized_kl': norm_kl,
            'normalized_margin': norm_margin,
            'calibrated_eig': eig_cal,
            'log_odds_gain': log_odds,
            'entropy_weighted_conf_gain': ewc_gain,
            'cross_entropy_reduction': ce_reduction,
            'kl_to_truth_reduction': kl_truth_reduction,
            'delta_surprisal': delta_surprisal,
            # 'sem_relevance_cosine': sem_relevance_cosine,
            # 'sem_relevance_euclidean': sem_relevance_euclidean,
            # 'sem_relevance_dot': sem_relevance_dot,
            'sem_info_cosine': sem_info_cosine,
            'sem_info_euclidean': sem_info_euclidean,
            'sem_info_dot': sem_info_dot
        }, prior




import re
def main(input_txt, output_txt, model_name, device=None, style='numbered', start_object=None, add_noise=False, shuffle_questions=False):
    # Read the entire file content at once
    with open(input_txt) as f:
        content = f.read()

    # Split by persona blocks
    blocks = content.split("Suppose you are this person")
    parsed_tuples = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # Persona is first few non-empty lines until a question is found
        lines = block.splitlines()
        persona_lines = []
        i = 0
        while i < len(lines):
            l = lines[i].strip()
            if not l:
                i += 1
                continue
            if re.match(r'^\d+\.\s+', l):
                break
            if l.startswith("EVAL POINT"):
                i += 1
                continue
            persona_lines.append(l)
            i += 1
        persona = "\n".join(persona_lines).strip()
        # Now, parse all question/answer pairs in the block
        qa_pairs = []
        while i < len(lines):
            qline = lines[i].strip()
            if not qline or qline.startswith("EVAL POINT"):
                i += 1
                continue
            q_match = re.match(r'^(\d+)\. \s*(.+)', qline)
            if q_match:
                question = q_match.group(2).strip()
                # Find answer in next non-empty, non-EVAL line
                i += 1
                while i < len(lines):
                    aline = lines[i].strip()
                    if not aline or aline.startswith("EVAL POINT"):
                        i += 1
                        continue
                    answer = aline
                    break
                else:
                    answer = ""
                qa_pairs.append((persona, question, answer))
            i += 1
        parsed_tuples.extend(qa_pairs)

    # Get unique objects (personas)
    objects = [t[0] for t in parsed_tuples]
    objects = list(dict.fromkeys(objects))
    if start_object is not None and start_object in objects:
        start_index = objects.index(start_object)
        objects = objects[start_index:] + objects[:start_index]

    calc = ObjectEIGCalculator(objects, model_name=model_name, device=device)
    per_q, per_g = [], []

    # Example noise questions to inject during ablation
    noise_qa_pairs = [
        ("What color is the sky?", "Blue"),
        ("How many legs does a spider have?", "Eight"),
        ("Is fire hot?", "Yes"),
        ("Can fish fly?", "No"),
        ("What is 2 + 2?", "Four")
    ]

    # Group by persona for each object
    from collections import defaultdict
    persona_to_qas = defaultdict(list)
    for persona, question, answer in parsed_tuples:
        persona_to_qas[persona].append((question, answer))

    for obj in objects:
        qa_pairs = persona_to_qas[obj]
        nturns = len(qa_pairs)
        ctx = ""
        prior = None
        metrics_acc = {k: [] for k in calc.compute_metrics(ctx, prior, objects[0])[0].keys()}

        qas = qa_pairs.copy()
        if shuffle_questions:
            random.shuffle(qas)

        if add_noise:
            n_noise = min(len(qas), len(noise_qa_pairs))
            interval = max(1, len(qas) // (n_noise + 1))
            new_qa_pairs = []
            noise_index = 0
            for i, pair in enumerate(qas):
                if (i + 1) % interval == 0 and noise_index < n_noise:
                    new_qa_pairs.append(noise_qa_pairs[noise_index])
                    noise_index += 1
                new_qa_pairs.append(pair)
            qas = new_qa_pairs

        for t, (q, a) in enumerate(qas):
            ctx = f"{q} {a}"
            m, prior = calc.compute_metrics(ctx, prior, obj, style=style)
            if a not in objects:
                for k in ['directional','rr_change','ndcg_gain']:
                    m[k] = float('nan')
            for k, v in m.items():
                metrics_acc[k].append(v)
            per_q.append({**{'object':obj,'turn':t,'question':q,'answer':a}, **m})
            pd.DataFrame(per_q).to_csv(f"{output_txt}.csv", index=False)
        agg = {'object':obj, 'n_turns':nturns}
        for k, arr in metrics_acc.items():
            agg[f'avg_{k}'] = np.mean(arr)
            agg[f'std_{k}'] = np.std(arr)
        per_g.append(agg)
        logger.info(f"Processed {obj}")

    pd.DataFrame(per_q).to_csv(f"{output_txt}.csv", index=False)


logger.info('============= Open Results (Llama-3-70B) =================')
file = 'model_model_results/website_preferences/meta-llama/Llama-3.3-70B-Instruct_per_turn_0_questions_open.txt'
model = 'meta-llama/Llama-3.3-70B-Instruct'
main(file, 'Open_GATE_quid_llama3-70b_numbered', model, style='numbered')
