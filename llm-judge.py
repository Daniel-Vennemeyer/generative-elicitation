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

test_samples =[
        "Website Name: New York Times\nTitle: In Norway, the Electric Vehicle Future Has Already Arrived",
        "Website Name: TIME Magazine\nTitle: Two Coronations, 70 Years Apart\nDescription: The Visual Parallels Between Elizabeth II and Charles III's Coronations",
        "Website Name: The Atlantic\nTitle: What Does Sentience Really Mean?",
        "Website Name: Bon Appetit\nTitle: Bon Appetit's Best Banana Bread Recipe",
        "Website Name: National Geographic\nTitle: There’s a new way to tour the Amazon rainforest: by crane",
        "Website Name: Travel and Leisure\nTitle: These Airlines Have the Most Luxurious Economy Seats",
        "Website Name: Popular Science\nTitle: The right amount of online scrolling could decrease your risk of dementia",
        "Website Name: Rolling Stone\nTitle: All of Nintendo’s Zelda Games, Ranked",
        "Website Name: Sports Illustrated\nTitle: 2024 NFL Draft: The Six Players Fans Need to Know, Including Caleb Williams",
        "Website Name: Architectural Digest\nTitle: Hang On, Is Artificial Grass Actually Chic?",
        "Website Name: Esquire\nTitle:The Best Summer Songs of 2023",
        "Website Name: Sunset Magazine\nTitle: 31 Perfect Recipes for Mother’s Day Brunch.",
        "Website Name: Vogue\nTitle: 40 Minimalist Earrings to Wear Now and Forever",
        "Website Name: Runner's World\nTitle: How to Prevent Running Injuries Before They Sideline You",
        "Website Name: Forbes\nTitle: Killing It",
        "Website Name: Tricycle: The Buddhist Review\nTitle: Why We Look for Happiness in the Wrong Places"
    ],

test_samples =[
        "New York Times\nTitle: In Norway, the Electric Vehicle Future Has Already Arrived",
        "TIME Magazine\nTitle: Two Coronations, 70 Years Apart\nDescription: The Visual Parallels Between Elizabeth II and Charles III's Coronations",
        "The Atlantic\nTitle: What Does Sentience Really Mean?",
        "Bon Appetit\nTitle: Bon Appetit's Best Banana Bread Recipe",
        "National Geographic\nTitle: There’s a new way to tour the Amazon rainforest: by crane",
        "Travel and Leisure\nTitle: These Airlines Have the Most Luxurious Economy Seats",
        "Popular Science\nTitle: The right amount of online scrolling could decrease your risk of dementia",
        "Rolling Stone\nTitle: All of Nintendo’s Zelda Games, Ranked",
        "Sports Illustrated\nTitle: 2024 NFL Draft: The Six Players Fans Need to Know, Including Caleb Williams",
        "Architectural Digest\nTitle: Hang On, Is Artificial Grass Actually Chic?",
        "Esquire\nTitle:The Best Summer Songs of 2023",
        "Sunset Magazine\nTitle: 31 Perfect Recipes for Mother’s Day Brunch.",
        "Vogue\nTitle: 40 Minimalist Earrings to Wear Now and Forever",
        "Runner's World\nTitle: How to Prevent Running Injuries Before They Sideline You",
        "Forbes\nTitle: Killing It",
        "Tricycle: The Buddhist Review\nTitle: Why We Look for Happiness in the Wrong Places"
    ],

test_samples =[
        "In Norway, the Electric Vehicle Future Has Already Arrived",
        "Two Coronations, 70 Years Apart\nDescription: The Visual Parallels Between Elizabeth II and Charles III's Coronations",
        "What Does Sentience Really Mean?",
        "Bon Appetit's Best Banana Bread Recipe",
        "here’s a new way to tour the Amazon rainforest: by crane",
        "These Airlines Have the Most Luxurious Economy Seats",
        "The right amount of online scrolling could decrease your risk of dementia",
        "All of Nintendo’s Zelda Games, Ranked",
        "NFL Draft: The Six Players Fans Need to Know, Including Caleb Williams",
        "Hang On, Is Artificial Grass Actually Chic?",
        "The Best Summer Songs of 2023",
        "Perfect Recipes for Mother’s Day Brunch.",
        "Minimalist Earrings to Wear Now and Forever",
        "How to Prevent Running Injuries Before They Sideline You",
        "Killing It",
        "Why We Look for Happiness in the Wrong Places"
    ],

# test_samples =[
#         "New York Times",
#         "TIME Magazine",
#         "The Atlantic",
#         "Bon Appetit",
#         "National Geographic",
#         "Travel and Leisure",
#         "Popular Science",
#         "Rolling Stone",
#         "Sports Illustrated",
#         "Architectural Digest",
#         "Esquire",
#         "Sunset Magazine",
#         "Vogue",
#         "Runner's World",
#         "Forbes",
#         "Tricycle: The Buddhist Review"
#     ],

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

    def rate_question_informativeness(self, context, question, scale, concept=None):
        prompt = (
            f"You are evaluating a question asked in a survey where the goal is to elicit preferences for recommending online articles. "
            f"Given the current conversation context:\n\n{context.strip()}\n\n"
            f"Here is the new question-answer: \"{question}\"\n\n"
            f"On a scale from 1 to {scale}, where {scale} means extremely informative and 1 means not informative at all, "
            f"how informative is this question?\n\n"
            f"ONLY respond with a single number from 1 to {scale}. Answer:\n"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=5,
                eos_token_id=self.tokenizer.eos_token_id
            )
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        try:
            import re
            match = [int(n) for n in re.findall(r'\b\d+\b', decoded)]
            if match:
                score = max(1, min(scale, int(match[-1])))
            else:
                score = -1
        except Exception as e:
            logger.error(f"Error parsing score for informativeness: {e}")
            score = -1
        return {'llm_score': score}, None

    def rate_question_multidimensional(self, context, question, scale, concept):
        def ask_single_metric(metric_name, prompt_detail, scale):
            import re
            prompt = (
                f"You are evaluating a question asked in a survey where the goal is to elicit preferences for recommending online articles. "
                f"For the given context and question, rate the question on a scale from 1 to {scale} for the following aspect:\n\n"
                f"{prompt_detail}\n\n"
                # f"Hidden Concept:\n{concept}\n\n"
                f"Context:\n{context.strip()}\n\n"
                f"Question-Answer: \"{question}\"\n\n"
                f"ONLY respond with a single number from 1 to {scale}. Answer: "
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            decoded = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
            # logger.info(f"Decoded response for {metric_name}")
            # logger.info(f"==================== {decoded} ======================")
            try:
                # Use regex to extract the last contiguous digits from the decoded string
                match = [int(n) for n in re.findall(r'\b\d+\b', decoded)]
                if match:
                    return max(1, min(scale, int(match[-1])))
            except Exception as e:
                logger.error(f"Error parsing score for {metric_name}: {e}")
            return -1

        informativeness = ask_single_metric("informativeness", "Informativeness: To what extent does the question reduce uncertainty about the task-relevant latent variable?", scale)
        relevance = ask_single_metric("relevance", "Relevance: Is the question appropriate and grounded in the current context or conversation state?", scale)
        answerability = ask_single_metric("answerability", "Answerability: Is the question well-formed such that an answer can be generated or obtained reliably?", scale)
        efficiency = ask_single_metric("efficiency", "Efficiency: Does the question progress the agent toward task completion with minimal redundancy?", scale)

        result = {
            'llm_informativeness': informativeness,
            'llm_relevance': relevance,
            'llm_answerability': answerability,
            'llm_efficiency': efficiency
        }
        valid_scores = [v for v in result.values() if v >= 0]
        result['llm_score'] = np.mean(valid_scores) if valid_scores else -1
        return result, None

    def compute_metrics(self, context, prior, object, question, scale, multidimensional=False):
        if multidimensional:
            return self.rate_question_multidimensional(context, question, scale, object)
        return self.rate_question_informativeness(context, question, scale, object)



import re
def main(input_txt, output_txt, model_name, device=None, multidimensional=False, start_object=None, add_noise=False, shuffle_questions=False):
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
        metrics_acc = {'llm_score': []}

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

        ctx = ""
        for t, (q, a) in enumerate(qas):
            # Call compute_metrics with multidimensional option
            m, _ = calc.compute_metrics(ctx, None, obj, f"{q} {a}", scale=5, multidimensional=multidimensional)
            metrics_acc['llm_score'].append(m['llm_score'])
            per_q.append({**{'object':obj,'turn':t,'question':q,'answer':a}, **m})
            pd.DataFrame(per_q).to_csv(f"{output_txt}_judge.csv", index=False)
            ctx += f"{q} {a}"
        agg = {'object':obj, 'n_turns':nturns}
        arr = metrics_acc['llm_score']
        agg['avg_llm_score'] = np.mean(arr)
        agg['std_llm_score'] = np.std(arr)
        per_g.append(agg)
        logger.info(f"Processed {obj}")

    pd.DataFrame(per_q).to_csv(f"{output_txt}_judge.csv", index=False)


logger.info('============= Open Results (Llama-3-70B Judge) =================')
file = 'model_model_results/website_preferences/meta-llama/Llama-3.3-70B-Instruct_per_turn_0_questions_open.txt'
# model = 'meta-llama/Llama-3.3-70B-Instruct'
model = 'gpt2'
main(file, 'Open_GATE_judge_llama3-70b', model, multidimensional=False)
