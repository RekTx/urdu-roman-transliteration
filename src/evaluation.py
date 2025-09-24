"""
Evaluation metrics for Urdu to Roman transliteration.
"""
import math
import numpy as np
from typing import List, Dict, Tuple


def calculate_bleu(reference: str, hypothesis: str) -> float:
    """
    Calculate BLEU score between reference and hypothesis.
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        
    Returns:
        BLEU score (0-100)
    """
    try:
        # Simple BLEU calculation (using 1-gram, 2-gram, 3-gram, 4-gram)
        reference_tokens = reference.split()
        hypothesis_tokens = hypothesis.split()
        
        if len(hypothesis_tokens) == 0:
            return 0.0
        
        # Calculate precision for each n-gram
        precisions = []
        for n in range(1, 5):
            reference_ngrams = list(zip(*[reference_tokens[i:] for i in range(n)]))
            hypothesis_ngrams = list(zip(*[hypothesis_tokens[i:] for i in range(n)]))
            
            if len(hypothesis_ngrams) == 0:
                precisions.append(0)
                continue
            
            # Count matching n-grams
            matches = sum(1 for ngram in hypothesis_ngrams if ngram in reference_ngrams)
            precision = matches / len(hypothesis_ngrams)
            precisions.append(precision)
        
        # Calculate brevity penalty
        if len(hypothesis_tokens) > len(reference_tokens):
            bp = 1.0
        else:
            bp = math.exp(1 - len(reference_tokens) / len(hypothesis_tokens)) if len(hypothesis_tokens) > 0 else 0.0
        
        # Calculate geometric mean of precisions
        if all(p > 0 for p in precisions):
            score = bp * math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        else:
            score = 0.0
        
        return score * 100  # Return as percentage
    except:
        return 0.0


def editdistance(s1: List, s2: List) -> int:
    """
    Calculate edit distance between two sequences.
    
    Args:
        s1: First sequence
        s2: Second sequence
        
    Returns:
        Edit distance
    """
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    
    distances = list(range(len(s1) + 1))
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    
    return distances[-1]


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate using edit distance.
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        
    Returns:
        CER (0-1)
    """
    try:
        ref_chars = list(reference.replace(' ', ''))
        hyp_chars = list(hypothesis.replace(' ', ''))

        # Calculate edit distance
        distance = editdistance(ref_chars, hyp_chars)

        # CER = (Substitutions + Deletions + Insertions) / Reference length
        cer = distance / len(ref_chars) if len(ref_chars) > 0 else 1.0
        return cer
    except:
        return 1.0


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate.
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        
    Returns:
        WER (0-1)
    """
    try:
        ref_words = reference.split()
        hyp_words = hypothesis.split()

        # Calculate edit distance
        distance = editdistance(ref_words, hyp_words)

        # WER = (Substitutions + Deletions + Insertions) / Reference length
        wer = distance / len(ref_words) if len(ref_words) > 0 else 1.0
        return wer
    except:
        return 1.0


def calculate_perplexity(loss: float) -> float:
    """
    Calculate perplexity from loss.
    
    Args:
        loss: Cross-entropy loss
        
    Returns:
        Perplexity
    """
    try:
        return math.exp(loss)
    except:
        return float('inf')


def evaluate_comprehensive(transliterator, test_pairs: List[Dict[str, str]], 
                         num_samples: int = None) -> Dict[str, float]:
    """
    Evaluate model with multiple metrics.
    
    Args:
        transliterator: Trained transliterator
        test_pairs: Test data pairs
        num_samples: Number of samples to evaluate (None for all)
        
    Returns:
        Dictionary with evaluation metrics
    """
    if num_samples is None:
        num_samples = len(test_pairs)

    # Sample test pairs
    sample_indices = np.random.choice(len(test_pairs), min(num_samples, len(test_pairs)), replace=False)

    # Initialize metrics
    bleu_scores = []
    cer_scores = []
    wer_scores = []
    exact_matches = 0

    # Store examples for qualitative analysis
    examples = []

    for idx in sample_indices:
        pair = test_pairs[idx]
        urdu_text = pair["ur_norm"]
        reference = pair["en_clean"]

        # Get model prediction
        prediction = transliterator.transliterate(urdu_text)

        # Calculate metrics
        bleu = calculate_bleu(reference, prediction)
        cer = calculate_cer(reference, prediction)
        wer = calculate_wer(reference, prediction)

        bleu_scores.append(bleu)
        cer_scores.append(cer)
        wer_scores.append(wer)

        if reference.strip() == prediction.strip():
            exact_matches += 1

        # Store examples
        examples.append({
            "urdu": urdu_text,
            "reference": reference,
            "prediction": prediction,
            "bleu": bleu,
            "cer": cer,
            "wer": wer
        })

    # Calculate average metrics
    avg_bleu = np.mean(bleu_scores)
    avg_cer = np.mean(cer_scores)
    avg_wer = np.mean(wer_scores)
    exact_match_rate = exact_matches / len(sample_indices)

    return {
        "avg_bleu": avg_bleu,
        "avg_cer": avg_cer,
        "avg_wer": avg_wer,
        "exact_match_rate": exact_match_rate,
        "examples": examples,
        "num_samples": len(sample_indices)
    }


def print_evaluation_summary(results: Dict[str, float], model_name: str = "Model"):
    """
    Print evaluation summary in a formatted way.
    
    Args:
        results: Evaluation results dictionary
        model_name: Name of the model being evaluated
    """
    print(f"\n{'='*50}")
    print(f"EVALUATION SUMMARY FOR {model_name}")
    print("="*50)
    print(f"BLEU Score: {results['avg_bleu']:.2f}")
    print(f"Character Error Rate (CER): {results['avg_cer']:.2%}")
    print(f"Word Error Rate (WER): {results['avg_wer']:.2%}")
    print(f"Exact Match Rate: {results['exact_match_rate']:.2%}")
    if 'perplexity' in results:
        print(f"Perplexity: {results['perplexity']:.2f}")
    print(f"Samples evaluated: {results['num_samples']}")
    print("="*50)


def show_examples(results: Dict[str, any], num_examples: int = 5, sort_by: str = "bleu"):
    """
    Show example predictions from evaluation results.
    
    Args:
        results: Evaluation results containing examples
        num_examples: Number of examples to show
        sort_by: Metric to sort by ("bleu", "cer", "wer")
    """
    examples = results["examples"]
    
    if sort_by == "bleu":
        # Show worst first (lowest BLEU)
        examples_sorted = sorted(examples, key=lambda x: x["bleu"])
        print(f"\nWorst {num_examples} Predictions (Low BLEU):")
    elif sort_by == "cer":
        # Show worst first (highest CER)
        examples_sorted = sorted(examples, key=lambda x: x["cer"], reverse=True)
        print(f"\nWorst {num_examples} Predictions (High CER):")
    else:
        examples_sorted = examples
        print(f"\nExample Predictions:")
    
    for i, example in enumerate(examples_sorted[:num_examples]):
        print(f"\nExample #{i+1} (BLEU: {example['bleu']:.2f}, CER: {example['cer']:.2%})")
        print(f"Urdu:      {example['urdu']}")
        print(f"Reference: {example['reference']}")
        print(f"Prediction: {example['prediction']}")
    
    # Show exact matches if any
    exact_matches = [ex for ex in examples if ex["reference"].strip() == ex["prediction"].strip()]
    if exact_matches:
        print(f"\nExact Matches ({len(exact_matches)} found):")
        for i, example in enumerate(exact_matches[:3]):
            print(f"\nMatch #{i+1}")
            print(f"Urdu:      {example['urdu']}")
            print(f"Match:     {example['reference']}")
