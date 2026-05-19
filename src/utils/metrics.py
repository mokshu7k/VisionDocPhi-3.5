"""
Evaluation Metrics for Document VQA

Metrics:
- ANLS (Average Normalized Levenshtein Similarity)
- Exact Match
"""
from typing import List, Union
from difflib import SequenceMatcher
import re
import string


def anls_score(predictions: List[str], ground_truth: List[str], threshold: float = 0.5) -> float:
    """
    Calculate ANLS (Average Normalized Levenshtein Similarity)
    
    ANLS = max(0, (1 - (edit_distance / max_len)) - threshold) / (1 - threshold)
    
    For multiple ground truth answers, takes the maximum score.
    
    Args:
        predictions: List of predicted answers
        ground_truth: List of ground truth answers
        threshold: Threshold for the ANLS calculation (default 0.5)
    
    Returns:
        ANLS score between 0 and 1
    """
    if not predictions or not ground_truth:
        return 0.0
    
    # Use first prediction
    pred = predictions[0] if predictions else ""
    
    # Calculate max ANLS across all ground truth answers
    max_anls = 0.0
    
    for gt in ground_truth:
        # Normalize text
        pred_norm = _normalize_text(pred)
        gt_norm = _normalize_text(gt)
        
        # Calculate normalized Levenshtein similarity
        similarity = _normalized_levenshtein(pred_norm, gt_norm)
        
        # Apply ANLS formula
        if similarity < threshold:
            anls = 0.0
        else:
            anls = (similarity - threshold) / (1 - threshold)
        
        max_anls = max(max_anls, anls)
    
    return max_anls


def _normalize_text(text: str) -> str:
    """
    Normalize text for comparison
    - Convert to lowercase
    - Remove articles (a, an, the)
    - Remove special characters and extra whitespace
    
    Args:
        text: Input text
    
    Returns:
        Normalized text
    """
    # Lowercase
    text = text.lower()
    
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def _normalized_levenshtein(s1: str, s2: str) -> float:
    """
    Calculate Normalized Levenshtein Similarity
    
    Returns:
        Similarity score between 0 and 1
        1.0 = identical strings
        0.0 = completely different
    """
    if len(s1) == 0 and len(s2) == 0:
        return 1.0
    
    if len(s1) == 0 or len(s2) == 0:
        return 0.0
    
    # Calculate Levenshtein distance
    distance = _levenshtein_distance(s1, s2)
    
    # Normalize by max length
    max_len = max(len(s1), len(s2))
    similarity = 1.0 - (distance / max_len)
    
    return similarity


def _levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein distance (edit distance) between two strings
    
    Uses dynamic programming for efficiency.
    
    Args:
        s1: First string
        s2: Second string
    
    Returns:
        Edit distance (integer)
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def exact_match(predictions: List[str], ground_truth: List[str]) -> bool:
    """
    Check if prediction exactly matches any ground truth (after normalization)
    
    Args:
        predictions: List of predicted answers
        ground_truth: List of ground truth answers
    
    Returns:
        True if exact match found, False otherwise
    """
    if not predictions or not ground_truth:
        return False
    
    pred_norm = _normalize_text(predictions[0])
    
    for gt in ground_truth:
        gt_norm = _normalize_text(gt)
        if pred_norm == gt_norm:
            return True
    
    return False


def calculate_metrics(all_predictions: List[str], all_ground_truths: List[List[str]]) -> dict:
    """
    Calculate multiple metrics for the entire dataset
    
    Args:
        all_predictions: List of predicted answers
        all_ground_truths: List of lists of ground truth answers
    
    Returns:
        Dictionary with various metrics
    """
    assert len(all_predictions) == len(all_ground_truths), \
        "Number of predictions must match number of ground truths"
    
    anls_scores = []
    exact_matches = 0
    
    for pred, gt in zip(all_predictions, all_ground_truths):
        # Calculate ANLS
        score = anls_score([pred], gt)
        anls_scores.append(score)
        
        # Calculate exact match
        if exact_match([pred], gt):
            exact_matches += 1
    
    metrics = {
        'anls': sum(anls_scores) / len(anls_scores) if anls_scores else 0.0,
        'exact_match': exact_matches / len(all_predictions) if all_predictions else 0.0,
        'total_samples': len(all_predictions),
    }
    
    return metrics


if __name__ == "__main__":
    # Test ANLS metric
    print("Testing ANLS Metric...")
    print("=" * 50)
    
    # Test case 1: Exact match
    pred1 = ["10.99"]
    gt1 = ["10.99", "$10.99"]
    score1 = anls_score(pred1, gt1)
    print(f"Test 1 (Exact match): {score1:.4f}")
    
    # Test case 2: Close match
    pred2 = ["JOHN SMITH"]
    gt2 = ["John Smith", "JOHN SMITH"]
    score2 = anls_score(pred2, gt2)
    print(f"Test 2 (Close match): {score2:.4f}")
    
    # Test case 3: Partial match
    pred3 = ["Hello"]
    gt3 = ["Hello World"]
    score3 = anls_score(pred3, gt3)
    print(f"Test 3 (Partial match): {score3:.4f}")
    
    # Test case 4: No match
    pred4 = ["xyz"]
    gt4 = ["abc"]
    score4 = anls_score(pred4, gt4)
    print(f"Test 4 (No match): {score4:.4f}")
    
    print("=" * 50)
