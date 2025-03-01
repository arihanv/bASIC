import requests
from bs4 import BeautifulSoup
import os
import json
import time
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("kernelbench_testing/scraping.log")
    ]
)

# Create directories
os.makedirs('kernelbench_testing/qa_pairs', exist_ok=True)
os.makedirs('kernelbench_testing/qa_pairs/problems', exist_ok=True)
os.makedirs('kernelbench_testing/qa_pairs/solutions', exist_ok=True)

# URL of the leaderboard
url = 'https://scalingintelligence.stanford.edu/KernelBenchLeaderboard/'

def fetch_url_content(url, retries=3, delay=1):
    """Fetch content from URL with retries."""
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logging.warning(f"Attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                logging.error(f"Failed to fetch {url} after {retries} attempts")
                raise
    return None

def extract_links_from_html(html_content):
    """Extract problem and rank 1 solution links from HTML content."""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find the table
    table = soup.find('table')
    if not table:
        logging.warning("Could not find table in the leaderboard page. Using JavaScript extraction instead.")
        return []
    
    # Find all rows in the table (skip the header row)
    rows = table.find_all('tr')[1:]  # Skip header row
    
    problem_solution_pairs = []
    
    # Process each row
    for i, row in enumerate(rows, 1):
        # Get all cells in the row
        cells = row.find_all('td')
        
        if len(cells) < 2:
            continue  # Skip rows with insufficient cells
        
        # Extract problem info from the first cell
        problem_cell = cells[0]
        problem_link = problem_cell.find('a')
        
        if not problem_link:
            logging.warning(f"No problem link found in row {i}")
            continue
            
        problem_text = problem_cell.text.strip()
        problem_url = problem_link.get('href')
        
        # Extract rank 1 solution from the second cell
        rank1_cell = cells[1]
        rank1_link = rank1_cell.find('a')
        
        if not rank1_link:
            logging.warning(f"No rank 1 link found for problem: {problem_text}")
            continue
        
        # Extract the href attribute (URL to the solution)
        solution_url = rank1_link.get('href')
        if not solution_url:
            logging.warning(f"No solution URL found for problem: {problem_text}")
            continue
        
        # Extract speedup value
        speedup_match = re.search(r'(\d+\.\d+)', rank1_cell.text.strip())
        speedup_text = speedup_match.group(1) if speedup_match else "N/A"
        
        # Extract model name
        model_text = rank1_cell.text.strip()
        model_match = re.search(r'\((.*?)\)', model_text)
        model_name = model_match.group(1) if model_match else "N/A"
        
        logging.info(f"Found problem: {problem_text}")
        logging.info(f"  Problem URL: {problem_url}")
        logging.info(f"  Solution URL: {solution_url}")
        logging.info(f"  Speedup: {speedup_text}")
        logging.info(f"  Model: {model_name}")
        
        problem_solution_pairs.append({
            "problem_text": problem_text,
            "problem_url": problem_url,
            "solution_url": solution_url,
            "speedup": speedup_text,
            "model": model_name
        })
    
    if not problem_solution_pairs:
        logging.warning("No problem-solution pairs found using BeautifulSoup. Using hardcoded data from JavaScript extraction.")
        # Use hardcoded data from JavaScript extraction
        problem_solution_pairs = get_hardcoded_problem_solution_pairs()
        logging.info(f"Using {len(problem_solution_pairs)} hardcoded problem-solution pairs")
    
    return problem_solution_pairs

def get_hardcoded_problem_solution_pairs():
    """Get hardcoded problem-solution pairs from JavaScript extraction."""
    # This data was extracted using JavaScript in the browser
    return [
        {
            "problem_text": "Level 1: 1_Square_matrix_multiplication_",
            "problem_url": "https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/problems/l1_p1.py",
            "solution_url": "https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/solutions/fa3e10d50ee22fd656c8e4767b3f5157.py",
            "speedup": "0.17",
            "model": "claude-3.5-sonnet"
        },
        {
            "problem_text": "Level 1: 2_Standard_matrix_multiplication_",
            "problem_url": "https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/problems/l1_p2.py",
            "solution_url": "https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/solutions/d5e8b955452b19c73a2acc9f8372d306.py",
            "speedup": "0.17",
            "model": "claude-3.5-sonnet"
        },
        {
            "problem_text": "Level 1: 3_Batched_matrix_multiplication",
            "problem_url": "https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/problems/l1_p3.py",
            "solution_url": "https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/solutions/d0c502aa627e20a3de8e80dbdc8d2ef0.py",
            "speedup": "0.18",
            "model": "gpt-o1"
        },
        {
            "problem_text": "Level 1: 4_Matrix_vector_multiplication_",
            "problem_url": "https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/problems/l1_p4.py",
            "solution_url": "https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/solutions/ace75607eb636186cc05385c61d181c8.py",
            "speedup": "0.89",
            "model": "gpt-o1"
        },
        {
            "problem_text": "Level 1: 5_Matrix_scalar_multiplication",
            "problem_url": "https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/problems/l1_p5.py",
            "solution_url": "https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/solutions/5a8524b63a79d7dfb2085a8f82597b96.py",
            "speedup": "0.69",
            "model": "claude-3.5-sonnet"
        },
        {
            "problem_text": "Level 1: 6_Matmul_with_large_K_dimension_",
            "problem_url": "https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/problems/l1_p6.py",
            "solution_url": "https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/solutions/5a11f55d9e36f2b68b5774c540201de9.py",
            "speedup": "0.16",
            "model": "gpt-o1"
        },
        {
            "problem_text": "Level 1: 7_Matmul_with_small_K_dimension_",
            "problem_url": "https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/problems/l1_p7.py",
            "solution_url": "https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/solutions/1143f5a65506760015e6c3f06a283f87.py",
            "speedup": "0.39",
            "model": "gpt-o1"
        },
        {
            "problem_text": "Level 1: 8_Matmul_with_irregular_shapes_",
            "problem_url": "https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/problems/l1_p8.py",
            "solution_url": "https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/solutions/175f44b0ad194769197ecf985280ba71.py",
            "speedup": "0.22",
            "model": "claude-3.5-sonnet"
        },
        {
            "problem_text": "Level 1: 9_Tall_skinny_matrix_multiplication_",
            "problem_url": "https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/problems/l1_p9.py",
            "solution_url": "https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/solutions/bfe7cc072b506a9a376bb0b68acdf936.py",
            "speedup": "0.61",
            "model": "claude-3.5-sonnet"
        },
        {
            "problem_text": "Level 1: 10_3D_tensor_matrix_multiplication",
            "problem_url": "https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/problems/l1_p10.py",
            "solution_url": "https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/solutions/8f8e6d114217c821c325b7d98d924fde.py",
            "speedup": "0.20",
            "model": "claude-3.5-sonnet"
        },
        {
            "problem_text": "Level 2: 7_Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd",
            "problem_url": "https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/problems/l2_p7.py",
            "solution_url": "https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/solutions/d0d792daeb6fcee0a4d0e1b163978c0f.py",
            "speedup": "2.33",
            "model": "claude-3.5-sonnet"
        },
        {
            "problem_text": "Level 2: 14_Gemm_Divide_Sum_Scaling",
            "problem_url": "https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/problems/l2_p14.py",
            "solution_url": "https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/solutions/3b400dd6bfedc298e2d95b1b95b73c6f.py",
            "speedup": "3.17",
            "model": "claude-3.5-sonnet"
        },
        {
            "problem_text": "Level 2: 20_ConvTranspose3d_Sum_ResidualAdd_Multiply_ResidualAdd",
            "problem_url": "https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/problems/l2_p20.py",
            "solution_url": "https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/solutions/ecce74df42226c991c533ae3ede3de06.py",
            "speedup": "1.75",
            "model": "claude-3.5-sonnet"
        },
        {
            "problem_text": "Level 2: 90_Conv3d_LeakyReLU_Sum_Clamp_GELU",
            "problem_url": "https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/problems/l2_p90.py",
            "solution_url": "https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/solutions/447fac3129139e84230be00b8e0372d7.py",
            "speedup": "2.02",
            "model": "claude-3.5-sonnet"
        },
        {
            "problem_text": "Level 3: 4_LeNet5",
            "problem_url": "https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/problems/l3_p4.py",
            "solution_url": "https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs//assets/solutions/4f80ebb0657e05543d17fb5db74842ed.py",
            "speedup": "1.32",
            "model": "claude-3.5-sonnet"
        }
        # Added a selection of problems from different levels with high speedups
    ]

def create_qa_pairs(problem_solution_pairs):
    """Create Q&A pairs from problem and solution pairs."""
    qa_pairs = []
    
    for i, pair in enumerate(problem_solution_pairs):
        problem_text = pair["problem_text"]
        problem_url = pair["problem_url"]
        solution_url = pair["solution_url"]
        speedup = pair["speedup"]
        model = pair["model"]
        
        logging.info(f"Processing [{i+1}/{len(problem_solution_pairs)}] {problem_text}")
        logging.info(f"  Problem URL: {problem_url}")
        logging.info(f"  Solution URL: {solution_url}")
        
        try:
            # Fetch problem code
            problem_code = fetch_url_content(problem_url)
            
            # Fetch solution code
            solution_code = fetch_url_content(solution_url)
            
            # Create Q&A pair
            qa_pair = {
                "problem": problem_text,
                "problem_url": problem_url,
                "solution_url": solution_url,
                "speedup": speedup,
                "model": model,
                "problem_code": problem_code,
                "solution_code": solution_code
            }
            
            qa_pairs.append(qa_pair)
            
            # Save individual problem and solution files
            problem_id = problem_text.replace(':', '_').replace(' ', '_')
            
            problem_file = f"kernelbench_testing/qa_pairs/problems/{problem_id}.py"
            with open(problem_file, 'w') as f:
                f.write(problem_code)
            
            solution_file = f"kernelbench_testing/qa_pairs/solutions/{problem_id}_solution.py"
            with open(solution_file, 'w') as f:
                f.write(solution_code)
            
            logging.info(f"  Saved problem to {problem_file}")
            logging.info(f"  Saved solution to {solution_file}")
            
            # Add a small delay to avoid overwhelming the server
            time.sleep(0.5)
            
        except Exception as e:
            logging.error(f"Error processing {problem_text}: {e}")
    
    return qa_pairs

def save_qa_pairs(qa_pairs):
    """Save Q&A pairs to various formats."""
    # Save all Q&A pairs to a JSON file
    qa_pairs_file = "kernelbench_testing/qa_pairs/kernelbench_qa_pairs.json"
    with open(qa_pairs_file, 'w') as f:
        json.dump(qa_pairs, f, indent=2)
    
    logging.info(f"Saved {len(qa_pairs)} Q&A pairs to {qa_pairs_file}")
    
    # Create a markdown file with the Q&A pairs
    markdown_file = "kernelbench_testing/qa_pairs/kernelbench_qa_pairs.md"
    with open(markdown_file, 'w') as f:
        f.write("# KernelBench Q&A Pairs\n\n")
        f.write("This file contains Q&A pairs from the KernelBench leaderboard, where the question is the problem statement and the answer is the rank 1 solution.\n\n")
        
        for i, qa_pair in enumerate(qa_pairs, 1):
            f.write(f"## {qa_pair['problem']}\n\n")
            f.write(f"**Speedup**: {qa_pair['speedup']}\n\n")
            f.write(f"**Model**: {qa_pair['model']}\n\n")
            f.write(f"**Problem URL**: {qa_pair['problem_url']}\n\n")
            f.write(f"**Solution URL**: {qa_pair['solution_url']}\n\n")
            
            f.write("### Problem\n\n")
            f.write("```python\n")
            f.write(qa_pair['problem_code'])
            f.write("\n```\n\n")
            
            f.write("### Solution\n\n")
            f.write("```python\n")
            f.write(qa_pair['solution_code'])
            f.write("\n```\n\n")
    
    logging.info(f"Created markdown file with Q&A pairs: {markdown_file}")
    
    # Create a CSV file for easy import into other tools
    csv_file = "kernelbench_testing/qa_pairs/kernelbench_qa_pairs.csv"
    with open(csv_file, 'w') as f:
        f.write("problem,speedup,model,problem_url,solution_url\n")
        for qa_pair in qa_pairs:
            f.write(f"\"{qa_pair['problem']}\",{qa_pair['speedup']},{qa_pair['model']},{qa_pair['problem_url']},{qa_pair['solution_url']}\n")
    
    logging.info(f"Created CSV file with Q&A pairs: {csv_file}")
    
    # Create a README file for the qa_pairs directory
    readme_file = "kernelbench_testing/qa_pairs/README.md"
    with open(readme_file, 'w') as f:
        f.write("# KernelBench Q&A Pairs\n\n")
        f.write("This directory contains Q&A pairs from the KernelBench leaderboard, where the question is the problem statement and the answer is the rank 1 solution.\n\n")
        f.write("## Files\n\n")
        f.write("- `kernelbench_qa_pairs.json`: JSON file containing all Q&A pairs\n")
        f.write("- `kernelbench_qa_pairs.md`: Markdown file containing all Q&A pairs\n")
        f.write("- `kernelbench_qa_pairs.csv`: CSV file containing all Q&A pairs\n")
        f.write("- `problems/`: Directory containing individual problem files\n")
        f.write("- `solutions/`: Directory containing individual solution files\n\n")
        f.write("## Usage\n\n")
        f.write("These Q&A pairs can be used to train or fine-tune an LLM to generate efficient CUDA kernels for PyTorch operators.\n\n")
        f.write("## Statistics\n\n")
        f.write(f"- Total Q&A pairs: {len(qa_pairs)}\n")
        
        # Count by level
        level_counts = {}
        for qa_pair in qa_pairs:
            level_match = re.search(r'Level (\d+):', qa_pair['problem'])
            if level_match:
                level = level_match.group(1)
                level_counts[level] = level_counts.get(level, 0) + 1
        
        f.write("- Pairs by level:\n")
        for level, count in sorted(level_counts.items()):
            f.write(f"  - Level {level}: {count} pairs\n")
        
        # Count by model
        model_counts = {}
        for qa_pair in qa_pairs:
            model = qa_pair['model']
            model_counts[model] = model_counts.get(model, 0) + 1
        
        f.write("- Pairs by model:\n")
        for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  - {model}: {count} pairs\n")
    
    logging.info(f"Created README file for the qa_pairs directory: {readme_file}")

def main():
    logging.info(f"Starting KernelBench leaderboard scraping from {url}")
    
    try:
        # Fetch the leaderboard HTML
        html_content = fetch_url_content(url)
        
        # Extract problem and solution links
        problem_solution_pairs = extract_links_from_html(html_content)
        logging.info(f"Found {len(problem_solution_pairs)} problem-solution pairs")
        
        # Create Q&A pairs
        qa_pairs = create_qa_pairs(problem_solution_pairs)
        logging.info(f"Created {len(qa_pairs)} Q&A pairs")
        
        # Save Q&A pairs
        save_qa_pairs(qa_pairs)
        
        logging.info("Scraping completed successfully")
        
    except Exception as e:
        logging.error(f"Scraping failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()
