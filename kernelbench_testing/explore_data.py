from datasets import load_dataset

# Load the full KernelBench dataset
dataset = load_dataset('ScalingIntelligence/KernelBench')

# The dataset is a DatasetDict with keys 'level_1', 'level_2', 'level_3', and 'level_4'
print(dataset.keys())  # Output: dict_keys(['level_1', 'level_2', 'level_3', 'level_4'])


level = 1
level_dataset = dataset[f'level_{level}']

# Print the number of problems in this level
print(f"Number of problems in Level {level}: {len(level_dataset)}")

# Get a specific problem by its ID
problem_id = 1  # Problem IDs start from 1
problem = level_dataset.filter(lambda x: x['problem_id'] == problem_id)

# Access the problem's code and name
code = problem['code'][0]
name = problem['name'][0]

print(f"Problem name: {name}")
print(f"Problem code:\n{code}")