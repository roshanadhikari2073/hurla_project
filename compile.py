import os

# Explicit list of files to include
file_list = [
    "config.py",
    "generate_test_with_anomalies.py",
    "hurla_pipeline.py",
    "run_experiment.py",
    "test_preprocessing.py",
    os.path.join("models", "autoencoder.py"),
    os.path.join("models", "q_learning_agent.py"),
    os.path.join("utils", "evaluation.py"),
    os.path.join("utils", "preprocessing.py")
]

# Output file name
output_file = "selected_code_dump.txt"

# Initialize content holder
compiled_content = []

# Read and append each file's content
for file_path in file_list:
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            compiled_content.append(file_path)
            compiled_content.append('')
            compiled_content.append(content)
            compiled_content.append('')
            compiled_content.append('#' * 80)
            compiled_content.append('')

        except Exception as e:
            print(f"Could not read {file_path}: {e}")
    else:
        print(f"File not found: {file_path}")

# Write compiled content to output
with open(output_file, "w", encoding="utf-8") as out:
    out.write('\n'.join(compiled_content))

print(f"Selected files have been compiled into '{output_file}'")
