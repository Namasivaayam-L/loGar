import os

def process_files(directory, operation):
  """
  Recursively processes files in a directory and its subdirectories.

  Args:
    directory: The root directory to start from.
    operation: A function that takes a file path as input and performs the desired operation.
  """

  for root, dirs, files in os.walk(directory):
    for file in files:
      file_path = os.path.join(root, file)
      operation(file_path)

# Example operation: printing file names
def print_file_name(file_path):
  print(file_path)

# Example usage:
root_dir = "./logs"
process_files(root_dir, print_file_name)
