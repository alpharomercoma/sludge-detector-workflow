import kagglehub

# Download latest version
path = kagglehub.dataset_download("jobisaacong/tiktok-sludge-dataset-500")

print("Path to dataset files:", path)