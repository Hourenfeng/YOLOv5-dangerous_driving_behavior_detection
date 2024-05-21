import torch

def main():
    if torch.cuda.is_available():
        print("CUDA is available.")
        print("Number of CUDA devices:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print("CUDA Device", i, ":", torch.cuda.get_device_name(i))
            print("CUDA Capability:", torch.cuda.get_device_capability(i))
    else:
        print("CUDA is not available.")

if __name__ == "__main__":
    main()
