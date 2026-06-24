from data_provider import all_datasets

if __name__ == '__main__':
    for name, generator in all_datasets.items():
        try:
            print(f"Downloading {name} ...")
            generator()
            print(f"{name} loaded \n")
        except Exception as e:
            print(f" Failed to load {name}: {e}\n")