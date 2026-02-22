import shutil
import os

def delete_cache(cache_directory):
    for item in os.listdir(cache_directory):
        item_path = os.path.join(cache_directory, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

    print(f"Cache deleted successfully from: {cache_directory}")