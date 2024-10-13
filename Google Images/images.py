from serpapi import GoogleSearch
import os
import requests
import sys as s
def download_images(query,output_dir):
    search = GoogleSearch({
        "q": query,
        "tbm": "isch",
        "num": 100,
        "api_key":"4085e936b654d4263f2742d3137e95ae2cd0c0e8db5208da02fb5d6410b02056"
    })

    results = search.get_dict()
    print(results)
    os.makedirs(output_dir, exist_ok=True)

    for i, image_info in enumerate(results['images_results']):
        image_url = image_info['original']
        image_data = requests.get(image_url).content
        with open(os.path.join(output_dir, f"{query}_{i+1}.jpg"), "wb") as image_file:
            image_file.write(image_data)
        print(f"Downloaded {image_url}")
        if i == 99:
            break
    
if __name__ == "__main__":
    download_images(s.argv[1],s.argv[2])
