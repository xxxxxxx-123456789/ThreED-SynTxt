import os
import requests

def fetch_words(n):
    """
    使用 Random Word API 抓取 n 個隨機單字。
    若取得的單字不夠獨特，則持續補足到 n 個。
    """
    url = f"https://random-word-api.herokuapp.com/word?number={n}"
    response = requests.get(url)
    if response.status_code == 200:
        words = response.json()
        # 若重複字太多，補足不足的部分
        unique_words = list(set(words))
        while len(unique_words) < n:
            needed = n - len(unique_words)
            extra_response = requests.get(f"https://random-word-api.herokuapp.com/word?number={needed}")
            if extra_response.status_code == 200:
                extra_words = extra_response.json()
                for word in extra_words:
                    if word not in unique_words:
                        unique_words.append(word)
            else:
                raise Exception("Error fetching extra words")
        return unique_words
    else:
        raise Exception("Error fetching words")

def main():
    num_files = 2000  # 產生的檔案數
    num_words = 4000  # 需要的單字數量
    folder_name = "words_folder"
    
    # 若資料夾不存在，則建立
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # 抓取 4000 個不重複的單字
    words = fetch_words(num_words)
    if len(words) < num_words:
        raise Exception("獲取的單字數量不足！")
    
    # 為每個檔案取一對單字並寫入檔案
    for i in range(num_files):
        file_name = os.path.join(folder_name, f"{i:05d}.txt")
        with open(file_name, "w", encoding="utf-8") as f:
            # 每個檔案取兩個單字
            word_pair = words[2*i] + " " + words[2*i+1]
            f.write(word_pair)
    
    print("成功產生 2000 個檔案在資料夾:", folder_name)

if __name__ == "__main__":
    main()
