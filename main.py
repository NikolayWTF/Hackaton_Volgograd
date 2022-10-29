import easyocr
import base64

def text_recognition(file_path):
    reader = easyocr.Reader(["ru", "en"])
    result = reader.readtext(file_path)

    return result

def main():
    file_path = input("Введите путь к файлу: ")
    words_array = text_recognition(file_path = file_path)
    ans = ""
    y = 0
    for i in words_array:

        if ans == "":
            ans += i[1] # Если в строке нет слов - добавляем
            y = i[0][0][1]
        else:
            if i[0][0][1] - 8 <= y and i[0][0][1] + 8 >= y: # Если координата по y отличается слабо - значит слово из той же строки
                ans += " " + i[1]
                y = i[0][0][1]
            else:
                print(ans)
                ans = i[1]
                y = i[0][0][1]
    if (ans != ""):
        print(ans)
if __name__ == "__main__":
    main()


#test_photo/00111.jpg