import easyocr

reader = easyocr.Reader(['en'])
def extract_text(img):
    result = reader.readtext(img)
    final_text = ""
    for (box, text, prob) in result:
        final_text += text
    return final_text

