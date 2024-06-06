import pymupdf
import PIL
from PIL import Image
import cv2
import numpy as np

def display_contour(cnt,shape):
    blank = np.zeros(shape, dtype='uint8')
    cv2.drawContours(blank, cnt, -1, (0,0,255))
    pil_img = cv2.cvtColor(blank, cv2.COLOR_BGR2RGB)
    img2  = Image.fromarray(pil_img)
    img2.show()

def display_img_pil(img):
    pil_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img  = Image.fromarray(pil_img)
    pil_img.show()

def segment_pdf(doc):
    num_pages=len(doc)
    zoom = 4
    mat = pymupdf.Matrix(zoom,zoom)
    for i in range(num_pages):
            page = doc[i]
            pix = page.get_pixmap(matrix=mat)
            pix.pil_save(f"pdf_pages\page_{i}_img.jpg")

    seg_count=[]
    for i in range(num_pages):
        count=0
        img = cv2.imread(f"pdf_pages\page_{i}_img.jpg")

        shape = img.shape
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100,35))

        dilated_img = cv2.dilate(thresh_img, rect_kernel, iterations = 1)
        contours, hierarchy = cv2.findContours(dilated_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # display_contour(contours,shape)

        img_copy = img.copy()
        img_copy = cv2.cvtColor(img_copy,cv2.COLOR_BGR2RGB)

        seg_count.append(len(contours))
        for cnt in contours[::-1]:
            x, y, w, h = cv2.boundingRect(cnt)
            rect = cv2.rectangle(img_copy, (x, y), (x+w,y+h), (0,255,0), 2)
            text_img = gray_img[y:y+h, x:x+w]
            pil_img = Image.fromarray(text_img)
            pil_img.save(f"text_seg\page_{i}_seg_img_{count}.jpg")
            count+=1

            # display_img_pil(text_img)

        # display_img_pil(img_copy)
    return seg_count
        

        




