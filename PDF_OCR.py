import PDF_Segment
import EasyOCR
import pymupdf
import sys
sys.path.append("FCNN")
import Model
from PIL import Image
import os
import torch
from torchvision import transforms

def rem_files_in_folder(path):
    for filename in os.listdir(path):
        os.remove(f"{path}/{filename}")

if __name__ == "__main__":

    model = Model.FCNN()
    PATH = "FCNN\model_params\model_state_dict"
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])

    for i in ("extracted_diagram","pdf_pages","extracted_text","text_seg"):
        rem_files_in_folder(i)

    doc = pymupdf.open(r"DOC-20240528-WA0001.pdf")
    seg_count = PDF_Segment.segment_pdf(doc)
    count=24
    for i in range(len(doc)):
        print(f"----------PAGE {i+1}---------")
        text_file = open(f"doc_text\page_{i}", "w")
        for j in range(seg_count[i]):
            img = Image.open(f"text_seg\page_{i}_seg_img_{j}.jpg")
            transform = transforms.Compose([ 
                transforms.ToTensor() 
            ]) 
            img_tens = transform(img)
            
            res = model(img_tens)
            if(res<0.05):
                img.save(fr"extracted_diagram\0_page{i}_img_{j}.jpg")
            else:
                result = EasyOCR.extract_text(f"text_seg\page_{i}_seg_img_{j}.jpg")
                result+="\n"
                img.save(fr"extracted_text\1_img_{count}.jpg")
                count+=1
                text_file.write(result)


            # result = EasyOCR.extract_text(f"text_seg\page_{i}_seg_img_{j}.jpg")
            # if result=="":
            #     # img.save(fr"doc_img\0_page{i}_img_{j}.jpg")
            #     continue
            # result+="\n"
            # img.save(fr"text_img\1_img_{count}.jpg")
            # count+=1
            # text_file.write(result)
        text_file.close()


