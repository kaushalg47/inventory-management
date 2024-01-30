import cv2
from PIL import Image, ImageEnhance
import easyocr
import io
import matplotlib.pyplot as plt

from roboflow import Roboflow
rf = Roboflow(api_key="cfdKq2OHtlZZFRxJqtV6")
project = rf.workspace().project("sku-110k")
model = project.version(4).model

# infer on a local image
results = model.predict(r"C:\Users\kaushal\Downloads\trial.jpg", confidence=40, overlap=30).json()
print(results['predictions'])

reader = easyocr.Reader(['en'])
count = 0

def enhance(img):
    enhanced_img = ImageEnhance.Contrast(img).enhance(2.0)
    return enhanced_img

def append_to_limit_group(limits_dict, live_data_name, live_data_number, low, high):
    for limits, name_list in limits_dict.items():
        lower_limit, upper_limit = limits
        if lower_limit <= live_data_number <= upper_limit:
            name_list.append(live_data_name)
            return

    new_lower_limit = low 
    new_upper_limit = high
    new_key = (new_lower_limit, new_upper_limit)
    limits_dict[new_key] = [live_data_name]

image_path = r"C:\Users\kaushal\Downloads\trial.jpg"
image_real = Image.open(image_path)
full_img = cv2.imread(image_path)

limit_dict = {}
for i, prediction in enumerate(results['predictions']):
    x, y, w, h, name = prediction['x'], prediction['y'], prediction['width'], prediction['height'], prediction['class']
    xywh_values.append((x, y, w, h))
    labels1.append(name)
    count+=1

    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)

    # BOUNDED BOX
    color = (0, 255, 0)
    thickness = 2
    cv2.rectangle(full_img, (x1, y1), (x2, y2), color, thickness)
    

    window = image_real.crop((x1, y1, x2, y2))

    img = window
    image_gray = enhance(img)
    image_gray = image_gray.convert('L')
   
    
    # Set the threshold value
    threshold = 128 
    
    # Image binary processing
    #image_binary = image_gray.point(lambda p: p > threshold and 255)
    
    # Save the processed image to a BytesIO object
    #image_bytes = io.BytesIO()
    #image_binary.save(image_bytes, format='PNG')
    #image_bytes = image_bytes.getvalue()
#     plt.imshow(image_gray)
#     plt.axis('off')  
#     plt.show()
    
    image_np = np.array(image_gray)

    reader = easyocr.Reader(['en'])
    result = reader.readtext(image_np)
    print(result)
    if result and result[0][1] =="amazon":
        extracted_text = result[0][1]+str(count)
        print(extracted_text)
    else:
        extracted_text = "unknown"+str(count)
    

    append_to_limit_group(limit_dict, extracted_text, y, y1, y2)

    label_position = (x1, y1 - 10)
    cv2.putText(full_img, extracted_text, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

print(limit_dict)
output_image_path = r"C:\Users\kaushal\Downloads\output_warehouse_with_boxes.jpg"
cv2.imwrite(output_image_path, full_img)
print(f"Image with bounding boxes and labels saved to: {output_image_path}")
print(f"number of products:{count}")

