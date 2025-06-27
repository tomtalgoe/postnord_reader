from datetime import datetime as time
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests

start_time = time.now()

# load image from the IAM database
image = Image.open("image.png").convert("RGB")

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
pixel_values = processor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Print the result
print(generated_text)
stop_time = time.now()

print(stop_time-start_time)