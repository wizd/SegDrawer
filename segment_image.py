import argparse
import numpy as np
from PIL import Image
from io import BytesIO
import zipfile
from base64 import b64encode
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

def seg_everything(image_path):
    # Load the model
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = 'cuda'
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # Load the image
    img = np.array(Image.open(image_path))

    # Generate masks
    masks = mask_generator.generate(img)
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)

    # Create a new image with the same size as the original image
    img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]), dtype=np.uint8)
    for idx, ann in enumerate(sorted_anns, 0):
        img[ann['segmentation']] = idx % 255 + 1

    zip_buffer = BytesIO()
    PIL_GLOBAL_IMAGE = Image.fromarray(img)
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for idx, ann in enumerate(sorted_anns, 0):
            left, top, right, bottom = ann["bbox"][0], ann["bbox"][1], ann["bbox"][0] + ann["bbox"][2], ann["bbox"][1] + ann["bbox"][3]
            cropped = PIL_GLOBAL_IMAGE.crop((left, top, right, bottom))
            transparent = Image.new("RGBA", cropped.size, (0, 0, 0, 0))
            mask = Image.fromarray(ann["segmentation"].astype(np.uint8) * 255)
            mask_cropped = mask.crop((left, top, right, bottom))
            result = Image.composite(cropped.convert("RGBA"), transparent, mask_cropped)

            result_bytes = BytesIO()
            result.save(result_bytes, format="PNG")
            result_bytes.seek(0)
            zip_file.writestr(f"seg_{idx}.png", result_bytes.read())
    
    # Save the zip file
    with open(f"{image_path.rsplit('.', 1)[0]}.zip", 'wb') as f:
        f.write(zip_buffer.getvalue())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Segment an image and save the result as a zip file.')
    parser.add_argument('image_path', type=str, help='The path of the image to be segmented.')
    args = parser.parse_args()
    seg_everything(args.image_path)