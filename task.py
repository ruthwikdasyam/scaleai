import requests
from requests.auth import HTTPBasicAuth
import json
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import center_of_mass
import os
import math
import csv

tasks_here = ["5f127f6f26831d0010e985e5", "5f127f6c3a6b1000172320ad", "5f127f699740b80017f9b170", "5f127f671ab28b001762c204", 
              "5f127f643a6b1000172320a5", "5f127f5f3a6b100017232099", "5f127f5ab1cb1300109e4ffc", "5f127f55fdc4150010e37244"]

WIDTH_HEIGHT_THRESHOLD = 10
ALIGNMENT_THRESHOLD = 5
SYMMETRY_THRESHOLD = 0.8
SYMM_DIFF_THRESHOLD = 0.22


class QualityCheck:
    def __init__(self, task_id):
        self.task_id = task_id
        self.url = f"https://api.scale.com/v1/task/{task_id}"
        self.headers = {"Accept": "application/json"}
        self.auth = HTTPBasicAuth('live_69275e78f13a40d59750ac805cd9c515', '')  # No password

    def get_task(self):
        response = requests.request("GET", self.url, headers=self.headers, auth=self.auth)
        if response.status_code == 200:
            self.task_data = response.json()
            return response.json()
        else:
            raise Exception(f"Error fetching task: {response.status_code} - {response.text}")
        
    def get_images(self):
        task_data = self.get_task()
        img = requests.get(task_data['params']['attachment'])

        with open(f"images/{self.task_id}.jpg", "wb") as f:
            f.write(img.content)
        self.image = Image.open(f"images/{self.task_id}.jpg")


    def get_centerofmass_error(self, x, y, w, h, label, cropped):
        save_dir = f"cropped_images/cropped_{self.task_id}"
        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)

        # cropping graying
        gray = cropped.convert("L")
        
        # finding center of mass
        cy, cx = center_of_mass(np.array(gray))
        geom_cx = w/ 2
        geom_cy = h / 2
        draw = ImageDraw.Draw(cropped)
        r = 2 
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill="red")
        draw.ellipse((geom_cx - r, geom_cy - r, geom_cx + r, geom_cy + r), fill="blue")  # geometric center

        error = math.sqrt((geom_cx - cx) ** 2 + (geom_cy - cy) ** 2)
        print(f"Saved: {x},{y} | Alignment error: {error:.2f} px")

        # Save image
        save_path = os.path.join(save_dir, f"{self.task_id}_{x}_{y}.jpg")
        cropped.convert("RGB").save(save_path)
        # print(f"Saved")
        return error

    def check_symmetry(self, x, y, w, h, cropped_):
        cropped = cropped_.convert("L")

        # normalize to square for symmetry comparison
        size = max(w, h)
        square = Image.new("L", (size, size), 255)
        square.paste(cropped, ((size - w) // 2, (size - h) // 2))
        arr = np.array(square)
        # --- Vertical symmetry ---
        mid_x = size // 2
        left_half = arr[:, :mid_x]
        right_half = arr[:, -mid_x:]
        right_half_flipped = np.fliplr(right_half)
        vert_diff = np.mean(np.abs(left_half - right_half_flipped))
        # --- Horizontal symmetry ---
        mid_y = size // 2
        top_half = arr[:mid_y, :]
        bottom_half = arr[-mid_y:, :]
        bottom_half_flipped = np.flipud(bottom_half)
        horiz_diff = np.mean(np.abs(top_half - bottom_half_flipped))
        # --- Normalize scores to 0–1 scale ---
        vert_score = vert_diff / 255
        horiz_score = horiz_diff / 255
        score_diff = abs(vert_score - horiz_score)

        # print(f"Symmetry Errors - Vertical: {vert_score:.3f}, Horizontal: {horiz_score:.3f}, Combined: {score_diff:.3f}")
        return vert_score, horiz_score, score_diff


    def extract_bounding_boxes(self):
        annotations = self.task_data.get("response", {}).get("annotations", [])
        label_boxes = {}
        for ann in annotations:
            label = ann["label"]
            left = ann["left"]
            top = ann["top"]
            width = ann["width"]
            height = ann["height"]
            # vvid = ann["vvid"]
            box = [left, top, width, height]
            if label not in label_boxes:
                label_boxes[label] = []
            label_boxes[label].append(box)
        return label_boxes
    
    def validate_box(self, alignment_error, vert_symm, horiz_symm, symmetry_diff, w, h):
        # Alignment check
        alignment_ok = alignment_error <= ALIGNMENT_THRESHOLD
        # Symmetry check
        symmetry_ok = symmetry_diff <= SYMM_DIFF_THRESHOLD

        both_symm_ok = vert_symm <= SYMMETRY_THRESHOLD and horiz_symm <= SYMMETRY_THRESHOLD


        size_ok = w > WIDTH_HEIGHT_THRESHOLD and h > WIDTH_HEIGHT_THRESHOLD

        if alignment_ok and symmetry_ok and size_ok:
            return "Good"
        elif alignment_ok and symmetry_ok and not size_ok:
            return "Size Too Small"
        elif alignment_ok and not symmetry_ok and not size_ok:
            return "Size Too Small and Recheck"
        else:
            return "Recheck"


    def run(self):
        self.get_task()
        self.get_images()
        
        # Write to CSV
        csv_path = os.path.join(f"combined.csv")
        # write_header = not os.path.exists(csv_path)  # only write header once
        with open(csv_path, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # if write_header:
            writer.writerow(["image_name","label", "x", "y", "width", "height", "alignment_error_px", "v_symm_score", "h_symm_score", "symmetry_diff", "status"])

        
        bounding_boxes = self.extract_bounding_boxes()
        print("Bounding Boxes:", bounding_boxes)
        for label, boxes in bounding_boxes.items():
            for box in boxes:   
                x, y, w, h = box
                # crop it
                cropped = self.image.crop((x, y, x + w, y + h))
                # check symmetry
                vert_score, horiz_score, symmetry_diff = self.check_symmetry(x, y, w, h, cropped)
                # check center of mass
                error = self.get_centerofmass_error(x, y, w, h, label, cropped)
                # Save to CSV
                with open(csv_path, mode='a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([task_id, label, x, y, w, h, round(error, 2), round(vert_score, 3), round(horiz_score, 3), round(symmetry_diff, 3), self.validate_box(error, vert_score, horiz_score, symmetry_diff, w, h)])


if __name__ == "__main__":
    for task_id in tasks_here:
        qc = QualityCheck(task_id)
        qc.run()