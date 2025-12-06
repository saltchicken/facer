import json
import time
import requests
import io
import cv2
import numpy as np
from pathlib import Path
import uuid



COMFY_URL = (
    "http://127.0.0.1:8188"  # Adjust if your ComfyUI is on a different port/host
)


class ComfyRunner:
    def __init__(self, base_url=COMFY_URL):
        self.base_url = base_url
        self.workflow_path = Path(__file__).parent / "workflow_api.json"

    def _load_workflow(self):
        if not self.workflow_path.exists():
            raise FileNotFoundError(
                f"Please save your exported API JSON to {self.workflow_path}"
            )

        with open(self.workflow_path, "r") as f:
            return json.load(f)

    def _upload_image(self, image_np):
        """Encodes numpy image to PNG and uploads to ComfyUI."""
        success, buffer = cv2.imencode(".png", image_np)
        if not success:
            raise ValueError("Could not encode image for upload")

        # Create a unique filename to avoid collisions
        filename = f"facer_upload_{uuid.uuid4().hex}.png"

        files = {"image": (filename, buffer.tobytes(), "image/png")}
        data = {"overwrite": "true"}

        resp = requests.post(f"{self.base_url}/upload/image", files=files, data=data)
        resp.raise_for_status()

        # Return the filename response to set in the workflow
        return resp.json().get("name") or filename

    def _find_node_by_class(self, workflow, class_type):
        """Helper to find a node ID by its class type."""
        for node_id, node_data in workflow.items():
            if node_data.get("class_type") == class_type:
                return node_id
        return None

    def run(self, image_np, prompt_text=None):
        """
        Full pipeline: Upload -> Patch Workflow -> Queue -> Wait -> Download
        """
        workflow = self._load_workflow()

        # 1. Upload Image
        print(f"📤 Uploading image to ComfyUI...")
        uploaded_filename = self._upload_image(image_np)

        # 2. Patch Workflow (Find LoadImage node and set input)
        load_node_id = self._find_node_by_class(workflow, "LoadImage")
        if not load_node_id:
            raise ValueError("Workflow must contain a 'LoadImage' node")

        workflow[load_node_id]["inputs"]["image"] = uploaded_filename


        if prompt_text:
            # Try to find the specific Qwen node
            qwen_node_id = self._find_node_by_class(workflow, "TextEncodeQwenImageEdit")

            if qwen_node_id:
                print(f"📝 Setting prompt for Qwen Node {qwen_node_id}: {prompt_text}")
                workflow[qwen_node_id]["inputs"]["prompt"] = prompt_text
            else:
                # Fallback: Check for standard CLIPTextEncode if Qwen isn't found
                print(
                    "⚠️ TextEncodeQwenImageEdit not found, checking for CLIPTextEncode..."
                )
                clip_id = self._find_node_by_class(workflow, "CLIPTextEncode")
                if clip_id:
                    workflow[clip_id]["inputs"]["text"] = prompt_text

        # Optional: Randomize seed if KSampler exists to ensure variation
        sampler_node_id = self._find_node_by_class(workflow, "KSampler")
        if sampler_node_id:
            workflow[sampler_node_id]["inputs"]["seed"] = np.random.randint(
                1, 1000000000
            )

        # 3. Queue Prompt
        print(f"🚀 Queuing workflow...")
        payload = {"prompt": workflow}
        resp = requests.post(f"{self.base_url}/prompt", json=payload)
        resp.raise_for_status()
        prompt_id = resp.json()["prompt_id"]

        # 4. Wait for completion (Polling)
        print(f"⏳ Waiting for ComfyUI (ID: {prompt_id})...")
        while True:
            history_resp = requests.get(f"{self.base_url}/history/{prompt_id}")
            history = history_resp.json()

            if prompt_id in history:
                # Execution finished
                outputs = history[prompt_id]["outputs"]
                break

            time.sleep(1)  # Wait 1 second before next poll

        # 5. Retrieve Images
        # We look for any output that has 'images' (SaveImage or PreviewImage nodes)
        output_images = []
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for img_info in node_output["images"]:
                    output_images.append(img_info)

        if not output_images:
            raise RuntimeError("Workflow finished but returned no images.")

        # 6. Download first result
        target_img = output_images[0]
        filename = target_img["filename"]
        subfolder = target_img["subfolder"]
        folder_type = target_img["type"]

        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        print(f"📥 Downloading result: {filename}")

        view_resp = requests.get(f"{self.base_url}/view", params=params)
        view_resp.raise_for_status()

        # 7. Decode back to Numpy
        img_array = np.frombuffer(view_resp.content, np.uint8)
        final_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        return final_image